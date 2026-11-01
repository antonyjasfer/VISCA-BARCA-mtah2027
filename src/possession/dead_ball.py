"""Dead-ball score and D segments.

The score is a noisy-OR of independent pieces of evidence, each in [0, 1] and each weighted below
1 so that no single soft cue (slow players, a still ball, a set-piece tag) can declare the ball
dead on its own: those need a second source to agree. Only "ball outside the pitch", the
pre-kickoff rule and the combined "ball gone while players are walking" cue are strong enough to
stand alone.

All parameters come from football reasoning plus label-free whole-match distributions (for
example the quantiles of collective speed); none was chosen by looking at a ground truth. See
METHOD.md for the reasoning behind each constant.
"""

import numpy as np
import pandas as pd

from .features import FPS

# --- evidence weights ---------------------------------------------------------------------------
W_OUT = 0.90        # ball beyond a line for >= 0.5 s: dead by the laws of the game
W_ABSENT_SLOW = 0.70  # ball unseen for seconds while everyone walks: a stoppage, not an aerial duel
W_SLOW = 0.45       # walking pace alone: also happens in slow build-up -> needs a partner
W_STATIC = 0.40     # ball motionless for ~1 s: also a goalkeeper holding it -> needs a partner
W_ANCHOR = 0.40     # a restart is coming (Smart Tagging) -> needs a partner
DEAD_THRESHOLD = 0.5

# --- collective speed: mean speed of tracked players, m/s, median over 3 s -----------------------
# Walking is ~1.4 m/s and a jog ~2.5 m/s. Whole-match quantiles of the mean (players include the
# goalkeeper and resting players) are 1.0 / 1.5 / 2.3 m/s at the 25th / 50th / 75th percentiles,
# so "everyone is walking" means clearly below the median: full evidence at <= 0.8, none at >= 1.6.
SPEED_WINDOW = 3 * FPS
SLOW_FULL, SLOW_NONE = 0.8, 1.6

# --- ball outside the pitch -----------------------------------------------------------------------
OUT_MIN_RUN = FPS // 2          # a run shorter than 0.5 s is tracker noise, not a ball out of play
OUT_HOLD = 10 * FPS             # carry the "outside" state across a ball gap for up to 10 s: a ball
#                                 that left the pitch is usually lost by the tracker until the restart

# --- long ball absence ------------------------------------------------------------------------------
ABSENT_FROM, ABSENT_FULL = 2 * FPS, 6 * FPS   # evidence ramps from 2 s to 6 s of continuous absence

# --- stationary ball --------------------------------------------------------------------------------
STATIC_SPEED = 0.8              # m/s: tracker jitter on a still ball is ~0.3 m/s at the 0.32 s baseline
STATIC_WINDOW = FPS             # must stay below that for +/-0.5 s

# --- Smart Tagging anchors ----------------------------------------------------------------------
# SET PIECES windows start at the restart; the stoppage precedes it. The 35 s phase windows and the
# 5 s kickoff windows are both anchored on start_frame, as the brief specifies.
ANCHOR_LEAD = 8 * FPS           # dead-ball evidence in the 8 s before a restart (throw-in/goal-kick set-up)
KICKOFF_TYPE = "kick off (start)"

# --- segments ----------------------------------------------------------------------------------------
SCORE_SMOOTH = FPS              # 1 s centred mean on the score
MIN_SEGMENT = 2 * FPS           # shortest stoppage kept: a fast throw-in is ~3-5 s, nothing real is under 2 s
MERGE_GAP = FPS                 # two stoppages less than 1 s apart are one stoppage


def _runs(flag):
    """(start, end_exclusive) of every True run."""
    padded = np.concatenate(([0], flag.astype(np.int8), [0]))
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def ramp_down(values, full, none):
    """1 at or below `full`, 0 at or above `none`, linear between; NaN stays NaN."""
    return np.clip((none - values) / (none - full), 0.0, 1.0)


def collective_slowness(mean_speed):
    smoothed = pd.Series(mean_speed.astype(np.float64)).rolling(
        SPEED_WINDOW, center=True, min_periods=SPEED_WINDOW // 3).median().to_numpy()
    return np.nan_to_num(ramp_down(smoothed, SLOW_FULL, SLOW_NONE), nan=0.0)


def ball_out_evidence(in_pitch):
    """1 while the ball is (or was last seen) outside the pitch, held across tracking gaps."""
    outside = in_pitch == 0.0
    starts, ends = _runs(outside)
    for s, e in zip(starts, ends):          # drop runs under 0.5 s
        if e - s < OUT_MIN_RUN:
            outside[s:e] = False

    state = pd.Series(np.where(np.isnan(in_pitch), np.nan, outside.astype(np.float64)))
    held = state.ffill(limit=OUT_HOLD).fillna(0.0).to_numpy()
    return held


def absence_evidence(ball_present):
    """Ramp over the length of the continuous run of missing ball that each frame sits in."""
    out = np.zeros(len(ball_present))
    starts, ends = _runs(~ball_present)
    for s, e in zip(starts, ends):
        out[s:e] = np.clip((e - s - ABSENT_FROM) / (ABSENT_FULL - ABSENT_FROM), 0.0, 1.0)
    return out


def static_ball_evidence(ball_speed):
    still = pd.Series((ball_speed < STATIC_SPEED).astype(np.float64)).where(
        ~np.isnan(ball_speed))
    # still for the whole +/-0.5 s window (NaN ball speed counts as "unknown", never as still)
    full = still.rolling(2 * STATIC_WINDOW + 1, center=True, min_periods=2 * STATIC_WINDOW + 1).min()
    return np.nan_to_num(full.to_numpy(), nan=0.0)


def anchor_evidence(n_frames, restarts):
    """1 in the ANCHOR_LEAD frames before each Smart Tagging set-piece start."""
    out = np.zeros(n_frames)
    for frame in restarts:
        out[max(0, frame - ANCHOR_LEAD):frame] = 1.0
    return out


def pre_kickoff(n_frames, blocks, events):
    """Frames from a half's first graded frame to its 'Kick Off (Start)' tag: the ball cannot be live."""
    out = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        starts = [e["start_frame"] for e in events
                  if (e["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE
                  and first <= e["end_frame"] <= last]
        if starts:
            out[first:max(first, min(starts))] = True
    return out


def dead_ball_score(feats, restarts, kickoff_dead):
    """Noisy-OR combination of the evidence, per frame, in [0, 1]."""
    present = ~np.isnan(feats.ball_x)
    slow = collective_slowness(feats.mean_speed)

    evidence = [
        (W_OUT, ball_out_evidence(feats.ball_in_pitch)),
        (W_ABSENT_SLOW, absence_evidence(present) * slow),
        (W_SLOW, slow),
        (W_STATIC, static_ball_evidence(feats.ball_speed)),
        (W_ANCHOR, anchor_evidence(len(present), restarts)),
    ]
    survive = np.ones(len(present))
    for weight, signal in evidence:
        survive *= 1.0 - weight * signal
    score = 1.0 - survive
    score = pd.Series(score).rolling(SCORE_SMOOTH, center=True, min_periods=1).mean().to_numpy().copy()
    score[kickoff_dead] = 1.0
    return score


def dead_segments(score, blocks):
    """Boolean mask of frames inside D segments: score >= threshold, small gaps closed, short runs dropped.
    Works half by half so a segment never bridges halftime."""
    mask = np.zeros(len(score), dtype=bool)
    for _, first, last in blocks:
        part = score[first:last + 1] >= DEAD_THRESHOLD
        starts, ends = _runs(part)
        keep = part.copy()
        for (s0, e0), (s1, e1) in zip(zip(starts[:-1], ends[:-1]), zip(starts[1:], ends[1:])):
            if s1 - e0 <= MERGE_GAP:
                keep[e0:s1] = True
        starts, ends = _runs(keep)
        for s, e in zip(starts, ends):
            if e - s < MIN_SEGMENT:
                keep[s:e] = False
        mask[first:last + 1] = keep
    return mask
