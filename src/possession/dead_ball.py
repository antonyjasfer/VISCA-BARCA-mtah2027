"""Dead-ball score, D segments and restarts.

The score is a noisy-OR of independent pieces of evidence, each in [0, 1]. Soft cues (slow players
alone, a set-piece tag with no restart found) are weighted below the threshold, so they need a
second source to agree; the rest are strong enough to stand alone.

All parameters come from football reasoning plus label-free whole-match distributions; none was
chosen by looking at a ground truth. See METHOD.md for the reasoning behind each constant.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from .control import AWAY, HOME, runs
from .features import FPS

# --- evidence weights ---------------------------------------------------------------------------
W_OUT = 0.90            # ball beyond a line for >= 0.5 s: dead by the laws of the game
W_RESTART = 0.85        # the run-up to a located restart
W_STILL = 0.80          # ball still for 1.5 s while players are slow
W_EDGE = 0.75           # ball lost or stopped at a line
W_ABSENT_SLOW = 0.70    # ball unseen for seconds while everyone walks
W_SLOW = 0.45           # walking pace alone: also slow build-up -> needs a partner
W_ANCHOR = 0.40         # set-piece tag where no restart was found -> needs a partner
DEAD_THRESHOLD = 0.5

# --- collective speed: mean speed of tracked players, m/s, median over 3 s -----------------------
# Effective playing time of 52-58 of 106 minutes puts the dead share near 47%. Whole-match quantiles
# of this signal are 1.0 / 1.5 / 2.3 m/s at the 25th / 50th / 75th percentiles, and the aggregate
# profile around set-piece tags sits at 1.0-1.3 in stoppages and 1.7-2.4 after them. So the ramp
# is centred at 1.4: full evidence at <= 1.0, none at >= 1.8.
SPEED_WINDOW = 3 * FPS
SLOW_FULL, SLOW_NONE = 1.0, 1.8

# --- ball outside the pitch -----------------------------------------------------------------------
OUT_MIN_RUN = FPS // 2          # a run shorter than 0.5 s is tracker noise
OUT_HOLD = 10 * FPS             # carry "outside" across a tracking gap for up to 10 s

# --- ball lost or stopped at a line ---------------------------------------------------------------
EDGE_MARGIN_M = 1.0             # within a metre of a line, or over it
EDGE_MIN = FPS                  # lost or stopped for at least 1 s
EDGE_HOLD = 15 * FPS            # a lost ball counts for at most 15 s, the length of a slow throw-in

# --- long ball absence ------------------------------------------------------------------------------
ABSENT_FROM, ABSENT_FULL = 2 * FPS, 6 * FPS   # evidence ramps from 2 s to 6 s of continuous absence

# --- stationary ball --------------------------------------------------------------------------------
STILL_SPEED = 0.8               # m/s: jitter on a still ball is ~0.3 m/s over the 0.32 s baseline
STILL_MIN = 37                  # 1.5 s: a goalkeeper's bounce or a pause at the feet is shorter

# --- restarts inside Smart Tagging SET PIECES -----------------------------------------------------
# The aggregate profile around tag starts shows the stoppage from about -15 s to +5 s and play
# back by +10 s, so the tag start falls inside the stoppage and the restart follows it.
SET_PIECE_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"
GOAL_KICKOFF_TYPE = "kick off (after goal)"
TEAM_SIDES = {"barcelona": HOME, "atletic club": AWAY}
RESTART_SEARCH = 15 * FPS       # look this far past the tag start
RESTART_SLOW = 2.0              # m/s: a ball being placed or carried to the spot
RESTART_FAST = 6.0              # m/s: the softest real restart, a short throw, is faster
RESTART_CALM_RUN = FPS          # at least 1 s of slow or unseen ball before the kick
RESTART_ACCEL_SPAN = FPS // 2   # slow to fast inside 0.5 s: struck, not jogged up to speed
RESTART_LEAD = 10 * FPS         # dead this long before the tag start
GOAL_LEAD = 30 * FPS            # a goal: celebration and walk back
RESTART_LIVE = 3 * FPS          # the ball is live this long after a restart, whatever else says
ANCHOR_AFTER = 5 * FPS          # no restart found: the stoppage runs to about +5 s
TEAM_BEFORE, TEAM_AFTER = 12, 2  # frames around the restart in which the taker is on the ball
SEGMENT_TEAM_REACH = FPS        # after a stoppage with no tag: look 1 s into play for the owner

# --- segments ----------------------------------------------------------------------------------------
SCORE_SMOOTH = FPS              # 1 s centred mean on the score
MIN_SEGMENT = FPS               # shortest stoppage kept
MERGE_GAP = FPS                 # two stoppages less than 1 s apart are one stoppage

Restart = namedtuple("Restart", "frame team dead_from")
Unlocated = namedtuple("Unlocated", "start dead_from")


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
    starts, ends = runs(outside)
    for s, e in zip(starts, ends):
        if e - s < OUT_MIN_RUN:
            outside[s:e] = False
    state = pd.Series(np.where(np.isnan(in_pitch), np.nan, outside.astype(np.float64)))
    return state.ffill(limit=OUT_HOLD).fillna(0.0).to_numpy()


def edge_evidence(ball_x, ball_y, ball_speed, pitch):
    """1 where the ball was lost at a line, or has stopped there."""
    length, width = pitch
    with np.errstate(invalid="ignore"):
        near = ((ball_x < EDGE_MARGIN_M) | (ball_x > length - EDGE_MARGIN_M)
                | (ball_y < EDGE_MARGIN_M) | (ball_y > width - EDGE_MARGIN_M))
        still = ball_speed < STILL_SPEED
    out = np.zeros(len(ball_x))

    starts, ends = runs(np.isnan(ball_x))
    for s, e in zip(starts, ends):
        if s > 0 and e - s >= EDGE_MIN and near[s - 1]:
            out[s:min(e, s + EDGE_HOLD)] = 1.0

    starts, ends = runs(near & still)
    for s, e in zip(starts, ends):
        if e - s >= EDGE_MIN:
            out[s:e] = 1.0
    return out


def absence_evidence(ball_present):
    """Ramp over the length of the continuous run of missing ball that each frame sits in."""
    out = np.zeros(len(ball_present))
    starts, ends = runs(~ball_present)
    for s, e in zip(starts, ends):
        out[s:e] = np.clip((e - s - ABSENT_FROM) / (ABSENT_FULL - ABSENT_FROM), 0.0, 1.0)
    return out


def still_ball_evidence(ball_speed, slow):
    """Collective slowness on frames where the ball has been motionless for at least 1.5 s."""
    with np.errstate(invalid="ignore"):
        still = ball_speed < STILL_SPEED
    out = np.zeros(len(ball_speed))
    starts, ends = runs(still)
    for s, e in zip(starts, ends):
        if e - s >= STILL_MIN:
            out[s:e] = slow[s:e]
    return out


def find_restart(ball_speed, start, stop):
    """First sharp acceleration after a slow period in [start, stop), or None.

    "Slow" includes an unseen ball: the tracker usually loses it during a stoppage. Returns the
    first frame of the acceleration.
    """
    with np.errstate(invalid="ignore"):
        fast = ball_speed >= RESTART_FAST
        calm = ~(ball_speed >= RESTART_SLOW)
    for i in range(max(start, 1), min(stop, len(ball_speed))):
        if not fast[i] or fast[i - 1]:
            continue
        kick = i
        while kick > i - RESTART_ACCEL_SPAN and kick > 0 and not calm[kick - 1]:
            kick -= 1
        if kick < RESTART_CALM_RUN or not calm[kick - 1]:
            continue
        if calm[kick - RESTART_CALM_RUN:kick].all():
            return kick
    return None


def infer_restart_team(d_home, d_away, frame, before=TEAM_BEFORE, after=TEAM_AFTER):
    """Team of the player nearest the ball around `frame`, or None if the ball is not seen."""
    low, high = max(0, frame - before), frame + after + 1
    home, away = d_home[low:high], d_away[low:high]
    if np.all(np.isnan(home)) or np.all(np.isnan(away)):
        return None
    return HOME if np.nanmedian(home) <= np.nanmedian(away) else AWAY


def locate_restarts(events, feats):
    """(restarts, unlocated) for every SET PIECES tag. Tagged team, else nearest team at the restart."""
    restarts, unlocated = [], []
    for event in events:
        if (event["code"] or "").strip().lower() != SET_PIECE_CODE:
            continue
        start = event["start_frame"]
        kind = (event["tags"].get("Type") or "").strip().lower()
        lead = GOAL_LEAD if kind == GOAL_KICKOFF_TYPE else RESTART_LEAD
        frame = find_restart(feats.ball_speed, start, start + RESTART_SEARCH)
        if frame is None:
            unlocated.append(Unlocated(start, max(0, start - lead)))
            continue
        team = TEAM_SIDES.get((event["tags"].get("Team") or "").strip().lower())
        if team is None:
            team = infer_restart_team(feats.d_home, feats.d_away, frame)
        restarts.append(Restart(frame, team, max(0, start - lead)))
    return restarts, unlocated


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


def dead_ball_score(feats, restarts, unlocated, kickoff_dead, pitch):
    """Noisy-OR combination of the evidence, per frame, in [0, 1]."""
    n = len(feats.ball_x)
    present = ~np.isnan(feats.ball_x)
    slow = collective_slowness(feats.mean_speed)

    run_up, anchor = np.zeros(n), np.zeros(n)
    for restart in restarts:
        run_up[restart.dead_from:restart.frame] = 1.0
    for tag in unlocated:
        anchor[tag.dead_from:tag.start + ANCHOR_AFTER] = 1.0

    evidence = [
        (W_OUT, ball_out_evidence(feats.ball_in_pitch)),
        (W_RESTART, run_up),
        (W_STILL, still_ball_evidence(feats.ball_speed, slow)),
        (W_EDGE, edge_evidence(feats.ball_x, feats.ball_y, feats.ball_speed, pitch)),
        (W_ABSENT_SLOW, absence_evidence(present) * slow),
        (W_SLOW, slow),
        (W_ANCHOR, anchor),
    ]
    survive = np.ones(n)
    for weight, signal in evidence:
        survive *= 1.0 - weight * signal
    score = pd.Series(1.0 - survive).rolling(
        SCORE_SMOOTH, center=True, min_periods=1).mean().to_numpy().copy()

    score[kickoff_dead] = 1.0
    for restart in restarts:
        score[restart.frame:restart.frame + RESTART_LIVE] = 0.0
    return score


def dead_segments(score, blocks):
    """Boolean mask of frames inside D segments: score >= threshold, small gaps closed, short runs
    dropped. Works half by half so a segment never bridges halftime."""
    mask = np.zeros(len(score), dtype=bool)
    for _, first, last in blocks:
        keep = score[first:last + 1] >= DEAD_THRESHOLD
        starts, ends = runs(keep)
        for s, e in zip(starts[1:], ends[:-1]):
            if s - e <= MERGE_GAP:
                keep[e:s] = True
        starts, ends = runs(keep)
        for s, e in zip(starts, ends):
            if e - s < MIN_SEGMENT:
                keep[s:e] = False
        mask[first:last + 1] = keep
    return mask


def segment_restarts(in_segment, restarts, feats):
    """A Restart at the end of every D segment that no located restart already explains."""
    known = np.array(sorted(r.frame for r in restarts), dtype=np.int64)
    out = []
    _, ends = runs(in_segment)
    for end in ends:
        if np.any(np.abs(known - end) <= 2 * FPS):
            continue
        team = infer_restart_team(feats.d_home, feats.d_away, int(end), 0, SEGMENT_TEAM_REACH)
        out.append(Restart(int(end), team, int(end)))
    return out
