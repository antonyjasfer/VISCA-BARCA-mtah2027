"""Three-state (H, A, D) hidden Markov model decoded with Viterbi. No training, no labels.

Emissions are hand-specified from football reasoning (METHOD.md); the transition matrix is
deliberately sticky so that a hand-over needs about a second of sustained evidence.
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

# --- transitions (per frame, 25 fps) ---------------------------------------------------------------
# A possession lasts 10-20 s and ends in a hand-over (most often) or a stoppage; a stoppage lasts
# 5-15 s. Exit rates: alive states leave with p = 0.0025/frame (mean 16 s), of which 60% to the
# other team and 40% to a stoppage; D leaves with p = 0.004/frame (mean 10 s).
# Switching teams therefore costs -ln(0.0015) = 6.5 nats, which the ownership evidence below
# (at most 0.25 nats/frame) pays off after >= 26 frames (~1 s) of unambiguous control.
P_HA = 0.0015
P_HD = 0.0010
P_DX = 0.0020  # D -> H and D -> A each
TRANSITION = np.array([
    [1 - P_HA - P_HD, P_HA, P_HD],
    [P_HA, 1 - P_HA - P_HD, P_HD],
    [P_DX, P_DX, 1 - 2 * P_DX],
])

# --- ownership emission ---------------------------------------------------------------------------
LLR_MAX = 0.25      # nats per frame at full-strength evidence
CLOSER_BY = 3.0     # m: a team nearer by this much (or more) gives the full LLR
CONTROL_NEAR = 2.5  # m: nearest player inside this is "on the ball": full weight
CONTROL_FAR = 8.0   # m: nearest player beyond this means a loose ball or a pass in flight: zero weight
FLIGHT_SLOW = 8.0   # m/s: faster than a sprinting dribbler ...
FLIGHT_FAST = 16.0  # m/s: ... and by here the ball is certainly passed or kicked, nobody controls it

# --- dead-ball emission ---------------------------------------------------------------------------
P_DEAD_IN_SEGMENT = 0.90     # inside a D segment (dead_ball.dead_segments)
P_DEAD_OUTSIDE_CAP = 0.30    # outside a segment the score can raise P(dead) at most this far
P_DEAD_FLOOR = 0.01


def ownership_llr(d_home, d_away, ball_speed):
    """log P(obs | home controls) - log P(obs | away controls), per frame; 0 where uninformative.

    Sign: positive when home is nearer. Weighted by how plausible it is that anyone controls the
    ball at all (nearest player close, ball not in flight)."""
    d_home = np.where(np.isnan(d_home), 60.0, d_home).astype(np.float64)
    d_away = np.where(np.isnan(d_away), 60.0, d_away).astype(np.float64)
    nearest = np.minimum(d_home, d_away)
    closer = np.clip((d_away - d_home) / CLOSER_BY, -1.0, 1.0)

    near = np.clip((CONTROL_FAR - nearest) / (CONTROL_FAR - CONTROL_NEAR), 0.0, 1.0)
    speed = np.where(np.isnan(ball_speed), 0.0, ball_speed)
    grounded = np.clip((FLIGHT_FAST - speed) / (FLIGHT_FAST - FLIGHT_SLOW), 0.0, 1.0)

    llr = LLR_MAX * closer * near * grounded
    llr[np.isnan(d_home + d_away)] = 0.0
    return llr


def dead_probability(score, in_segment):
    p = np.where(in_segment, np.maximum(score, P_DEAD_IN_SEGMENT),
                 np.minimum(score, P_DEAD_OUTSIDE_CAP))
    return np.clip(p, P_DEAD_FLOOR, 1.0 - P_DEAD_FLOOR)


def log_emissions(feats, score, in_segment, ball_present):
    llr = ownership_llr(feats.d_home, feats.d_away, feats.ball_speed)
    llr[~ball_present] = 0.0      # no ball: no ownership evidence; the HMM carries the last owner
    p_dead = dead_probability(score, in_segment)
    log = np.empty((len(p_dead), 3))
    log[:, H] = np.log1p(-p_dead) + llr / 2.0
    log[:, A] = np.log1p(-p_dead) - llr / 2.0
    log[:, D] = np.log(p_dead)
    return log


def viterbi(log_emit, transition=TRANSITION):
    """Most likely state path. Deterministic: ties go to the lower state index (argmax)."""
    n = len(log_emit)
    log_t = np.log(transition)
    delta = log_emit[0] + np.log(1.0 / 3.0)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        scores = delta[:, None] + log_t          # [from, to]
        best = scores.argmax(axis=0)
        back[t] = best
        delta = scores[best, np.arange(3)] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(feats, score, in_segment, blocks):
    """Labels for every frame: Viterbi inside each graded half, 'X' elsewhere."""
    ball_present = ~np.isnan(feats.ball_x)
    log_emit = log_emissions(feats, score, in_segment, ball_present)
    labels = np.full(len(score), "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = LABELS[viterbi(log_emit[first:last + 1])]
    return labels
