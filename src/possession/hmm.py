"""Three-state (H, A, D) hidden Markov model decoded with Viterbi. No training, no labels.

Team switches are gated: a direct H -> A is only possible on frames of an away control event or an
away restart, and the mirror for A -> H (control.py, dead_ball.py). The owner can also change across
a D segment, because the end of a stoppage is a restart; D -> H and D -> A are not gated, and where
the restarting team is unknown the owner after the stoppage is a tie, which goes to H.
"""

import numpy as np

from . import control

H, A, D = control.HOME, control.AWAY, 2
LABELS = np.array(["H", "A", "D"])

# --- transitions (per frame, 25 fps) ---------------------------------------------------------------
# P_SWITCH applies only on open frames. It was 0.0015 on every frame in v1 (6.5 nats); with the gate
# doing the filtering a control event needs far less, so a switch now costs -ln(0.02) = 3.9 nats and
# self-persistence on an open frame falls from 0.9975 to 0.9785.
P_SWITCH = 0.02
# About 110 stoppages in about 55 live minutes: one every 30 s (750 frames).
P_TO_DEAD = 0.0015
# About 50 dead minutes over those 110 stoppages: 27 s each (680 frames), split between the teams.
P_FROM_DEAD = 0.00075

RESTART_SPAN = 25       # 1 s after a restart: the restarting team holds the gate and the evidence

# --- dead-ball emission ---------------------------------------------------------------------------
# D exists only inside a D segment. If it were reachable anywhere, a one-frame H -> D -> A would be a
# way round the switch gate; a stoppage is what makes a change of owner at its end a restart.
P_DEAD_IN_SEGMENT = 0.90     # at least this inside a D segment (dead_ball.dead_segments)
P_DEAD_CEILING = 0.99

NEVER = -1e9


def log_emissions(llr, score, in_segment):
    p_dead = np.where(in_segment, np.clip(score, P_DEAD_IN_SEGMENT, P_DEAD_CEILING), 0.0)
    log = np.empty((len(p_dead), 3))
    log[:, H] = np.log1p(-p_dead) + llr / 2.0
    log[:, A] = np.log1p(-p_dead) - llr / 2.0
    log[:, D] = np.log(np.where(in_segment, p_dead, 1.0))
    log[~in_segment, D] = NEVER
    return log


def restart_evidence(restarts, n_frames):
    """Same shape as control.switch_evidence: the restarting team takes the ball at the restart."""
    llr = np.zeros(n_frames)
    open_home = np.zeros(n_frames, dtype=bool)
    open_away = np.zeros(n_frames, dtype=bool)
    for restart in restarts:
        if restart.team is None:
            continue
        home = restart.team == control.HOME
        stop = restart.frame + RESTART_SPAN
        llr[restart.frame:stop] += control.LLR_CONTROL if home else -control.LLR_CONTROL
        (open_home if home else open_away)[restart.frame:stop] = True
    return llr, open_home, open_away


def viterbi(log_emit, open_home, open_away):
    """Most likely path. A team switch into frame t needs that team's gate open at t.

    Deterministic: ties go to the lower state index (argmax).
    """
    n = len(log_emit)
    stay = np.log(1.0 - P_TO_DEAD)
    stay_open = np.log(1.0 - P_TO_DEAD - P_SWITCH)
    switch, to_dead = np.log(P_SWITCH), np.log(P_TO_DEAD)
    from_dead, stay_dead = np.log(P_FROM_DEAD), np.log(1.0 - 2.0 * P_FROM_DEAD)

    delta = log_emit[0] + np.log(1.0 / 3.0)
    back = np.zeros((n, 3), dtype=np.int8)
    log_t = np.array([[stay, NEVER, to_dead], [NEVER, stay, to_dead],
                      [from_dead, from_dead, stay_dead]])
    for t in range(1, n):
        # the leaving side of a pair pays for the open gate of the team that may arrive
        log_t[H, H] = stay_open if open_away[t] else stay
        log_t[H, A] = switch if open_away[t] else NEVER
        log_t[A, A] = stay_open if open_home[t] else stay
        log_t[A, H] = switch if open_home[t] else NEVER
        scores = delta[:, None] + log_t          # [from, to]
        best = scores.argmax(axis=0)
        back[t] = best
        delta = scores[best, np.arange(3)] + log_emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(score, in_segment, events, restarts, blocks):
    """Labels for every frame: gated Viterbi inside each graded half, 'X' elsewhere."""
    n = len(score)
    llr_c, open_home_c, open_away_c = control.switch_evidence(events, n)
    llr_r, open_home_r, open_away_r = restart_evidence(restarts, n)
    log_emit = log_emissions(llr_c + llr_r, score, in_segment)
    open_home, open_away = open_home_c | open_home_r, open_away_c | open_away_r

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in blocks:
        span = slice(first, last + 1)
        labels[span] = LABELS[viterbi(log_emit[span], open_home[span], open_away[span])]
    return labels
