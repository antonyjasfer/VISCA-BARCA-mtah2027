"""Control events: the only thing, besides a restart, that may hand possession to the other team.

CHALLENGE.md: a touch, a deflection or a duel is not a change of possession; a player of the other
team gaining *controlled* possession is. Here control means a player stays on the ball and the ball
moves with them, for long enough that it cannot have been a touch.

Parameters come from football reasoning; see METHOD.md.
"""

from collections import namedtuple

import numpy as np

HOME, AWAY = 0, 1

CONTROL_RADIUS_M = 1.5      # a stride plus the ball: the player can play it without moving to it
MATCH_SPEED_MS = 3.0        # ball velocity within this of the player's: dribble touches sway the
#                             ball 1-3 m/s around the carrier, a passing ball is 10+ m/s off
MIN_CONTROL_FRAMES = 12     # 0.5 s; a first-time touch or deflection is over well before this
BRIDGE_FRAMES = 3           # a flicker of 0.12 s is tracker noise, not a lost ball

# log P(frame | team controls) - log P(frame | other team controls), per control frame. A minimum
# event gives 12 x 0.5 = 6 nats: above one team switch (hmm.P_SWITCH, 3.9 nats) but below the two
# switches needed to flip and flip back (7.8), so a lone minimum-length event between two controls
# of the same team stays a touch, while the same event with no answer changes the owner.
LLR_CONTROL = 0.5

MAX_SILENT_GAP = 50         # 2 s: a longer ball gap inside a possession becomes X

ControlEvent = namedtuple("ControlEvent", "team start end")     # frames, end exclusive


def runs(flag):
    """(start, end_exclusive) of every True run, as two arrays."""
    padded = np.concatenate(([0], flag.astype(np.int8), [0]))
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def _team_events(team, flag):
    starts, ends = runs(flag)
    keep = np.ones(len(starts), dtype=bool)
    for i in range(1, len(starts)):
        if starts[i] - ends[i - 1] <= BRIDGE_FRAMES:
            starts[i] = starts[i - 1]
            keep[i - 1] = False
    return [ControlEvent(team, int(s), int(e)) for s, e in zip(starts[keep], ends[keep])
            if e - s >= MIN_CONTROL_FRAMES]


def control_events(d_home, d_away, rel_home, rel_away):
    """Every control event of both teams, in time order.

    In a duel only the closer team can hold a control frame, so a real contest breaks up into
    fragments shorter than an event and produces none. NaN comparisons are False, so a missing ball
    or player is never control.
    """
    far_home = np.nan_to_num(d_home, nan=np.inf)
    far_away = np.nan_to_num(d_away, nan=np.inf)
    home = (far_home <= CONTROL_RADIUS_M) & (rel_home <= MATCH_SPEED_MS) & (far_home <= far_away)
    away = (far_away <= CONTROL_RADIUS_M) & (rel_away <= MATCH_SPEED_MS) & (far_away < far_home)
    events = _team_events(HOME, home) + _team_events(AWAY, away)
    return sorted(events, key=lambda event: event.start)


def switch_evidence(events, n_frames):
    """(llr, open_home, open_away): evidence per frame (+ is home) and where each team may take over."""
    llr = np.zeros(n_frames)
    open_home = np.zeros(n_frames, dtype=bool)
    open_away = np.zeros(n_frames, dtype=bool)
    for event in events:
        home = event.team == HOME
        llr[event.start:event.end] += LLR_CONTROL if home else -LLR_CONTROL
        (open_home if home else open_away)[event.start:event.end] = True
    return llr, open_home, open_away


def abstain_gaps(labels, ball_present, events, blocks):
    """Copy of `labels` with X on long ball gaps strictly inside one possession.

    The scorer repeats the previous label over an X, so a gap with the same team on both sides and
    no opposing control fills back to exactly what was there. A gap touching either end of the
    possession is left alone, so an X can never carry a label across a switch.
    """
    out = labels.copy()
    control = {"H": np.zeros(len(labels), dtype=bool), "A": np.zeros(len(labels), dtype=bool)}
    for event in events:
        control["H" if event.team == HOME else "A"][event.start:event.end] = True
    opposing = {"H": control["A"], "A": control["H"]}

    gap_starts, gap_ends = runs(~ball_present)
    for _, first, last in blocks:
        change = np.flatnonzero(labels[first + 1:last + 1] != labels[first:last]) + first + 1
        run_starts = np.concatenate(([first], change))
        run_ends = np.concatenate((change, [last + 1]))
        for start, end in zip(run_starts, run_ends):
            label = labels[start]
            if label not in opposing:
                continue
            inside = (gap_starts > start) & (gap_ends < end)
            for s, e in zip(gap_starts[inside], gap_ends[inside]):
                if e - s > MAX_SILENT_GAP and not opposing[label][s:e].any():
                    out[s:e] = "X"
    return out
