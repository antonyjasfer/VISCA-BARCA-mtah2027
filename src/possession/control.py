"""Control events: the only thing, besides a restart, that may hand possession to the other team.

CHALLENGE.md: a touch, a deflection or a duel is not a change of possession; a player of the other
team gaining *controlled* possession is. Here control means a player stays on the ball and the ball
moves with them, for long enough that it cannot have been a touch.

Parameters come from football reasoning; see METHOD.md.
"""

from collections import namedtuple

import numpy as np

HOME, AWAY = 0, 1

CONTROL_RADIUS_M = 1.5  # a stride plus the ball: the player can play it without moving to it
MATCH_SPEED_MS = 3.0  # ball velocity within this of the player's: dribble touches sway the
#                             ball 1-3 m/s around the carrier, a passing ball is 10+ m/s off
MIN_CONTROL_FRAMES = 12  # 0.5 s; a first-time touch or deflection is over well before this
BRIDGE_FRAMES = 3  # a flicker of 0.12 s is tracker noise, not a lost ball

# log P(frame | team controls) - log P(frame | other team controls), per control frame. A minimum
# event gives 12 x 0.5 = 6 nats: above one team switch (hmm.P_SWITCH, 3.9 nats) but below the two
# switches needed to flip and flip back (7.8), so a lone minimum-length event between two controls
# of the same team stays a touch, while the same event with no answer changes the owner.
LLR_CONTROL = 0.5

MAX_TOUCH_FRAMES = 37  # under 1.5 s: a touch, a block and the ball coming straight back. A team
#                             that really wins the ball needs a receive plus one action, about 1.5 s

ControlEvent = namedtuple("ControlEvent", "team start end")  # frames, end exclusive


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
    return [
        ControlEvent(team, int(s), int(e))
        for s, e in zip(starts[keep], ends[keep])
        if e - s >= MIN_CONTROL_FRAMES
    ]


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
        llr[event.start : event.end] += LLR_CONTROL if home else -LLR_CONTROL
        (open_home if home else open_away)[event.start : event.end] = True
    return llr, open_home, open_away


def merge_touches(labels, blocks):
    """Copy of `labels` with every short live run between two runs of the other team relabelled.

    A rulebook touch or deflection is not a change of possession, so in X-Y-X with no D in between,
    a Y of under 1.5 s goes back to X. Repeats left to right until nothing changes, so a ping-pong
    collapses into the possession around it. Runs next to a D, or at the edge of a half, are left
    alone: a stoppage ends the sequence.
    """
    out = labels.copy()
    for _, first, last in blocks:
        part = out[first : last + 1]
        changed = True
        while changed:
            changed = False
            starts = np.flatnonzero(np.concatenate(([True], part[1:] != part[:-1])))
            ends = np.append(starts[1:], len(part))
            for i in range(1, len(starts) - 1):
                before, here, after = part[starts[i - 1]], part[starts[i]], part[starts[i + 1]]
                if (
                    before == after
                    and {before, here} == {"H", "A"}
                    and ends[i] - starts[i] <= MAX_TOUCH_FRAMES
                ):
                    part[starts[i] : ends[i]] = before
                    changed = True
                    break
    return out
