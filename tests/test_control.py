"""Control events and switch gating, on synthetic arrays only."""

import numpy as np

from src.possession.control import AWAY, HOME, ControlEvent, control_events, switch_evidence

N = 200


def far():
    return np.full(N, 20.0)


def calm():
    return np.full(N, 1.0)


def on_ball(start, length):
    d = far()
    d[start : start + length] = 0.5
    return d


def test_half_second_at_feet_is_a_control_event():
    events = control_events(on_ball(50, 12), far(), calm(), calm())
    assert events == [ControlEvent(HOME, 50, 62)]


def test_a_touch_shorter_than_half_a_second_is_not_control():
    assert control_events(on_ball(50, 11), far(), calm(), calm()) == []


def test_ball_flying_past_a_nearby_player_is_not_control():
    fast = np.full(N, 12.0)
    assert control_events(on_ball(50, 40), far(), fast, calm()) == []


def test_nearest_player_beyond_control_radius_is_not_control():
    d = far()
    d[50:90] = 2.0
    assert control_events(d, far(), calm(), calm()) == []


def test_missing_values_are_not_control():
    d = on_ball(50, 40)
    rel = calm()
    rel[50:90] = np.nan
    assert control_events(d, np.full(N, np.nan), rel, calm()) == []


def test_a_short_tracking_flicker_does_not_split_an_event():
    d = on_ball(50, 30)
    d[60:62] = 20.0
    assert control_events(d, far(), calm(), calm()) == [ControlEvent(HOME, 50, 80)]


def test_in_a_duel_only_the_closer_team_can_hold_control():
    home, away = on_ball(50, 40), on_ball(50, 40)
    away[50:90] = 0.3
    assert control_events(home, away, calm(), calm()) == [ControlEvent(AWAY, 50, 90)]


def test_switch_evidence_opens_the_gate_only_during_the_event():
    llr, open_home, open_away = switch_evidence([ControlEvent(AWAY, 10, 22)], 40)
    assert llr[10] < 0 and llr[9] == 0 and llr[22] == 0
    assert open_away[10:22].all() and open_away.sum() == 12
    assert not open_home.any()
