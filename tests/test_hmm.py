"""Gated Viterbi and player-relative ball speed, on synthetic arrays only."""

import numpy as np

from src.possession import control, features, hmm


def emissions(n, llr):
    """Live play: no D segment anywhere."""
    return hmm.log_emissions(llr, np.zeros(n), np.zeros(n, dtype=bool))


def test_dead_is_unreachable_outside_a_segment_so_it_cannot_bypass_the_gate():
    log = hmm.log_emissions(np.zeros(4), np.full(4, 0.9), np.array([False, False, True, True]))
    assert (log[:2, hmm.D] == hmm.NEVER).all()
    assert (log[2:, hmm.D] > log[2:, hmm.H]).all()


def test_a_restart_after_a_stoppage_may_change_the_owner():
    n = 300
    llr = np.zeros(n)
    llr[200:225] = -control.LLR_CONTROL
    in_segment = np.zeros(n, dtype=bool)
    in_segment[100:200] = True
    log = hmm.log_emissions(llr, np.zeros(n), in_segment)
    closed = np.zeros(n, dtype=bool)
    path = hmm.viterbi(start_home(log), closed, closed)
    assert (path[:100] == hmm.H).all() and (path[100:200] == hmm.D).all()
    assert (path[200:] == hmm.A).all()


def start_home(log):
    log[:5, hmm.H] += 50.0
    return log


def test_no_team_switch_while_the_gate_is_closed_however_strong_the_evidence():
    n = 200
    llr = np.zeros(n)
    llr[100:] = -2.0
    closed = np.zeros(n, dtype=bool)
    path = hmm.viterbi(start_home(emissions(n, llr)), closed, closed)
    assert (path == hmm.H).all()


def test_control_event_of_half_a_second_switches_owner_at_its_start():
    n = 200
    llr = np.zeros(n)
    llr[100:112] = -hmm.control.LLR_CONTROL
    open_away = np.zeros(n, dtype=bool)
    open_away[100:112] = True
    path = hmm.viterbi(start_home(emissions(n, llr)), np.zeros(n, dtype=bool), open_away)
    assert (path[:100] == hmm.H).all() and (path[100:] == hmm.A).all()


def test_lone_touch_between_two_home_controls_does_not_flip_possession():
    n = 300
    llr = np.zeros(n)
    llr[80:120] = hmm.control.LLR_CONTROL
    llr[140:152] = -hmm.control.LLR_CONTROL
    llr[160:200] = hmm.control.LLR_CONTROL
    open_home = llr > 0
    open_away = llr < 0
    path = hmm.viterbi(start_home(emissions(n, llr)), open_home, open_away)
    assert (path == hmm.H).all()


def test_relative_speed_is_zero_when_ball_and_player_move_together():
    n, half = 60, features.SPEED_HALF_SPAN
    t = np.arange(n, dtype=np.float64) / features.FPS
    ball_x, ball_y = 5.0 * t, np.zeros(n)
    px, py = (5.0 * t + 0.5)[:, None], np.zeros((n, 1))
    rel = features.relative_speed(px, py, ball_x, ball_y, np.zeros(n, dtype=np.int64))
    assert np.nanmax(np.abs(rel)) < 1e-9
    assert np.isnan(rel[:half]).all() and np.isfinite(rel[half:-half]).all()


def test_relative_speed_sees_a_ball_passing_a_standing_player():
    n = 60
    t = np.arange(n, dtype=np.float64) / features.FPS
    rel = features.relative_speed(np.zeros((n, 1)), np.zeros((n, 1)), 15.0 * t, np.zeros(n),
                                  np.zeros(n, dtype=np.int64))
    assert abs(np.nanmedian(rel) - 15.0) < 1e-9


def test_nearest_slot_is_reported_with_its_distance():
    px = np.array([[0.0, 3.0, np.nan]])
    py = np.zeros((1, 3))
    distance, slot = features.nearest_player(px, py, np.array([2.5]), np.zeros(1),
                                             np.array([True, True, True]))
    assert distance[0] == 0.5 and slot[0] == 1
