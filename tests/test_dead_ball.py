"""Restart finder, restart team and dead-ball evidence, on synthetic arrays only."""

import numpy as np

from src.possession import dead_ball
from src.possession.control import AWAY, HOME

PITCH = (105.0, 68.0)


def speed_with_kick(n=400, kick=200):
    speed = np.full(n, 0.3)
    speed[kick:] = 10.0
    return speed


def test_restart_is_the_first_sharp_acceleration_after_a_slow_period():
    assert dead_ball.find_restart(speed_with_kick(), 100, 350) == 200


def test_a_ramp_of_exactly_half_a_second_is_still_a_restart():
    speed = speed_with_kick()
    speed[200:212] = 4.0  # neither calm nor fast for 12 frames
    assert dead_ball.find_restart(speed, 100, 350) == 200


def test_a_ramp_longer_than_half_a_second_is_not_a_restart():
    speed = speed_with_kick()
    speed[200:213] = 4.0
    assert dead_ball.find_restart(speed, 100, 350) is None


def test_restart_needs_a_slow_period_first():
    assert dead_ball.find_restart(np.full(400, 10.0), 100, 350) is None


def test_a_ball_that_never_accelerates_has_no_restart():
    assert dead_ball.find_restart(np.full(400, 0.3), 100, 350) is None


def test_earlier_open_play_acceleration_without_a_slow_period_is_skipped():
    speed = speed_with_kick()
    speed[110:113] = 1.0  # a brief dip is not a stoppage
    speed[:110] = 9.0
    speed[113:200] = 0.3
    speed[113:125] = 9.0  # re-acceleration after the dip
    assert dead_ball.find_restart(speed, 100, 350) == 200


def test_unseen_ball_counts_as_part_of_the_slow_period():
    speed = speed_with_kick()
    speed[150:200] = np.nan
    assert dead_ball.find_restart(speed, 100, 350) == 200


def test_throw_in_team_is_the_team_nearest_the_ball_at_the_restart():
    d_home, d_away = np.full(100, 9.0), np.full(100, 9.0)
    d_home[40:52] = 0.6
    assert dead_ball.infer_restart_team(d_home, d_away, 50) == HOME
    assert dead_ball.infer_restart_team(d_away, d_home, 50) == AWAY


def test_restart_team_is_unknown_without_a_ball():
    missing = np.full(100, np.nan)
    assert dead_ball.infer_restart_team(missing, missing, 50) is None


def test_ball_lost_at_the_touchline_is_dead_until_it_reappears():
    n = 300
    x, y = np.full(n, 50.0), np.full(n, 67.5)
    x[100:200] = y[100:200] = np.nan
    evidence = dead_ball.edge_evidence(x, y, np.full(n, 3.0), PITCH)
    assert evidence[100:200].all() and evidence[:100].sum() == 0 and evidence[200:].sum() == 0


def test_ball_lost_in_midfield_is_not_edge_evidence():
    n = 300
    x, y = np.full(n, 50.0), np.full(n, 30.0)
    x[100:200] = y[100:200] = np.nan
    assert dead_ball.edge_evidence(x, y, np.full(n, 3.0), PITCH).sum() == 0


def test_ball_stopped_at_the_goal_line_is_edge_evidence():
    n = 300
    x, y = np.full(n, 104.5), np.full(n, 30.0)
    speed = np.full(n, 6.0)
    speed[100:160] = 0.1
    evidence = dead_ball.edge_evidence(x, y, speed, PITCH)
    assert evidence[100:160].all() and evidence[:100].sum() == 0


def test_ball_running_along_the_line_is_not_edge_evidence():
    n = 300
    x, y = np.linspace(10, 90, n), np.full(n, 67.5)
    assert dead_ball.edge_evidence(x, y, np.full(n, 6.0), PITCH).sum() == 0


def test_still_ball_needs_a_second_and_a_half():
    speed = np.full(300, 6.0)
    speed[100:136] = 0.1  # 36 frames: under 1.5 s
    speed[200:237] = 0.1  # 37 frames: long enough
    slow = np.ones(300)
    evidence = dead_ball.still_ball_evidence(speed, slow)
    assert evidence[100:136].sum() == 0 and evidence[200:237].all()


def test_still_ball_with_players_running_is_not_evidence():
    speed = np.full(300, 0.1)
    assert dead_ball.still_ball_evidence(speed, np.zeros(300)).sum() == 0


def test_short_stoppage_of_one_second_survives_and_shorter_does_not():
    score = np.zeros(400)
    score[100:125] = 1.0
    score[300:320] = 1.0
    mask = dead_ball.dead_segments(score, [(1, 0, 399)])
    assert mask[100:125].all() and mask[300:320].sum() == 0
