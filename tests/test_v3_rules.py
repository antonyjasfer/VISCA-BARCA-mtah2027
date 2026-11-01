"""v3 rules, on synthetic arrays only: no graded X, restart team, touch merge, extra stoppages."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.possession import control, dead_ball, pipeline
from src.possession.control import AWAY, HOME
from src.possession.dead_ball import Restart

PITCH = (105.0, 68.0)
N = 2000


def labels_of(text):
    return np.array(list(text), dtype="<U1")


def tag(code, team, start, end=None, kind=None):
    tags = {"Team": team}
    if kind:
        tags["Type"] = kind
    return {"code": code, "start_frame": start, "end_frame": end or start + 100, "tags": tags}


def restart(frame, team=None):
    return Restart(frame, team, frame - 100, "throw-in", None)


def pitch_feats():
    """Ball in midfield, both teams far away, unless a test says otherwise."""
    return SimpleNamespace(
        ball_x=np.full(N, 50.0),
        ball_y=np.full(N, 30.0),
        ball_in_pitch=np.ones(N),
        d_home=np.full(N, 9.0),
        d_away=np.full(N, 9.0),
    )


def ball_goes_out(feats, frame, stop):
    feats.ball_y[frame:stop] = 69.0
    feats.ball_in_pitch[frame:stop] = 0.0


def segment(start, stop):
    mask = np.zeros(N, dtype=bool)
    mask[start:stop] = True
    return mask


# --- 1. no graded X --------------------------------------------------------------------------------


def test_abstaining_on_graded_frames_is_gone():
    assert not hasattr(control, "abstain_gaps")


def test_check_labels_rejects_x_on_a_graded_frame():
    with pytest.raises(ValueError):
        pipeline.check_labels(labels_of("HXD"), np.ones(3, dtype=bool))


def test_check_labels_rejects_a_label_on_an_ungraded_frame():
    with pytest.raises(ValueError):
        pipeline.check_labels(labels_of("HH"), np.array([True, False]))


def test_check_labels_accepts_a_clean_submission():
    pipeline.check_labels(labels_of("XHADX"), np.array([False, True, True, True, False]))


# --- 2. restart team -------------------------------------------------------------------------------


def test_crossing_is_where_the_ball_leaves_the_pitch():
    feats = pitch_feats()
    ball_goes_out(feats, 500, 600)
    frames = dead_ball.crossing_frames(feats.ball_in_pitch, feats.ball_x, feats.ball_y, PITCH)
    assert list(frames) == [500]


def test_a_flicker_over_the_line_is_not_a_crossing():
    feats = pitch_feats()
    ball_goes_out(feats, 500, 505)
    frames = dead_ball.crossing_frames(feats.ball_in_pitch, feats.ball_x, feats.ball_y, PITCH)
    assert len(frames) == 0


def test_a_ball_lost_at_the_line_is_a_crossing():
    feats = pitch_feats()
    feats.ball_y[:] = 67.5
    feats.ball_x[500:600] = feats.ball_y[500:600] = feats.ball_in_pitch[500:600] = np.nan
    frames = dead_ball.crossing_frames(feats.ball_in_pitch, feats.ball_x, feats.ball_y, PITCH)
    assert list(frames) == [500]


def test_a_ball_lost_in_midfield_is_not_a_crossing():
    feats = pitch_feats()
    feats.ball_x[500:600] = feats.ball_y[500:600] = feats.ball_in_pitch[500:600] = np.nan
    frames = dead_ball.crossing_frames(feats.ball_in_pitch, feats.ball_x, feats.ball_y, PITCH)
    assert len(frames) == 0


def test_last_touch_is_the_team_nearest_in_the_second_before_the_crossing():
    d_home, d_away = np.full(100, 9.0), np.full(100, 9.0)
    d_away[75:80] = 0.4  # the kick that sent it out
    d_home[90:] = 3.0  # a defender closest when it crossed
    assert dead_ball.last_touch_team(d_home, d_away, 95) == AWAY


def test_an_exact_tie_favours_neither_team():
    same = np.full(100, 2.0)
    assert dead_ball.last_touch_team(same, same, 95) is None
    assert dead_ball.infer_restart_team(same, same, 50) is None


def test_last_touch_is_unknown_without_a_ball():
    missing = np.full(100, np.nan)
    assert dead_ball.last_touch_team(missing, missing, 95) is None


def test_ball_out_gives_the_restart_to_the_other_team():
    feats = pitch_feats()
    feats.d_home[480:500] = 0.5
    ball_goes_out(feats, 500, 600)
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)], segment(510, 700), feats, [], PITCH
    )
    assert out[0].team == AWAY and counts["out"] == 1


def test_in_pitch_stoppage_gives_the_restart_to_the_nearest_team():
    feats = pitch_feats()
    feats.d_away[690:702] = 0.5
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)], segment(510, 700), feats, [], PITCH
    )
    assert out[0].team == AWAY and counts["nearest"] == 1


def test_crossing_before_the_stoppage_began_is_a_different_stoppage():
    feats = pitch_feats()
    feats.d_home[280:300] = 0.5
    ball_goes_out(feats, 300, 320)
    feats.d_away[690:702] = 0.5
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)], segment(600, 700), feats, [], PITCH
    )
    assert out[0].team == AWAY and counts["nearest"] == 1


def test_a_tagged_set_piece_team_is_kept():
    feats = pitch_feats()
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)._replace(team=HOME)], segment(510, 700), feats, [], PITCH
    )
    assert out[0].team == HOME and counts["tag"] == 1


def test_fallback_is_the_next_team_tagged_phase():
    feats = pitch_feats()
    feats.d_home[:] = feats.d_away[:] = np.nan
    tags = [
        tag("BUILD UP", "Atletic Club", 100),
        tag("BUILD UP", "Atletic Club", 800),
        tag("PROGRESSION", "Barcelona", 900),
    ]
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)], segment(510, 700), feats, tags, PITCH
    )
    assert out[0].team == AWAY and counts["phase"] == 1


def test_a_defending_tag_names_the_other_team():
    assert (
        dead_ball.next_tagged_team([tag("DEFENDING IN MIDDLE THIRD", "Barcelona", 800)], 700)
        == AWAY
    )


def test_untagged_and_na_phases_are_not_used_as_fallback():
    tags = [tag("BUILD UP", "N/A", 800), tag("BALL IN THE BOX", "Barcelona", 810)]
    assert dead_ball.next_tagged_team(tags, 700) is None


@pytest.mark.parametrize("kind", [dead_ball.KICKOFF_TYPE, dead_ball.GOAL_KICKOFF_TYPE])
def test_a_kickoff_never_uses_the_ball_out_rule(kind):
    feats = pitch_feats()
    feats.d_home[480:500] = 0.5  # a home keeper nearest as the ball crossed the goal line
    ball_goes_out(feats, 500, 600)
    feats.d_home[690:702] = 0.5  # the home side kicks off, having conceded
    start = restart(700)._replace(kind=kind)
    out, counts = dead_ball.resolve_restart_teams([start], segment(510, 700), feats, [], PITCH)
    assert out[0].team == HOME and counts["nearest"] == 1


def test_unknown_restart_carries_the_owner_from_before_the_stoppage():
    labels = labels_of("AAADDDHH")
    out = dead_ball.carry_previous_owner([restart(6)], labels)
    assert out[0].team == AWAY and out[0].rule is None


def test_known_restart_is_not_overwritten_by_the_carry():
    labels = labels_of("AAADDDHH")
    assert dead_ball.carry_previous_owner([restart(6, HOME)], labels)[0].team == HOME


def test_carry_finds_nothing_at_the_start_of_the_recording():
    assert dead_ball.carry_previous_owner([restart(3)], labels_of("DDDHH"))[0].team is None


def test_a_restart_with_no_evidence_stays_unknown_rather_than_home():
    feats = pitch_feats()
    feats.d_home[:] = feats.d_away[:] = np.nan
    out, counts = dead_ball.resolve_restart_teams(
        [restart(700)], segment(510, 700), feats, [], PITCH
    )
    assert out[0].team is None and counts["unknown"] == 1


# --- 3. touch merge --------------------------------------------------------------------------------


def run_labels(*pairs):
    return np.concatenate([np.full(length, label, dtype="<U1") for label, length in pairs])


def whole(labels):
    return [(1, 0, len(labels) - 1)]


def test_a_short_run_between_two_runs_of_the_other_team_is_a_touch():
    labels = run_labels(("H", 100), ("A", 37), ("H", 100))
    out = control.merge_touches(labels, whole(labels))
    assert (out == "H").all()


def test_a_run_of_a_second_and_a_half_or_more_stays():
    labels = run_labels(("H", 100), ("A", 38), ("H", 100))
    out = control.merge_touches(labels, whole(labels))
    assert (out == labels).all()


def test_no_merge_across_a_dead_ball():
    labels = run_labels(("H", 100), ("D", 30), ("A", 20), ("H", 100))
    out = control.merge_touches(labels, whole(labels))
    assert (out == labels).all()


def test_no_merge_when_the_short_run_ends_in_a_dead_ball():
    labels = run_labels(("H", 100), ("A", 20), ("D", 100), ("H", 50))
    out = control.merge_touches(labels, whole(labels))
    assert (out == labels).all()


def test_short_run_at_the_end_of_a_half_stays():
    labels = run_labels(("H", 100), ("A", 20))
    out = control.merge_touches(labels, whole(labels))
    assert (out == labels).all()


def test_ping_pong_collapses_into_the_surrounding_possession():
    labels = run_labels(("H", 100), ("A", 10), ("H", 10), ("A", 10), ("H", 100))
    out = control.merge_touches(labels, whole(labels))
    assert (out == "H").all()


def test_merge_stays_inside_each_half():
    labels = run_labels(("H", 100), ("A", 20), ("H", 100), ("A", 20), ("H", 100))
    blocks = [(1, 0, 119), (2, 120, 339)]
    out = control.merge_touches(labels, blocks)
    assert (out[100:120] == "A").all() and (out[220:240] == "H").all()


def test_ungraded_frames_stay_x():
    labels = run_labels(("X", 10), ("H", 100), ("A", 20), ("H", 100), ("X", 10))
    out = control.merge_touches(labels, [(1, 10, 229)])
    assert (out[:10] == "X").all() and (out[10:230] == "H").all() and (out[230:] == "X").all()


# --- 4. extra stoppages ----------------------------------------------------------------------------


def test_goal_is_where_the_longest_ball_absence_in_the_tag_starts():
    present = np.ones(N, dtype=bool)
    present[300:340] = False  # a lost ball in the attack before the goal
    present[800:1400] = False  # the celebration
    assert dead_ball.goal_frame(tag("GOALS", "Barcelona", 200, 1500), present) == 800


def test_a_goal_with_the_ball_always_seen_has_no_frame():
    assert (
        dead_ball.goal_frame(tag("GOALS", "Barcelona", 200, 1500), np.ones(N, dtype=bool)) is None
    )


def test_a_short_tracking_dropout_is_not_a_goal():
    present = np.ones(N, dtype=bool)
    present[300:424] = False  # just under 5 s
    assert dead_ball.goal_frame(tag("GOALS", "Barcelona", 200, 1500), present) is None


def test_a_goal_with_no_frame_marks_nothing():
    tags = [tag("GOALS", "Barcelona", 200, 1500)]
    assert dead_ball.goal_stoppages(N, tags, [1450], np.ones(N, dtype=bool)).sum() == 0


def test_dead_from_the_goal_until_the_kickoff_restart():
    present = np.ones(N, dtype=bool)
    present[800:1400] = False
    tags = [tag("GOALS", "Barcelona", 200, 1500)]
    mask = dead_ball.goal_stoppages(N, tags, [1450], present)
    assert mask[800:1450].all() and mask.sum() == 650


def test_a_goal_with_no_following_kickoff_marks_nothing():
    present = np.ones(N, dtype=bool)
    present[800:1400] = False
    mask = dead_ball.goal_stoppages(N, [tag("GOALS", "Barcelona", 200, 1500)], [100], present)
    assert mask.sum() == 0


def standstill_inputs():
    return np.full(N, 0.5), np.full(N, 0.1)  # smoothed collective speed, ball speed


def test_eight_seconds_of_standstill_is_a_stoppage():
    speed, ball = standstill_inputs()
    speed[:] = 2.0
    speed[500:700] = 0.5
    assert dead_ball.standstill_evidence(speed, ball)[500:700].all()


def test_just_under_eight_seconds_is_not():
    speed, ball = standstill_inputs()
    speed[:] = 2.0
    speed[500:699] = 0.5
    assert dead_ball.standstill_evidence(speed, ball).sum() == 0


def test_missing_ball_counts_as_stationary():
    speed, ball = standstill_inputs()
    ball[:] = np.nan
    assert dead_ball.standstill_evidence(speed, ball).all()


def test_keeper_holding_the_ball_with_players_moving_is_not_a_standstill():
    speed, ball = standstill_inputs()
    speed[:] = 1.6
    assert dead_ball.standstill_evidence(speed, ball).sum() == 0


def test_slow_players_with_the_ball_moving_is_not_a_standstill():
    speed, ball = standstill_inputs()
    ball[:] = 4.0
    assert dead_ball.standstill_evidence(speed, ball).sum() == 0


def test_a_brief_ball_nudge_does_not_end_a_standstill():
    speed, ball = standstill_inputs()
    speed[:] = 2.0
    speed[500:700] = 0.5
    ball[600:610] = 3.0
    assert dead_ball.standstill_evidence(speed, ball)[500:700].all()
