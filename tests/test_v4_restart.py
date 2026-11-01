"""v4 flags, the first_control restart rule and restart validation, on synthetic arrays only."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.possession import dead_ball, diagnostics, pipeline
from src.possession.config import ModelConfig
from src.possession.control import AWAY, HOME, ControlEvent
from src.possession.dead_ball import Restart

PITCH = (105.0, 68.0)
N = 2000


def restart(frame, team=None):
    return Restart(frame, team, frame - 100, "throw-in", None)


def quiet_feats():
    """Ball in midfield, nobody tracked: no v3 rule can answer."""
    return SimpleNamespace(
        ball_x=np.full(N, 50.0),
        ball_y=np.full(N, 30.0),
        ball_in_pitch=np.ones(N),
        d_home=np.full(N, np.nan),
        d_away=np.full(N, np.nan),
    )


def segment(start, stop):
    mask = np.zeros(N, dtype=bool)
    mask[start:stop] = True
    return mask


# --- flags -----------------------------------------------------------------------------------------


def test_config_accepts_both_restart_rules():
    assert ModelConfig(restart_rule="v3").restart_rule == "v3"
    assert ModelConfig(restart_rule="first_control").restart_rule == "first_control"


def test_config_rejects_an_unknown_restart_rule():
    with pytest.raises(ValueError):
        ModelConfig(restart_rule="nearest")


def test_touch_merge_flag_on_merges():
    labels = np.array(["H"] * 100 + ["A"] * 20 + ["H"] * 100, dtype="<U1")
    out = pipeline.finish_labels(labels, [(1, 0, 219)], ModelConfig(touch_merge=True))
    assert (out == "H").all()


def test_touch_merge_flag_off_leaves_labels_alone():
    labels = np.array(["H"] * 100 + ["A"] * 20 + ["H"] * 100, dtype="<U1")
    out = pipeline.finish_labels(labels, [(1, 0, 219)], ModelConfig(touch_merge=False))
    assert (out == labels).all()


# --- first_control -----------------------------------------------------------------------------------


def test_first_control_takes_the_first_event_after_the_restart():
    events = [ControlEvent(AWAY, 720, 740), ControlEvent(HOME, 750, 770)]
    assert dead_ball.first_control(events, 700) == (AWAY, 20)


def test_first_control_ignores_an_event_that_began_before_the_restart():
    assert dead_ball.first_control([ControlEvent(HOME, 690, 720)], 700) is None


def test_first_control_reaches_exactly_four_seconds():
    assert dead_ball.first_control([ControlEvent(AWAY, 800, 820)], 700) == (AWAY, 100)
    assert dead_ball.first_control([ControlEvent(AWAY, 801, 820)], 700) is None


def test_first_control_overrides_a_tagged_team():
    out, latencies = dead_ball.apply_first_control(
        [restart(700, HOME)], [ControlEvent(AWAY, 720, 740)]
    )
    assert out[0].team == AWAY and out[0].rule == "first_control" and latencies == [20]


def test_no_control_event_keeps_the_tagged_team():
    out, latencies = dead_ball.apply_first_control([restart(700, HOME)], [])
    assert out[0].team == HOME and out[0].rule is None and latencies == []


def test_resolve_counts_the_first_control_rule_and_keeps_its_team():
    done, _ = dead_ball.apply_first_control([restart(700)], [ControlEvent(AWAY, 720, 740)])
    out, counts = dead_ball.resolve_restart_teams(done, segment(510, 700), quiet_feats(), [], PITCH)
    assert out[0].team == AWAY and counts["first_control"] == 1 and counts["unknown"] == 0


def test_no_control_event_and_no_tag_falls_through_to_the_v3_chain():
    feats = quiet_feats()
    feats.d_home[:] = 9.0
    feats.d_away[:] = 9.0
    feats.d_away[690:702] = 0.5
    done, _ = dead_ball.apply_first_control([restart(700)], [])
    out, counts = dead_ball.resolve_restart_teams(done, segment(510, 700), feats, [], PITCH)
    assert out[0].team == AWAY and counts["nearest"] == 1


# --- validation --------------------------------------------------------------------------------------


def test_agreement_counts_only_answers_given():
    assert diagnostics.team_agreement([HOME, AWAY, HOME], [HOME, HOME, None]) == (1, 2)


def test_consistency_compares_with_the_first_possession_after_the_stoppage():
    labels = np.array(["A"] * 10 + ["D"] * 10 + ["H"] * 10, dtype="<U1")
    assert diagnostics.restart_consistency([restart(20, HOME)], labels) == (1, 1)
    assert diagnostics.restart_consistency([restart(20, AWAY)], labels) == (0, 1)


def test_consistency_skips_past_a_stoppage_that_outlasts_the_restart():
    labels = np.array(["D"] * 30 + ["A"] * 10, dtype="<U1")
    assert diagnostics.restart_consistency([restart(20, AWAY)], labels) == (1, 1)


def test_consistency_ignores_unknown_teams_and_ungraded_frames():
    labels = np.array(["D"] * 20 + ["X"] * 20, dtype="<U1")
    assert diagnostics.restart_consistency([restart(10), restart(25, HOME)], labels) == (0, 0)


def test_blind_validation_never_lets_a_rule_see_the_tag():
    feats = quiet_feats()
    tagged = [restart(700, HOME)]
    events = [ControlEvent(AWAY, 720, 740)]
    out = pipeline.restart_validation(tagged, events, segment(510, 700), feats, PITCH)
    assert out["first_control"] == (0, 1)  # the control event disagrees; the tag is not consulted
    assert out["v3"] == (0, 0)  # nothing but the tag could answer, so nothing is compared
