"""Label-free diagnostics, on synthetic arrays only. Nothing here reads data/ or evaluation/gt/."""

import numpy as np

from src.possession.diagnostics import (
    fill_abstentions,
    half_summary,
    set_piece_coverage,
    tag_agreement,
)


def labels_of(text):
    return np.array(list(text), dtype="<U1")


def tag(code, team, start, end):
    return {"code": code, "start_frame": start, "end_frame": end, "tags": {"Team": team}}


def test_fill_abstentions_repeats_previous_label_and_backfills_the_head():
    assert "".join(fill_abstentions(labels_of("XXHHXXDX"))) == "HHHHHHDD"


def test_half_summary_counts_runs_and_live_minutes():
    labels = np.array(["H"] * 25 + ["D"] * 25 + ["H"] * 10 + ["A"] * 50, dtype="<U1")
    out = half_summary(labels, fps=25)
    assert out["possessions"] == 3
    assert out["under_2s"] == 2  # 25 and 10 frames; 50 frames is exactly 2 s
    assert abs(out["d_share"] - 25 / 110) < 1e-9
    assert abs(out["live_min"] - 85 / 25 / 60) < 1e-9


def test_tag_agreement_mirrors_the_team_for_defending_tags():
    labels = labels_of("HHHHAAAA")
    events = [
        tag("BUILD UP", "Barcelona", 0, 4),
        tag("DEFENDING IN MIDDLE THIRD", "Barcelona", 4, 8),
    ]
    out = tag_agreement(labels, events, [(1, 0, 7)])
    assert out["attacking"] == 1.0
    assert out["defending"] == 1.0


def test_tag_agreement_ignores_dead_frames_and_na_teams():
    labels = labels_of("HHDDAAAA")
    events = [tag("BUILD UP", "Barcelona", 0, 6), tag("PROGRESSION", "N/A", 0, 8)]
    assert tag_agreement(labels, events, [(1, 0, 7)])["attacking"] == 0.5


def test_set_piece_coverage_needs_a_second_of_dead_ball_near_the_start():
    labels = np.array(["H"] * 1000, dtype="<U1")
    labels[300:325] = "D"  # exactly 1 s, inside start +/- 10 s
    events = [
        tag("SET PIECES", "N/A", 400, 500),
        tag("SET PIECES", "N/A", 800, 900),
        tag("BUILD UP", "Barcelona", 0, 10),
    ]
    assert set_piece_coverage(labels, events, fps=25) == 0.5
