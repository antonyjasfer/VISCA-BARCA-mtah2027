"""Label-free plausibility checks for a submission. Never touches any ground truth.

    python -m src.possession.diagnostics evaluation/results/v1_hmm.csv evaluation/results/baseline_smooth50.csv

Prints aggregates only. Abstentions are filled the way the scorer fills them (repeat the last label
in the half, back-fill the head), so these numbers describe what the ranking metric will see.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks
from starter.load_smart_tagging import load as load_smart_tagging

from .control import AWAY, HOME

ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition"}
DEFENDING_PREFIX = "defending"
SET_PIECE_CODE = "set pieces"
TEAM_SIDES = {"barcelona": "H", "atletic club": "A"}
OTHER_SIDE = {"H": "A", "A": "H"}
COVERAGE_REACH_S = 10
COVERAGE_MIN_DEAD_S = 1
SHORT_POSSESSION_S = 2


def _code(event):
    return (event["code"] or "").strip().lower()


def fill_abstentions(labels):
    """X -> previous label; leading X -> first real label. What the scorer does within a half."""
    series = pd.Series(labels).where(labels != "X")
    return series.ffill().bfill().to_numpy(dtype="<U1")


def possession_lengths(labels):
    """Frame length of every maximal H or A run."""
    change = np.concatenate(([True], labels[1:] != labels[:-1]))
    starts = np.flatnonzero(change)
    lengths = np.diff(np.append(starts, len(labels)))
    return lengths[np.isin(labels[starts], ("H", "A"))]


def half_summary(labels, fps):
    """Summary of one half of already-filled labels."""
    lengths = possession_lengths(labels)
    seconds = lengths / float(fps)
    p10, p50, p90 = np.percentile(seconds, (10, 50, 90)) if len(seconds) else (0.0, 0.0, 0.0)
    return {
        "live_min": np.sum(labels != "D") / float(fps) / 60.0,
        "d_share": float(np.mean(labels == "D")),
        "possessions": len(lengths),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "under_2s": int(np.sum(seconds < SHORT_POSSESSION_S)),
    }


def tag_agreement(labels, events, blocks):
    """Share of live frames inside team-tagged phases whose label matches the tag.

    Attacking phases expect the tagged team on the ball, DEFENDING ones the other team. Dead frames
    are skipped: a stoppage inside a phase is neither agreement nor disagreement.
    """
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in blocks:
        graded[first : last + 1] = True

    hits = {"attacking": 0, "defending": 0}
    totals = {"attacking": 0, "defending": 0}
    for event in events:
        side = TEAM_SIDES.get((event["tags"].get("Team") or "").strip().lower())
        code = _code(event)
        if side is None:
            continue
        if code in ATTACKING_CODES:
            kind, expected = "attacking", side
        elif code.startswith(DEFENDING_PREFIX):
            kind, expected = "defending", OTHER_SIDE[side]
        else:
            continue
        part = labels[event["start_frame"] : event["end_frame"]]
        part = part[graded[event["start_frame"] : event["end_frame"]]]
        live = part[np.isin(part, ("H", "A"))]
        hits[kind] += int(np.sum(live == expected))
        totals[kind] += len(live)
    return {
        kind: hits[kind] / float(totals[kind]) if totals[kind] else float("nan") for kind in hits
    }


def set_piece_coverage(labels, events, fps):
    """Share of SET PIECES instances with at least 1 s of D within the start +/- 10 s."""
    reach = COVERAGE_REACH_S * fps
    pieces = [event for event in events if _code(event) == SET_PIECE_CODE]
    covered = sum(
        int(np.sum(labels[max(0, e["start_frame"] - reach) : e["start_frame"] + reach + 1] == "D"))
        >= COVERAGE_MIN_DEAD_S * fps
        for e in pieces
    )
    return covered / float(len(pieces)) if pieces else float("nan")


def team_agreement(expected, answered):
    """(agreeing, compared) over the pairs where a rule gave an answer."""
    pairs = [(a, b) for a, b in zip(expected, answered) if b is not None]
    return sum(a == b for a, b in pairs), len(pairs)


def restart_consistency(restarts, labels):
    """(agreeing, compared): does the restart team own the first possession after the stoppage?"""
    sides = {HOME: "H", AWAY: "A"}
    agreeing = compared = 0
    for restart in restarts:
        if restart.team is None:
            continue
        frame = restart.frame
        while frame < len(labels) and labels[frame] == "D":
            frame += 1
        if frame < len(labels) and labels[frame] in ("H", "A"):
            compared += 1
            agreeing += labels[frame] == sides[restart.team]
    return agreeing, compared


def report(path, events, blocks, fps):
    return summarize(pd.read_csv(path)["label"].to_numpy(dtype="<U1"), events, blocks, fps)


def summarize(raw, events, blocks, fps):
    """Everything report() prints, for labels already in memory."""
    labels = raw.copy()
    rows = []
    for half, first, last in blocks:
        labels[first : last + 1] = fill_abstentions(raw[first : last + 1])
        rows.append((half, half_summary(labels[first : last + 1], fps)))
    graded = np.concatenate([raw[first : last + 1] for _, first, last in blocks])
    return {
        "halves": rows,
        "agreement": tag_agreement(labels, events, blocks),
        "coverage": set_piece_coverage(labels, events, fps),
        "x_share": float(np.mean(graded == "X")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+")
    args = parser.parse_args()

    blocks = half_video_blocks(GAME)
    fps = GAMES[GAME]["fps"]
    events = load_smart_tagging(GAME)

    print(
        "{:<20} {:>4} {:>8} {:>7} {:>5} {:>6} {:>6} {:>6} {:>5}".format(
            "submission", "half", "live min", "D share", "poss", "p10 s", "p50 s", "p90 s", "<2 s"
        )
    )
    summaries = []
    for path in args.submissions:
        name = os.path.splitext(os.path.basename(path))[0]
        out = report(path, events, blocks, fps)
        summaries.append((name, out))
        for half, row in out["halves"]:
            print(
                "{:<20} {:>4} {:>8.1f} {:>7.3f} {:>5} {:>6.1f} {:>6.1f} {:>6.1f} {:>5}".format(
                    name,
                    half,
                    row["live_min"],
                    row["d_share"],
                    row["possessions"],
                    row["p10"],
                    row["p50"],
                    row["p90"],
                    row["under_2s"],
                )
            )
    print()
    print(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>9} {:>8}".format(
            "submission",
            "live total",
            "poss total",
            "attack ok",
            "defend ok",
            "set piece",
            "X share",
        )
    )
    for name, out in summaries:
        print(
            "{:<20} {:>10.1f} {:>10} {:>10.3f} {:>10.3f} {:>9.3f} {:>8.3f}".format(
                name,
                sum(r["live_min"] for _, r in out["halves"]),
                sum(r["possessions"] for _, r in out["halves"]),
                out["agreement"]["attacking"],
                out["agreement"]["defending"],
                out["coverage"],
                out["x_share"],
            )
        )


if __name__ == "__main__":
    main()
