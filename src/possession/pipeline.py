"""v3 possession model: features -> dead ball and restarts -> control events -> gated Viterbi ->
touch merge -> CSV.

    python -m src.possession.pipeline --out evaluation/results/v3_rules.csv

Run from the repository root. Prints only aggregates (counts, shares, runtime), never rows.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks
from starter.baseline import write_submission
from starter.load_smart_tagging import load as load_smart_tagging

from . import control, dead_ball, hmm
from .diagnostics import possession_lengths
from .features import load_features


def check_labels(labels, graded):
    """Every graded frame carries H, A or D; every other frame carries X."""
    if not set(np.unique(labels[graded])) <= {"H", "A", "D"}:
        raise ValueError("a graded frame is not H, A or D")
    if not set(np.unique(labels[~graded])) <= {"X"}:
        raise ValueError("an ungraded frame is not X")


def tag_cross_check(restarts, in_segment, feats, tags, pitch):
    """(agreeing, compared): on restarts whose tag names a team, do the other rules name it too?"""
    tagged = [r for r in restarts if r.team is not None]
    blind, _ = dead_ball.resolve_restart_teams(
        [r._replace(team=None) for r in tagged], in_segment, feats, [], pitch
    )
    pairs = [(r.team, b.team) for r, b in zip(tagged, blind) if b.team is not None]
    return sum(a == b for a, b in pairs), len(pairs)


def run(game=GAME, cache=None):
    blocks = half_video_blocks(game)
    feats = load_features(game, cache=cache)
    n_frames = len(feats.ball_x)
    pitch = GAMES[game]["pitch"]

    tags = load_smart_tagging(game)
    restarts, unlocated = dead_ball.locate_restarts(tags, feats)
    kickoffs = dead_ball.after_goal_kickoffs(restarts, unlocated)
    goal_dead = dead_ball.goal_stoppages(n_frames, tags, kickoffs, ~np.isnan(feats.ball_x))
    forced_dead = dead_ball.pre_kickoff(n_frames, blocks, tags) | goal_dead
    score = dead_ball.dead_ball_score(feats, restarts, unlocated, forced_dead, pitch)
    in_segment = dead_ball.dead_segments(score, blocks)

    every_restart = restarts + dead_ball.segment_restarts(in_segment, restarts)
    resolved, rules = dead_ball.resolve_restart_teams(every_restart, in_segment, feats, tags, pitch)

    events = control.control_events(feats.d_home, feats.d_away, feats.rel_home, feats.rel_away)
    decoded = hmm.decode(score, in_segment, events, resolved, blocks)
    if rules["unknown"]:
        carried = dead_ball.carry_previous_owner(resolved, decoded)
        decoded = hmm.decode(score, in_segment, events, carried, blocks)
    labels = control.merge_touches(decoded, blocks)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first : last + 1] = True
    check_labels(labels, graded)

    summary = {
        "located": len(restarts),
        "unlocated": len(unlocated),
        "segment_restarts": len(every_restart) - len(restarts),
        "rules": rules,
        "tag_check": tag_cross_check(restarts, in_segment, feats, tags, pitch),
        "control_events": len(events),
        "segments": int(np.sum(np.diff(in_segment.astype(int)) == 1)),
        "merged_frames": int(np.sum(labels != decoded)),
        "goal_stoppages": int(np.sum(np.diff(goal_dead.astype(int)) == 1)),
        "goal_minutes": float(np.sum(goal_dead)) / 25.0 / 60.0,
    }
    return labels, graded, summary, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("evaluation", "results", "v3_rules.csv"))
    parser.add_argument(
        "--cache",
        default=None,
        help="optional .npz for the extracted features (speeds up reruns; output is identical)",
    )
    args = parser.parse_args()

    started = time.time()
    labels, graded, summary, blocks = run(cache=args.cache)
    write_submission(args.out, labels)

    shares = {k: float(np.mean(labels[graded] == k)) for k in "HADX"}
    lengths = np.concatenate(
        [possession_lengths(labels[first : last + 1]) for _, first, last in blocks]
    )
    print(f"wrote {args.out}")
    print(
        "graded class shares  H {:.3f}  A {:.3f}  D {:.3f}  X {:.3f}".format(
            shares["H"], shares["A"], shares["D"], shares["X"]
        )
    )
    print(f"possessions {len(lengths)}  mean duration {lengths.mean() / 25.0:.1f} s")
    print(
        "restarts located {}  not located {}  from D segments {}".format(
            summary["located"], summary["unlocated"], summary["segment_restarts"]
        )
    )
    print(
        "restart team by rule  {}".format(
            "  ".join(f"{k} {v}" for k, v in summary["rules"].items())
        )
    )
    print(
        "tagged restarts where the other rules agree with the tag: {} of {}".format(
            *summary["tag_check"]
        )
    )
    print(
        "control events {}  D segments {}  frames relabelled as touches {}".format(
            summary["control_events"], summary["segments"], summary["merged_frames"]
        )
    )
    print(f"goal stoppages {summary['goal_stoppages']}  forced D {summary['goal_minutes']:.1f} min")
    print(f"runtime {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
