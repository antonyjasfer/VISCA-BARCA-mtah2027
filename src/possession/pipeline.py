"""v4 possession model: features -> dead ball and restarts -> control events -> gated Viterbi ->
touch merge -> CSV. Switches are in config.py.

    python -m src.possession.pipeline --out evaluation/results/v4_restart.csv

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
from .config import ModelConfig
from .diagnostics import possession_lengths, restart_consistency, team_agreement
from .features import load_features


def check_labels(labels, graded):
    """Every graded frame carries H, A or D; every other frame carries X."""
    if not set(np.unique(labels[graded])) <= {"H", "A", "D"}:
        raise ValueError("a graded frame is not H, A or D")
    if not set(np.unique(labels[~graded])) <= {"X"}:
        raise ValueError("an ungraded frame is not X")


def restart_validation(restarts, events, in_segment, feats, pitch):
    """(agreeing, compared) per rule, on restarts whose tag names a team.

    Each rule runs blind: the tag and the tagged phases are hidden, because the v3 chain starts from
    the tag and first_control falls back to it, so either would otherwise agree by construction.
    """
    tagged = [r for r in restarts if r.team is not None]
    truth = [r.team for r in tagged]
    blind = [r._replace(team=None) for r in tagged]

    v3, _ = dead_ball.resolve_restart_teams(blind, in_segment, feats, [], pitch)
    with_control, _ = dead_ball.apply_first_control(blind, events)
    full, _ = dead_ball.resolve_restart_teams(with_control, in_segment, feats, [], pitch)
    return {
        "v3": team_agreement(truth, [r.team for r in v3]),
        "first_control": team_agreement(truth, [r.team for r in full]),
        "control_only": team_agreement(truth, [r.team for r in with_control]),
    }


def finish_labels(decoded, blocks, config):
    return control.merge_touches(decoded, blocks) if config.touch_merge else decoded


def run(game=GAME, cache=None, config=None):
    config = config or ModelConfig()
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

    events = control.control_events(feats.d_home, feats.d_away, feats.rel_home, feats.rel_away)
    every_restart = restarts + dead_ball.segment_restarts(in_segment, restarts)
    latencies = []
    if config.restart_rule == "first_control":
        every_restart, latencies = dead_ball.apply_first_control(every_restart, events)
    resolved, rules = dead_ball.resolve_restart_teams(every_restart, in_segment, feats, tags, pitch)

    decoded = hmm.decode(score, in_segment, events, resolved, blocks)
    if rules["unknown"]:
        carried = dead_ball.carry_previous_owner(resolved, decoded)
        decoded = hmm.decode(score, in_segment, events, carried, blocks)
    labels = finish_labels(decoded, blocks, config)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first : last + 1] = True
    check_labels(labels, graded)

    summary = {
        "located": len(restarts),
        "unlocated": len(unlocated),
        "segment_restarts": len(every_restart) - len(restarts),
        "rules": rules,
        "validation": restart_validation(restarts, events, in_segment, feats, pitch),
        "consistency": restart_consistency(resolved, labels),
        "control_latency_s": float(np.median(latencies)) / 25.0 if latencies else None,
        "control_events": len(events),
        "segments": int(np.sum(np.diff(in_segment.astype(int)) == 1)),
        "merged_frames": int(np.sum(labels != decoded)),
        "goal_stoppages": int(np.sum(np.diff(goal_dead.astype(int)) == 1)),
        "goal_minutes": float(np.sum(goal_dead)) / 25.0 / 60.0,
    }
    return labels, graded, summary, blocks


def format_validation(validation):
    return "  ".join("{} {}/{}".format(rule, *pair) for rule, pair in validation.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("evaluation", "results", "v4_restart.csv"))
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
    print("blind agreement with the set-piece tag  " + format_validation(summary["validation"]))
    print("restart team owns the next possession  {} of {}".format(*summary["consistency"]))
    if summary["control_latency_s"] is not None:
        print(f"median restart to first control {summary['control_latency_s']:.2f} s")
    print(
        "control events {}  D segments {}  frames relabelled as touches {}".format(
            summary["control_events"], summary["segments"], summary["merged_frames"]
        )
    )
    print(f"goal stoppages {summary['goal_stoppages']}  forced D {summary['goal_minutes']:.1f} min")
    print(f"runtime {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
