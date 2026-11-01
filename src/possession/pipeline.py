"""v2 possession model: features -> dead ball and restarts -> control events -> gated Viterbi -> CSV.

    python -m src.possession.pipeline --out evaluation/results/v2_control.csv

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
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

from . import control, dead_ball, hmm  # noqa: E402
from .diagnostics import fill_abstentions, possession_lengths  # noqa: E402
from .features import load_features  # noqa: E402


def run(game=GAME, cache=None):
    blocks = half_video_blocks(game)
    feats = load_features(game, cache=cache)
    n_frames = len(feats.ball_x)
    pitch = GAMES[game]["pitch"]

    tags = load_smart_tagging(game)
    restarts, unlocated = dead_ball.locate_restarts(tags, feats)
    kickoff_dead = dead_ball.pre_kickoff(n_frames, blocks, tags)
    score = dead_ball.dead_ball_score(feats, restarts, unlocated, kickoff_dead, pitch)
    in_segment = dead_ball.dead_segments(score, blocks)
    all_restarts = restarts + dead_ball.segment_restarts(in_segment, restarts, feats)

    events = control.control_events(feats.d_home, feats.d_away, feats.rel_home, feats.rel_away)
    decoded = hmm.decode(score, in_segment, events, all_restarts, blocks)
    labels = control.abstain_gaps(decoded, ~np.isnan(feats.ball_x), events, blocks)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    assert set(np.unique(decoded[graded])) <= {"H", "A", "D"}, "a stray label on a graded frame"
    assert set(np.unique(labels[~graded])) <= {"X"}, "a label outside the graded halves"
    for _, first, last in blocks:
        filled = fill_abstentions(labels[first:last + 1])
        assert (filled == decoded[first:last + 1]).all(), "an X changed what the scorer will see"

    summary = {
        "located": len(restarts), "unlocated": len(unlocated),
        "restart_team_unknown": sum(r.team is None for r in all_restarts),
        "segment_restarts": len(all_restarts) - len(restarts),
        "control_events": len(events),
        "segments": int(np.sum(np.diff(in_segment.astype(int)) == 1)),
    }
    return labels, decoded, graded, summary, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("evaluation", "results", "v2_control.csv"))
    parser.add_argument("--cache", default=None,
                        help="optional .npz for the extracted features (speeds up reruns; output is identical)")
    args = parser.parse_args()

    started = time.time()
    labels, decoded, graded, summary, blocks = run(cache=args.cache)
    write_submission(args.out, labels)

    shares = {k: float(np.mean(labels[graded] == k)) for k in "HADX"}
    lengths = np.concatenate([possession_lengths(decoded[first:last + 1]) for _, first, last in blocks])
    print("wrote {}".format(args.out))
    print("graded class shares  H {:.3f}  A {:.3f}  D {:.3f}  X {:.3f}".format(
        shares["H"], shares["A"], shares["D"], shares["X"]))
    print("possessions {}  mean duration {:.1f} s".format(len(lengths), lengths.mean() / 25.0))
    print("restarts located {}  not located {}  team unknown {}  from D segments {}".format(
        summary["located"], summary["unlocated"], summary["restart_team_unknown"],
        summary["segment_restarts"]))
    print("control events {}  D segments {}".format(summary["control_events"], summary["segments"]))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
