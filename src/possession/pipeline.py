"""v1 possession model: features -> dead-ball segments -> HMM/Viterbi -> frame,label CSV.

    python -m src.possession.pipeline --out evaluation/results/v1_hmm.csv

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
from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

from . import dead_ball, hmm  # noqa: E402
from .features import load_features  # noqa: E402

SET_PIECE_CODE = "set pieces"


def run(game=GAME, cache=None):
    blocks = half_video_blocks(game)
    feats = load_features(game, cache=cache)
    n_frames = len(feats.ball_x)

    events = load_smart_tagging(game)
    restarts = sorted(e["start_frame"] for e in events
                      if (e["code"] or "").strip().lower() == SET_PIECE_CODE)

    kickoff_dead = dead_ball.pre_kickoff(n_frames, blocks, events)
    score = dead_ball.dead_ball_score(feats, restarts, kickoff_dead)
    in_segment = dead_ball.dead_segments(score, blocks)
    labels = hmm.decode(feats, score, in_segment, blocks)

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    assert set(np.unique(labels[graded])) <= {"H", "A", "D"}, "an X or stray label on a graded frame"
    assert set(np.unique(labels[~graded])) <= {"X"}, "a label outside the graded halves"
    return labels, graded, in_segment


def possession_stats(labels, graded):
    """Count of possessions (maximal H or A runs) and mean duration, summed over the two halves."""
    count, frames = 0, 0
    for half_labels in np.split(labels, np.flatnonzero(np.diff(graded.astype(int)) != 0) + 1):
        if half_labels[0] == "X":
            continue
        change = np.concatenate(([True], half_labels[1:] != half_labels[:-1]))
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], len(half_labels))
        for s, e in zip(starts, ends):
            if half_labels[s] in ("H", "A"):
                count += 1
                frames += e - s
    return count, (frames / count / 25.0 if count else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join("evaluation", "results", "v1_hmm.csv"))
    parser.add_argument("--cache", default=None,
                        help="optional .npz for the extracted features (speeds up reruns; output is identical)")
    args = parser.parse_args()

    started = time.time()
    labels, graded, in_segment = run(cache=args.cache)
    write_submission(args.out, labels)

    shares = {k: float(np.mean(labels[graded] == k)) for k in "HAD"}
    count, mean_s = possession_stats(labels, graded)
    print("wrote {}".format(args.out))
    print("graded class shares  H {:.3f}  A {:.3f}  D {:.3f}".format(shares["H"], shares["A"], shares["D"]))
    print("possessions {}  mean duration {:.1f} s  D segments {}".format(
        count, mean_s, int(np.sum(np.diff(in_segment.astype(int)) == 1))))
    print("runtime {:.0f} s".format(time.time() - started))


if __name__ == "__main__":
    main()
