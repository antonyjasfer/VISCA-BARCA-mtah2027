"""Aggregate timing around Smart Tagging GOALS tags. Never touches any ground truth.

    python scripts/goal_profile.py

Run from the repository root. Prints tag lengths, the gap to the next after-goal kickoff, and
collective speed and ball visibility by 10 s offset from the tag start, across all tags. This is what
shows the tag starting well before the goal (METHOD.md, TECHNICAL_REPORT.md). No rows are printed.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.possession.features import FPS, load_features
from starter.load_smart_tagging import load

OFFSETS_S = range(-40, 80, 10)
BIN_S = 10
KICKOFF_TYPE = "kick off (after goal)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=None, help="optional .npz of extracted features")
    args = parser.parse_args()

    feats = load_features(cache=args.cache)
    tags = load()
    goals = [e for e in tags if (e["code"] or "").strip().lower() == "goals"]
    kickoffs = [
        e["start_frame"] for e in tags if (e["tags"].get("Type") or "").lower() == KICKOFF_TYPE
    ]

    lengths = [(g["end_frame"] - g["start_frame"]) / FPS for g in goals]
    gaps = [
        (
            min((k for k in kickoffs if k > g["start_frame"]), default=g["start_frame"])
            - g["start_frame"]
        )
        / FPS
        for g in goals
    ]
    print(f"goals {len(goals)}  after-goal kickoffs {len(kickoffs)}")
    print(
        "tag length s, min / median / max: {:.0f} / {:.0f} / {:.0f}".format(
            *np.percentile(lengths, (0, 50, 100))
        )
    )
    print(
        "tag start to kickoff s, min / median / max: {:.0f} / {:.0f} / {:.0f}".format(
            *np.percentile(gaps, (0, 50, 100))
        )
    )

    print("\noffset s     median speed  ball seen")
    for offset in OFFSETS_S:
        speeds, seen = [], []
        for goal in goals:
            low = goal["start_frame"] + offset * FPS
            high = low + BIN_S * FPS
            speeds.append(np.nanmedian(feats.mean_speed[low:high]))
            seen.append(np.mean(~np.isnan(feats.ball_x[low:high])))
        print(
            f"{offset:+4d}..{offset + BIN_S:+4d}   {np.median(speeds):>10.2f}  {np.mean(seen):>9.2f}"
        )


if __name__ == "__main__":
    main()
