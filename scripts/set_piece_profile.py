"""Aggregate timing around Smart Tagging SET PIECES tags. Never touches any ground truth.

    python scripts/set_piece_profile.py

Run from the repository root. Prints window lengths per type, then collective speed, ball visibility
and ball speed by 5 s offset from the tag start, as medians and means across all tags. This is what
shows the tag start falling inside the stoppage (METHOD.md, TECHNICAL_REPORT.md). No rows are printed.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.possession.features import FPS, load_features
from starter.load_smart_tagging import load

OFFSETS_S = range(-30, 40, 5)
BIN_S = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=None, help="optional .npz of extracted features")
    args = parser.parse_args()

    feats = load_features(cache=args.cache)
    pieces = [e for e in load() if (e["code"] or "").strip().lower() == "set pieces"]

    lengths = {}
    for piece in pieces:
        kind = piece["tags"].get("Type")
        lengths.setdefault(kind, []).append((piece["end_frame"] - piece["start_frame"]) / FPS)
    for kind, values in sorted(lengths.items(), key=lambda item: str(item[0])):
        print(f"{kind!s:<24} n={len(values):>2}  median window {np.median(values):.1f} s")

    groups = {
        "phase windows": [p for p in pieces if "Kick Off" not in p["tags"].get("Type", "")],
        "kickoffs": [p for p in pieces if "Kick Off" in p["tags"].get("Type", "")],
    }
    for name, group in groups.items():
        print(f"\n{name}, n={len(group)}")
        print("offset s     median speed  ball seen  median ball speed")
        for offset in OFFSETS_S:
            speeds, seen, ball = [], [], []
            for piece in group:
                low = piece["start_frame"] + offset * FPS
                high = low + BIN_S * FPS
                speeds.append(np.nanmedian(feats.mean_speed[low:high]))
                seen.append(np.mean(~np.isnan(feats.ball_x[low:high])))
                ball.append(feats.ball_speed[low:high])
            ball = np.concatenate(ball)
            ball = ball[~np.isnan(ball)]
            ball_median = np.median(ball) if len(ball) else float("nan")
            print(
                f"{offset:+4d}..{offset + BIN_S:+4d}   {np.nanmedian(speeds):>10.2f}"
                f"  {np.mean(seen):>9.2f}  {ball_median:>17.2f}"
            )


if __name__ == "__main__":
    main()
