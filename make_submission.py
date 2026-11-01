"""Regenerate the VISCA BARCA submission from the match data, validate it and print its hash.

    python make_submission.py

Needs data/tactical/ in place (see README.md). Writes submissions/VISCA BARCA.csv with LF line
endings on every OS. Prints aggregates only, never rows.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.possession.pipeline import run, write_labels

TEAM = "VISCA BARCA"
OUTPUT = os.path.join(ROOT, "submissions", f"{TEAM}.csv")
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")


def file_sha256(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache",
        default=None,
        help="optional .npz for the extracted features (speeds up reruns; output is identical)",
    )
    args = parser.parse_args()

    started = time.time()
    labels, _, _, _ = run(cache=args.cache)
    write_labels(OUTPUT, labels)
    subprocess.run([sys.executable, VALIDATOR, OUTPUT], check=True)

    with open(OUTPUT, "rb") as handle:
        content = handle.read()
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}")
    print(f"SHA-256 {file_sha256(content)}")
    print(f"runtime {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
