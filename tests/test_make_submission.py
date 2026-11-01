"""The submission writer and the printed hash, on synthetic labels only."""

import hashlib

import numpy as np

import make_submission
from src.possession import pipeline


def labels_of(text):
    return np.array(list(text), dtype="<U1")


def test_labels_are_written_with_lf_line_endings(tmp_path):
    path = tmp_path / "out.csv"
    pipeline.write_labels(str(path), labels_of("XH"))
    assert path.read_bytes() == b"frame,label\n0,X\n1,H\n"


def test_writer_creates_a_missing_folder(tmp_path):
    path = tmp_path / "new" / "out.csv"
    pipeline.write_labels(str(path), labels_of("D"))
    assert path.read_bytes() == b"frame,label\n0,D\n"


def test_hash_is_the_sha256_of_the_bytes():
    content = b"frame,label\n0,X\n"
    assert make_submission.file_sha256(content) == hashlib.sha256(content).hexdigest()
