"""The hashes make_submission.py prints, on synthetic bytes only."""

import make_submission


def test_lf_hash_is_the_same_for_both_line_endings():
    crlf = b"frame,label\r\n0,X\r\n"
    lf = b"frame,label\n0,X\n"
    assert make_submission.lf_sha256(crlf) == make_submission.lf_sha256(lf)


def test_file_hash_differs_between_line_endings():
    crlf = b"frame,label\r\n0,X\r\n"
    lf = b"frame,label\n0,X\n"
    assert make_submission.file_sha256(crlf) != make_submission.file_sha256(lf)


def test_lf_hash_changes_when_a_label_changes():
    assert make_submission.lf_sha256(b"0,H\n") != make_submission.lf_sha256(b"0,A\n")
