import csv

import numpy as np
import pytest

from src.io_utils import ResumableCsvWriter, atomic_save_npz

COLS = ["qid", "x", "lst"]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_resume_skips_done(tmp_path):
    p = tmp_path / "out.csv"
    with ResumableCsvWriter(p, COLS, flush_every=1) as w:
        w.write({"qid": "a", "x": 1, "lst": [1, 2]})
        w.write({"qid": "b", "x": 2, "lst": []})
    with ResumableCsvWriter(p, COLS) as w:
        assert w.done == {"a", "b"}
        w.write({"qid": "c", "x": 3, "lst": None})
    rows = read(p)
    assert [r["qid"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["lst"] == "[1, 2]"


def test_truncated_last_line_recovered(tmp_path):
    p = tmp_path / "out.csv"
    with ResumableCsvWriter(p, COLS, flush_every=1) as w:
        w.write({"qid": "a", "x": 1})
    with open(p, "a", encoding="utf-8") as f:
        f.write("b,2")                       # 끊겨서 줄바꿈 없이 잘린 행
    with ResumableCsvWriter(p, COLS) as w:
        assert w.done == {"a"}
        w.write({"qid": "b", "x": 2})
    assert [r["qid"] for r in read(p)] == ["a", "b"]


def test_column_mismatch_raises(tmp_path):
    p = tmp_path / "out.csv"
    ResumableCsvWriter(p, COLS).close()
    with pytest.raises(ValueError):
        ResumableCsvWriter(p, ["qid", "other"])


def test_atomic_npz(tmp_path):
    p = tmp_path / "sub" / "q1.npz"
    atomic_save_npz(p, a=np.arange(3))
    assert p.exists() and not (tmp_path / "sub" / "q1.npz.tmp").exists()
    assert np.load(p)["a"].tolist() == [0, 1, 2]
