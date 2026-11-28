"""src/track.py 검증 (GPU 없이): 감싼 실행이 그대로 돌고, 실행마다 한 줄이 기록되는지."""
import csv
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest

from src import track as TR

DUMMY = textwrap.dedent('''
    import json, sys
    def main():
        args = sys.argv[1:]
        with open(args[args.index("--out") + 1], "w") as f:
            json.dump(args, f)
        if "--fail" in args:
            raise SystemExit(3)
        if "--boom" in args:
            raise RuntimeError("boom")
    if __name__ == "__main__":
        main()
''')


@pytest.fixture()
def dummy(tmp_path, monkeypatch):
    (tmp_path / "dummy_track_mod.py").write_text(DUMMY)
    monkeypatch.syspath_prepend(str(tmp_path))
    log = tmp_path / "compute" / "runs.csv"
    monkeypatch.setenv("COMPUTE_LOG", str(log))
    return tmp_path, log


def rows(log):
    return list(csv.DictReader(open(log, newline="", encoding="utf-8")))


def test_runs_module_with_args_and_records_row(dummy):
    tmp, log = dummy
    out = tmp / "o.json"
    argv_before = list(sys.argv)
    code = TR.main(["dummy_track_mod", "--model", "qwen2.5-vl-7b", "--questions", "q.jsonl", "--out", str(out)])
    assert code == 0 and sys.argv == argv_before                      # sys.argv 원복
    assert out.read_text().startswith('["--model", "qwen2.5-vl-7b"')  # 대상 모듈이 인자를 그대로 받음
    r, = rows(log)
    assert list(r) == TR.COLUMNS
    assert r["model"] == "qwen2.5-vl-7b" and r["module"] == "dummy_track_mod" and r["questions"] == "q.jsonl"
    assert r["out"] == str(out) and r["exit_code"] == "0" and float(r["elapsed_s"]) >= 0
    assert r["cmd"].startswith("python -m dummy_track_mod --model qwen2.5-vl-7b")


def test_records_exit_code_and_exceptions(dummy):
    tmp, log = dummy
    assert TR.main(["dummy_track_mod", "--out", str(tmp / "a.json"), "--fail"]) == 3
    with pytest.raises(RuntimeError):
        TR.main(["dummy_track_mod", "--out", str(tmp / "b.json"), "--boom"])
    assert [r["exit_code"] for r in rows(log)] == ["3", "1"]       # 예외로 끝나도 한 줄 남음


def test_vram_columns_empty_without_gpu(dummy):
    tmp, log = dummy
    TR.main(["dummy_track_mod", "--out", str(tmp / "c.json")])
    r, = rows(log)
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        assert float(r["vram_peak_gb"]) >= float(r["vram_mean_gb"]) >= 0
    else:
        assert r["gpu"] == r["vram_peak_gb"] == r["vram_mean_gb"] == ""


def test_concurrent_appends_do_not_mix(tmp_path):
    log = tmp_path / "runs.csv"

    def write(i):
        TR.append_row(log, {c: f"{c}-{i}" for c in TR.COLUMNS})

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(write, range(200)))
    rs = rows(log)
    assert len(rs) == 200 and sorted(int(r["start"].split("-")[1]) for r in rs) == list(range(200))
    assert all(r["cmd"] == f"cmd-{r['start'].split('-')[1]}" for r in rs)  # 한 줄 안의 값이 섞이지 않음
    assert open(log, encoding="utf-8").read().count("start,end,") == 1   # 머리줄 한 번


def test_usage_error():
    assert TR.main([]) == 2
