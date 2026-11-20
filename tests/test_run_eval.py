"""가짜 VLM으로 run_eval 루프(재시작 포함)를 CPU에서 검증."""
import csv
import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from src import models as M  # noqa: E402
from src import run_eval as R  # noqa: E402
from src.segments import split_segments  # noqa: E402

VS, VE, IMS, PAD = 5, 6, 7, 8


class FakeTok:
    def decode(self, ids):
        return f"<{ids[0]}>"


class FakeVLM:
    name = "fake"
    size = {"longest_edge": 100}
    answer_tokens = M.AnswerTokens(yes=1, no=2, variants={" Yes": 3, " No": 4, "yes": 1, "no": 2})

    def __init__(self):
        self.processor = type("P", (), {"tokenizer": FakeTok()})()
        self.calls = 0

    def build_inputs(self, images, question):
        ids = [IMS, 0, IMS, 0] + sum(([VS, PAD, PAD, VE] for _ in images), []) + [0, 0]
        return {"input_ids": torch.tensor([ids]), "n": len(images)}

    def forward_last_logits(self, inputs):
        self.calls += 1
        logits = torch.zeros(10)
        logits[1] = float(inputs["n"])     # yes logit = N
        logits[2] = 3.0                    # no logit
        return logits

    def segments(self, inputs):
        return split_segments(inputs["input_ids"][0].tolist(), VS, VE, IMS, inputs["n"])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    qs = []
    for N in (2, 4):
        for t, ans in (("forall_yes", "Yes"), ("forall_no", "No")):
            qs.append({"qid": f"q{N}{t}", "pair_id": f"p{N}", "cls": "cat", "N": N, "qtype": t,
                       "answer": ans, "question": "?", "image_ids": list(range(N)),
                       "image_paths": ["x"] * N, "present_positions": [], "absent_positions": [0],
                       "area_ratios": [0.0] * N})
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in qs))
    cfg = {"coco": {"root": "."}, "model": {"name": "fake", "attn_impl": "sdpa", "max_pixels": 100}}
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    fake = FakeVLM()
    monkeypatch.setattr(M, "load_model", lambda *a, **k: fake)
    monkeypatch.setattr(M, "load_images", lambda paths, root: [None] * len(paths))
    return qpath, cpath, tmp_path / "out.csv", fake


def test_run_and_resume(setup):
    qpath, cpath, out, fake = setup
    base = ["--questions", str(qpath), "--config", str(cpath), "--out", str(out), "--device", "cpu"]
    R.main(base + ["--limit", "2"])
    assert fake.calls == 2
    R.main(base)
    assert fake.calls == 4                    # 앞의 2개는 skip
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert len(rows) == 4 and len({r["qid"] for r in rows}) == 4
    for r in rows:
        N = int(r["N"])
        assert float(r["m"]) == pytest.approx(N - 3.0)
        assert r["pred"] == ("Yes" if N > 3 else "No")
        assert json.loads(r["n_image_tokens"]) == [2] * N
        assert 0.0 <= float(r["p_yes"]) + float(r["p_no"]) <= 1.0 + 1e-6
