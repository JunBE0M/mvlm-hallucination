"""E2 probe 검증.

1) 수식: last_row_attention == transformers eager_attention_forward 전체 행렬의 마지막 행 (GQA, 마스크 포함)
2) 소형 랜덤 Qwen2.5-VL: probe 가 모델 출력을 바꾸지 않고, eager attn_weights 마지막 행과 일치하며,
   언어모델 층만 기록하고, 구간 합이 1 이 되고, 종료 후 레지스트리가 원복되는지.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward  # noqa: E402

from src.attn_probe import LastRowAttentionProbe, last_row_attention, repeat_kv  # noqa: E402


class _M(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.num_key_value_groups = groups


@pytest.mark.parametrize("H,Hkv", [(4, 4), (8, 2), (28, 4)])
@pytest.mark.parametrize("mask_kind", ["none", "causal_float", "padded_bool"])
def test_last_row_matches_eager(H, Hkv, mask_kind):
    g = torch.Generator().manual_seed(0)
    B, L, d = 1, 13, 16
    q = torch.randn(B, H, L, d, generator=g)
    k = torch.randn(B, Hkv, L, d, generator=g)
    v = torch.randn(B, Hkv, L, d, generator=g)
    causal = torch.ones(L, L).tril().bool()
    if mask_kind == "padded_bool":
        causal[:, :2] = False          # 앞쪽 2개 토큰을 패딩처럼 가림
    if mask_kind == "none":
        mask_f = mask_in = None
    else:
        mask_f = torch.where(causal, 0.0, torch.finfo(torch.float32).min)[None, None]
        mask_in = causal[None, None] if mask_kind == "padded_bool" else mask_f
    _, w = eager_attention_forward(_M(H // Hkv), q, k, v, mask_f, scaling=d ** -0.5)
    p = last_row_attention(q, k, mask_in, d ** -0.5)
    assert p.shape == (B, H, L)
    assert torch.allclose(p, w[:, :, -1, :], atol=1e-6)
    assert torch.allclose(p.sum(-1), torch.ones(B, H))


def test_repeat_kv_order():
    x = torch.arange(2 * 3).float().reshape(1, 2, 3, 1)       # 2 kv heads
    r = repeat_kv(x, 3)
    assert r.shape == (1, 6, 3, 1)
    assert torch.equal(r[0, :3], x[0, [0, 0, 0]]) and torch.equal(r[0, 3:], x[0, [1, 1, 1]])


# ---------------------------------------------------------------- tiny model
@pytest.fixture(scope="module")
def vlm():
    from tiny_qwen import make_tiny_vlm
    return make_tiny_vlm("eager")


def _inputs(vlm, n=3):
    imgs = [Image.new("RGB", (640, 480) if i % 2 else (480, 640), (60 * i, 30, 200)) for i in range(n)]
    return vlm.build_inputs(imgs, f"Is there a cat in all {n} images? Please answer with Yes or No.")


def test_probe_on_tiny_model(vlm):
    inp = _inputs(vlm, 3)
    before = vlm.forward_last_logits(inp)
    assert "eager" not in ALL_ATTENTION_FUNCTIONS._local_mapping

    with LastRowAttentionProbe(vlm, impl="eager", verify=True) as probe:
        seg = vlm.segments(inp)
        probe.set_segments(seg, inp["input_ids"].device)
        during = vlm.forward_last_logits(inp)
        res = probe.result()

    assert "eager" not in ALL_ATTENTION_FUNCTIONS._local_mapping          # 원복
    assert torch.allclose(before, during)                                 # 모델 출력 불변
    assert probe.max_abs_diff < 1e-5                                       # eager 마지막 행과 일치
    L, H = vlm.num_layers, vlm.num_heads
    assert res["img_sum"].shape == (L, H, 3) and res["img_mean"].shape == (L, H, 3)
    assert res["text_sum"].shape == (L, H)
    total = res["img_sum"].sum(-1) + res["text_sum"] + res["sys_sum"] + res["marker_sum"]
    np.testing.assert_allclose(total, 1.0, atol=1e-5)
    np.testing.assert_allclose(res["img_mean"] * res["n_tokens"], res["img_sum"], rtol=1e-5)
    assert res["n_tokens"].tolist() == seg.n_tokens().tolist()


def test_probe_matches_output_attentions(vlm):
    """작은 입력에서는 output_attentions=True 결과와 직접 대조 가능."""
    inp = _inputs(vlm, 2)
    with torch.inference_mode():
        out = vlm.model(**inp, output_attentions=True, use_cache=False)
    att = out.attentions                                                   # tuple of [1, H, L, L]
    with LastRowAttentionProbe(vlm) as probe:
        seg = vlm.segments(inp)
        probe.set_segments(seg, inp["input_ids"].device)
        vlm.forward_last_logits(inp)
        res = probe.result()
    for layer, a in enumerate(att):
        last = a[0, :, -1, :].float().numpy()
        for i, (s, e) in enumerate(seg.image_spans):
            np.testing.assert_allclose(res["img_sum"][layer, :, i], last[:, s:e].sum(-1), atol=1e-5)
        np.testing.assert_allclose(res["sys_sum"][layer], last[:, seg.labels == seg.system_label].sum(-1),
                                   atol=1e-5)


def test_probe_restores_on_error(vlm):
    with pytest.raises(RuntimeError):
        with LastRowAttentionProbe(vlm):
            raise RuntimeError("boom")
    assert "eager" not in ALL_ATTENTION_FUNCTIONS._local_mapping


def test_attn_probe_cli_end_to_end(vlm, tmp_path, monkeypatch):
    import csv
    import json

    import yaml

    from src import attn_probe as A
    from src import models as M
    root = tmp_path / "coco"
    (root / "val2017").mkdir(parents=True)
    qs = []
    for k, N in enumerate((2, 3)):
        paths = []
        for i in range(N):
            p = f"val2017/{k}_{i}.jpg"
            Image.new("RGB", (320, 240), (50 * i, 100, 150)).save(root / p)
            paths.append(p)
        qs.append({"qid": f"q{k}", "pair_id": None, "cls": "cat", "N": N, "qtype": "forall_no", "answer": "No",
                   "question": f"Is there a cat in all {N} images? Please answer with Yes or No.",
                   "image_ids": list(range(N)), "image_paths": paths, "present_positions": list(range(1, N)),
                   "absent_positions": [0], "area_ratios": [0.0] + [0.1] * (N - 1)})
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in qs))
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump({"coco": {"root": str(root)},
                                     "model": {"name": "tiny", "attn_impl": "eager", "max_pixels": 12544}}))
    calls = []
    monkeypatch.setattr(M, "load_model", lambda *a, **k: calls.append(k) or vlm)
    out = tmp_path / "e2.csv"
    args = ["--questions", str(qpath), "--config", str(cpath), "--out", str(out), "--device", "cpu", "--verify"]
    A.main(args)
    assert calls[0]["attn_impl"] == "eager"
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert [r["qid"] for r in rows] == ["q0", "q1"]
    for r in rows:
        z = np.load(r["npz"])
        assert z["img_sum"].shape == (vlm.num_layers, vlm.num_heads, int(r["N"]))
        assert float(r["verify_max_abs_diff"]) < 1e-5
    A.main(args)                     # 재실행: 모두 skip → 모델 로드 안 함
    assert len(calls) == 1
