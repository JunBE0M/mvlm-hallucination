"""src/dab.py (DAB 이식) 검증.

1) dab_delta 수식: 행 합 유지, Sᵢ' = (1−α)Sᵢ + α·avg, base_ratio 이하 행은 그대로, α=0 이면 변화 없음,
   이미지 토큰 수가 같을 때 공식 구현(DAB repo reallocation_img_attn_ratio_avg)의 식과 일치.
2) rows_attention == transformers eager_attention_forward 의 해당 행 (GQA, causal 마스크 / 마스크 없음).
3) 소형 랜덤 Qwen2.5-VL: α=0 이면 logits 불변, α>0 이면 전체 행렬로 (P+Δ)V 를 직접 계산한 기준과 일치,
   예외 시 레지스트리 원복, CLI end-to-end.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from PIL import Image  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import eager_attention_forward  # noqa: E402

from src.attn_probe import repeat_kv  # noqa: E402
from src.dab import DABIntervention, dab_delta, rows_attention  # noqa: E402


def _probs(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(*shape, generator=g) * 2, dim=-1)


SPANS = [(2, 6), (6, 11), (13, 17)]          # 토큰 수 4, 5, 4 (일부러 다르게)


def test_dab_delta_preserves_rows_and_moves_shares_to_mean():
    P = _probs((3, 5, 20))
    a = 0.5
    d, mask = dab_delta(P, SPANS, a, 0.0)
    assert bool(mask.all())
    torch.testing.assert_close((P + d).sum(-1), P.sum(-1))
    S = torch.stack([P[..., s:e].sum(-1) for s, e in SPANS])
    S2 = torch.stack([(P + d)[..., s:e].sum(-1) for s, e in SPANS])
    torch.testing.assert_close(S2, (1 - a) * S + a * S.mean(0, keepdim=True))
    other = torch.ones(20, dtype=torch.bool)
    for s, e in SPANS:
        other[s:e] = False
    assert torch.count_nonzero(d[..., other]) == 0
    d0, _ = dab_delta(P, SPANS, 0.0, 0.0)
    assert torch.count_nonzero(d0) == 0


def test_dab_delta_base_ratio_mask():
    P = _probs((2, 6, 20), seed=1)
    T = torch.stack([P[..., s:e].sum(-1) for s, e in SPANS]).sum(0)
    br = float(T.median())
    d, mask = dab_delta(P, SPANS, 0.7, br)
    torch.testing.assert_close(mask, T > br)
    assert torch.count_nonzero(d[~mask]) == 0
    assert torch.count_nonzero(d[mask]) > 0


def test_dab_delta_matches_official_formula_with_equal_tokens():
    """공식 구현의 식을 그대로 옮겨 계산: 이미지마다 (Sᵢ − T/n)·α / img_token_num 을 토큰마다 뺀다 (T > base_ratio 인 곳만)."""
    P = _probs((4, 7, 24), seed=2)
    starts, tok = [3, 9, 15], 5
    spans = [(s, s + tok) for s in starts]
    alpha, br = 0.5, 0.2
    ref = P.clone()
    sums = torch.stack([P[..., s:s + tok].sum(-1) for s in starts])
    total = sums.sum(0)
    scale = (sums - total / len(starts)) * alpha / tok * (total > br)
    for i, s in enumerate(starts):
        ref[..., s:s + tok] -= scale[i].unsqueeze(-1)
    d, _ = dab_delta(P, spans, alpha, br)
    torch.testing.assert_close(P + d, ref)


class _M(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.num_key_value_groups = groups


@pytest.mark.parametrize("H,Hkv", [(4, 4), (8, 2)])
@pytest.mark.parametrize("with_mask", [True, False])
def test_rows_attention_matches_eager(H, Hkv, with_mask):
    torch.manual_seed(0)
    L, d, q0 = 12, 8, 7
    q, k, v = torch.randn(1, H, L, d), torch.randn(1, Hkv, L, d), torch.randn(1, Hkv, L, d)
    causal = torch.zeros(1, 1, L, L).masked_fill(torch.ones(L, L, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
    _, w = eager_attention_forward(_M(H // Hkv), q, k, v, causal, scaling=d ** -0.5)
    got = rows_attention(q, k, causal if with_mask else None, d ** -0.5, q0)
    torch.testing.assert_close(got, w[:, :, q0:, :].float(), atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- tiny model
@pytest.fixture(scope="module")
def vlm():
    from tiny_qwen import make_tiny_vlm
    return make_tiny_vlm("eager")


def _inputs(vlm, n=3):
    imgs = [Image.new("RGB", (640, 480) if i % 2 else (480, 640), (60 * i, 30, 200)) for i in range(n)]
    return vlm.build_inputs(imgs, f"Is there a cat in all {n} images? Please answer with Yes or No.")


class _FullMatrixDAB:
    """기준 구현: 언어모델 attention 에서 전체 [H, L, L] 확률을 fp32 로 만들고 q0: 행에 Δ 를 더해 (P+Δ)V."""

    def __init__(self, vlm, spans, q0, alpha, br):
        self.vlm, self.spans, self.q0, self.alpha, self.br = vlm, spans, q0, alpha, br

    def __enter__(self):
        ALL_ATTENTION_FUNCTIONS["eager"] = self.fn
        return self

    def __exit__(self, *exc):
        del ALL_ATTENTION_FUNCTIONS["eager"]

    def fn(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if not self.vlm.is_text_attention(module):
            return eager_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)
        n_rep = query.shape[1] // key.shape[1]
        k, v = repeat_kv(key, n_rep).float(), repeat_kv(value, n_rep).float()
        s = torch.matmul(query.float(), k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            s = s + attention_mask[:, :, :, : k.shape[-2]].float()
        P = torch.softmax(s, dim=-1)
        d, _ = dab_delta(P[:, :, self.q0:], self.spans, self.alpha, self.br)
        P = P.clone()
        P[:, :, self.q0:] += d
        return torch.matmul(P, v).transpose(1, 2).to(query.dtype).contiguous(), None


def test_alpha_zero_keeps_logits(vlm):
    inp = _inputs(vlm, 3)
    before = vlm.forward_last_logits(inp)
    with DABIntervention(vlm, set(range(vlm.num_layers)), alpha=0.0, base_ratio=0.2) as iv:
        iv.set_segments(vlm.segments(inp))
        during = vlm.forward_last_logits(inp)
    assert "eager" not in ALL_ATTENTION_FUNCTIONS._local_mapping
    torch.testing.assert_close(during, before, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("alpha,br", [(0.5, 0.0), (1.0, 0.0), (0.5, 0.05)])
def test_matches_full_matrix_reference(vlm, alpha, br):
    inp = _inputs(vlm, 3)
    seg = vlm.segments(inp)
    base = vlm.forward_last_logits(inp)
    with DABIntervention(vlm, set(range(vlm.num_layers)), alpha=alpha, base_ratio=br, verify=True) as iv:
        iv.set_segments(seg)
        got = vlm.forward_last_logits(inp)
        log, frac = list(iv.verify_log), iv.masked_frac()
    with _FullMatrixDAB(vlm, list(seg.image_spans), seg.image_spans[-1][1], alpha, br):
        ref = vlm.forward_last_logits(inp)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    assert len(log) == vlm.num_layers and frac > 0
    assert max(r for _, r, _ in log) < 1e-4                                  # 직접 계산한 P 가 sdpa 와 일치
    ratios = [x for _, _, x in log if x == x]
    np.testing.assert_allclose(ratios, 1 - alpha, atol=1e-4)                 # 이미지 몫 편차가 (1−α)배
    if br == 0.0:
        assert not torch.allclose(got, base)                                 # 실제로 출력이 바뀜


def test_restores_registry_on_error(vlm):
    with pytest.raises(RuntimeError):
        with DABIntervention(vlm, {0}, 0.5, 0.2):
            raise RuntimeError("boom")
    assert "eager" not in ALL_ATTENTION_FUNCTIONS._local_mapping


def test_dab_cli_end_to_end(vlm, tmp_path, monkeypatch):
    import csv
    import json

    import yaml

    from src import dab as D
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
                                     "model": {"name": "tiny", "attn_impl": "sdpa", "max_pixels": 12544}}))
    calls = []
    monkeypatch.setattr(M, "load_model", lambda *a, **k: calls.append(k) or vlm)
    out = tmp_path / "dab.csv"
    args = ["--questions", str(qpath), "--config", str(cpath), "--out", str(out), "--device", "cpu",
            "--alpha", "0.5", "--base_ratio", "0.0", "--verify"]
    D.main(args)
    assert calls[0]["attn_impl"] == "eager"
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert [r["qid"] for r in rows] == ["q0", "q1"]
    assert all(float(r["alpha"]) == 0.5 and float(r["masked_frac"]) > 0 for r in rows)
    D.main(args)                     # 재실행: 모두 skip -> 모델 로드 안 함
    assert len(calls) == 1
