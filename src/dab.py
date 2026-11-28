"""DAB (Dynamic Attention Balancing, MIHBench 논문 §4.2) 이식 + 실행 (docs/split_coco.md §4).

    python -m src.dab --model qwen2.5-vl-7b --questions data/splits/coco2017_seed1/val.jsonl --alpha 0.5 --base_ratio 0.2 \
        --out results/raw/coco2017_s1_dab_a0.5_val_qwen7b.csv [--limit 5 --verify]

공식 구현 (github.com/pgtrece/DAB, Qwen2.5-VL/modeling_qwen2_5_vl.py 790–834행, 2026-11-28 원문 확인):
  모든 decoder 층 attention 의 softmax 뒤 확률 P 에서, 마지막 이미지 뒤 텍스트 query 행마다
      Sᵢ = Σ_{k∈이미지 i} P[h, q, k],  T = Σᵢ Sᵢ,  avg = T / n,  δᵢ = Sᵢ − avg
      이미지 i 의 각 토큰에서 α·δᵢ / (이미지당 토큰 수) 를 뺀다  ->  Sᵢ' = (1−α)·Sᵢ + α·avg
  (행 합 유지. 재정규화·clamp 없음. 개별 값이 음수가 될 수 있다.) T > base_ratio 인 (head, 행)에만 적용.
  기본값 α=0.5, base_ratio=0.2 (Qwen 스크립트). 학습 없음.
이식에서 다른 점 (결과 해석 시 명시):
  - 이미지 구간: 공식은 vision_start 위치부터 "이미지당 같은 토큰 수"를 가정(672px resize)해 한 칸 어긋난다.
    여기서는 vlm.segments(inputs) 의 실제 이미지 토큰 구간을 쓰고 이미지별 토큰 수 nᵢ 로 나눈다
    (토큰 수가 같으면 공식과 같은 식). resize 가 필요 없어서 5개 모델 모두에 적용된다.
  - query 행: 공식은 마지막 이미지의 마지막 토큰부터. 여기서는 마지막 이미지 구간 바로 뒤(vision_end 등)부터 끝까지.
  - 정밀도: 공식은 bf16 확률을 고쳐 bf16 matmul. 여기서는 기본 출력을 sdpa(fp32 누적)로 계산하고,
    P 와 Δ 를 fp32 로 직접 계산해 출력[q0:] += Δ·V 를 더한다 ((P + Δ)V = PV + ΔV).
  - 래퍼 방식은 intervene.AttentionIntervention 과 같다: 모델을 attn_implementation="eager" 로 로드하고
    ALL_ATTENTION_FUNCTIONS["eager"] 에 래퍼를 임시 등록한다 (비전 인코더 attention 은 건드리지 않음).
--verify: 개입한 층마다 (a) 직접 계산한 P·V 와 sdpa 출력의 상대오차, (b) 적용된 (head, 행)에서
          이미지 몫 표준편차 비율(기대값 1−α)을 기록하고 문항마다 요약을 출력한다.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

from .attn_probe import repeat_kv
from .data import load_config
from .intervene import AttentionIntervention, layer_set


def dab_delta(P, spans, alpha: float, base_ratio: float):
    """P: [..., Q, L] fp32 확률 (행 = 텍스트 query). spans: 이미지 토큰 구간 [(k0, k1), ...] (half-open).
    반환 (delta, mask). delta 는 P 와 같은 모양이고 이미지 열에만 값이 있다. mask: [..., Q] (DAB 적용 여부)."""
    import torch
    S = torch.stack([P[..., a:b].sum(-1) for a, b in spans], dim=0)          # [n, ..., Q]
    T = S.sum(0)
    avg = T / len(spans)
    mask = T > base_ratio
    delta = torch.zeros_like(P)
    for i, (a, b) in enumerate(spans):
        d = alpha * (S[i] - avg) / (b - a) * mask                             # 토큰마다 뺄 양 [..., Q]
        delta[..., a:b] = -d.unsqueeze(-1)
    return delta, mask


def rows_attention(query, key, attention_mask, scaling: float, q0: int):
    """query 행 q0: 의 attention 확률 [B, H, Q, Lk] (fp32). attn_probe.last_row_attention 을 여러 행으로 일반화.
    attention_mask 가 None 이면 causal 마스크를 직접 만든다."""
    import torch
    n_rep = query.shape[1] // key.shape[1]
    k = repeat_kv(key, n_rep).float()
    q = query[:, :, q0:, :].float()
    scores = torch.matmul(q, k.transpose(2, 3)) * scaling                    # [B, H, Q, Lk]
    Lq, Lk = query.shape[2], k.shape[-2]
    if attention_mask is not None:
        m = attention_mask[:, :, q0:, :Lk]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()
    else:
        pos_q = torch.arange(q0, Lq, device=q.device)[:, None] + (Lk - Lq)
        pos_k = torch.arange(Lk, device=q.device)[None, :]
        scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
    return torch.softmax(scores, dim=-1)


class DABIntervention(AttentionIntervention):
    """with DABIntervention(vlm, layers, alpha, base_ratio) as iv: iv.set_segments(seg); vlm.forward_last_logits(inputs)"""

    def __init__(self, vlm, layers: set[int], alpha: float, base_ratio: float, verify: bool = False):
        super().__init__(vlm, layers, verify=verify, base="sdpa")
        self.alpha = float(alpha)
        self.base_ratio = float(base_ratio)
        self._seg = None
        self.mask_log: list[tuple[int, float]] = []
        self.verify_log: list[tuple[int, float, float]] = []

    def set_segments(self, seg):
        """문항마다 호출. 이미지 구간과 query 시작 위치(마지막 이미지 구간 끝)를 정한다."""
        if not seg.image_spans:
            raise ValueError("no image spans")
        self._seg = (list(seg.image_spans), int(seg.image_spans[-1][1]))
        self.mask_log, self.verify_log = [], []

    def masked_frac(self) -> float:
        """이번 forward 에서 DAB 가 적용된 (head, 행) 비율의 층 평균."""
        return float(sum(f for _, f in self.mask_log) / len(self.mask_log)) if self.mask_log else 0.0

    def _wrapped(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch
        base = self._registry._global_mapping["sdpa"]
        out, weights = base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        if (self._seg is None or self.alpha == 0.0 or not self.vlm.is_text_attention(module)
                or module.layer_idx not in self.layers):
            return out, weights
        if query.shape[0] != 1:
            raise ValueError("DAB intervention supports batch size 1 only")
        if query.shape[2] != key.shape[2]:
            raise ValueError(f"query len {query.shape[2]} != key len {key.shape[2]} (KV cache in use?)")
        spans, q0 = self._seg
        P = rows_attention(query, key, attention_mask, scaling, q0)            # [1, H, Q, Lk]
        delta, mask = dab_delta(P, spans, self.alpha, self.base_ratio)
        v = repeat_kv(value, query.shape[1] // value.shape[1]).float()        # [1, H, Lk, d]
        corr = torch.matmul(delta, v).transpose(1, 2)                         # [1, Q, H, d]  (출력은 [B, L, H, d])
        new = out.clone()
        new[:, q0:] = (out[:, q0:].float() + corr).to(out.dtype)
        self.mask_log.append((module.layer_idx, float(mask.float().mean())))
        if self.verify:
            pv = torch.matmul(P, v).transpose(1, 2)
            ref = out[:, q0:].float()
            rel = float((pv - ref).abs().max() / ref.abs().max().clamp_min(1e-12))
            S = torch.stack([P[..., a:b].sum(-1) for a, b in spans], 0)
            S2 = torch.stack([(P + delta)[..., a:b].sum(-1) for a, b in spans], 0)
            m = mask[0]
            ratio = float(S2[:, 0][:, m].std(0).mean() / S[:, 0][:, m].std(0).mean().clamp_min(1e-12)) \
                if bool(m.any()) else float("nan")
            self.verify_log.append((module.layer_idx, rel, ratio))
        return new, weights


def main(argv=None):
    from .run_eval import COLUMNS as E1_COLUMNS
    from .run_eval import common_args, score_from_logits, select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--base_ratio", type=float, default=0.2)
    ap.add_argument("--layers", default="all", help="all | early | late | range:a-b (공식: all)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model

    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)
    columns = E1_COLUMNS + ["alpha", "base_ratio", "layers", "masked_frac"]
    with ResumableCsvWriter(args.out, columns, flush_every=args.flush_every) as w:
        todo = [q for q in qs if q["qid"] not in w.done]
        print(f"{len(qs)} questions, {len(qs) - len(todo)} already done, {len(todo)} to run")
        if not todo:
            return
        vlm = load_model(args.model or mc["name"], attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=args.max_pixels or mc.get("max_pixels"), min_pixels=mc.get("min_pixels"),
                         device=args.device, image_labels=args.image_labels, adapter=args.adapter)
        layers = layer_set(args.layers, vlm.num_layers)
        t0, n_err = time.time(), 0
        with DABIntervention(vlm, layers, args.alpha, args.base_ratio, verify=args.verify) as iv:
            for i, q in enumerate(todo, 1):
                try:
                    images = load_images(q["image_paths"], cfg["coco"]["root"])
                    inputs = vlm.build_inputs(images, q["question"])
                    iv.set_segments(vlm.segments(inputs))
                    logits = vlm.forward_last_logits(inputs)
                    row = score_from_logits(vlm, logits, q, inputs)
                    row.update(alpha=args.alpha, base_ratio=args.base_ratio, layers=args.layers,
                               masked_frac=iv.masked_frac())
                    w.write(row)
                    if args.verify and iv.verify_log:
                        rels = [r for _, r, _ in iv.verify_log]
                        ratios = [x for _, _, x in iv.verify_log if x == x]
                        print(f"[verify] {q['qid']}: P·V vs sdpa rel err max {max(rels):.2e} | "
                              f"image-share std ratio mean {sum(ratios) / max(len(ratios), 1):.3f} "
                              f"(expected {1 - args.alpha:.3f}) | masked_frac {iv.masked_frac():.3f}", flush=True)
                except Exception:
                    n_err += 1
                    print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                    if n_err > 20:
                        raise RuntimeError("too many errors, aborting")
                if i % 50 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
