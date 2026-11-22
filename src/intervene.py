"""E3: attention 개입 (인과 검증). 특정 이미지로 가는 attention 을 α배로 조절했을 때 답(m)이 어떻게 바뀌는지.

    python -m src.intervene --model qwen2.5-vl-3b --qtypes forall_no \
        --conditions none:1 absent:2 absent:4 absent:0 present:2 present:4 \
        --out_dir results/raw/e3_qwen3b [--layers all|early|late] [--query post_image|last] [--limit 5 --verify]

구현:
  - attn_probe 와 같은 방식으로 ALL_ATTENTION_FUNCTIONS["eager"] 에 래퍼를 임시 등록 (모델은 eager 로 로드).
    실제 attention 계산은 기본적으로 sdpa (fp32 누적) 로 한다. eager 는 QKᵀ·softmax·PV 를 bf16 으로 해서
    7B 에서 m 이 sdpa 대비 크게 어긋났다 (N=2 평균 m: sdpa -2.26 vs eager +0.40, 답 일치율 73%).
  - 래퍼는 언어모델 attention 모듈(선택한 층)일 때만 additive mask 에 bias 를 더한 뒤 원래 eager 함수를 호출:
        bias[q, k] = log α   (q ∈ query 행, k ∈ 대상 이미지 토큰),  α=0 이면 -inf(=dtype min) → knockout
    softmax 전 score 에 log α 를 더하면 해당 행에서 대상 이미지 토큰들의 (정규화 전) 가중치가 정확히 α배가 된다.
  - query 행: post_image = 마지막 이미지(vision_end) 뒤의 모든 토큰(질문, 템플릿, 마지막 토큰),
              last = 마지막 토큰만,
              after_target = 대상 이미지 바로 뒤부터 끝까지(뒤쪽 이미지 토큰 포함). post_image 는 뒤쪽 이미지
                             토큰을 거쳐 대상 이미지 정보가 우회해 들어오는 경로를 막지 못하므로 이를 막는 변형.
  - 대상: absent = forall_no 의 객체 없는 이미지, present = 객체 있는 이미지 중 하나(qid 로 고정된 난수),
          none = 개입 없음(같은 eager 경로의 기준선).
  - 비전 인코더 attention 은 건드리지 않는다.
--verify: 지정 층에서 마지막 행의 대상 이미지 attention 몫이 기대대로 바뀌었는지(α·s/(α·s+1-s)) 출력.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
import traceback
from pathlib import Path

from .data import load_config


def parse_conditions(specs):
    out = []
    for s in specs:
        target, a = s.split(":")
        if target not in ("none", "absent", "present"):
            raise ValueError(f"unknown target {target}")
        out.append((target, float(a)))
    return out


def layer_set(mode: str, n_layers: int) -> set[int]:
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(n_layers // 2))
    if mode == "late":
        return set(range(n_layers // 2, n_layers))
    if mode.startswith("range:"):
        a, b = map(int, mode.split(":")[1].split("-"))
        return set(range(a, b + 1))
    raise ValueError(mode)


def pick_target(q: dict, target: str) -> int | None:
    if target == "none":
        return None
    if target == "absent":
        if len(q["absent_positions"]) != 1:
            raise ValueError(f"{q['qid']}: absent target needs exactly one absent image")
        return q["absent_positions"][0]
    if target == "present":
        rng = random.Random(f"present-target|{q['qid']}")
        return rng.choice(q["present_positions"])
    raise ValueError(target)


class AttentionIntervention:
    def __init__(self, vlm, layers: set[int], verify: bool = False, base: str = "sdpa"):
        """base: 실제 attention 계산 함수. "sdpa"(기본, fp32 누적) | "eager"(bf16 matmul — 7B 에서 부정확).
        모델은 attn_implementation="eager" 로 로드해 레지스트리 조회가 이 래퍼를 거치게 하고, 계산은 base 로 한다."""
        self.base = base
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.vlm = vlm
        self.layers = layers
        self.verify = verify
        self._registry = ALL_ATTENTION_FUNCTIONS
        self._bias_spec = None      # (q_rows slice, k_cols slice, log_alpha)
        self.verify_log: list[tuple[int, float, float, float]] = []

    def __enter__(self):
        reg = self._registry
        self._had_local = "eager" in reg._local_mapping
        self._prev_local = reg._local_mapping.get("eager")
        reg["eager"] = self._wrapped
        return self

    def __exit__(self, *exc):
        reg = self._registry
        if self._had_local:
            reg["eager"] = self._prev_local
        else:
            del reg["eager"]

    def set(self, q_rows: tuple[int, int] | None, k_cols: tuple[int, int] | None, alpha: float):
        """단일 블록: rows [q0,q1) x cols [k0,k1) 에 log alpha."""
        if q_rows is None or alpha == 1.0:
            self.set_blocks([], 1.0)
        else:
            self.set_blocks([(q_rows[0], q_rows[1], k_cols[0], k_cols[1])], alpha)

    def set_blocks(self, blocks: list[tuple[int, int, int, int]], alpha: float):
        """여러 블록 (q0, q1, k0, k1) 에 같은 alpha 적용. blocks 가 비면 개입 없음."""
        self._bias_spec = (blocks, alpha) if blocks and alpha != 1.0 else None
        self.verify_log = []

    def _wrapped(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        import torch
        if self.base == "sdpa":
            base = self._registry._global_mapping["sdpa"]
        else:
            base = sys.modules[type(module).__module__].eager_attention_forward
        spec = self._bias_spec
        if spec is not None and self.vlm.is_text_attention(module) and module.layer_idx in self.layers:
            if query.shape[0] != 1:
                raise ValueError("intervention supports batch size 1 only")
            blocks, alpha = spec
            Lq, Lk = query.shape[2], key.shape[2]
            if Lq != Lk:
                raise ValueError(f"query len {Lq} != key len {Lk} (KV cache in use?)")
            if attention_mask is None:
                m = torch.zeros(1, 1, Lq, Lk, dtype=query.dtype, device=query.device)
                m = m.masked_fill(torch.ones(Lq, Lk, dtype=torch.bool, device=query.device).triu(1),
                                  torch.finfo(query.dtype).min)
            elif attention_mask.dtype == torch.bool:
                m = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device)
                m = m.masked_fill(~attention_mask, torch.finfo(query.dtype).min)
            else:
                m = attention_mask[:, :, :, :Lk].clone()
            fmin = torch.finfo(m.dtype).min
            for q0, q1, k0, k1 in blocks:
                if alpha == 0.0:
                    m[:, :, q0:q1, k0:k1] = fmin
                else:
                    m[:, :, q0:q1, k0:k1] = (m[:, :, q0:q1, k0:k1] + math.log(alpha)).clamp_min(fmin)
            if self.verify and len(blocks) == 1:
                from .attn_probe import last_row_attention
                q0, q1, k0, k1 = blocks[0]
                s0 = last_row_attention(query, key, attention_mask, scaling)[0, :, k0:k1].sum(-1)
                after = last_row_attention(query, key, m, scaling)[0, :, k0:k1].sum(-1).mean().item()
                expected = (alpha * s0 / (alpha * s0 + 1 - s0)).mean().item()
                self.verify_log.append((module.layer_idx, s0.mean().item(), after, expected))
            attention_mask = m
        return base(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)


def main(argv=None):
    from .run_eval import COLUMNS as E1_COLUMNS
    from .run_eval import common_args, score_from_logits, select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # common_args 의 --out 은 required 이므로 out_dir 로 대체
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conditions", nargs="+", default=["none:1", "absent:2", "absent:4", "absent:0",
                                                        "present:2", "present:4"])
    ap.add_argument("--layers", default="all", help="all | early | late | range:a-b")
    ap.add_argument("--query", default="post_image", choices=["post_image", "last", "after_target"])
    ap.add_argument("--verify", action="store_true")
    common_args(ap)
    args = ap.parse_args(argv + ["--out", "_unused"] if argv else sys.argv[1:] + ["--out", "_unused"])

    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model

    cfg = load_config(args.config)
    mc = cfg["model"]
    qtypes = args.qtypes or ["forall_no"]
    qs = select_questions(args.questions, args.limit, args.Ns, qtypes)
    conds = parse_conditions(args.conditions)
    out_dir = Path(args.out_dir)
    columns = E1_COLUMNS + ["target", "alpha", "target_pos", "layers", "query"]
    model_name = args.model or mc["name"]

    vlm = None
    for target, alpha in conds:
        tag = f"{target}_a{alpha:g}_{args.layers.replace(':', '')}_{args.query}"
        out = out_dir / f"{tag}.csv"
        with ResumableCsvWriter(out, columns, flush_every=args.flush_every) as w:
            todo = [q for q in qs if q["qid"] not in w.done]
            print(f"[{tag}] {len(qs)} questions, {len(todo)} to run", flush=True)
            if not todo:
                continue
            if vlm is None:
                vlm = load_model(model_name, attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                                 max_pixels=args.max_pixels or mc.get("max_pixels"),
                                 min_pixels=mc.get("min_pixels"), device=args.device,
                                 image_labels=args.image_labels)
                layers = layer_set(args.layers, vlm.num_layers)
            t0, n_err = time.time(), 0
            with AttentionIntervention(vlm, layers, verify=args.verify) as iv:
                for i, q in enumerate(todo, 1):
                    try:
                        images = load_images(q["image_paths"], cfg["coco"]["root"])
                        inputs = vlm.build_inputs(images, q["question"])
                        seg = vlm.segments(inputs)
                        L = int(inputs["input_ids"].shape[1])
                        tpos = pick_target(q, target)
                        if tpos is None:
                            iv.set(None, None, 1.0)
                        else:
                            k0, k1 = seg.image_spans[tpos]
                            if args.query == "post_image":
                                q0 = seg.image_spans[-1][1] + 1
                            elif args.query == "last":
                                q0 = L - 1
                            else:   # after_target: 대상 이미지 뒤의 모든 토큰(뒤쪽 이미지 토큰 포함) -> 우회 경로 차단
                                q0 = k1
                            iv.set((q0, L), (k0, k1), alpha)
                        logits = vlm.forward_last_logits(inputs)
                        row = score_from_logits(vlm, logits, q, inputs)
                        row.update(target=target, alpha=alpha, target_pos=tpos, layers=args.layers,
                                   query=args.query)
                        w.write(row)
                        if args.verify and iv.verify_log:
                            for li, b, a_, e in iv.verify_log[:: max(1, len(iv.verify_log) // 4)]:
                                print(f"  [verify] {q['qid']} layer {li}: target share {b:.4f} -> {a_:.4f} "
                                      f"(expected {e:.4f})")
                    except Exception:
                        n_err += 1
                        print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                        if n_err > 20:
                            raise RuntimeError("too many errors, aborting")
                    if i % 100 == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"  {i}/{len(todo)}  {el / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
