"""E2: 마지막 query 토큰(답을 내는 위치)이 각 이미지/텍스트/시스템 구간에 주는 attention 총량을 층·헤드별로 저장.

    python -m src.attn_probe --model qwen2.5-vl-3b --out results/raw/e2_qwen3b.csv [--limit 2 --verify]

구현 방식 (transformers 5.22.2 소스 확인 결과에 근거):
  - Qwen2_5_VLAttention.forward 는 q/k 에 mrope 를 적용(+KV cache 갱신)한 뒤
    `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` 로 고른 함수를
    `fn(module, query, key, value, attention_mask, scaling=..., dropout=..., **kw)` 형태로 호출한다.
    query: [B, H, Lq, d], key: [B, H_kv, Lk, d] (GQA: H = H_kv * module.num_key_value_groups).
  - "eager" 는 레지스트리에 기본 등록되어 있지 않아 모델 파일의 eager_attention_forward 가 쓰인다.
    여기서 ALL_ATTENTION_FUNCTIONS["eager"] 에 래퍼를 *임시로* 등록(인스턴스 local mapping, 종료 시 원복)하면
    모든 attention 호출이 래퍼를 거친다.
  - 래퍼는 (1) 원래 eager 함수로 출력을 계산해 그대로 반환하고(모델 동작 불변),
    (2) 언어모델 attention 모듈일 때만(비전 인코더 attention 은 같은 함수를 쓰므로 isinstance 로 제외)
        q_last = query[:, :, -1], K = repeat_kv(key) 로 softmax(q_last·Kᵀ·scaling + mask[-1]) 를 fp32 로 직접 계산,
        곧바로 구간(one-hot [S, L])에 곱해 [H, S] 로 축약해 CPU 로 옮긴다.
    → output_attentions=True 처럼 [H, L, L] 를 모든 층에 대해 쌓지 않는다 (층당 저장량 H×S).
  - --verify: eager 가 함께 반환하는 attn_weights 의 마지막 행과 직접 계산한 행의 최대 절대오차를 기록.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .data import load_config


def repeat_kv(x, n_rep: int):
    """[B, H_kv, L, d] -> [B, H_kv*n_rep, L, d] (transformers repeat_kv 와 동일한 순서: kv head 마다 연속 복제)."""
    if n_rep == 1:
        return x
    b, h, l, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)


def last_row_attention(query, key, attention_mask, scaling: float):
    """마지막 query 행의 attention 확률 [B, H, Lk] (fp32)."""
    import torch
    n_rep = query.shape[1] // key.shape[1]
    k = repeat_kv(key, n_rep).float()
    q = query[:, :, -1:, :].float()
    scores = torch.matmul(q, k.transpose(2, 3)).squeeze(2) * scaling          # [B, H, Lk]
    if attention_mask is not None:
        m = attention_mask[:, :, -1, : k.shape[-2]]                             # [B, 1, Lk]
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()
    return torch.softmax(scores, dim=-1)


class LastRowAttentionProbe:
    """with LastRowAttentionProbe(vlm) as probe: probe.set_segments(seg); vlm.forward_last_logits(inputs); probe.result()"""

    def __init__(self, vlm, impl: str = "eager", verify: bool = False, base: str | None = None):
        """impl: 래퍼를 등록할 레지스트리 키(모델 로드 시 attn_implementation). base: 실제 출력 계산 함수.
        base="sdpa" 가 기본 (eager 의 bf16 matmul 은 7B 에서 출력이 크게 어긋남). --verify 는 base="eager" 필요."""
        self.base = base if base is not None else ("eager" if verify else "sdpa")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.vlm = vlm
        self.impl = impl
        self.verify = verify
        self._registry = ALL_ATTENTION_FUNCTIONS
        self._onehot = None
        self._store: dict[int, np.ndarray] = {}
        self.max_abs_diff = 0.0

    # --- registry patch -------------------------------------------------
    def __enter__(self):
        reg = self._registry
        self._had_local = self.impl in reg._local_mapping
        self._prev_local = reg._local_mapping.get(self.impl)
        self._sdpa_fn = reg._global_mapping.get("sdpa")
        reg[self.impl] = self._wrapped
        return self

    def __exit__(self, *exc):
        reg = self._registry
        if self._had_local:
            reg[self.impl] = self._prev_local
        else:
            del reg[self.impl]

    def _base_fn(self, module):
        if self.base == "sdpa":
            return self._sdpa_fn
        if self.impl == "eager":
            return sys.modules[type(module).__module__].eager_attention_forward
        if self.impl == "sdpa":
            return self._sdpa_fn
        raise ValueError(f"unsupported base impl {self.impl}")

    # --- per-question state ---------------------------------------------
    def set_segments(self, seg, device):
        import torch
        labels = torch.as_tensor(seg.labels, device=device)
        self._onehot = torch.nn.functional.one_hot(labels, seg.n_labels).float().T.contiguous()   # [S, L]
        self._seg = seg
        self._store = {}
        self.max_abs_diff = 0.0

    def _wrapped(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        out = self._base_fn(module)(module, query, key, value, attention_mask,
                                    scaling=scaling, dropout=dropout, **kwargs)
        if self._onehot is not None and self.vlm.is_text_attention(module):
            if query.shape[0] != 1:
                raise ValueError("attn probe supports batch size 1 only")
            p = last_row_attention(query, key, attention_mask, scaling)[0]        # [H, L]
            if p.shape[-1] != self._onehot.shape[1]:
                raise ValueError(f"key length {p.shape[-1]} != segment length {self._onehot.shape[1]}"
                                 " (KV cache in use?)")
            self._store[module.layer_idx] = (p @ self._onehot.T).cpu().numpy()   # [H, S]
            if self.verify:
                w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                if w is None:
                    raise RuntimeError("--verify needs the eager implementation (attn_weights returned)")
                d = (w[0, :, -1, :].float() - p).abs().max().item()
                self.max_abs_diff = max(self.max_abs_diff, d)
        return out

    def result(self) -> dict[str, np.ndarray]:
        n_layers = self.vlm.num_layers
        missing = [i for i in range(n_layers) if i not in self._store]
        if missing:
            raise RuntimeError(f"no attention recorded for layers {missing}")
        a = np.stack([self._store[i] for i in range(n_layers)])                 # [L, H, S]
        seg = self._seg
        n = seg.n_images
        n_tok = seg.n_tokens()
        img_sum = a[:, :, :n]
        return {
            "img_sum": img_sum.astype(np.float32),                                  # [L, H, N]
            "img_mean": (img_sum / np.maximum(n_tok, 1)).astype(np.float32),        # [L, H, N] 토큰당 평균
            "text_sum": a[:, :, seg.text_label].astype(np.float32),                 # [L, H]
            "sys_sum": a[:, :, seg.system_label].astype(np.float32),                # [L, H]
            "marker_sum": a[:, :, seg.marker_label].astype(np.float32),             # [L, H] vision_start/end 토큰
            "n_tokens": n_tok.astype(np.int32),                                     # [N]
            "n_text_tokens": np.int32((seg.labels == seg.text_label).sum()),
            "n_sys_tokens": np.int32((seg.labels == seg.system_label).sum()),
        }


def main(argv=None):
    from .run_eval import COLUMNS as E1_COLUMNS
    from .run_eval import common_args, score_from_logits, select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--npz_dir", default=None, help="기본: <out 경로 stem>_npz/")
    ap.add_argument("--verify", action="store_true", help="eager attn_weights 마지막 행과 대조")
    args = ap.parse_args(argv)

    from .io_utils import ResumableCsvWriter, atomic_save_npz
    from .models import load_images, load_model

    cfg = load_config(args.config)
    mc = cfg["model"]
    out = Path(args.out)
    npz_dir = Path(args.npz_dir) if args.npz_dir else out.with_name(out.stem + "_npz")
    columns = E1_COLUMNS + ["npz", "verify_max_abs_diff"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)

    with ResumableCsvWriter(out, columns, flush_every=args.flush_every) as w:
        todo = [q for q in qs if not (q["qid"] in w.done and (npz_dir / f"{q['qid']}.npz").exists())]
        print(f"{len(qs)} questions, {len(qs) - len(todo)} already done, {len(todo)} to run")
        if not todo:
            return
        vlm = load_model(args.model or mc["name"], attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=args.max_pixels or mc.get("max_pixels"),
                         min_pixels=mc.get("min_pixels"), device=args.device,
                         image_labels=args.image_labels)
        t0, n_err = time.time(), 0
        with LastRowAttentionProbe(vlm, impl="eager", verify=args.verify,
                                   base="eager" if args.verify else "sdpa") as probe:
            for i, q in enumerate(todo, 1):
                try:
                    images = load_images(q["image_paths"], cfg["coco"]["root"])
                    inputs = vlm.build_inputs(images, q["question"])
                    seg = vlm.segments(inputs)
                    probe.set_segments(seg, inputs["input_ids"].device)
                    logits = vlm.forward_last_logits(inputs)
                    res = probe.result()
                    row = score_from_logits(vlm, logits, q, inputs)
                    npz_path = npz_dir / f"{q['qid']}.npz"
                    atomic_save_npz(npz_path, **res, logit_yes=np.float32(row["logit_yes"]),
                                    logit_no=np.float32(row["logit_no"]))
                    row["npz"] = str(npz_path)
                    row["verify_max_abs_diff"] = probe.max_abs_diff if args.verify else None
                    if args.verify:
                        print(f"[verify] {q['qid']}: max|eager_last_row - probe| = {probe.max_abs_diff:.2e}")
                    w.write(row)
                except Exception:
                    n_err += 1
                    print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                    if n_err > 20:
                        raise RuntimeError("too many errors, aborting")
                if i % 20 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"{i}/{len(todo)}  {el / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
