"""층별 logit lens: 마지막 위치 hidden state 를 각 층에서 final norm + lm_head 로 읽어 m_l = logit(Yes) - logit(No).

    python -m src.lens --model qwen2.5-vl-3b --out results/raw/lens_qwen3b.csv [--limit 5]

출력: E1 메타 열 + m_layers (json, 길이 = 층 수 + 1, 0 = 임베딩) + m_final (실제 forward 의 m, 검증용).
분석(src.agg_multi --lens)에서 층마다 β 를 피팅해 ∃/∀ 집계가 어느 층에서 형성되는지 본다.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from .data import load_config

COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "image_ids", "present_positions", "absent_positions",
        "m_layers", "m_final", "lens_last_minus_final", "model"]


def main(argv=None):
    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model
    from .run_eval import select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--qtypes", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch
    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)
    with ResumableCsvWriter(args.out, COLS, flush_every=20) as w:
        todo = [q for q in qs if q["qid"] not in w.done]
        print(f"{len(qs)} questions, {len(todo)} to run", flush=True)
        if not todo:
            return
        vlm = load_model(args.model, attn_impl="sdpa", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"), device=args.device)
        at = vlm.answer_tokens
        lm = vlm.model.model.language_model
        norm = lm.norm
        W = vlm.model.lm_head.weight.detach()
        w_diff = (W[at.yes].float() - W[at.no].float())          # m = (W_yes - W_no) · norm(h)
        t0, n_err = time.time(), 0
        for i, q in enumerate(todo, 1):
            try:
                images = load_images(q["image_paths"], cfg["coco"]["root"])
                inputs = vlm.build_inputs(images, q["question"])
                with torch.inference_mode():
                    out = vlm.model.model(**inputs, use_cache=False, output_hidden_states=True)
                    hs = out.hidden_states                     # tuple(L+1) of [1, T, D]
                    last = torch.stack([h[0, -1] for h in hs])  # [L+1, D]
                    # 마지막 원소가 이미 norm 되어 있을 수 있으므로 final 은 out[0] (=last_hidden_state, norm 후) 로 따로 계산
                    m_l = (norm(last).float() @ w_diff).tolist()
                    m_final = float(out[0][0, -1].float() @ w_diff)
                    m_l[-1] = m_final
                w.write(dict(qid=q["qid"], pair_id=q["pair_id"], cls=q["cls"], N=q["N"], qtype=q["qtype"],
                             answer=q["answer"], image_ids=q["image_ids"], present_positions=q["present_positions"],
                             absent_positions=q["absent_positions"], m_layers=json.dumps([round(x, 4) for x in m_l]),
                             m_final=m_final, lens_last_minus_final=float(norm(last[-1:]).float()[0] @ w_diff) - m_final,
                             model=vlm.name))
            except Exception:
                n_err += 1
                print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                if n_err > 20:
                    raise RuntimeError("too many errors, aborting")
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
