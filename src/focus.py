"""M3: 찾고 -> 집중하기 (detect-then-focus). attention 수정 + 이미지별 판독 결합.

1단계(검출): M2(isolated) 결과의 이미지별 m_i 로 결정적 이미지를 고른다.
             forall -> argmin_i m_i (객체가 없을 가능성이 가장 큰 이미지), exists -> argmax_i m_i.
2단계(집중): 원래 다중 이미지 forward 를 다시 하되, 그 이미지로 가는 attention 을 α배로 키워 최종 m 을 읽는다.

    python -m src.focus --model qwen2.5-vl-3b --m2 results/raw/m2_post_qwen3b.csv \
        --alphas 4 16 --query post_image after_target --out_dir results/raw/m3_qwen3b
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from .data import load_config


def main(argv=None):
    import pandas as pd

    from .intervene import AttentionIntervention, layer_set
    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model
    from .run_eval import COLUMNS as E1_COLUMNS
    from .run_eval import score_from_logits, select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--m2", required=True, help="isolated.py 출력 csv (m_list)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[4.0, 16.0])
    ap.add_argument("--query", nargs="+", default=["post_image"], choices=["post_image", "after_target"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mc = cfg["model"]
    m2 = pd.read_csv(args.m2).set_index("qid")
    qs = [q for q in select_questions(args.questions, args.limit) if q["qid"] in m2.index]
    cols = E1_COLUMNS + ["focus_pos", "alpha", "query"]
    vlm = None
    for qmode in args.query:
        for alpha in args.alphas:
            out = Path(args.out_dir) / f"focus_a{alpha:g}_{qmode}.csv"
            with ResumableCsvWriter(out, cols, flush_every=20) as w:
                todo = [q for q in qs if q["qid"] not in w.done]
                print(f"[{out.stem}] {len(qs)} questions, {len(todo)} to run", flush=True)
                if not todo:
                    continue
                if vlm is None:
                    vlm = load_model(args.model or mc["name"], attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                                     max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"),
                                     device=args.device)
                t0, n_err = time.time(), 0
                with AttentionIntervention(vlm, layer_set("all", vlm.num_layers)) as iv:
                    for i, q in enumerate(todo, 1):
                        try:
                            ms = json.loads(m2.loc[q["qid"], "m_list"])
                            fpos = min(range(len(ms)), key=ms.__getitem__) if q["qtype"].startswith("forall") \
                                else max(range(len(ms)), key=ms.__getitem__)
                            images = load_images(q["image_paths"], cfg["coco"]["root"])
                            inputs = vlm.build_inputs(images, q["question"])
                            seg = vlm.segments(inputs)
                            L = int(inputs["input_ids"].shape[1])
                            k0, k1 = seg.image_spans[fpos]
                            q0 = seg.image_spans[-1][1] + 1 if qmode == "post_image" else k1
                            iv.set((q0, L), (k0, k1), alpha)
                            logits = vlm.forward_last_logits(inputs)
                            row = score_from_logits(vlm, logits, q, inputs)
                            row.update(focus_pos=fpos, alpha=alpha, query=qmode)
                            w.write(row)
                        except Exception:
                            n_err += 1
                            print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                            if n_err > 20:
                                raise RuntimeError("too many errors, aborting")
                        if i % 100 == 0 or i == len(todo):
                            print(f"  {i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
