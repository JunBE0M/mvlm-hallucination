"""M1 기준선: 이미지마다 따로 "Is there a {obj} in the image?" 를 묻고 m_i 를 얻은 뒤 규칙으로 집계.

    python -m src.per_image --model qwen2.5-vl-3b --out results/raw/m1_single_qwen3b.csv

1) questions.jsonl 에 등장하는 고유 (cls, image_id) 쌍마다 단일 이미지 m_i = logit(Yes) - logit(No) 를 계산 (재시작 가능).
2) 집계는 analyze 단계에서: forall -> min_i m_i, exists -> max_i m_i (그 외 mean 도 비교용으로 저장).
   python -m src.per_image --aggregate --single results/raw/m1_single_qwen3b.csv --out results/raw/m1_agg_qwen3b.csv
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from .data import load_config, read_jsonl

SINGLE_TEMPLATE = "Is there a {obj} in the image? Please answer with Yes or No."
SINGLE_COLS = ["key", "cls", "image_id", "image_path", "present", "logit_yes", "logit_no", "m", "model"]
AGG_COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "m_min", "m_max", "m_mean", "m", "pred", "correct",
            "rule", "model"]


def run_single(args):
    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model
    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = read_jsonl(args.questions)
    items = {}
    for q in qs:
        for iid, p, pos in zip(q["image_ids"], q["image_paths"], range(q["N"])):
            key = f"{q['cls'].replace(' ', '_')}|{iid}"
            items.setdefault(key, dict(key=key, cls=q["cls"], image_id=iid, image_path=p,
                                       present=int(pos in q["present_positions"])))
    with ResumableCsvWriter(args.out, SINGLE_COLS, key="key", flush_every=50) as w:
        todo = [it for k, it in items.items() if k not in w.done]
        print(f"{len(items)} unique (cls, image) pairs, {len(todo)} to run", flush=True)
        if not todo:
            return
        vlm = load_model(args.model or mc["name"], attn_impl="sdpa", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"), device=args.device)
        at = vlm.answer_tokens
        t0 = time.time()
        for i, it in enumerate(todo, 1):
            try:
                imgs = load_images([it["image_path"]], cfg["coco"]["root"])
                inputs = vlm.build_inputs(imgs, SINGLE_TEMPLATE.format(obj=it["cls"]))
                lg = vlm.forward_last_logits(inputs)
                ly, ln = float(lg[at.yes]), float(lg[at.no])
                w.write({**it, "logit_yes": ly, "logit_no": ln, "m": ly - ln, "model": vlm.name})
            except Exception:
                print(f"[error] {it['key']}\n{traceback.format_exc()}")
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.3f}s/img", flush=True)


def aggregate(args):
    import pandas as pd
    s = pd.read_csv(args.single).set_index("key")
    qs = read_jsonl(args.questions)
    rows = []
    for q in qs:
        keys = [f"{q['cls'].replace(' ', '_')}|{iid}" for iid in q["image_ids"]]
        ms = s.loc[keys, "m"].to_numpy()
        m_min, m_max, m_mean = float(ms.min()), float(ms.max()), float(ms.mean())
        rule = "min" if q["qtype"].startswith("forall") else "max"
        m = m_min if rule == "min" else m_max
        pred = "Yes" if m > 0 else "No"
        rows.append(dict(qid=q["qid"], pair_id=q["pair_id"], cls=q["cls"], N=q["N"], qtype=q["qtype"],
                         answer=q["answer"], m_min=m_min, m_max=m_max, m_mean=m_mean, m=m, pred=pred,
                         correct=int(pred == q["answer"]), rule=rule, model=s["model"].iloc[0]))
    pd.DataFrame(rows, columns=AGG_COLS).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--single", default=None, help="--aggregate 시 단일 이미지 결과 csv")
    args = ap.parse_args(argv)
    aggregate(args) if args.aggregate else run_single(args)


if __name__ == "__main__":
    main()
