"""E1: ∀/∃ 문항에 대한 yes/no logit 측정 (generate 없이 forward 1회).

    python -m src.run_eval --questions data/questions.jsonl --model qwen2.5-vl-3b \
        --out results/raw/e1_qwen2.5-vl-3b.csv [--limit 5]

같은 --out 으로 다시 실행하면 이미 처리된 qid는 건너뛴다 (중간에 끊겨도 이어서 실행).
"""
from __future__ import annotations

import argparse
import time
import traceback

from .data import load_config, read_jsonl

META_COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "image_ids",
             "present_positions", "absent_positions", "area_ratios"]
RESULT_COLS = ["logit_yes", "logit_no", "m", "pred", "correct",
               "logit_yes_space", "logit_no_space", "logit_yes_lower", "logit_no_lower",
               "p_yes", "p_no", "top1_id", "top1_token",
               "n_input_tokens", "n_image_tokens", "model", "max_pixels"]
COLUMNS = META_COLS + RESULT_COLS


def score_from_logits(vlm, logits, q: dict, inputs) -> dict:
    """마지막 위치 logits [V] -> 결과 행."""
    import torch
    at = vlm.answer_tokens
    ly, ln = float(logits[at.yes]), float(logits[at.no])
    probs = torch.softmax(logits.float(), dim=-1)
    top1 = int(torch.argmax(logits))
    seg = vlm.segments(inputs)
    m = ly - ln
    pred = "Yes" if m > 0 else "No"
    row = {c: q.get(c) for c in META_COLS}
    row.update(
        logit_yes=ly, logit_no=ln, m=m, pred=pred, correct=int(pred == q["answer"]),
        logit_yes_space=float(logits[at.variants[" Yes"]]), logit_no_space=float(logits[at.variants[" No"]]),
        logit_yes_lower=float(logits[at.variants["yes"]]), logit_no_lower=float(logits[at.variants["no"]]),
        p_yes=float(probs[at.yes]), p_no=float(probs[at.no]),
        top1_id=top1, top1_token=vlm.processor.tokenizer.decode([top1]),
        n_input_tokens=int(inputs["input_ids"].shape[1]),
        n_image_tokens=seg.n_tokens().tolist(),
        model=vlm.name, max_pixels=vlm.size["longest_edge"],
    )
    return row


def select_questions(path, limit=None, Ns=None, qtypes=None):
    qs = read_jsonl(path)
    if Ns:
        qs = [q for q in qs if q["N"] in Ns]
    if qtypes:
        qs = [q for q in qs if q["qtype"] in qtypes]
    if limit:
        qs = qs[:limit]
    return qs


def common_args(ap: argparse.ArgumentParser):
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None, help="config의 model.name override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="jsonl 앞에서부터 K개만 (디버깅)")
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--qtypes", nargs="*", default=None)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--flush_every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_labels", action="store_true", help='각 이미지 앞에 "Image i: " 삽입')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_args(ap)
    ap.add_argument("--attn_impl", default=None, help="config의 model.attn_impl override (sdpa 권장)")
    args = ap.parse_args(argv)

    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model

    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)
    with ResumableCsvWriter(args.out, COLUMNS, flush_every=args.flush_every) as w:
        todo = [q for q in qs if q["qid"] not in w.done]
        print(f"{len(qs)} questions, {len(qs) - len(todo)} already done, {len(todo)} to run")
        if not todo:
            return
        vlm = load_model(args.model or mc["name"], attn_impl=args.attn_impl or mc["attn_impl"],
                         dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=args.max_pixels or mc.get("max_pixels"),
                         min_pixels=mc.get("min_pixels"), device=args.device,
                         image_labels=args.image_labels)
        t0, n_err = time.time(), 0
        for i, q in enumerate(todo, 1):
            try:
                images = load_images(q["image_paths"], cfg["coco"]["root"])
                inputs = vlm.build_inputs(images, q["question"])
                logits = vlm.forward_last_logits(inputs)
                w.write(score_from_logits(vlm, logits, q, inputs))
            except Exception:
                n_err += 1
                print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                if n_err > 20:
                    raise RuntimeError("too many errors, aborting")
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)}  {el / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
