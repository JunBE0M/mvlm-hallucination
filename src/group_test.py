"""실험 D: 그룹 테스팅 판독 (log N 비용).

M2 처럼 이미지는 전부 한 번에 넣고, 질문 토큰(마지막 이미지 뒤 토큰들)이 이미지 부분집합 S 만 보게 다른 이미지로 가는
attention 을 막는다 (AttentionIntervention.set_blocks, α=0, sdpa 경로). 이진 분할:
    S = 전체;  while |S| > 1:  S 를 앞/뒤 절반 A, B 로 나눠 m(A), m(B) 를 구하고, m 이 낮은 쪽으로 내려간다.
    점수 = 지금까지 본 m 중 최솟값,  located = 마지막에 남은 이미지(객체가 없다고 추정한 이미지).
forward 수 ≈ 2·⌈log2 N⌉ (m_full 1회는 비교용으로 따로 계산해 n_forward 에 넣지 않는다).
마지막 단계의 단일 이미지 m 은 M2(post) 의 m_i 와 같은 개입이다.

    python -m src.group_test --model qwen2.5-vl-7b --questions data/questions.jsonl --Ns 8 \
        --out results/raw/D_group_qwen7b.csv
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

from .data import load_config

COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "absent_positions", "m_full", "m_seen", "subsets",
        "score_min", "located", "n_forward", "model"]


def main(argv=None):
    from .intervene import AttentionIntervention, layer_set
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
    ap.add_argument("--qtypes", nargs="*", default=["forall_yes", "forall_no"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)
    with ResumableCsvWriter(args.out, COLS, flush_every=20) as w:
        todo = [q for q in qs if q["qid"] not in w.done]
        print(f"{len(qs)} questions, {len(todo)} to run", flush=True)
        if not todo:
            return
        vlm = load_model(args.model, attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"), device=args.device)
        at = vlm.answer_tokens
        t0, n_err = time.time(), 0
        with AttentionIntervention(vlm, layer_set("all", vlm.num_layers)) as iv:
            for i, q in enumerate(todo, 1):
                try:
                    images = load_images(q["image_paths"], cfg["coco"]["root"])
                    inputs = vlm.build_inputs(images, q["question"])
                    seg = vlm.segments(inputs)
                    L = int(inputs["input_ids"].shape[1])
                    post0 = seg.image_spans[-1][1] + 1

                    def m_subset(S):
                        blocks = [(post0, L, k0, k1) for j, (k0, k1) in enumerate(seg.image_spans) if j not in S]
                        iv.set_blocks(blocks, 0.0)
                        lg = vlm.forward_last_logits(inputs)
                        return float(lg[at.yes] - lg[at.no])

                    iv.set_blocks([], 1.0)
                    lg = vlm.forward_last_logits(inputs)
                    m_full = float(lg[at.yes] - lg[at.no])
                    S, seen, subsets = list(range(q["N"])), [], []
                    while len(S) > 1:
                        A, B = S[: len(S) // 2], S[len(S) // 2:]
                        mA, mB = m_subset(A), m_subset(B)
                        seen += [mA, mB]
                        subsets += [A, B]
                        S = A if mA <= mB else B
                    iv.set_blocks([], 1.0)
                    w.write(dict(qid=q["qid"], pair_id=q["pair_id"], cls=q["cls"], N=q["N"], qtype=q["qtype"],
                                 answer=q["answer"], absent_positions=q["absent_positions"], m_full=m_full,
                                 m_seen=json.dumps([round(x, 5) for x in seen]), subsets=json.dumps(subsets),
                                 score_min=min(seen), located=S[0], n_forward=len(seen), model=vlm.name))
                except Exception:
                    n_err += 1
                    print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                    if n_err > 20:
                        raise RuntimeError("too many errors, aborting")
                if i % 50 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
