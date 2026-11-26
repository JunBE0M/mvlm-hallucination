"""M2: 이미지별 격리 판독 (image-isolated readout).

이미지는 전부 한 번에 넣고(다중 이미지 문맥 유지), 답을 읽는 질문 토큰들이 i번째 이미지만 보도록
다른 이미지들로 가는 attention 을 막은(knockout) forward 를 이미지마다 한 번씩 수행해 m_i 를 얻는다.
그 뒤 규칙으로 집계: forall -> min_i m_i, exists -> max_i m_i.

    python -m src.isolated --model qwen2.5-vl-3b --qtypes forall_yes forall_no --mode post \
        --out results/raw/m2_post_qwen3b.csv

mode:
  post   : 마지막 이미지 뒤 토큰(질문/템플릿/마지막 토큰) -> 다른 이미지 차단. 이미지 토큰끼리는 서로 볼 수 있음.
  strict : 추가로 각 차단 이미지 뒤의 모든 토큰(뒤쪽 이미지 토큰 포함)에서도 차단 -> i번째 이미지 정보만 남음.
질문 문장은 원래 ∀/∃ 질문을 그대로 쓴다 (모델은 "모든 이미지에 있나?"를 i번째 이미지만 보고 답하는 셈).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np

from .data import load_config

COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "present_positions", "absent_positions",
        "m_list", "m_min", "m_max", "m_mean", "m_full", "m", "pred", "correct", "rule", "mode", "model"]


def main(argv=None):
    from .intervene import AttentionIntervention, layer_set
    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model
    from .run_eval import select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--Ns", type=int, nargs="*", default=None)
    ap.add_argument("--qtypes", nargs="*", default=None)
    ap.add_argument("--mode", default="post", choices=["post", "strict"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter", default=None, help="peft LoRA 디렉터리 (실험 B student)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mc = cfg["model"]
    qs = select_questions(args.questions, args.limit, args.Ns, args.qtypes)
    with ResumableCsvWriter(args.out, COLS, flush_every=20) as w:
        todo = [q for q in qs if q["qid"] not in w.done]
        print(f"{len(qs)} questions, {len(todo)} to run", flush=True)
        if not todo:
            return
        vlm = load_model(args.model or mc["name"], attn_impl="eager", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"), device=args.device,
                         adapter=args.adapter)
        at = vlm.answer_tokens
        layers = layer_set("all", vlm.num_layers)
        t0, n_err = time.time(), 0
        with AttentionIntervention(vlm, layers) as iv:
            for i, q in enumerate(todo, 1):
                try:
                    images = load_images(q["image_paths"], cfg["coco"]["root"])
                    inputs = vlm.build_inputs(images, q["question"])
                    seg = vlm.segments(inputs)
                    L = int(inputs["input_ids"].shape[1])
                    post0 = seg.image_spans[-1][1] + 1
                    iv.set_blocks([], 1.0)
                    lg = vlm.forward_last_logits(inputs)
                    m_full = float(lg[at.yes] - lg[at.no])
                    ms = []
                    for keep in range(q["N"]):
                        blocks = []
                        for j, (k0, k1) in enumerate(seg.image_spans):
                            if j == keep:
                                continue
                            q0 = post0 if args.mode == "post" else k1
                            blocks.append((q0, L, k0, k1))
                        iv.set_blocks(blocks, 0.0)
                        lg = vlm.forward_last_logits(inputs)
                        ms.append(float(lg[at.yes] - lg[at.no]))
                    iv.set_blocks([], 1.0)
                    ms_a = np.array(ms)
                    rule = "min" if q["qtype"].startswith("forall") else "max"
                    m = float(ms_a.min() if rule == "min" else ms_a.max())
                    pred = "Yes" if m > 0 else "No"
                    w.write(dict(qid=q["qid"], pair_id=q["pair_id"], cls=q["cls"], N=q["N"], qtype=q["qtype"],
                                 answer=q["answer"], present_positions=q["present_positions"],
                                 absent_positions=q["absent_positions"], m_list=json.dumps([round(x, 5) for x in ms]),
                                 m_min=float(ms_a.min()), m_max=float(ms_a.max()), m_mean=float(ms_a.mean()),
                                 m_full=m_full, m=m, pred=pred, correct=int(pred == q["answer"]), rule=rule,
                                 mode=args.mode, model=vlm.name))
                except Exception:
                    n_err += 1
                    print(f"[error] {q['qid']}\n{traceback.format_exc()}")
                    if n_err > 20:
                        raise RuntimeError("too many errors, aborting")
                if i % 50 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)}  {(time.time() - t0) / i:.2f}s/q  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
