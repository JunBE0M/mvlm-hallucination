"""기준선: FOCUS 방식 (arXiv 2508.13744) — 이미지 i 만 남기고 나머지를 노이즈로 가린 입력으로 N 회 + 노이즈 전용 1 회.

논문에서 확인한 내용 (arXiv HTML, 2026-11-26 확인):
  - 가리기: 시각 인코더 *이전* 이미지에 적용. ImageNet mean/std 로 정규화한 공간에서
        v' = (1-λ)·v + λ·U(0,1),  이후 역정규화하고 [0,1] 로 clip  (Appendix C)
  - forward: 대상 이미지마다 1회(나머지 이미지 가림) + 모든 이미지를 가린 노이즈 기준 1회 = N+1
  - 집계: ℓ_final = Σ_i (ℓ_i − α·ℓ_noise)  (logit 합, 대조). λ, α 는 모델·벤치마크별 검증셋으로 튜닝
    (ablation 기본 λ=0.3, α=0.4 [Qwen2.5-VL-7B, Mantis]). plausibility 제약 없음.
여기서는 yes/no logit 차 m 을 쓴다: m_i (이미지 i 만 원본), m_noise. 집계는 분석 단계에서
  - min_i m_i            : M2 와 같은 집계 (주 비교)
  - FOCUS 원 집계        : Σ_i (m_i − α·m_noise),  α ∈ {0.1, 0.4, 1.0}
검증셋 튜닝은 하지 않았다 (λ ∈ {0.3, 1.0} 고정 실행, α 는 사후에 여러 값 보고).
노이즈는 원본 해상도에서 (qid, 이미지 위치, λ) 로 시드를 고정해 만들고, 같은 문항 안에서는 문맥이 달라도 같은 노이즈 이미지를 쓴다.
(논문은 전처리된 텐서에 적용했을 수 있으나 확인 못 함 — 원본 해상도 노이즈는 processor 리사이즈로 일부 평균된다.)

    python -m src.focus_noise --model qwen2.5-vl-7b --lam 0.3 --out results/raw/F_focus_l0.3_qwen7b.csv
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
import zlib

import numpy as np

from .data import load_config

COLS = ["qid", "pair_id", "cls", "N", "qtype", "answer", "absent_positions", "present_positions", "m_list",
        "m_noise", "m_full", "lam", "model"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def corrupt(img, lam: float, seed: int):
    from PIL import Image
    x = np.asarray(img, dtype=np.float64) / 255.0
    v = (x - IMAGENET_MEAN) / IMAGENET_STD
    u = np.random.default_rng(seed).uniform(0.0, 1.0, size=v.shape)
    v2 = (1 - lam) * v + lam * u
    x2 = np.clip(v2 * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
    return Image.fromarray((x2 * 255 + 0.5).astype(np.uint8))


def main(argv=None):
    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model
    from .run_eval import select_questions

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.3)
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
        vlm = load_model(args.model, attn_impl="sdpa", dtype=mc.get("dtype", "bfloat16"),
                         max_pixels=mc.get("max_pixels"), min_pixels=mc.get("min_pixels"), device=args.device)
        at = vlm.answer_tokens

        def m_of(imgs, question):
            lg = vlm.forward_last_logits(vlm.build_inputs(imgs, question))
            return float(lg[at.yes] - lg[at.no])

        t0, n_err = time.time(), 0
        for i, q in enumerate(todo, 1):
            try:
                images = load_images(q["image_paths"], cfg["coco"]["root"])
                noisy = [corrupt(im, args.lam, zlib.crc32(f"{q['qid']}|{j}|{args.lam}".encode()))
                         for j, im in enumerate(images)]
                ms = [m_of([images[j] if j == k else noisy[j] for j in range(len(images))], q["question"])
                      for k in range(len(images))]
                w.write(dict(qid=q["qid"], pair_id=q["pair_id"], cls=q["cls"], N=q["N"], qtype=q["qtype"],
                             answer=q["answer"], absent_positions=q["absent_positions"],
                             present_positions=q["present_positions"], m_list=json.dumps([round(x, 5) for x in ms]),
                             m_noise=m_of(noisy, q["question"]), m_full=m_of(images, q["question"]), lam=args.lam,
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
