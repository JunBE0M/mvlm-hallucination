"""실험 B: M2(teacher, N+1 forward) -> 단일 forward student 자기증류 (라벨 미사용).

    # 1) 클래스 분할 + held-out 평가 문항 파일 (CPU)
    python -m src.distill split
    # 2) teacher: 학습 클래스의 seed-1 문항에 M2 (src.isolated, ∀ -> min, ∃ -> max)
    python -m src.isolated --model qwen2.5-vl-7b --questions data/s1_train.jsonl --mode post --out results/raw/B_teacher_qwen7b.csv
    # 3) student: LoRA (언어모델 attention·MLP 만), 손실 mse | rank | both
    python -m src.distill train --model qwen2.5-vl-7b --teacher results/raw/B_teacher_qwen7b.csv --loss both \
        --out_dir ckpt/B_qwen7b_both
    # 4) 평가: 기존 스크립트에 --adapter (run_eval / per_image / isolated), 분석은 src.distill_eval (CPU)

설계
  - 라벨(answer)은 쓰지 않는다. 문항의 양화사(∀/∃, 질문 문장에서 알 수 있음)로 teacher 집계(min/max)만 고른다.
  - student 점수 m_s = (W_yes - W_no)·h_last  (lm_head 는 fp32, 실제 forward 와 같은 경로: model.model(...)[0][:, -1]).
  - 정규화: 양화사 family(∀/∃)별로 학습셋에서 teacher t 의 (μ_t, σ_t), base m(=teacher csv 의 m_full)의 (μ_b, σ_b)를 고정.
      z_t = (t - μ_t)/σ_t,  z_s = (m_s - μ_b)/σ_b
    (a) mse : mean (z_s - z_t)^2
    (b) rank: 같은 (family, N) 묶음 G개 안에서 z_t_i - z_t_j > margin 인 쌍에 RankNet log(1 + exp(-(z_s_i - z_s_j)))
    (c) both: (a) + (b)
  - 한 optimizer step = 같은 (family, N) 의 문항 G개 (rank 손실용). 문항마다 forward(그래프 유지) 후 한 번에 backward.
  - LoRA: 정규식으로 model.language_model.layers.*.{q,k,v,o}_proj, mlp.{gate,up,down}_proj 만.
    (Qwen2.5-VL 비전 블록도 mlp.gate_proj/up_proj/down_proj 라는 같은 이름을 쓰므로 이름 목록으로 지정하면 안 됨 — 직접 확인)
  - attention 은 sdpa (eager 금지). GPU 단독 사용 + checkpointing 없음이 기본 (--grad_ckpt 로 켬).
  - 7B 는 G=8 문항 그래프를 동시에 들면 OOM -> rank/both 는 2-pass, mse 는 1-pass (train() 주석).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

from .data import load_config, read_jsonl, write_jsonl

CLASSES = ["cat", "dog", "horse", "elephant", "giraffe", "zebra", "car", "bus", "train", "motorcycle",
           "airplane", "boat", "person", "pizza", "laptop", "tv", "toilet", "refrigerator", "umbrella", "teddy bear"]
SPLIT_PATH = "results/novelty/B_split.json"
LORA_REGEX = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
TRAIN_LOG_COLS = ["step", "epoch", "family", "N", "loss", "mse", "rank", "n_pairs", "lr", "sec"]


def family(qtype: str) -> str:
    return "forall" if qtype.startswith("forall") else "exists"


# ------------------------------------------------------------------ split
def split(args):
    train = sorted(random.Random(0).sample(CLASSES, 10))
    held = sorted(set(CLASSES) - set(train))
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    json.dump({"train": train, "heldout": held, "rule": "sorted(random.Random(0).sample(CLASSES, 10))"},
              open(SPLIT_PATH, "w"), indent=1)
    s1 = read_jsonl(args.s1)
    write_jsonl([q for q in s1 if q["cls"] in train], "data/s1_train.jsonl")
    for src, dst in [("data/questions.jsonl", "data/heldout_questions.jsonl"),
                     ("data/questions_bigN.jsonl", "data/heldout_bigN.jsonl"),
                     ("data/questions_var.jsonl", "data/heldout_var.jsonl")]:
        write_jsonl([q for q in read_jsonl(src) if q["cls"] in held], dst)
    print(f"train classes: {train}\nheld-out: {held}")
    for f in ["data/s1_train.jsonl", "data/heldout_questions.jsonl", "data/heldout_bigN.jsonl", "data/heldout_var.jsonl"]:
        print(f, sum(1 for _ in open(f)))


# ------------------------------------------------------------------ train
def teacher_table(path, train_q):
    import pandas as pd
    T = pd.read_csv(path)
    T["m_list"] = T.m_list.map(json.loads)
    T["fam"] = T.qtype.map(family)
    T["t"] = [min(v) if f == "forall" else max(v) for v, f in zip(T.m_list, T.fam)]
    qs = {q["qid"]: q for q in read_jsonl(train_q)}
    T = T[T.qid.isin(qs)].reset_index(drop=True)
    stats = {}
    for f, g in T.groupby("fam"):
        stats[f] = dict(mu_t=float(g.t.mean()), sd_t=float(g.t.std()), mu_b=float(g.m_full.mean()), sd_b=float(g.m_full.std()))
    T["z_t"] = [(t - stats[f]["mu_t"]) / stats[f]["sd_t"] for t, f in zip(T.t, T.fam)]
    return T, qs, stats


def make_groups(T, G, rng):
    groups = []
    for (f, N), g in T.groupby(["fam", "N"]):
        ids = list(g.index)
        rng.shuffle(ids)
        groups += [(f, N, ids[i:i + G]) for i in range(0, len(ids), G) if len(ids[i:i + G]) >= 2]
    rng.shuffle(groups)
    return groups


def add_lora(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_REGEX, bias="none")
    pm = get_peft_model(model, cfg)
    names = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    n_layers = model.config.text_config.num_hidden_layers
    bad = [n for n in names if "language_model" not in n]
    if bad or len(names) != 7 * n_layers:
        raise RuntimeError(f"unexpected LoRA targets: {len(names)} (expected {7 * n_layers}), outside LM: {bad[:3]}")
    return pm


def train(args):
    import pandas as pd
    import torch

    from .io_utils import ResumableCsvWriter
    from .models import load_images, load_model

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = load_config(args.config)
    mc = cfg["model"]
    T, qs, stats = teacher_table(args.teacher, args.train_q)
    print(f"{len(T)} training questions; stats {json.dumps(stats)}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(vars(args), stats=stats, lora_regex=LORA_REGEX), open(f"{args.out_dir}/train_config.json", "w"),
              indent=1)

    vlm = load_model(args.model, attn_impl="sdpa", dtype=mc.get("dtype", "bfloat16"), max_pixels=mc.get("max_pixels"),
                     min_pixels=mc.get("min_pixels"), device=args.device)
    model = vlm.model
    for p in model.parameters():
        p.requires_grad_(False)
    pm = add_lora(model, args.r, args.alpha, args.dropout)
    pm.print_trainable_parameters()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    W = model.lm_head.weight.detach()
    w_diff = (W[vlm.answer_tokens.yes].float() - W[vlm.answer_tokens.no].float())    # m = w_diff · h (fp32)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(make_groups(T, args.group, random.Random(0)))
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    def student_m(q):
        images = load_images(q["image_paths"], cfg["coco"]["root"])
        inputs = vlm.build_inputs(images, q["question"])
        h = model.model(**inputs, use_cache=False)[0][0, -1]
        return h.float() @ w_diff

    log = ResumableCsvWriter(f"{args.out_dir}/train_log.csv", TRAIN_LOG_COLS, key="step", flush_every=10)
    step, t0 = 0, time.time()
    model.train()
    for ep in range(args.epochs):
        for f, N, ids in make_groups(T, args.group, rng):
            st = stats[f]
            # 2-pass (메모리 = 문항 1개): ① grad 없이 G 개 z_s -> 손실의 ∂L/∂z_s,
            # ② 문항마다 다시 forward 해 (z_s_i · g_i) 를 backward. 그래디언트는 한 번에 backward 한 것과 같다
            # (pass 1 은 eval 모드라 LoRA dropout 마스크만 다르다). 7B 에서 8 문항 그래프를 동시에 들면 OOM.
            if args.loss == "mse":   # 문항별 독립 -> 1-pass: 문항마다 forward+backward (평균이므로 1/G)
                opt.zero_grad(set_to_none=True)
                zl = []
                for i in ids:
                    z = (student_m(qs[T.qid[i]]) - st["mu_b"]) / st["sd_b"]
                    (((z - T.z_t[i]) ** 2) / len(ids)).backward()
                    zl.append(float(z.detach()))
                zs0 = torch.tensor(zl)
                zt = torch.tensor([T.z_t[i] for i in ids], dtype=torch.float32)
                mse = ((zs0 - zt) ** 2).mean(); rank = torch.tensor(0.0); loss = mse; n_pairs = 0
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                log.write(dict(step=step, epoch=ep, family=f, N=N, loss=float(loss), mse=float(mse), rank=0.0,
                               n_pairs=0, lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1)))
                if args.max_steps and step >= args.max_steps:
                    break
                if step % 25 == 0:
                    print(f"ep {ep} step {step}/{total} mse {float(mse):.4f} {(time.time() - t0) / step:.2f}s/step",
                          flush=True)
                continue
            model.eval()
            with torch.no_grad():
                zs0 = torch.stack([(student_m(qs[T.qid[i]]) - st["mu_b"]) / st["sd_b"] for i in ids])
            model.train()
            zs = zs0.clone().requires_grad_(True)
            zt = torch.tensor([T.z_t[i] for i in ids], device=zs.device, dtype=torch.float32)
            mse = ((zs - zt) ** 2).mean()
            d_t = zt[:, None] - zt[None, :]
            mask = d_t > args.margin
            n_pairs = int(mask.sum())
            rank = (torch.nn.functional.softplus(-(zs[:, None] - zs[None, :]))[mask].mean()
                    if n_pairs else zs.sum() * 0.0)
            loss = {"mse": mse, "rank": rank, "both": mse + args.rank_weight * rank}[args.loss]
            g = torch.autograd.grad(loss, zs)[0].detach()
            opt.zero_grad(set_to_none=True)
            for gi, i in zip(g.tolist(), ids):
                if gi != 0.0:
                    ((student_m(qs[T.qid[i]]) - st["mu_b"]) / st["sd_b"] * gi).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            log.write(dict(step=step, epoch=ep, family=f, N=N, loss=float(loss.detach()), mse=float(mse.detach()), rank=float(rank.detach()),
                           n_pairs=n_pairs, lr=sched.get_last_lr()[0], sec=round(time.time() - t0, 1)))
            if args.max_steps and step >= args.max_steps:
                break
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{total} loss {float(loss):.4f} mse {float(mse):.4f} rank {float(rank):.4f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        pm.save_pretrained(f"{args.out_dir}/epoch{ep + 1}")
        print(f"saved {args.out_dir}/epoch{ep + 1}", flush=True)
    log.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--s1", default="data/questions_s1.jsonl")
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--teacher", required=True)
    t.add_argument("--train_q", default="data/s1_train.jsonl")
    t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--loss", choices=["mse", "rank", "both"], default="both")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=32)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--group", type=int, default=8)
    t.add_argument("--margin", type=float, default=0.1)
    t.add_argument("--rank_weight", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (GPU 공유 시 메모리 절약)")
    t.add_argument("--max_steps", type=int, default=0, help="디버그용: epoch 마다 이 step 수에서 중단")
    t.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    split(args) if args.cmd == "split" else train(args)


if __name__ == "__main__":
    main()
