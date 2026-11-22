"""방법 비교 (forall 과제): 있는 결과 파일만 읽어 model × method × N 별 지표 표를 만든다.

    python -m src.compare_methods --out results/summary_methods.csv

지표 (forall_yes = signal, forall_no = noise):
    acc_yes  : forall_yes 정답률 (hit)
    acc_no   : forall_no 정답률 (correct rejection) = 1 - false-Yes
    bal_acc  : (acc_yes + acc_no) / 2
    d_prime, criterion : log-linear 보정 SDT (analyze.py 와 같은 방식)
방법:
    base      : 원래 질문, 한 번에 N장 (E1)
    labels    : "Image i:" 번호 (E1lab)
    M0_neg    : "없는 이미지가 하나라도 있나?" 로 바꿔 묻고 답을 뒤집음
    M1_min    : 이미지마다 단일 이미지 질문 -> min
    M2_post / M2_strict : 다중 이미지 문맥 + 질문 토큰이 한 이미지만 보게 격리 -> min  (M2_*_mean: 평균 집계, 대조용)
    M3_*      : M2 로 찾은 최저 이미지에 attention α배 -> 원래 질문의 m
    M4_fuse   : m_full + w * m2_min (w 는 leave-one-class-out 으로 선택)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

MODELS = {"qwen3b": "qwen2.5-vl-3b", "qwen7b": "qwen2.5-vl-7b"}


def sdt(hit_k, hit_n, fa_k, fa_n):
    h = (hit_k + 0.5) / (hit_n + 1)
    f = (fa_k + 0.5) / (fa_n + 1)
    return norm.ppf(h) - norm.ppf(f), -(norm.ppf(h) + norm.ppf(f)) / 2


def frame(qid, qtype, N, cls, score, pred=None):
    d = pd.DataFrame(dict(qid=qid, qtype=qtype, N=N, cls=cls, score=score))
    d["pred_yes"] = (d.score > 0) if pred is None else (pd.Series(pred, index=d.index) == "Yes")
    return d


def load_methods(raw="results/raw"):
    out = {}
    for tag in MODELS:
        M = {}
        def add(name, path, fn):
            if os.path.exists(path):
                d = pd.read_csv(path)
                if len(d):
                    M[name] = fn(d)
        fa = lambda d: d[d.qtype.str.startswith("forall")]
        add("base", f"{raw}/e1_{tag}.csv", lambda d: frame(fa(d).qid, fa(d).qtype, fa(d).N, fa(d).cls, fa(d).m))
        add("labels", f"{raw}/e1lab_{tag}.csv", lambda d: frame(fa(d).qid, fa(d).qtype, fa(d).N, fa(d).cls, fa(d).m))
        # M0: 부정 질문의 m 은 "없는 이미지가 있다" 쪽 점수 -> 원래 질문 점수로는 -m
        add("M0_neg", f"{raw}/m0neg_{tag}.csv", lambda d: frame(d.qid, d.qtype, d.N, d.cls, -d.m))
        add("M1_min", f"{raw}/m1_agg_{tag}.csv", lambda d: frame(fa(d).qid, fa(d).qtype, fa(d).N, fa(d).cls, fa(d).m_min))
        add("M1_mean", f"{raw}/m1_agg_{tag}.csv", lambda d: frame(fa(d).qid, fa(d).qtype, fa(d).N, fa(d).cls, fa(d).m_mean))
        for mode in ("post", "strict"):
            add(f"M2_{mode}", f"{raw}/m2_{mode}_{tag}.csv", lambda d: frame(d.qid, d.qtype, d.N, d.cls, d.m_min))
            add(f"M2_{mode}_mean", f"{raw}/m2_{mode}_{tag}.csv", lambda d: frame(d.qid, d.qtype, d.N, d.cls, d.m_mean))
        for f in sorted(glob.glob(f"{raw}/m3_{tag}/focus_*.csv")):
            name = "M3_" + os.path.basename(f)[len("focus_"):-4]
            add(name, f, lambda d: frame(d.qid, d.qtype, d.N, d.cls, d.m))
        for f in sorted(glob.glob(f"{raw}/e3_{tag}/*.csv")):
            pass  # 개입 실험은 forall_no 만 있어 여기서 비교하지 않음 (summary_intervention 참고)
        # M4: m_full + w * m2_min, w 는 leave-one-class-out
        p = f"{raw}/m2_post_{tag}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            if len(d):
                M["M4_fuse"] = fuse_loco(d)
        out[tag] = M
    return out


def bal_acc(yes_mask, pred_yes):
    y = pred_yes[yes_mask].mean() if yes_mask.any() else np.nan
    n = 1 - pred_yes[~yes_mask].mean() if (~yes_mask).any() else np.nan
    return (y + n) / 2


def fuse_loco(d, grid=(0, 0.1, 0.25, 0.5, 1, 2, 4, 1e3)):
    d = d.copy()
    yes = (d.qtype == "forall_yes").to_numpy()
    scores = np.empty(len(d))
    for c in d.cls.unique():
        tr = (d.cls != c).to_numpy()
        best = max(grid, key=lambda w: bal_acc(yes[tr], ((d.m_full + w * d.m_min) > 0).to_numpy()[tr]))
        te = ~tr
        scores[te] = (d.m_full + best * d.m_min).to_numpy()[te]
    return frame(d.qid, d.qtype, d.N, d.cls, scores)


def summarize(methods):
    rows = []
    for tag, M in methods.items():
        for name, d in M.items():
            for N, g in list(d.groupby("N")) + [("all", d)]:
                y = g[g.qtype == "forall_yes"]
                n = g[g.qtype == "forall_no"]
                if not len(y) or not len(n):
                    continue
                hk, fk = int(y.pred_yes.sum()), int(n.pred_yes.sum())
                dp, c = sdt(hk, len(y), fk, len(n))
                rows.append(dict(model=MODELS[tag], method=name, N=N, n_yes=len(y), n_no=len(n),
                                 acc_yes=hk / len(y), acc_no=1 - fk / len(n),
                                 bal_acc=(hk / len(y) + 1 - fk / len(n)) / 2, d_prime=dp, criterion=c))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results/summary_methods.csv")
    args = ap.parse_args(argv)
    s = summarize(load_methods(args.raw))
    s.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    piv = s.pivot_table(index=["model", "method"], columns="N", values="bal_acc", sort=False)
    print("balanced accuracy (forall_yes vs forall_no)\n", piv.round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
