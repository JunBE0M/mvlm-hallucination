"""COCO 2017 자체 데이터(coco2017_custom) 바닐라 분석. GPU 불필요 (docs/coco2017_vanilla.md).

    python -m src.vanilla_eval [--tags qwen3b qwen7b ...] [--seeds 1 2 3] [--splits val test]

입력: results/coco2017_custom/raw/coco2017_s{S}_e1_{split}_{tag}.csv (src.run_eval 원자료, 없는 파일은 건너뜀)
출력: results/coco2017_custom/analysis/vanilla/
    accuracy.csv                tag·seed·split·N·유형별 정확도, Yes 비율, 거짓 Yes 비율, 평균 m   (analyze.accuracy_table)
    accuracy_mean.csv           위 표의 seed 평균 ± 표준편차
    delta.csv                   N별 Δ = m(forall_yes) − m(forall_no) (짝 단위)와 Δ×N            (analyze.pair_deltas, delta_summary)
    gamma.csv                   Δ ∝ N^−γ 의 γ와 95% CI. seed 별 + seed 를 합친 것(seed=all)      (agg_analysis.a1 과 같은 식)
    sdt.csv                     d′(구분 능력), c(<0 이면 Yes 쪽 치우침)                          (analyze.sdt_table)
    position.csv                한 장만 다른 사진의 자리별 정확도·평균 m (seed 합침)              (analyze.position_table)
    class_auc.csv               클래스별 ∀/∃ AUC. seed 별 + seed 합침
    answer_format.csv           첫 토큰이 Yes/No 가 아닌 비율, p_yes + p_no 평균·최솟값 (logit 방식이 유효한지)
    label_audit_candidates.csv  라벨 검수 후보: test 의 forall_no 에서 모델들이 강하게 Yes 라고 한 '객체 없는' 사진
                                (m 을 tag·seed·N 안에서 z 점수로 바꾼 뒤 사진별로 평균, 상위 --top_k 개)
    summary.txt                 위 내용 요약
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from . import analyze as A
from .dab_eval import auc

RAW = "results/coco2017_custom/raw"
OUT = "results/coco2017_custom/analysis/vanilla"
TAGS = ["qwen3b", "qwen7b", "qwen3vl8b", "internvl8b", "llavaov7b"]
SEEDS = [1, 2, 3]
SPLITS = ["val", "test"]
KEYS = ["tag", "seed", "split"]


# ------------------------------------------------------------------ 입력
def raw_path(raw: str, tag: str, seed: int, split: str) -> str:
    return f"{raw}/coco2017_s{seed}_e1_{split}_{tag}.csv"


def load_all(raw, tags, seeds, splits) -> dict[tuple, pd.DataFrame]:
    """(tag, seed, split) -> 원자료 DataFrame (analyze.load_results 로 읽어 목록 열을 풀고 yes 열을 붙임)."""
    out = {}
    for tag in tags:
        for seed in seeds:
            for split in splits:
                p = raw_path(raw, tag, seed, split)
                if os.path.exists(p):
                    df = A.load_results([p])
                    df["tag"], df["seed"], df["split"] = tag, seed, split
                    out[(tag, seed, split)] = df
    return out


def _with_keys(t: pd.DataFrame, tag, seed, split) -> pd.DataFrame:
    t = t.copy()
    t.insert(0, "split", split)
    t.insert(0, "seed", seed)
    t.insert(0, "tag", tag)
    return t


def _pooled(data, tag, split) -> pd.DataFrame | None:
    ds = [d for (t, _, s), d in data.items() if t == tag and s == split]
    return pd.concat(ds, ignore_index=True) if ds else None


# ------------------------------------------------------------------ 지표
def gamma_fit(pairs: pd.DataFrame, nb: int = 1000, rng=None) -> dict:
    """log(평균 Δ) = c − γ·log N 의 γ (agg_analysis.a1 과 같은 식) + N 안에서 짝을 다시 뽑는 bootstrap 95% CI."""
    rng = np.random.default_rng(0) if rng is None else rng
    groups = {int(n): g["delta"].to_numpy(float) for n, g in pairs.groupby("N")}
    Ns = sorted(groups)
    if len(Ns) < 2:
        return dict(gamma=np.nan, gamma_lo=np.nan, gamma_hi=np.nan, n_pairs=len(pairs), Ns=str(Ns))

    def gamma(gr):
        y = np.log([max(gr[n].mean(), 1e-6) for n in Ns])
        return -np.polyfit(np.log(Ns), y, 1)[0]

    g = gamma(groups)
    bs = [gamma({n: rng.choice(groups[n], len(groups[n])) for n in Ns}) for _ in range(nb)]
    return dict(gamma=g, gamma_lo=float(np.percentile(bs, 2.5)), gamma_hi=float(np.percentile(bs, 97.5)),
                n_pairs=len(pairs), Ns=" ".join(map(str, Ns)))


def class_auc_rows(df: pd.DataFrame, tag, seed, split) -> list[dict]:
    rows = []
    fam = df["qtype"].str.split("_").str[0]
    for f in ("forall", "exists"):
        d = df[fam == f]
        for cls, g in d.groupby("cls"):
            rows.append(dict(tag=tag, seed=seed, split=split, family=f, cls=cls, n=len(g),
                             auc=auc(g["m"].to_numpy(float), g["answer"].eq("Yes").to_numpy())))
    return rows


def answer_format_row(df: pd.DataFrame, tag, seed, split) -> dict:
    top = df["top1_token"].astype(str).str.strip()
    mass = df["p_yes"].astype(float) + df["p_no"].astype(float)
    return dict(tag=tag, seed=seed, split=split, n=len(df), top1_not_yesno=float((~top.isin(["Yes", "No"])).mean()),
                top1_other_examples=" | ".join(top[~top.isin(["Yes", "No"])].value_counts().head(3).index),
                pmass_mean=float(mass.mean()), pmass_min=float(mass.min()))


def label_audit(data: dict, top_k: int = 200) -> pd.DataFrame:
    """test 의 forall_no 에서 '객체 없는' 사진별로, 모델들이 Yes 쪽으로 얼마나 기울었는지(z 점수 평균)를 매긴다.
    COCO 라벨 누락(실제로는 객체가 있음) 의심 사진을 위로 올리는 용도."""
    parts = []
    for (tag, seed, split), df in data.items():
        if not split.startswith("test"):
            continue
        d = df[df["qtype"] == "forall_no"].copy()
        if d.empty:
            continue
        d["z"] = d.groupby("N")["m"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
        d["absent_image_id"] = [ids[a[0]] for ids, a in zip(d["image_ids"], d["absent_positions"])]
        parts.append(d[["tag", "seed", "split", "qid", "cls", "N", "m", "z", "pred", "absent_image_id"]])
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    g = d.groupby(["cls", "absent_image_id"])
    out = g.agg(n_questions=("qid", "size"), n_models=("tag", "nunique"), mean_z=("z", "mean"),
                mean_m=("m", "mean"), max_m=("m", "max"), frac_pred_yes=("pred", lambda p: float((p == "Yes").mean())),
                example_qid=("qid", "first")).reset_index()
    out["image_path"] = [f"val2017/{int(i):012d}.jpg" for i in out["absent_image_id"]]
    return out.sort_values(["mean_z", "max_m"], ascending=False).head(top_k).reset_index(drop=True)


# ------------------------------------------------------------------ 실행
def evaluate(data: dict, nb: int = 1000, top_k: int = 200) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    acc, delta, gam, sdt, cls_rows, fmt = [], [], [], [], [], []
    pairs_by = {}
    for (tag, seed, split), df in data.items():
        acc.append(_with_keys(A.accuracy_table(df), tag, seed, split))
        pairs = A.pair_deltas(df)
        pairs_by[(tag, seed, split)] = pairs
        if len(pairs):
            delta.append(_with_keys(A.delta_summary(pairs, n_boot=nb), tag, seed, split))
            gam.append(dict(tag=tag, seed=seed, split=split, **gamma_fit(pairs, nb, rng)))
        s = A.sdt_table(df)
        if len(s):
            sdt.append(_with_keys(s, tag, seed, split))
        cls_rows += class_auc_rows(df, tag, seed, split)
        fmt.append(answer_format_row(df, tag, seed, split))
    pos = []
    for tag, split in sorted({(t, s) for t, _, s in data}):
        pooled = _pooled(data, tag, split)
        n_seeds = len({sd for t, sd, s in data if t == tag and s == split})
        pos.append(_with_keys(A.position_table(pooled), tag, "all", split).assign(n_seeds=n_seeds))
        cls_rows += class_auc_rows(pooled, tag, "all", split)
        pp = [p for (t, _, s), p in pairs_by.items() if t == tag and s == split and len(p)]
        if pp:
            gam.append(dict(tag=tag, seed="all", split=split, **gamma_fit(pd.concat(pp, ignore_index=True), nb, rng)))
    accuracy = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()
    if not accuracy.empty:
        cols = ["accuracy", "yes_rate", "false_yes_rate", "mean_m"]
        accuracy["qtype"] = accuracy["qtype"].astype(str)
        g = accuracy.groupby(["tag", "split", "N", "qtype"])
        acc_mean = g[cols].agg(["mean", "std"])
        acc_mean.columns = [f"{c}_{s}" for c, s in acc_mean.columns]
        acc_mean = acc_mean.join(g["seed"].nunique().rename("n_seeds")).reset_index()
    else:
        acc_mean = pd.DataFrame()
    return {
        "accuracy": accuracy, "accuracy_mean": acc_mean,
        "delta": pd.concat(delta, ignore_index=True) if delta else pd.DataFrame(),
        "gamma": pd.DataFrame(gam), "sdt": pd.concat(sdt, ignore_index=True) if sdt else pd.DataFrame(),
        "position": pd.concat(pos, ignore_index=True) if pos else pd.DataFrame(),
        "class_auc": pd.DataFrame(cls_rows), "answer_format": pd.DataFrame(fmt),
        "label_audit_candidates": label_audit(data, top_k),
    }


def summarize(res: dict, split: str = "test") -> str:
    L = [f"# 바닐라 요약 (split={split}, seed 평균 ± 표준편차)"]
    am = res["accuracy_mean"]
    if not am.empty:
        a = am[am.split == split]
        L.append("\n## 거짓 Yes 비율 (forall_no) / 정확도 (forall_yes, exists_yes, exists_no)")
        for tag, g in a.groupby("tag"):
            for n, h in g.groupby("N"):
                q = h.set_index("qtype")

                def v(t, c):
                    return (f"{q.loc[t, c + '_mean']:.2f}±{0 if np.isnan(q.loc[t, c + '_std']) else q.loc[t, c + '_std']:.2f}"
                            if t in q.index else "-")
                L.append(f"{tag:11s} N={n:2d} | ∀No 거짓Yes {v('forall_no', 'false_yes_rate')} | ∀Yes 정확도 "
                         f"{v('forall_yes', 'accuracy')} | ∃Yes {v('exists_yes', 'accuracy')} | ∃No {v('exists_no', 'accuracy')}")
    gm = res["gamma"]
    if not gm.empty:
        L.append("\n## Δ ∝ N^−γ (seed 합침)")
        for _, r in gm[(gm.split == split) & (gm.seed == "all")].iterrows():
            L.append(f"{r.tag:11s} γ = {r.gamma:.2f} [{r.gamma_lo:.2f}, {r.gamma_hi:.2f}]  (짝 {r.n_pairs}, N {r.Ns})")
    sd = res["sdt"]
    if not sd.empty:
        L.append("\n## Yes 편향 c (N 평균, <0 이면 Yes 쪽)")
        for tag, g in sd[sd.split == split].groupby("tag"):
            L.append(f"{tag:11s} c = {g.criterion.mean():+.2f}  d′ = {g.d_prime.mean():.2f}")
    af = res["answer_format"]
    if not af.empty:
        bad = af[af.top1_not_yesno > 0.01]
        L.append("\n## 답 형식")
        L.append("첫 토큰이 Yes/No 가 아닌 비율이 1% 넘는 실행: " + ("없음" if bad.empty else
                 ", ".join(f"{r.tag} s{r.seed} {r.split} {r.top1_not_yesno:.1%} ({r.top1_other_examples})"
                           for _, r in bad.iterrows())))
        L.append(f"p_yes + p_no 최솟값: {af.pmass_min.min():.3f}")
    la = res["label_audit_candidates"]
    L.append(f"\n## 라벨 검수 후보: {len(la)}장 (label_audit_candidates.csv)")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out_dir", default=OUT)
    ap.add_argument("--nb", type=int, default=1000)
    ap.add_argument("--top_k", type=int, default=200)
    args = ap.parse_args(argv)
    data = load_all(args.raw, args.tags, args.seeds, args.splits)
    if not data:
        print(f"원자료가 없음: {args.raw}/coco2017_s*_e1_*.csv")
        return
    res = evaluate(data, args.nb, args.top_k)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, t in res.items():
        if not t.empty:
            t.to_csv(f"{args.out_dir}/{name}.csv", index=False, float_format="%.6g")
    split = "test" if "test" in args.splits else args.splits[0]
    text = summarize(res, split)
    with open(f"{args.out_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"{len(data)}개 원자료 -> {args.out_dir}/")


if __name__ == "__main__":
    main()
