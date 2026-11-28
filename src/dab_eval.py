"""COCO 2017 분할(coco2017_seed1·2·3) 방법 비교 (docs/split_coco.md): base / DAB / M2 / M1 / B 를 같은 문항에서 비교한다. GPU 불필요.

    python -m src.dab_eval [--tags qwen3b qwen7b ...] [--seeds 1 2 3] [--splits val test]

입력 (results/coco2017_custom/raw, 없는 파일은 건너뜀. 문항 목록은 data/splits/coco2017_seed{S}/{split}.jsonl 기준):
    coco2017_s{S}_e1_{split}_{tag}.csv                base     (src.run_eval)
    coco2017_s{S}_dab_a{α}_{split}_{tag}.csv          DAB(α)   (src.dab), α 마다 한 파일
    coco2017_s{S}_m2_post_{split}_{tag}.csv           M2       (src.isolated, ∀ min / ∃ max)
    coco2017_s{S}_m1_agg_{split}_{tag}.csv            M1       (src.per_image --aggregate, ∀ min / ∃ max)
    coco2017_s{S}_m1_single_{split}_{tag}.csv         (β 계산용 이미지별 증거)
    coco2017_s{S}_B_e1_{split}_{tag}_{loss}.csv       B[loss]  (src.run_eval --adapter, 학습 seed = S)
출력 (--out_dir, 기본 results/coco2017_custom/analysis):
    auc.csv        tag × seed × split × method × family(forall/exists) × N(개별 + 전체) : AUC + 95% CI
    diff.csv       같은 셀에서 method − base, method − M2 : AUC 차이 + 짝지은 bootstrap 95% CI + p
    dab_alpha.csv  seed 별 val 에서 α 별 AUC (α 선택용. 그 seed 의 val 로 고르고 docs/ 에 동결한 뒤 그 seed 의 test 를 1회)
    beta.csv       ∀/∃ 집계 온도 β (agg_analysis.fit_beta), 예측 P3
    summary.txt    test 의 seed 평균 ± 표준편차, 예측 P1–P3 판정용 숫자
bootstrap: ∀ 는 pair(forall_yes, forall_no) 단위, ∃ 는 Yes/No 를 따로 다시 뽑는다(층화). AUC 식은 agg_analysis.auc 와 같다.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .agg_analysis import fit_beta
from .data import read_jsonl

RAW = "results/coco2017_custom/raw"
QROOT = "data/splits"
TAGS = ["qwen3b", "qwen7b", "qwen3vl8b", "internvl8b", "llavaov7b"]
SEEDS = [1, 2, 3]
SPLITS = ["val", "test"]
FAMS = ("forall", "exists")


# ------------------------------------------------------------------ AUC / bootstrap
def auc(score, yes) -> float:
    """P(score_yes > score_no) + 0.5 P(=). agg_analysis.auc 와 같은 식 (scipy rankdata 로 빠르게)."""
    score, yes = np.asarray(score, float), np.asarray(yes, bool)
    n1, n0 = int(yes.sum()), int((~yes).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[yes].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _resamples(fam: str, d: pd.DataFrame, nb: int, rng) -> list[np.ndarray]:
    """bootstrap 행 인덱스 목록. ∀: pair 단위(두 행 함께), ∃: Yes / No 를 따로 다시 뽑음."""
    idx = np.arange(len(d))
    if fam == "forall":
        codes, _ = pd.factorize(d["pair_id"].astype(str))
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(np.r_[-1, codes[order], -2]))
        groups = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        out = []
        for _ in range(nb):
            pick = rng.integers(0, len(groups), len(groups))
            out.append(np.concatenate([groups[k] for k in pick]))
        return out
    yes = d["yes"].to_numpy(bool)
    pos, neg = idx[yes], idx[~yes]
    return [np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))] for _ in range(nb)]


def boot(fam, d, cols, nb, rng):
    """같은 bootstrap 표본으로 cols 의 AUC 를 계산. 반환 (점추정 dict, bootstrap 배열 dict)."""
    y = d["yes"].to_numpy(bool)
    S = {c: d[c].to_numpy(float) for c in cols}
    point = {c: auc(S[c], y) for c in cols}
    B = {c: np.empty(nb) for c in cols}
    for b, ix in enumerate(_resamples(fam, d, nb, rng)):
        yy = y[ix]
        for c in cols:
            B[c][b] = auc(S[c][ix], yy)
    return point, B


def ci_of(point, B, c):
    return point[c], float(np.nanpercentile(B[c], 2.5)), float(np.nanpercentile(B[c], 97.5))


def diff_of(point, B, c, ref):
    """(c − ref) 의 차이, 95% CI, 양측 bootstrap p."""
    dd = B[c] - B[ref]
    p = min(1.0, 2 * min(float(np.mean(dd <= 0)), float(np.mean(dd >= 0))))
    return point[c] - point[ref], float(np.nanpercentile(dd, 2.5)), float(np.nanpercentile(dd, 97.5)), p


# ------------------------------------------------------------------ 입력
def _read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def qfile(seed: int, split: str, qroot: str = QROOT) -> str:
    return f"{qroot}/coco2017_seed{seed}/{split}.jsonl"


def load_methods(tag: str, seed: int, split: str, raw: str = RAW) -> dict[str, pd.Series]:
    """method 이름 -> qid 로 색인한 m (점수) Series."""
    out: dict[str, pd.Series] = {}
    pre = f"{raw}/coco2017_s{seed}_"

    def put(name, df):
        if df is not None and len(df):
            out[name] = df.drop_duplicates("qid").set_index("qid")["m"].astype(float)

    put("base", _read(f"{pre}e1_{split}_{tag}.csv"))
    for f in sorted(glob.glob(f"{pre}dab_a*_{split}_{tag}.csv")):
        m = re.fullmatch(rf"coco2017_s{seed}_dab_a([0-9.]+)_{re.escape(split)}_{re.escape(tag)}\.csv", os.path.basename(f))
        if m:
            put(f"DAB(a={float(m.group(1)):g})", _read(f))
    put("M2", _read(f"{pre}m2_post_{split}_{tag}.csv"))
    put("M1", _read(f"{pre}m1_agg_{split}_{tag}.csv"))
    for f in sorted(glob.glob(f"{pre}B_e1_{split}_{tag}_*.csv")):
        m = re.fullmatch(rf"coco2017_s{seed}_B_e1_{re.escape(split)}_{re.escape(tag)}_(\w+)\.csv", os.path.basename(f))
        if m:
            put(f"B[{m.group(1)}]", _read(f))
    return out


def question_table(seed: int, split: str, qroot: str = QROOT) -> pd.DataFrame:
    qs = read_jsonl(qfile(seed, split, qroot))
    T = pd.DataFrame([{k: q[k] for k in ("qid", "pair_id", "cls", "N", "qtype", "answer", "image_ids")} for q in qs])
    T["family"] = T.qtype.str.split("_").str[0]
    T["yes"] = T.answer.eq("Yes")
    return T.set_index("qid")


def _cells(T):
    yield "all", T
    for n in sorted(T.N.unique()):
        yield str(n), T[T.N == n]


# ------------------------------------------------------------------ 분석
def evaluate(tags, seeds, splits, raw=RAW, qroot=QROOT, nb=1000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    auc_rows, diff_rows = [], []
    for seed in seeds:
        for split in splits:
            if not os.path.exists(qfile(seed, split, qroot)):
                continue
            Q = question_table(seed, split, qroot)
            for tag in tags:
                meths = load_methods(tag, seed, split, raw)
                if not meths:
                    continue
                T = Q.join(pd.DataFrame(meths), how="left")
                names = list(meths)
                key = dict(tag=tag, seed=seed, split=split)
                for fam in FAMS:
                    for nlab, d in _cells(T[T.family == fam]):
                        # 모든 문항이 있는 방법은 한 번의 bootstrap 을 공유하고, 빠진 문항이 있는 방법은 따로 계산
                        full = [c for c in names if d[c].notna().all()]
                        jobs = [(full, d)] if full else []
                        jobs += [([c] + [r for r in ("base", "M2") if r in full], d[d[c].notna()])
                                 for c in names if c not in full]
                        for cols, sub in jobs:
                            sub = sub.reset_index()
                            if not cols or len(sub) == 0 or sub.yes.all() or (~sub.yes).all():
                                continue
                            point, B = boot(fam, sub, cols, nb, rng)
                            record = cols if cols is full else cols[:1]
                            for c in record:
                                a, lo, hi = ci_of(point, B, c)
                                auc_rows.append(dict(**key, method=c, family=fam, N=nlab, n=len(sub),
                                                     auc=a, lo=lo, hi=hi))
                            for ref in ("base", "M2"):
                                if ref not in cols:
                                    continue
                                for c in record:
                                    if c == ref:
                                        continue
                                    df_, lo, hi, p = diff_of(point, B, c, ref)
                                    diff_rows.append(dict(**key, method=c, ref=ref, family=fam, N=nlab,
                                                          n=len(sub), diff=df_, lo=lo, hi=hi, p=p))
    return pd.DataFrame(auc_rows), pd.DataFrame(diff_rows)


def dab_alpha_table(auc_df: pd.DataFrame) -> pd.DataFrame:
    """seed 별 val 의 DAB α 별 AUC (N 전체). 선택은 ∀ AUC 최대 (동점이면 ∃ AUC)."""
    d = auc_df[(auc_df.split == "val") & (auc_df.N == "all") & auc_df.method.str.startswith("DAB")]
    if d.empty:
        return d
    w = d.pivot_table(index=["tag", "seed", "method"], columns="family", values="auc").reset_index()
    w["alpha"] = w.method.str.extract(r"a=([0-9.]+)", expand=False).astype(float)
    w["selected"] = False
    for _, g in w.groupby(["tag", "seed"]):
        w.loc[g.sort_values(["forall", "exists"], ascending=False).index[0], "selected"] = True
    return w.sort_values(["tag", "seed", "alpha"])


def beta_table(tags, seeds, splits, raw=RAW, qroot=QROOT, nboot=100) -> pd.DataFrame:
    """method 점수 m 을 이미지별 증거 E(M1 단일 이미지 m)의 softmin 으로 설명하는 β. <0 max 쪽, 0 평균, >0 min 쪽."""
    rows = []
    for seed in seeds:
        for split in splits:
            if not os.path.exists(qfile(seed, split, qroot)):
                continue
            Q = question_table(seed, split, qroot)
            for tag in tags:
                single = _read(f"{raw}/coco2017_s{seed}_m1_single_{split}_{tag}.csv")
                if single is None:
                    continue
                ev = single.drop_duplicates("key").set_index("key")["m"]
                for name, m in load_methods(tag, seed, split, raw).items():
                    if name == "M1":
                        continue
                    for fam in FAMS:
                        d = Q[(Q.family == fam) & Q.index.isin(m.index)]
                        keys = [[f"{c.replace(' ', '_')}|{i}" for i in ids] for c, ids in zip(d.cls, d.image_ids)]
                        ok = [all(k in ev.index for k in ks) for ks in keys]
                        d = d[ok]
                        if len(d) < 20:
                            continue
                        E = [ev.loc[ks].to_numpy(float) for ks, o in zip(keys, ok) if o]
                        c, best, lo, hi = fit_beta(m.loc[d.index].to_numpy(float), E, d.N.to_numpy(), nboot=nboot)
                        esd = float(np.concatenate(E).std())
                        rows.append(dict(tag=tag, seed=seed, split=split, method=name, family=fam, n=len(d),
                                         beta=best, lo=lo, hi=hi, beta_norm=best * esd, beta_norm_lo=lo * esd,
                                         beta_norm_hi=hi * esd, r2_best=float(c.max())))
    return pd.DataFrame(rows)


def summarize(auc_df, diff_df, beta_df, test_split="test") -> str:
    t = auc_df[auc_df.split == test_split]
    if t.empty:
        return "test 결과 없음 (seed 마다 val 로 설정을 동결한 뒤 test 를 돌린다)\n"
    lines = ["## test AUC (N 전체), seed 평균 ± 표준편차"]
    g = t[t.N == "all"].groupby(["tag", "method", "family"]).auc.agg(["mean", "std", "count"]).reset_index()
    for _, r in g.iterrows():
        sd = 0.0 if np.isnan(r["std"]) else r["std"]
        lines.append(f"{r.tag:11s} {r.method:12s} {r.family:7s} {r['mean']:.3f} ± {sd:.3f} (seed {int(r['count'])})")
    lines.append("\n## 예측 판정용 숫자 (seed 평균, DAB 는 α 별)")
    dt = diff_df[(diff_df.split == test_split) & (diff_df.ref == "base")]
    for (tag, meth), g2 in dt[dt.method.str.startswith("DAB")].groupby(["tag", "method"]):
        fa = g2[(g2.family == "forall") & (g2.N == "all")]["diff"].mean()
        ex = g2[(g2.family == "exists") & (g2.N == "all")]["diff"].mean()
        m2 = dt[(dt.tag == tag) & (dt.method == "M2") & (dt.family == "forall") & (dt.N == "all")]["diff"].mean()
        per_n = g2[(g2.family == "forall") & (g2.N != "all")].groupby("N")["diff"].mean()
        rho = spearmanr(per_n.index.astype(int), per_n.values)[0] if len(per_n) > 2 else float("nan")
        lines.append(f"{tag:11s} {meth:12s} P1: ∀ 개선 {fa:+.3f} (M2 {m2:+.3f}), N 과 개선의 순위상관 {rho:+.2f} | "
                     f"P2: ∃ 변화 {ex:+.3f}")
    if beta_df is not None and not beta_df.empty:
        lines.append("\n## P3: 정규화 β (seed 평균; <0 max 쪽, 0 평균, >0 min 쪽)")
        b = beta_df[beta_df.split == test_split].groupby(["tag", "method", "family"]).beta_norm.mean().unstack()
        lines.append(b.round(2).to_string())
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=TAGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--qroot", default=QROOT)
    ap.add_argument("--out_dir", default="results/coco2017_custom/analysis")
    ap.add_argument("--nb", type=int, default=1000)
    ap.add_argument("--no_beta", action="store_true")
    args = ap.parse_args(argv)
    auc_df, diff_df = evaluate(args.tags, args.seeds, args.splits, args.raw, args.qroot, args.nb)
    if auc_df.empty:
        print("결과 파일이 없음")
        return
    os.makedirs(args.out_dir, exist_ok=True)
    auc_df.to_csv(f"{args.out_dir}/auc.csv", index=False)
    diff_df.to_csv(f"{args.out_dir}/diff.csv", index=False)
    alpha = dab_alpha_table(auc_df)
    if not alpha.empty:
        alpha.to_csv(f"{args.out_dir}/dab_alpha.csv", index=False)
        print(alpha.to_string(index=False))
    beta_df = None if args.no_beta else beta_table(args.tags, args.seeds, args.splits, args.raw, args.qroot)
    if beta_df is not None and not beta_df.empty:
        beta_df.to_csv(f"{args.out_dir}/beta.csv", index=False)
    text = summarize(auc_df, diff_df, beta_df)
    open(f"{args.out_dir}/summary.txt", "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
