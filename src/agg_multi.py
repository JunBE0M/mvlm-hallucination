"""여러 모델에 대한 집계 분석 (A1 γ, A2 β, 방법 AUC, 질문 변형, 큰 N). GPU 불필요.

    python -m src.agg_multi --tags qwen3b qwen7b qwen3vl8b internvl8b llavaov7b --out_dir results/multi

입력 (results/raw, 없는 파일은 건너뜀):
    e1_{tag}.csv, m1_single_{tag}.csv            필수 (A1, A2)
    e1big_{tag}.csv, m1_single_big_{tag}.csv     N=12,16 (A1 γ 를 N=2..16 으로)
    e1var_{tag}.csv                              질문 변형 (template 별 β, 정확도)
    m2_post_{tag}.csv, m0neg_{tag}.csv           방법 AUC
출력: results/multi/*.csv + 표준출력 요약.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import agg_analysis as A
from .data import read_jsonl

NAMES = {"qwen3b": "Qwen2.5-VL-3B", "qwen7b": "Qwen2.5-VL-7B", "qwen3vl8b": "Qwen3-VL-8B",
         "internvl8b": "InternVL3-8B", "llavaov7b": "LLaVA-OV-7B"}


def loads(s):
    return json.loads(s) if isinstance(s, str) else s


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def m1_series(raw, tag):
    import glob
    parts = [read(f"{raw}/m1_single_{tag}.csv")] + [read(f) for f in sorted(glob.glob(f"{raw}/m1_single_big_{tag}*.csv"))]
    s = pd.concat([p for p in parts if p is not None]).drop_duplicates("key").set_index("key")["m"]
    return s


def attach_E(df, m1):
    for c in ("image_ids", "present_positions", "absent_positions"):
        if c in df:
            df[c] = df[c].map(loads)
    keys = [[f"{c.replace(' ', '_')}|{i}" for i in ids] for c, ids in zip(df.cls, df.image_ids)]
    ok = np.array([all(k in m1.index for k in ks) for ks in keys])
    df = df[ok].copy()
    df["E"] = [m1.loc[ks].to_numpy() for ks, o in zip(keys, ok) if o]
    return df


def sdt(h, f, nh, nf):
    from scipy.stats import norm
    H = (h * nh + 0.5) / (nh + 1); F = (f * nf + 0.5) / (nf + 1)
    return norm.ppf(H) - norm.ppf(F), -(norm.ppf(H) + norm.ppf(F)) / 2


def behaviour(e1, model):
    rows = []
    for N, g in e1.groupby("N"):
        r = dict(model=model, N=N)
        for qt in ("forall_yes", "forall_no", "exists_yes", "exists_no"):
            r[f"acc_{qt}"] = g[g.qtype == qt].correct.mean()
        fy, fn = g[g.qtype == "forall_yes"], g[g.qtype == "forall_no"]
        r["false_yes_forall"] = (fn.pred == "Yes").mean()
        r["dprime"], r["c"] = sdt((fy.pred == "Yes").mean(), (fn.pred == "Yes").mean(), len(fy), len(fn))
        r["auc_forall"] = A.auc(fy.m, fn.m)
        ey, en = g[g.qtype == "exists_yes"], g[g.qtype == "exists_no"]
        r["auc_exists"] = A.auc(ey.m, en.m)
        rows.append(r)
    return rows


def a1_gamma(e1, model, label):
    f = e1[e1.qtype.str.startswith("forall")]
    full = f.groupby("pair_id").qtype.nunique()
    e1 = f[f.pair_id.isin(full[full == 2].index)]
    r, s = A.a1({"e1": e1}, model)
    s["Ns"] = label
    r["Ns"] = label
    return r, s


def a2_beta(d, model, question, label):
    d = d[d.N > 1]
    c, best, lo, hi = A.fit_beta(d["m"].to_numpy(), list(d["E"]), d.N.to_numpy(), nboot=NBOOT_BETA)
    r2 = dict(zip(A.BETAS, c))
    e_sd = float(np.concatenate(list(d["E"])).std())   # β 는 e 의 스케일에 의존 -> β·sd(e) 로 모델 간 비교
    row = dict(model=model, question=question, template=label, n=len(d), beta=best, beta_lo=lo, beta_hi=hi,
               e_sd=e_sd, beta_norm=best * e_sd, beta_norm_lo=lo * e_sd, beta_norm_hi=hi * e_sd,
               r2_best=c.max(), r2_mean=r2[0.0], r2_min=r2[10.0], r2_max=r2[-10.0])
    curve = [dict(model=model, question=question, template=label, beta=b, r2=v) for b, v in zip(A.BETAS, c)]
    return row, curve


def methods(D, model):
    e1 = D["e1"]
    f = e1[e1.qtype.str.startswith("forall")].set_index("qid")
    S = {"base": f.m, "M1_min": f.E.map(np.min), "M1_mean": f.E.map(np.mean)}
    if D.get("m2") is not None:
        S["M2_post_min"] = D["m2"].set_index("qid").m_min
    if D.get("m0") is not None:
        S["M0_neg"] = -D["m0"].set_index("qid").m
    T = f[["pair_id", "N", "qtype"]].join(pd.DataFrame(S), how="left")
    rows = []
    for Nsel in sorted(T.N.unique()) + ["all"]:
        TT = T[T.N <= 8] if Nsel == "all" else T[T.N == Nsel]   # "all" = N 2-8 (모든 방법이 있는 범위)
        yes = TT.qtype == "forall_yes"
        pr = TT.pair_id.unique()
        by = {p: g for p, g in TT.groupby("pair_id")}
        boots = []
        for _ in range(NBOOT_AUC):
            s = pd.concat([by[p] for p in RNG.choice(pr, len(pr))])
            y = s.qtype == "forall_yes"
            boots.append({k: A.auc(s.loc[y, k].dropna(), s.loc[~y, k].dropna()) for k in S})
        B = pd.DataFrame(boots)
        for k in S:
            a, b = TT.loc[yes, k].dropna(), TT.loc[~yes, k].dropna()
            if len(a) == 0:
                continue
            rows.append(dict(model=model, N=Nsel, method=k, auc=A.auc(a, b), lo=B[k].quantile(.025),
                             hi=B[k].quantile(.975), acc_thr0=((a > 0).mean() + (b <= 0).mean()) / 2,
                             diff_vs_base=A.auc(a, b) - A.auc(TT.loc[yes, "base"], TT.loc[~yes, "base"]),
                             p_vs_base=min(1, 2 * min(((B[k] - B["base"]) <= 0).mean(),
                                                      ((B[k] - B["base"]) >= 0).mean())) if k != "base" else np.nan))
    return rows


RNG = np.random.default_rng(0)
NBOOT_BETA = 200
NBOOT_AUC = 500


def main(argv=None):
    global NBOOT_BETA, NBOOT_AUC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out_dir", default="results/multi")
    ap.add_argument("--tags", nargs="+", default=list(NAMES))
    ap.add_argument("--nboot_beta", type=int, default=200)
    ap.add_argument("--nboot_auc", type=int, default=500)
    args = ap.parse_args(argv)
    NBOOT_BETA, NBOOT_AUC = args.nboot_beta, args.nboot_auc
    A.NB = 1000
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    out = {k: [] for k in ["behaviour", "a1", "a1s", "a2", "a2c", "methods", "var_acc", "lens", "e3", "small"]}
    for tag in args.tags:
        model = NAMES.get(tag, tag)
        e1 = read(f"{args.raw}/e1_{tag}.csv")
        if e1 is None or not os.path.exists(f"{args.raw}/m1_single_{tag}.csv"):
            print(f"skip {tag}: e1 or m1 missing"); continue
        m1 = m1_series(args.raw, tag)
        e1 = attach_E(e1, m1)
        big = read(f"{args.raw}/e1big_{tag}.csv")
        big = attach_E(big, m1) if big is not None else None
        out["behaviour"] += behaviour(e1 if big is None else pd.concat([e1, big]), model)
        r, s = a1_gamma(e1, model, "2-8"); out["a1"].append(r); out["a1s"].append(s)
        if big is not None and len(big):
            r, s = a1_gamma(pd.concat([e1, big], ignore_index=True), model, "2-16")
            out["a1"].append(r[r.N > 8]); out["a1s"].append(s)
        for fam in ("forall", "exists"):
            row, c = a2_beta(e1[e1.qtype.str.startswith(fam)], model, fam, "base")
            out["a2"].append(row); out["a2c"] += c
        var = read(f"{args.raw}/e1var_{tag}.csv")
        if var is not None:
            qv = {q["qid"]: q for q in read_jsonl("data/questions_var.jsonl")}
            var["template"] = var.qid.map(lambda x: qv[x]["template"])
            var["quant"] = var.qid.map(lambda x: qv[x]["quant"])
            var = attach_E(var, m1)
            for (tp, qu), g in var.groupby(["template", "quant"]):
                row, c = a2_beta(g, model, qu, tp); out["a2"].append(row); out["a2c"] += c
                for N, gg in g.groupby("N"):
                    pos = gg.answer == "Yes"
                    out["var_acc"].append(dict(model=model, template=tp, quant=qu, N=N, acc=gg.correct.mean(),
                                               acc_yes=gg[pos].correct.mean(), acc_no=gg[~pos].correct.mean(),
                                               auc=A.auc(gg[pos].m, gg[~pos].m)))
        sm = read(f"{args.raw}/e1small_{tag}.csv")
        if sm is not None and os.path.exists(f"{args.raw}/m1_single_small_{tag}.csv"):
            m1s = pd.concat([m1.reset_index(), read(f"{args.raw}/m1_single_small_{tag}.csv")[["key", "m"]]]
                            ).drop_duplicates("key").set_index("key")["m"]
            sm = attach_E(sm, m1s)
            for r in behaviour(sm, model):
                out["small"].append({**r, "set": "small(area<=5%)"})
            for r in behaviour(e1[e1.N.isin(sm.N.unique())], model):
                out["small"].append({**r, "set": "all"})
            for fam in ("forall", "exists"):
                row, _ = a2_beta(sm[sm.qtype.str.startswith(fam)], model, fam, "small")
                out["a2"].append(row)
        lens = read(f"{args.raw}/lens_{tag}.csv")
        if lens is not None:
            lens["m_layers"] = lens.m_layers.map(loads)
            lens = attach_E(lens, m1)
            nL = len(lens.m_layers.iloc[0])
            for fam in ("forall", "exists"):
                g = lens[lens.qtype.str.startswith(fam) & (lens.N > 1)]
                if g.empty:
                    continue
                for l in range(1, nL):
                    gg = g.assign(m=g.m_layers.map(lambda v: v[l]))
                    c, best, _, _ = A.fit_beta(gg.m.to_numpy(), list(gg.E), gg.N.to_numpy(), nboot=1)
                    yes = gg.qtype.str.endswith("_yes")
                    out["lens"].append(dict(model=model, question=fam, layer=l, rel_depth=l / (nL - 1), beta=best,
                                            r2_best=c.max(), r2_mean=dict(zip(A.BETAS, c))[0.0],
                                            auc=A.auc(gg[yes].m, gg[~yes].m) if yes.any() and (~yes).any() else np.nan))
        e3dir = f"{args.raw}/e3_{tag}"
        base3 = read(f"{e3dir}/none_a1_all_post_image.csv")
        if base3 is not None:
            b = base3.set_index("qid").m
            for cond in ("none_a1", "absent_a4", "absent_a0", "present_a4"):
                d3 = read(f"{e3dir}/{cond}_all_post_image.csv")
                if d3 is None:
                    continue
                d3 = d3.set_index("qid")
                for N in sorted(d3.N.unique()) + ["all"]:
                    g = d3 if N == "all" else d3[d3.N == N]
                    dm = (g.m - b.reindex(g.index)).dropna()
                    out["e3"].append(dict(model=model, cond=cond, N=N, n=len(g), false_yes=(g.pred == "Yes").mean(),
                                          dm_mean=dm.mean(), dm_lo=dm.mean() - 1.96 * dm.std() / np.sqrt(len(dm)),
                                          dm_hi=dm.mean() + 1.96 * dm.std() / np.sqrt(len(dm)),
                                          dm_over_sd=dm.mean() / b.std()))
        import glob
        m2s = [read(f) for f in sorted(glob.glob(f"{args.raw}/m2_post_{tag}.csv")
                                       + glob.glob(f"{args.raw}/m2_post_{tag}_N*.csv")
                                       + glob.glob(f"{args.raw}/m2_post_big_{tag}*.csv"))]
        m2 = pd.concat(m2s, ignore_index=True).drop_duplicates("qid") if m2s else None
        e1all = e1 if big is None else pd.concat([e1, big], ignore_index=True)
        D = {"e1": e1all, "m2": m2, "m0": read(f"{args.raw}/m0neg_{tag}.csv")}
        out["methods"] += methods(D, model)
        print(f"done {tag}", flush=True)
    T = {}
    for k, v in out.items():
        if not v:
            continue
        T[k] = pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
        T[k].to_csv(f"{args.out_dir}/{k}.csv", index=False)
    if "behaviour" in T:
        print("== behaviour\n", T["behaviour"].round(3).to_string(index=False))
    if "a1s" in T:
        print("\n== A1 γ (Δ ∝ N^-γ) and item-level R²\n", pd.DataFrame(T["a1s"]).round(3).to_string(index=False))
    if "a1" in T:
        print("\n== A1 Δ×N\n", T["a1"].pivot_table(index="model", columns="N", values="deltaxN").round(2).to_string())
    if "a2" in T:
        print("\n== A2 β (>0 min-like, 0 mean, <0 max-like)\n", T["a2"].round(3).to_string(index=False))
    if "methods" in T:
        print("\n== methods AUC (forall)\n",
              T["methods"].pivot_table(index=["model", "method"], columns="N", values="auc", sort=False).round(3).to_string())
    if "small" in T:
        print("\n== small objects vs all (N=4,8)\n", T["small"][["model", "set", "N", "acc_forall_yes", "acc_forall_no",
              "false_yes_forall", "auc_forall", "auc_exists"]].round(3).to_string(index=False))
    if "e3" in T:
        print("\n== E3 false-Yes (forall_no) by condition\n",
              T["e3"].pivot_table(index=["model", "cond"], columns="N", values="false_yes", sort=False).round(3).to_string())
        print("\n== E3 Δm / sd(m) vs none\n",
              T["e3"].pivot_table(index=["model", "cond"], columns="N", values="dm_over_sd", sort=False).round(3).to_string())
    if "lens" in T:
        L = T["lens"].copy(); L["depth"] = (L.rel_depth * 10).round() / 10
        print("\n== lens β by relative depth\n",
              L.pivot_table(index=["model", "question"], columns="depth", values="beta").round(2).to_string())
        print("\n== lens AUC by relative depth\n",
              L.pivot_table(index=["model", "question"], columns="depth", values="auc").round(2).to_string())
    if "var_acc" in T:
        print("\n== variants AUC\n",
              T["var_acc"].pivot_table(index=["model", "template"], columns="N", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()
