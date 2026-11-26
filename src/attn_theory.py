"""실험 A: "attention 은 존재 탐지기" 가설과 1/N 이론 (학습 없음).

가설: softmax attention 은 내용(객체)이 있는 곳에 몰린다. 그래서 ∃(있는 곳 찾기)는 사실상 max 가 되지만,
∀(없는 곳 찾기)는 그렇게 되지 않아 평균에 머문다. 이미지 블록 key 가 비슷하면 각 이미지 가중치 ≈ 1/N 이므로 Δ ∝ 1/N.

두 단계 (둘 다 GPU 불필요):
  1) compact : E2 npz([L,H,N], gitignore 대상) -> 커밋 가능한 요약
       results/raw/e2c_{tag}.csv.gz  문항 × 층: 이미지 간 분포 share(질량 가중 헤드 평균), share_h(헤드별 분포의 평균),
                                    share_tok(토큰 수 보정), img/text/sys 질량
       results/raw/e2h_{tag}.csv     (set, qtype, N, layer, head) 셀 평균: odd 이미지 share×N, 정규화 엔트로피
     python -m src.attn_theory compact --tag qwen7b --e2 results/raw/e2x_qwen7b*.csv --neg results/raw/e2xneg_qwen7b*.csv
  2) analyze : 위 csv + results/multi/a1s.csv(행동 γ) + E1/M1 csv 만 읽는다.
     python -m src.attn_theory analyze --out_dir results/novelty

정의
  p_i      = img_sum_i / Σ_j img_sum_j       (마지막 query 토큰 -> 이미지 i 의 attention 질량을 이미지끼리만 정규화)
  share×N  : 균등이면 1. forall_no 의 absent 이미지, exists_yes 의 present 이미지("odd")에 대해 본다.
  H/log N  : 이미지 간 분포의 정규화 엔트로피 (1 = 균등)
  band     : late = 뒤쪽 절반 층 (실험 기록 1 E2 와 같은 정의), all = 전 층 평균
  γ_attn   : forall_no absent share ∝ N^-γ_attn (N=2..16) 을 log-log 로 피팅. 균등(1/N)이면 γ_attn = 1.
             "평균 집계 + attention 가중치" 모형이라면 Δ ∝ share_absent 이므로 γ_behav ≈ γ_attn 을 예측.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from . import agg_analysis as A
from .agg_multi import NAMES, attach_E, m1_series

ORDER = ["qwen3b", "qwen7b", "qwen3vl8b", "internvl8b", "llavaov7b"]
RNG = np.random.default_rng(0)


def loads(s):
    return json.loads(s) if isinstance(s, str) else s


def odd_pos(row):
    """forall_no -> absent 위치, exists_yes -> present 위치, 그 외 None."""
    if row["qtype"] == "forall_no" and len(row["absent_positions"]) == 1:
        return row["absent_positions"][0]
    if row["qtype"] == "exists_yes" and len(row["present_positions"]) == 1:
        return row["present_positions"][0]
    return None


# ------------------------------------------------------------------ compact (npz -> csv)
def compact(args):
    rows, heads = [], {}
    for set_name, path in [(st, p) for st, ps in (("base", args.e2), ("neg", args.neg or [])) for p in ps]:
        if not os.path.exists(path):
            continue
        meta = pd.read_csv(path)
        for c in ("present_positions", "absent_positions"):
            meta[c] = meta[c].map(loads)
        n_miss = 0
        for _, r in meta.iterrows():
            npz = r["npz"]
            if not os.path.exists(npz):
                n_miss += 1
                continue
            z = np.load(npz)
            s = z["img_sum"].astype(np.float64)                     # [L, H, N]
            ntok = z["n_tokens"].astype(np.float64)
            hm = s.mean(1)                                          # [L, N]  헤드 평균 질량
            share = hm / hm.sum(-1, keepdims=True)
            ph = s / np.clip(s.sum(-1, keepdims=True), 1e-12, None)  # [L, H, N] 헤드별 분포
            share_h = ph.mean(1)
            tm = hm / ntok
            share_tok = tm / tm.sum(-1, keepdims=True)
            text, sysm = z["text_sum"].mean(1), z["sys_sum"].mean(1)
            N = s.shape[-1]
            for l in range(s.shape[0]):
                rows.append(dict(set=set_name, qid=r["qid"], N=N, qtype=r["qtype"], layer=l,
                                 share=json.dumps(np.round(share[l], 5).tolist()),
                                 share_h=json.dumps(np.round(share_h[l], 5).tolist()),
                                 share_tok=json.dumps(np.round(share_tok[l], 5).tolist()),
                                 img_mass=round(float(hm[l].sum()), 5), text_mass=round(float(text[l]), 5),
                                 sys_mass=round(float(sysm[l]), 5)))
            # 헤드별 셀 집계
            k = odd_pos(r)
            ent = -(ph * np.log(np.clip(ph, 1e-12, None))).sum(-1) / np.log(N) if N > 1 else np.zeros(ph.shape[:2])
            key = (set_name, r["qtype"], N)
            acc = heads.setdefault(key, {"n": 0, "ent": 0.0, "odd": 0.0})
            acc["n"] += 1
            acc["ent"] = acc["ent"] + ent
            if k is not None:
                acc["odd"] = acc["odd"] + ph[:, :, k] * N
        if n_miss:
            print(f"[warn] {path}: {n_miss} npz missing")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    out.to_csv(f"{args.out_prefix}e2c_{args.tag}.csv.gz", index=False, compression="gzip")
    H = []
    for (st, qt, N), a in heads.items():
        ent = a["ent"] / a["n"]
        odd = a["odd"] / a["n"] if not np.isscalar(a["odd"]) else None
        for l in range(ent.shape[0]):
            for h in range(ent.shape[1]):
                H.append(dict(set=st, qtype=qt, N=N, layer=l, head=h, n=a["n"], entropy_norm=round(ent[l, h], 5),
                              odd_shareN=round(odd[l, h], 5) if odd is not None else np.nan))
    pd.DataFrame(H).to_csv(f"{args.out_prefix}e2h_{args.tag}.csv", index=False)
    print(f"[{args.tag}] wrote {len(out)} rows (e2c), {len(H)} head rows (e2h)")


# ------------------------------------------------------------------ analysis helpers
def load_tag(raw, tag):
    """E2 메타 + compact share. 반환: (items[set,qid,...,share_late,share_all,...], per-layer long df)."""
    p = f"{raw}/e2c_{tag}.csv.gz"
    if not os.path.exists(p):
        return None
    C = pd.read_csv(p)
    for c in ("share", "share_h", "share_tok"):
        C[c] = C[c].map(lambda s: np.array(json.loads(s)))
    import glob
    metas = []
    for st, pat in [("base", f"{raw}/e2x_{tag}.csv"), ("base", f"{raw}/e2x_{tag}_*.csv"),
                    ("neg", f"{raw}/e2xneg_{tag}.csv"), ("neg", f"{raw}/e2xneg_{tag}_*.csv")]:
        for f in sorted(glob.glob(pat)):
            m = pd.read_csv(f)
            m["set"] = st
            metas.append(m)
    meta = pd.concat(metas, ignore_index=True).drop_duplicates(["set", "qid"])
    for c in ("present_positions", "absent_positions", "image_ids"):
        meta[c] = meta[c].map(loads)
    meta["odd"] = [odd_pos(r) for _, r in meta.iterrows()]
    nL = C.layer.max() + 1
    C["late"] = C.layer >= nL // 2
    agg = {}
    for band, sel in [("late", C.late), ("all", np.ones(len(C), bool))]:
        g = C[sel].groupby(["set", "qid"])
        for c in ("share", "share_h", "share_tok"):
            agg[f"{c}_{band}"] = g[c].apply(lambda v: np.mean(np.stack(list(v)), 0))
        agg[f"img_mass_{band}"] = g.img_mass.mean()
    S = pd.DataFrame(agg).reset_index()
    items = meta.merge(S, on=["set", "qid"], how="inner")
    return items, C, nL


def odd_share(items, col):
    return np.array([s[int(k)] if not pd.isna(k) else np.nan
                     for s, k in zip(items[col], items["odd"])])


def norm_entropy(v):
    v = np.clip(np.asarray(v, float), 1e-12, None)
    return float(-(v * np.log(v)).sum() / np.log(len(v))) if len(v) > 1 else np.nan


def fit_gamma(Ns, vals_by_N, nb=1000):
    """log mean(v) = c - γ log N. 문항 bootstrap CI."""
    def g(d):
        x = np.log(Ns)
        y = np.log([np.mean(d[n]) for n in Ns])
        return -np.polyfit(x, y, 1)[0]
    est = g(vals_by_N)
    bs = [g({n: RNG.choice(vals_by_N[n], len(vals_by_N[n])) for n in Ns}) for _ in range(nb)]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def partial_corr(x, y, Z):
    """Z(더미 행렬)로 x, y 를 잔차화한 뒤 상관."""
    Z = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def r2_fit(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - (y - X @ c).var() / y.var())


def softmin_rows(E_list, N, beta):
    """행 순서를 유지한 softmin_β (N 별로 묶어 계산)."""
    out = np.empty(len(E_list))
    for n in np.unique(N):
        ix = np.where(N == n)[0]
        out[ix] = A.softmin(np.stack([E_list[i] for i in ix]), beta)
    return out


def boot_ci(f, n, nb=1000):
    bs = [f(RNG.choice(n, n)) for _ in range(nb)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ------------------------------------------------------------------ analyze
def analyze(args):
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    a1s = pd.read_csv(args.a1s)
    out = {k: [] for k in ["share", "gamma", "item", "aggr", "m0", "layers", "heads"]}
    for tag in args.tags:
        L = load_tag(args.raw, tag)
        if L is None:
            print(f"skip {tag}: no e2c"); continue
        items, C, nL = L
        model = NAMES.get(tag, tag)
        base = items[items.set == "base"].copy()
        # ---- 1) 셀별 share 통계 (late / all band, 3가지 share 정의)
        for band in ("late", "all"):
            for col in ("share", "share_h", "share_tok"):
                c = f"{col}_{band}"
                base["odd_sh"] = odd_share(base, c)
                base["ent"] = base[c].map(norm_entropy)
                for (qt, N), g in base.groupby(["qtype", "N"]):
                    v = g.odd_sh.dropna().to_numpy()
                    lo, hi = boot_ci(lambda ix: np.mean(v[ix]) * N, len(v), 500) if len(v) else (np.nan, np.nan)
                    out["share"].append(dict(model=model, band=band, share_def=col, qtype=qt, N=N, n=len(g),
                                             odd_share=np.mean(v) if len(v) else np.nan,
                                             odd_shareN=np.mean(v) * N if len(v) else np.nan, lo=lo, hi=hi,
                                             entropy_norm=g.ent.mean(), img_mass=g[f"img_mass_{band}"].mean()))
        # ---- 2) γ_attn (forall_no absent share ∝ N^-γ)  vs γ_behav
        for band in ("late", "all"):
            for col in ("share", "share_h", "share_tok"):
                fn = base[base.qtype == "forall_no"]
                v = odd_share(fn, f"{col}_{band}")
                Ns = sorted(fn.N.unique())
                d = {n: v[(fn.N == n).to_numpy()] for n in Ns}
                g, lo, hi = fit_gamma(Ns, d)
                ey = base[base.qtype == "exists_yes"]
                ve = odd_share(ey, f"{col}_{band}")
                de = {n: ve[(ey.N == n).to_numpy()] for n in sorted(ey.N.unique())}
                ge, _, _ = fit_gamma(sorted(de), de, nb=1)
                gb = a1s[(a1s.model == model) & (a1s.Ns == "2-16")]
                out["gamma"].append(dict(model=model, band=band, share_def=col, gamma_attn=g, lo=lo, hi=hi,
                                         gamma_attn_exists_present=ge,
                                         gamma_behav=gb.gamma.iloc[0] if len(gb) else np.nan,
                                         gamma_behav_lo=gb.gamma_lo.iloc[0] if len(gb) else np.nan,
                                         gamma_behav_hi=gb.gamma_hi.iloc[0] if len(gb) else np.nan))
        # ---- 3) 층별 odd share×N (forall_no absent, exists_yes present), base
        Cb = C[C.set == "base"].merge(base[["qid", "odd"]], on="qid")
        Cb = Cb[Cb.odd.notna()]
        Cb["oddN"] = [s[int(k)] * n for s, k, n in zip(Cb.share, Cb.odd, Cb.N)]
        Cb["ent"] = Cb.share.map(norm_entropy)
        for (qt, N, l), g in Cb.groupby(["qtype", "N", "layer"]):
            out["layers"].append(dict(model=model, qtype=qt, N=N, layer=l, rel_depth=l / (nL - 1),
                                      odd_shareN=g.oddN.mean(), entropy_norm=g.ent.mean()))
        # ---- 4) 문항 단위: absent share 와 Δ
        m1 = m1_series(args.raw, tag)
        fy = base[base.qtype == "forall_yes"].set_index("pair_id")
        fn = base[base.qtype == "forall_no"].set_index("pair_id")
        P = fy[["N", "m", "image_ids"]].join(fn[["m", "odd", "image_ids", "share_late", "share_all", "share_tok_late"]],
                                              rsuffix="_no", how="inner")
        P["delta"] = P.m - P.m_no
        for band, col in [("late", "share_late"), ("all", "share_all"), ("late_tok", "share_tok_late")]:
            sh = np.array([s[int(k)] for s, k in zip(P[col], P.odd)])
            N = P.N.to_numpy()
            pos_rel = P.odd.to_numpy() / (N - 1)
            cell = pd.get_dummies(pd.Series([f"{n}|{int(k)}" for n, k in zip(N, P.odd)]), drop_first=True).to_numpy(float)
            Nd = pd.get_dummies(pd.Series(N), drop_first=True).to_numpy(float)
            # e_rep - e_abs (M1) 로 "평균 모형" 예측: 균등 1/N vs attention 가중 share
            keys_y = [[f"{c.replace(' ', '_')}|{i}" for i in ids] for c, ids in zip(P.index.str.split("_N").str[0], P.image_ids)]
            keys_n = [[f"{c.replace(' ', '_')}|{i}" for i in ids] for c, ids in zip(P.index.str.split("_N").str[0], P.image_ids_no)]
            ok = np.array([all(k in m1.index for k in a + b) for a, b in zip(keys_y, keys_n)])
            de = np.array([m1[a[int(k)]] - m1[b[int(k)]] if o else np.nan
                           for a, b, k, o in zip(keys_y, keys_n, P.odd, ok)])
            row = dict(model=model, band=band, n=len(P), corr_share_delta=float(np.corrcoef(sh, P.delta)[0, 1]),
                       pcorr_share_delta_N=partial_corr(sh, P.delta.to_numpy(), Nd),
                       pcorr_share_delta_Npos=partial_corr(sh, P.delta.to_numpy(), cell),
                       corr_shareN_delta_withinN=np.nanmean([np.corrcoef(sh[N == n], P.delta[N == n])[0, 1]
                                                             for n in np.unique(N)]),
                       corr_pos_share=float(np.corrcoef(pos_rel, sh * N)[0, 1]))
            if ok.sum() > 20:
                d, s_, n_ = P.delta.to_numpy()[ok], sh[ok], N[ok]
                row.update(r2_uniform=r2_fit(d, de[ok] / n_), r2_attnweighted=r2_fit(d, de[ok] * s_),
                           r2_both=r2_fit(d, np.column_stack([de[ok] / n_, de[ok] * s_])),
                           pcorr_share_delta_Npos_de=partial_corr(s_, d, np.column_stack([cell[ok], de[ok] / n_])))
            out["item"].append(row)
        # ---- 5) attention 가중 평균이 ∃=max / ∀=mean 집계를 재현하는가 (M1 증거 e_i 필요)
        bE = attach_E(base.drop(columns=[c for c in ("E",) if c in base]).copy(), m1)
        for fam, qts in [("forall", ("forall_yes", "forall_no")), ("exists", ("exists_yes", "exists_no"))]:
            g = bE[bE.qtype.isin(qts) & (bE.N > 1)]
            if g.empty:
                continue
            mv, Nv = g.m.to_numpy(), g.N.to_numpy()
            Nd = pd.get_dummies(pd.Series(Nv), drop_first=True).to_numpy(float)
            for band in ("late", "all"):
                xa = np.array([np.dot(s, e) for s, e in zip(g[f"share_{band}"], g.E)])
                xm = np.array([np.mean(e) for e in g.E])
                xmax = np.array([np.max(e) for e in g.E]); xmin = np.array([np.min(e) for e in g.E])
                c, best, _, _ = A.fit_beta(mv, list(g.E), Nv, nboot=1)
                out["aggr"].append(dict(model=model, question=fam, band=band, n=len(g),
                                        qtypes="+".join(sorted(g.qtype.unique())),
                                        r2_mean=r2_fit(mv, np.column_stack([Nd, xm])),
                                        r2_attn=r2_fit(mv, np.column_stack([Nd, xa])),
                                        r2_max=r2_fit(mv, np.column_stack([Nd, xmax])),
                                        r2_min=r2_fit(mv, np.column_stack([Nd, xmin])),
                                        r2_softmin_best=float(c.max()), beta_best=float(best),
                                        # attention 가중 평균의 '유효 β': Σp·e 와 상관이 가장 큰 softmin_β(e)
                                        beta_eff_attn=float(A.BETAS[np.argmax(
                                            [np.corrcoef(xa, softmin_rows(list(g.E), Nv, b))[0, 1] for b in A.BETAS])])))
        # ---- 6) M0: 부재를 묻는 질문(neg)에서도 absent 에 attention 이 안 가는가 (같은 이미지 묶음, paired)
        neg = items[(items.set == "neg") & (items.qtype == "forall_no")]
        if len(neg):
            b = base[base.qtype == "forall_no"].set_index("qid")
            n_ = neg.set_index("qid")
            common = n_.index.intersection(b.index)
            for band in ("late", "all"):
                sb = odd_share(b.loc[common].reset_index(), f"share_{band}") * b.loc[common].N.to_numpy()
                sn = odd_share(n_.loc[common].reset_index(), f"share_{band}") * n_.loc[common].N.to_numpy()
                Nv = b.loc[common].N.to_numpy()
                for Nsel in list(np.unique(Nv)) + ["all"]:
                    sel = np.ones(len(Nv), bool) if Nsel == "all" else Nv == Nsel
                    dd = sn[sel] - sb[sel]
                    lo, hi = boot_ci(lambda ix: dd[ix].mean(), len(dd), 1000)
                    out["m0"].append(dict(model=model, band=band, N=Nsel, n=int(sel.sum()),
                                          absent_shareN_forall=sb[sel].mean(), absent_shareN_neg=sn[sel].mean(),
                                          diff=dd.mean(), lo=lo, hi=hi,
                                          frac_neg_above_uniform=(sn[sel] > 1).mean()))
        # ---- 7) 헤드별: absent 를 찾는 헤드가 있는가 (late 층, forall_no share×N < 1 이 강한 헤드)
        hp = f"{args.raw}/e2h_{tag}.csv"
        if os.path.exists(hp):
            Hh = pd.read_csv(hp)
            Hh = Hh[Hh.set == "base"]
            for (qt, N), g in Hh[Hh.qtype.isin(["forall_no", "exists_yes"])].groupby(["qtype", "N"]):
                gl = g[g.layer >= nL // 2]
                out["heads"].append(dict(model=model, qtype=qt, N=N, n_heads=len(gl),
                                         median=gl.odd_shareN.median(), q05=gl.odd_shareN.quantile(.05),
                                         q95=gl.odd_shareN.quantile(.95), frac_gt2=(gl.odd_shareN > 2).mean(),
                                         frac_lt05=(gl.odd_shareN < .5).mean(),
                                         entropy_median=gl.entropy_norm.median()))
        print(f"done {tag}", flush=True)
    T = {k: pd.DataFrame(v) for k, v in out.items() if v}
    for k, v in T.items():
        v.to_csv(f"{args.out_dir}/A_{k}.csv", index=False)
    summarize(T)
    if not args.no_fig:
        figures(T, args.fig_dir)


def summarize(T):
    if "share" in T:
        S = T["share"][(T["share"].band == "late") & (T["share"].share_def == "share")]
        print("\n== odd-image share × N (late layers; 1 = uniform). forall_no: absent, exists_yes: present")
        print(S.pivot_table(index=["model", "qtype"], columns="N", values="odd_shareN", sort=False).round(2).to_string())
        print("\n== normalized entropy H/log N (late layers)")
        print(S.pivot_table(index=["model", "qtype"], columns="N", values="entropy_norm", sort=False).round(3).to_string())
    if "gamma" in T:
        G = T["gamma"]
        print("\n== γ_attn (absent share ∝ N^-γ) vs γ_behav (Δ ∝ N^-γ, N=2-16)")
        print(G.round(3).to_string(index=False))
        for (band, sd), g in G.groupby(["band", "share_def"]):
            g = g.dropna(subset=["gamma_behav"])
            if len(g) >= 3:
                from scipy.stats import pearsonr, spearmanr
                r, p = pearsonr(g.gamma_attn, g.gamma_behav)
                rs, ps = spearmanr(g.gamma_attn, g.gamma_behav)
                print(f"  [{band}/{sd}] across {len(g)} models: pearson r={r:.2f} (p={p:.2f}), spearman={rs:.2f}")
    for k, title in [("item", "item-level: absent share vs Δ"), ("aggr", "attention-weighted aggregation (R²)"),
                     ("m0", "M0 negated question: absent share×N (paired, same images)"), ("heads", "per-head (late)")]:
        if k in T:
            print(f"\n== {title}\n{T[k].round(3).to_string(index=False)}")


def figures(T, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_multi import COLOR, ORDER, TEXT2, models_in
    os.makedirs(fig_dir, exist_ok=True)
    if "gamma" in T:
        G = T["gamma"][(T["gamma"].band == "late") & (T["gamma"].share_def == "share")]
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        for _, r in G.iterrows():
            ax.errorbar(r.gamma_attn, r.gamma_behav, xerr=[[r.gamma_attn - r.lo], [r.hi - r.gamma_attn]],
                        yerr=[[r.gamma_behav - r.gamma_behav_lo], [r.gamma_behav_hi - r.gamma_behav]],
                        fmt="o", color=COLOR.get(r.model, "k"), label=r.model, lw=1, ms=6)
        lim = [min(G.lo.min(), G.gamma_behav_lo.min()) - .1, max(G.hi.max(), G.gamma_behav_hi.max()) + .1]
        ax.plot(lim, lim, ls=":", color=TEXT2, lw=1)
        ax.set_xlabel("γ_attn  (absent-image attention share ∝ N^-γ)")
        ax.set_ylabel("γ_behav  (Δ ∝ N^-γ)")
        ax.legend(fontsize=7, loc="best")
        fig.savefig(f"{fig_dir}/novelty_attn_gamma.png"); plt.close(fig)
    if "share" in T:
        S = T["share"][(T["share"].band == "late") & (T["share"].share_def == "share")]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
        for ax, qt, title in [(axes[0], "forall_no", "∀ forall_no: absent image"),
                              (axes[1], "exists_yes", "∃ exists_yes: present image")]:
            for m in models_in(S):
                g = S[(S.model == m) & (S.qtype == qt)].sort_values("N")
                ax.errorbar(g.N, g.odd_shareN, yerr=[g.odd_shareN - g.lo, g.hi - g.odd_shareN], marker="o",
                            color=COLOR[m], label=m, lw=1.5, ms=4, capsize=0)
            ax.axhline(1, color=TEXT2, lw=0.8, ls=":")
            ax.set_xscale("log", base=2); Ns = sorted(S.N.unique()); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
            ax.set_title(title, fontsize=9); ax.set_xlabel("number of images N")
        axes[0].set_ylabel("attention share × N  (1 = uniform)")
        axes[1].legend(fontsize=7, loc="upper left")
        fig.savefig(f"{fig_dir}/novelty_attn_share.png"); plt.close(fig)
    if "layers" in T:
        Ly = T["layers"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
        for ax, qt in zip(axes, ["forall_no", "exists_yes"]):
            for m in models_in(Ly):
                g = Ly[(Ly.model == m) & (Ly.qtype == qt) & (Ly.N == 8)].sort_values("rel_depth")
                ax.plot(g.rel_depth, g.odd_shareN, color=COLOR[m], label=m, lw=1.5)
            ax.axhline(1, color=TEXT2, lw=0.8, ls=":")
            ax.set_xlabel("relative layer depth"); ax.set_title(f"{qt} N=8: odd image share × N", fontsize=9)
            if qt == "exists_yes":
                ax.set_yscale("log")
        axes[1].legend(fontsize=7)
        fig.savefig(f"{fig_dir}/novelty_attn_layers.png"); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compact")
    c.add_argument("--tag", required=True)
    c.add_argument("--e2", nargs="+", required=True)
    c.add_argument("--neg", nargs="*", default=None)
    c.add_argument("--out_prefix", default="results/raw/")
    a = sub.add_parser("analyze")
    a.add_argument("--raw", default="results/raw")
    a.add_argument("--tags", nargs="+", default=ORDER)
    a.add_argument("--a1s", default="results/multi/a1s.csv")
    a.add_argument("--out_dir", default="results/novelty")
    a.add_argument("--fig_dir", default="results/figs")
    a.add_argument("--no_fig", action="store_true")
    args = ap.parse_args(argv)
    compact(args) if args.cmd == "compact" else analyze(args)


if __name__ == "__main__":
    main()
