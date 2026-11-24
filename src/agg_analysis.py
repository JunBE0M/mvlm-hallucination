"""집계 방식 분석 (GPU 불필요, 기존 results/raw csv만 사용).

    python -m src.agg_analysis --out_dir results/agg

A1  Δ 법칙: forall 쌍 Δ = m(forall_yes) - m(forall_no) 를 Δ ∝ N^-γ 로 피팅 (bootstrap CI).
    + 문항 단위 검증: M1 단일 이미지 점수 e_i 로 "평균 집계라면 Δ ≈ k (e_replaced - e_absent) / N" 를 예측해
      실제 Δ 와 비교 (1/N 없는 모형, min 모형과 R² 비교).
A2  집계 온도 β: m_base ≈ a_N + b * softmin_β(e_1..e_N),  softmin_β(e) = -(1/β) log mean exp(-β e)
    (β>0 -> min 쪽, β=0 -> 평균, β<0 -> max 쪽). forall / exists 질문에서 따로 피팅 -> 양화사에 따라 집계를 바꾸는지.
    + 위치별 가중치: m_base ≈ a + Σ_p w_p e_p.
A3  M2 라벨 없는 임계값: 이미지별 점수 m_i 에 2성분 가우시안 혼합(비지도) -> P(present | m_i),
    forall: Yes iff Π_i P(present|m_i) > 0.5.  비교: 임계값 0, leave-one-class-out(라벨 사용).
A4  pair 단위 bootstrap 으로 방법 간 AUC 차이 CI.
A5  exists 과제: base vs M1_max.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.special import logsumexp

TAGS = {"qwen3b": "qwen2.5-vl-3b", "qwen7b": "qwen2.5-vl-7b"}
RNG = np.random.default_rng(0)
NB = 2000


# ---------------------------------------------------------------- utils
def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(=)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def softmin(E, beta):
    """E: [n, N]. beta>0 -> min 쪽, beta<0 -> max 쪽."""
    if abs(beta) < 1e-9:
        return E.mean(1)
    return -(logsumexp(-beta * E, axis=1) - np.log(E.shape[1])) / beta


def r2_with_N_intercept(y, x, N):
    """y ≈ a_N + b x (N별 절편, 공통 기울기). R² 반환."""
    Ns = np.unique(N)
    X = np.column_stack([(N == n).astype(float) for n in Ns] + [x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return 1 - res.var() / y.var(), coef[-1]


def loads(s):
    return json.loads(s) if isinstance(s, str) else s


def load(raw, tag):
    e1 = pd.read_csv(f"{raw}/e1_{tag}.csv")
    m1 = pd.read_csv(f"{raw}/m1_single_{tag}.csv").set_index("key")["m"]
    for c in ("image_ids", "present_positions", "absent_positions"):
        e1[c] = e1[c].map(loads)
    e1["E"] = [np.array([m1[f"{c.replace(' ', '_')}|{i}"] for i in ids])
               for c, ids in zip(e1.cls, e1.image_ids)]
    m2 = pd.read_csv(f"{raw}/m2_post_{tag}.csv")
    m2["m_list"] = m2.m_list.map(loads).map(np.array)
    out = dict(e1=e1, m1=m1, m2=m2)
    for name, path in [("m0", f"{raw}/m0neg_{tag}.csv"), ("m5", f"{raw}/m5_k5_{tag}/focus_a16_post_image.csv")]:
        if os.path.exists(path):
            out[name] = pd.read_csv(path)
    return out


# ---------------------------------------------------------------- A1
def a1(D, model):
    e1 = D["e1"]
    y = e1[e1.qtype == "forall_yes"].set_index("pair_id")
    n = e1[e1.qtype == "forall_no"].set_index("pair_id")
    P = y[["N", "m", "E", "image_ids"]].join(n[["m", "E", "image_ids", "absent_positions"]], rsuffix="_no")
    P["delta"] = P.m - P.m_no
    rows = []
    Ns = sorted(P.N.unique())
    groups = {N: P[P.N == N].delta.to_numpy() for N in Ns}
    for N in Ns:
        d = groups[N]
        bs = [RNG.choice(d, len(d)).mean() for _ in range(NB)]
        rows.append(dict(model=model, N=N, n_pairs=len(d), delta=d.mean(), lo=np.percentile(bs, 2.5),
                         hi=np.percentile(bs, 97.5), deltaxN=d.mean() * N))
    # γ: log mean Δ = c - γ log N
    def gamma(gr):
        x = np.log(Ns)
        yv = np.log([max(gr[N].mean(), 1e-6) for N in Ns])
        return -np.polyfit(x, yv, 1)[0]
    g = gamma(groups)
    gb = [gamma({N: RNG.choice(groups[N], len(groups[N])) for N in Ns}) for _ in range(NB)]
    summ = dict(model=model, gamma=g, gamma_lo=np.percentile(gb, 2.5), gamma_hi=np.percentile(gb, 97.5))

    # 문항 단위: e_replaced - e_absent
    p = P.absent_positions.map(lambda a: a[0]).to_numpy()
    e_rep = np.array([E[k] for E, k in zip(P.E, p)])
    e_abs = np.array([E[k] for E, k in zip(P.E_no, p)])
    de = e_rep - e_abs
    N = P.N.to_numpy().astype(float)
    dmin = np.array([a.min() - b.min() for a, b in zip(P.E, P.E_no)])
    dl = P.delta.to_numpy()

    def r2(x):
        X = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(X, dl, rcond=None)
        return 1 - (dl - X @ c).var() / dl.var()
    summ.update(r2_mean_model=r2(de / N), r2_no_dilution=r2(de), r2_min_model=r2(dmin),
                corr_mean_model=np.corrcoef(de / N, dl)[0, 1])
    return pd.DataFrame(rows), summ


# ---------------------------------------------------------------- A2
BETAS = np.round(np.concatenate([-np.logspace(1, -2, 13), [0], np.logspace(-2, 1, 13)]), 4)


def fit_beta(m, E_list, N, nboot=300):
    """같은 N끼리 묶어 softmin 계산 -> N별 절편 + 공통 기울기 회귀 R² 가 최대인 β."""
    idx_by_N = {n: np.where(N == n)[0] for n in np.unique(N)}

    def curve(sel):
        out = []
        for b in BETAS:
            x = np.empty(len(sel))
            for n, ix in idx_by_N.items():
                mask = np.isin(sel, ix)
                if mask.any():
                    x[mask] = softmin(np.stack([E_list[i] for i in sel[mask]]), b)
            out.append(r2_with_N_intercept(m[sel], x, N[sel])[0])
        return np.array(out)
    allsel = np.arange(len(m))
    c = curve(allsel)
    best = BETAS[c.argmax()]
    bb = []
    for _ in range(nboot):
        s = RNG.choice(allsel, len(allsel))
        bb.append(BETAS[curve(s).argmax()])
    return c, best, np.percentile(bb, 2.5), np.percentile(bb, 97.5)


def a2(D, model):
    e1, m2 = D["e1"], D["m2"]
    rows, curves = [], []
    specs = [("forall", "M1", e1[e1.qtype.str.startswith("forall")], "E", "m"),
             ("exists", "M1", e1[e1.qtype.str.startswith("exists")], "E", "m"),
             ("forall", "M2in-context", m2, "m_list", "m_full")]
    for fam, ev, d, ecol, mcol in specs:
        d = d[d.N > 1]
        c, best, lo, hi = fit_beta(d[mcol].to_numpy(), list(d[ecol]), d.N.to_numpy())
        r2 = dict(zip(BETAS, c))
        rows.append(dict(model=model, question=fam, evidence=ev, beta=best, beta_lo=lo, beta_hi=hi,
                         r2_best=c.max(), r2_mean=r2[0.0], r2_min=r2[10.0], r2_max=r2[-10.0]))
        curves += [dict(model=model, question=fam, evidence=ev, beta=b, r2=v) for b, v in zip(BETAS, c)]
    # 위치별 가중치 (M1 증거, N별)
    W = []
    for fam in ("forall", "exists"):
        d = e1[e1.qtype.str.startswith(fam)]
        for N, g in d.groupby("N"):
            X = np.column_stack([np.ones(len(g))] + [np.stack(g.E)[:, p] for p in range(N)])
            c, *_ = np.linalg.lstsq(X, g.m.to_numpy(), rcond=None)
            w = c[1:]
            W += [dict(model=model, question=fam, N=N, pos=p, w=w[p], w_norm=w[p] / w.sum()) for p in range(N)]
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(W)


# ---------------------------------------------------------------- A3
def gmm2(x, iters=500):
    """1D 2성분 가우시안 혼합 EM. (mu_lo, sd_lo, mu_hi, sd_hi, pi_lo) 반환."""
    x = np.asarray(x, float)
    q = np.quantile(x, [0.1, 0.9])
    mu = np.array(q); sd = np.array([x.std(), x.std()]) / 2; pi = np.array([0.5, 0.5])
    for _ in range(iters):
        lp = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(pi)
        r = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        nk = r.sum(0)
        pi = nk / len(x)
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-3
    o = np.argsort(mu)
    return mu[o], sd[o], pi[o]


def post_present(x, g):
    mu, sd, pi = g
    lp = -0.5 * ((np.asarray(x)[..., None] - mu) / sd) ** 2 - np.log(sd) + np.log(pi)
    return np.exp(lp[..., 1] - logsumexp(lp, axis=-1))


def calib_scores(d, ecol, per_N=True):
    """라벨 없이: 혼합 피팅 -> log Π P(present|m_i) - log 0.5. >0 이면 Yes."""
    s = np.empty(len(d))
    params = {}
    keys = sorted(d.N.unique()) if per_N else ["all"]
    for k in keys:
        sel = (d.N == k).to_numpy() if per_N else np.ones(len(d), bool)
        allm = np.concatenate(list(d.loc[sel, ecol]))
        g = gmm2(allm)
        params[k] = g
        s[sel] = [np.log(np.clip(post_present(E, g), 1e-12, 1)).sum() - np.log(0.5) for E in d.loc[sel, ecol]]
    return s, params


def loco_threshold(d, score, per_N=True):
    yes = (d.qtype == "forall_yes").to_numpy()
    pred = np.zeros(len(d), bool)
    cand = np.unique(score)
    for c in d.cls.unique():
        te = (d.cls == c).to_numpy()
        for N in (d.N.unique() if per_N else [None]):
            sel_te = te & ((d.N == N).to_numpy() if N is not None else True)
            tr = ~te & ((d.N == N).to_numpy() if N is not None else True)
            ba = [((score[tr & yes] > t).mean() + (score[tr & ~yes] <= t).mean()) / 2 for t in cand[::7]]
            t = cand[::7][int(np.argmax(ba))]
            pred[sel_te] = score[sel_te] > t
    return pred


def metr(d, pred_yes, model, method):
    yes = (d.qtype == "forall_yes").to_numpy()
    rows = []
    for N in list(sorted(d.N.unique())) + ["all"]:
        sel = np.ones(len(d), bool) if N == "all" else (d.N == N).to_numpy()
        h = pred_yes[sel & yes].mean(); cr = 1 - pred_yes[sel & ~yes].mean()
        rows.append(dict(model=model, method=method, N=N, acc_yes=h, acc_no=cr, bal_acc=(h + cr) / 2))
    return rows


def a3(D, model):
    m2 = D["m2"].copy()
    rows = []
    rows += metr(m2, (m2.m_min > 0).to_numpy(), model, "M2_thr0")
    rows += metr(m2, loco_threshold(m2, m2.m_min.to_numpy()), model, "M2_LOCO(label)")
    sN, pN = calib_scores(m2, "m_list", per_N=True)
    sA, pA = calib_scores(m2, "m_list", per_N=False)
    rows += metr(m2, sN > 0, model, "M2_GMM_perN(no label)")
    rows += metr(m2, sA > 0, model, "M2_GMM_pooled(no label)")
    # 대조: 혼합 경계 하나로 min 임계 (곱 대신)  -> EVT 효과 분리
    th = []
    for E in m2.m_list:
        th.append(np.all(post_present(E, pA["all"]) > 0.5))
    rows += metr(m2, np.array(th), model, "M2_GMM_pooled_minrule(no label)")
    # M1 단일 이미지 증거에도 같은 보정
    e1 = D["e1"]; f = e1[e1.qtype.str.startswith("forall")].reset_index(drop=True)
    rows += metr(f, (f.E.map(np.min) > 0).to_numpy(), model, "M1_thr0")
    s1, _ = calib_scores(f, "E", per_N=False)
    rows += metr(f, s1 > 0, model, "M1_GMM_pooled(no label)")
    rows += metr(f, (f.m > 0).to_numpy(), model, "base_thr0")
    m2["score_gmm"] = sA
    gm = pd.DataFrame([dict(model=model, N=k, mu_lo=g[0][0], mu_hi=g[0][1], sd_lo=g[1][0], sd_hi=g[1][1],
                            pi_lo=g[2][0]) for k, g in {**pN, **pA}.items()])
    return pd.DataFrame(rows), gm, m2


# ---------------------------------------------------------------- A4
def a4(D, m2cal, model):
    e1 = D["e1"]
    S = {}
    f = e1[e1.qtype.str.startswith("forall")]
    S["base"] = f.set_index("qid").m
    S["M1_min"] = f.set_index("qid").E.map(np.min)
    S["M2_post_min"] = D["m2"].set_index("qid").m_min
    S["M2_post_GMM"] = m2cal.set_index("qid").score_gmm
    if "m0" in D:
        S["M0_neg"] = -D["m0"].set_index("qid").m
    if "m5" in D:
        S["M5_k5_a16"] = D["m5"].set_index("qid").m
    meta = f.set_index("qid")[["pair_id", "N", "qtype"]]
    T = meta.join(pd.DataFrame(S), how="inner").dropna()
    pairs = T.pair_id.unique()
    comps = [("M2_post_min", "base"), ("M2_post_min", "M0_neg"), ("M2_post_min", "M1_min"),
             ("M5_k5_a16", "base"), ("M5_k5_a16", "M0_neg"), ("M0_neg", "base"), ("M2_post_GMM", "M2_post_min")]
    rows = []
    for Nsel in ["all", 8]:
        TT = T if Nsel == "all" else T[T.N == Nsel]
        pr = TT.pair_id.unique()
        by = {p: g for p, g in TT.groupby("pair_id")}

        def aucs(df):
            yes = df.qtype == "forall_yes"
            return {k: auc(df.loc[yes, k], df.loc[~yes, k]) for k in S if k in df}
        base_auc = aucs(TT)
        boots = []
        for _ in range(NB):
            samp = RNG.choice(pr, len(pr))
            boots.append(aucs(pd.concat([by[p] for p in samp])))
        B = pd.DataFrame(boots)
        for a, b in comps:
            if a not in B or b not in B:
                continue
            diff = B[a] - B[b]
            rows.append(dict(model=model, N=Nsel, A=a, B=b, auc_A=base_auc[a], auc_B=base_auc[b],
                             diff=base_auc[a] - base_auc[b], lo=diff.quantile(.025), hi=diff.quantile(.975),
                             p_two_sided=min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- A5
def a5(D, model):
    e1 = D["e1"]
    x = e1[e1.qtype.str.startswith("exists")].reset_index(drop=True)
    yes = (x.qtype == "exists_yes").to_numpy()
    sc = {"base": x.m.to_numpy(), "M1_max": x.E.map(np.max).to_numpy(), "M1_mean": x.E.map(np.mean).to_numpy()}
    rows = []
    for N in list(sorted(x.N.unique())) + ["all"]:
        sel = np.ones(len(x), bool) if N == "all" else (x.N == N).to_numpy()
        for k, s in sc.items():
            rows.append(dict(model=model, N=N, method=k, auc=auc(s[sel & yes], s[sel & ~yes]),
                             acc_yes=(s[sel & yes] > 0).mean(), acc_no=(s[sel & ~yes] <= 0).mean()))
    # 위치 편향: exists_yes N=8 present 위치별 정답률
    pos = []
    g = x[(x.N == 8) & yes]
    pp = g.present_positions.map(lambda a: a[0])
    for k in ("base", "M1_max"):
        s = sc[k][((x.N == 8) & yes).to_numpy()]
        for p in range(8):
            pos.append(dict(model=model, method=k, pos=p, acc=(s[(pp == p).to_numpy()] > 0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(pos)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out_dir", default="results/agg")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args(argv)
    global NB
    NB = args.nboot
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    acc = {k: [] for k in ["a1", "a1s", "a2", "a2c", "a2w", "a3", "a3g", "a4", "a5", "a5p"]}
    for tag, model in TAGS.items():
        D = load(args.raw, tag)
        r, s = a1(D, model); acc["a1"].append(r); acc["a1s"].append(pd.DataFrame([s]))
        r, c, w = a2(D, model); acc["a2"].append(r); acc["a2c"].append(c); acc["a2w"].append(w)
        r, g, m2cal = a3(D, model); acc["a3"].append(r); acc["a3g"].append(g)
        acc["a4"].append(a4(D, m2cal, model))
        r, p = a5(D, model); acc["a5"].append(r); acc["a5p"].append(p)
    T = {k: pd.concat(v, ignore_index=True) for k, v in acc.items()}
    for k, v in T.items():
        v.to_csv(f"{args.out_dir}/{k}.csv", index=False)
    print("== A1 Δ per N\n", T["a1"].round(3).to_string(index=False))
    print("\n== A1 summary (γ: Δ ∝ N^-γ;  item-level R²)\n", T["a1s"].round(3).to_string(index=False))
    print("\n== A2 aggregation temperature β (>0 min-like, <0 max-like, 0 mean)\n", T["a2"].round(3).to_string(index=False))
    wp = T["a2w"][T["a2w"].N == 8].pivot_table(index=["model", "question"], columns="pos", values="w_norm")
    print("\n== A2 position weights N=8 (normalized)\n", wp.round(3).to_string())
    piv = T["a3"].pivot_table(index=["model", "method"], columns="N", values="bal_acc", sort=False)
    print("\n== A3 calibration: balanced acc\n", piv.round(3).to_string())
    piv = T["a3"][T["a3"].N == "all"].set_index(["model", "method"])[["acc_yes", "acc_no", "bal_acc"]]
    print("\n== A3 (all N) acc_yes / acc_no\n", piv.round(3).to_string())
    print("\n== A3 GMM params\n", T["a3g"].round(2).to_string(index=False))
    print("\n== A4 paired bootstrap AUC differences\n", T["a4"].round(3).to_string(index=False))
    piv = T["a5"].pivot_table(index=["model", "method"], columns="N", values="auc", sort=False)
    print("\n== A5 exists AUC\n", piv.round(3).to_string())
    piv = T["a5p"].pivot_table(index=["model", "method"], columns="pos", values="acc")
    print("\n== A5 exists_yes N=8 acc by present position\n", piv.round(2).to_string())


if __name__ == "__main__":
    main()
