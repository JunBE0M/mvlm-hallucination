"""실험 기록 3 (노벨티 강화) 분석: B 자기증류, C 일반 양화사, D 그룹 테스팅, E 라벨 없는 보정, F FOCUS 기준선.
GPU 불필요 — results/raw 의 csv 만 읽는다 (agg_multi 스타일). 출력: results/novelty/{B,C,D,E,F}_*.csv, 표준출력 요약.

    python -m src.novelty_eval B C D E F [--tags qwen3b qwen7b ...]

공통: AUC = P(score_yes > score_no) (+0.5 동점). bootstrap 은 ∀ 는 pair 단위, 그 외는 문항 단위 (seed 0).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import agg_analysis as A
from .agg_multi import NAMES, attach_E, m1_series

RAW = "results/raw"
OUT = "results/novelty"
ORDER = ["qwen3b", "qwen7b", "qwen3vl8b", "internvl8b", "llavaov7b"]
RNG = np.random.default_rng(0)
NB = 1000


def loads(s):
    return json.loads(s) if isinstance(s, str) else s


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def read_many(pats):
    fs = sorted({f for p in pats for f in glob.glob(p)})
    ds = [pd.read_csv(f) for f in fs]
    return pd.concat(ds, ignore_index=True).drop_duplicates("qid") if ds else None


def auc_of(df, col, pos):
    a, b = df.loc[pos, col].dropna(), df.loc[~pos, col].dropna()
    return A.auc(a, b) if len(a) and len(b) else np.nan


def bal_acc(score, yes, t=0.0):
    score, yes = np.asarray(score, float), np.asarray(yes, bool)
    return ((score[yes] > t).mean() + (score[~yes] <= t).mean()) / 2


def boot_auc(T, cols, pos_col="yes", unit="pair_id", nb=NB, ref=None):
    """unit 단위 bootstrap. 반환 {col: (lo, hi)}, ref 가 있으면 {col: (diff, lo, hi, p)} 도."""
    units = T[unit].unique()
    by = {u: g for u, g in T.groupby(unit)}
    B = []
    for _ in range(nb):
        s = pd.concat([by[u] for u in RNG.choice(units, len(units))])
        y = s[pos_col].to_numpy(bool)
        B.append({c: auc_of(s, c, y) for c in cols})
    B = pd.DataFrame(B)
    ci = {c: (B[c].quantile(.025), B[c].quantile(.975)) for c in cols}
    diff = {}
    if ref is not None:
        y = T[pos_col].to_numpy(bool)
        for c in cols:
            if c == ref:
                continue
            d = B[c] - B[ref]
            diff[c] = (auc_of(T, c, y) - auc_of(T, ref, y), d.quantile(.025), d.quantile(.975),
                       min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return ci, diff


def heldout_classes():
    return json.load(open(f"{OUT}/B_split.json"))["heldout"]


def m2_forall(tag):
    return read_many([f"{RAW}/m2_post_{tag}.csv", f"{RAW}/m2_post_{tag}_N*.csv", f"{RAW}/m2_post_big_{tag}*.csv"])


# ------------------------------------------------------------------ B: 자기증류
def eval_B(tags, out):
    held = set(heldout_classes())
    rows, rows_beta, rows_var, rows_single, rows_fid = [], [], [], [], []
    for tag in tags:
        model = NAMES.get(tag, tag)
        base = read(f"{RAW}/e1_{tag}.csv")
        if base is None:
            continue
        base = base[base.cls.isin(held)]
        bigb = read(f"{RAW}/e1big_{tag}.csv")
        bigb = bigb[bigb.cls.isin(held)] if bigb is not None else None
        m2 = m2_forall(tag)
        m2x = read(f"{RAW}/m2ex_post_{tag}.csv")
        m1b = m1_series(RAW, tag)
        for sfile in sorted(glob.glob(f"{RAW}/B_e1_{tag}_*.csv")):
            S = os.path.basename(sfile)[len("B_e1_"):-4]          # {tag}_{loss}_{epoch}
            variant = S[len(tag) + 1:]
            st = read(sfile)
            stbig = read(f"{RAW}/B_e1big_{S}.csv")
            # ---- ∀ / ∃ on held-out seed-0 (N 2-8) + N=12/16
            for fam, qy, qn in [("forall", "forall_yes", "forall_no"), ("exists", "exists_yes", "exists_no")]:
                parts = [(base, st)] + ([(bigb, stbig)] if bigb is not None and stbig is not None else [])
                for b_df, s_df in parts:
                    b = b_df[b_df.qtype.isin([qy, qn])].set_index("qid")
                    D = b[["pair_id", "cls", "N", "qtype"]].copy()
                    D["unit"] = D.pair_id.fillna(pd.Series(D.index, index=D.index))
                    D["yes"] = D.qtype == qy
                    D["base"] = b.m
                    D["student"] = s_df.set_index("qid").m.reindex(D.index)
                    if fam == "forall" and m2 is not None:
                        D["M2"] = m2.set_index("qid").m_min.reindex(D.index)
                    if fam == "exists" and m2x is not None:
                        D["M2"] = m2x.set_index("qid").m_max.reindex(D.index)
                    cols = [c for c in ("base", "M2", "student") if c in D and D[c].notna().all()]
                    for Nsel in sorted(D.N.unique()) + (["2-8"] if D.N.max() <= 8 else ["12-16"]):
                        DD = D[D.N == Nsel] if not isinstance(Nsel, str) else D
                        ci, diff = boot_auc(DD, cols, unit="unit", ref="base", nb=300 if isinstance(Nsel, str) else 100)
                        _, diff2 = boot_auc(DD, [c for c in cols if c != "base"], unit="unit", ref="M2", nb=300) \
                            if "M2" in cols and isinstance(Nsel, str) else ({}, {})
                        y = DD.yes.to_numpy()
                        for c in cols:
                            r = dict(model=model, variant=variant, question=fam, N=Nsel, method=c, n=len(DD),
                                     auc=auc_of(DD, c, y), lo=ci[c][0], hi=ci[c][1],
                                     bal_acc_thr0=bal_acc(DD[c], y), acc_yes=(DD.loc[y, c] > 0).mean(),
                                     acc_no=(DD.loc[~y, c] <= 0).mean())
                            if c in diff:
                                r.update(d_vs_base=diff[c][0], d_lo=diff[c][1], d_hi=diff[c][2], p_vs_base=diff[c][3])
                            if c in diff2:
                                r.update(d_vs_M2=diff2[c][0], d2_lo=diff2[c][1], d2_hi=diff2[c][2], p_vs_M2=diff2[c][3])
                            rows.append(r)
                    if fam == "forall" and "M2" in D:     # teacher 충실도
                        ok = D.M2.notna()
                        rows_fid.append(dict(model=model, variant=variant, N="2-8" if D.N.max() <= 8 else "12-16",
                                             corr_student_M2=np.corrcoef(D.student[ok], D.M2[ok])[0, 1],
                                             corr_base_M2=np.corrcoef(D.base[ok], D.M2[ok])[0, 1]))
            # ---- β 재피팅 (held-out N 2-8): base, student(base 증거 e), student(자기 증거 e)
            m1s_parts = [read(f"{RAW}/B_m1_{S}.csv"), read(f"{RAW}/B_m1big_{S}.csv")]
            m1s = pd.concat([p for p in m1s_parts if p is not None]).drop_duplicates("key").set_index("key")["m"] \
                if any(p is not None for p in m1s_parts) else None
            for fam in ("forall", "exists"):
                bb = base[base.qtype.str.startswith(fam) & (base.N > 1)].copy()
                ss = st.set_index("qid").m
                specs = [("base", bb.assign(m=bb.m), m1b), ("student|e_base", bb.assign(m=ss.reindex(bb.qid).values), m1b)]
                if m1s is not None:
                    specs.append(("student|e_student", bb.assign(m=ss.reindex(bb.qid).values), m1s))
                for name, d, m1 in specs:
                    d = attach_E(d.copy(), m1)
                    d = d[d.m.notna()]
                    c, best, lo, hi = A.fit_beta(d.m.to_numpy(), list(d.E), d.N.to_numpy(), nboot=100)
                    esd = float(np.concatenate(list(d.E)).std())
                    r2 = dict(zip(A.BETAS, c))
                    rows_beta.append(dict(model=model, variant=variant, question=fam, which=name, n=len(d), beta=best,
                                          beta_lo=lo, beta_hi=hi, e_sd=esd, beta_norm=best * esd, beta_norm_lo=lo * esd,
                                          beta_norm_hi=hi * esd, r2_best=c.max(), r2_mean=r2[0.0], r2_min=r2[10.0],
                                          r2_max=r2[-10.0]))
            # ---- 단일 이미지 질문 (부작용): present vs absent
            if m1s is not None:
                keys = m1s.index.intersection(m1b.index)
                sing = pd.concat([read(f"{RAW}/m1_single_{tag}.csv")] +
                                 [read(f) for f in glob.glob(f"{RAW}/m1_single_big_{tag}*.csv")]).drop_duplicates("key")
                pres = sing.set_index("key").present.reindex(keys).astype(bool).to_numpy()
                for name, sc in [("base", m1b.reindex(keys).to_numpy()), ("student", m1s.reindex(keys).to_numpy())]:
                    rows_single.append(dict(model=model, variant=variant, method=name, n=len(keys),
                                            auc=A.auc(sc[pres], sc[~pres]), bal_acc_thr0=bal_acc(sc, pres),
                                            acc_present=(sc[pres] > 0).mean(), acc_absent=(sc[~pres] <= 0).mean()))
            # ---- 문장 변형 (held-out)
            vs = read(f"{RAW}/B_e1var_{S}.csv")
            vb = read(f"{RAW}/e1var_{tag}.csv")
            if vs is not None and vb is not None:
                qv = {}
                for l in open("data/questions_var.jsonl"):
                    q = json.loads(l)
                    qv[q["qid"]] = q["template"]
                for name, d in [("base", vb[vb.cls.isin(held)]), ("student", vs)]:
                    d = d.assign(template=d.qid.map(qv))
                    for (tp, N), g in d.groupby(["template", "N"]):
                        y = (g.answer == "Yes").to_numpy()
                        rows_var.append(dict(model=model, variant=variant, method=name, template=tp, N=N,
                                             auc=auc_of(g, "m", y), bal_acc_thr0=bal_acc(g.m, y)))
                    for tp, g in d.groupby("template"):
                        y = (g.answer == "Yes").to_numpy()
                        rows_var.append(dict(model=model, variant=variant, method=name, template=tp, N="2-8",
                                             auc=auc_of(g, "m", y), bal_acc_thr0=bal_acc(g.m, y)))
    T = {"B_auc": rows, "B_beta": rows_beta, "B_var": rows_var, "B_single": rows_single, "B_fidelity": rows_fid}
    T = {k: pd.DataFrame(v) for k, v in T.items() if v}
    for k, v in T.items():
        v.to_csv(f"{out}/{k}.csv", index=False)
    if "B_auc" in T:
        X = T["B_auc"][T["B_auc"].N.isin(["2-8", "12-16"])]
        print("\n== B: held-out AUC / bal.acc(thr0)\n",
              X[["model", "variant", "question", "N", "method", "auc", "lo", "hi", "bal_acc_thr0", "d_vs_base",
                 "p_vs_base", "d_vs_M2", "p_vs_M2"]].round(3).to_string(index=False))
    for k in ("B_beta", "B_single", "B_fidelity"):
        if k in T:
            print(f"\n== {k}\n", T[k].round(3).to_string(index=False))
    if "B_var" in T:
        print("\n== B_var (all N)\n", T["B_var"][T["B_var"].N == "2-8"].pivot_table(
            index=["model", "variant", "template"], columns="method", values="auc").round(3).to_string())
    return T


# ------------------------------------------------------------------ C: 일반 양화사
def order_stat(E, r):
    """r 번째로 큰 값 (r=1 -> max)."""
    return np.sort(E)[::-1][r - 1]


SYMB = {
    "all": lambda E, N, k: E.min(),
    "any": lambda E, N, k: E.max(),
    "atleast2": lambda E, N, k: order_stat(E, 2),
    "atleast_half": lambda E, N, k: order_stat(E, N // 2),
    "most": lambda E, N, k: np.median(E),
    "most_exact": lambda E, N, k: order_stat(E, N // 2 + 1),
    "none": lambda E, N, k: -E.max(),
}


def boundary(template, c, N):
    kk = {"all": N, "any": 1, "atleast2": 2, "atleast_half": N // 2, "most": N // 2 + 1, "none": 1}[template]
    return c in (kk - 1, kk)


def eval_C(tags, out):
    rows, rows_beta = [], []
    for tag in tags:
        model = NAMES.get(tag, tag)
        e1 = read(f"{RAW}/C_e1_{tag}.csv")
        m2 = read(f"{RAW}/C_m2_{tag}.csv")
        m1 = read(f"{RAW}/C_m1_{tag}.csv")
        if e1 is None:
            continue
        Q = {}
        for l in open("data/questions_quant.jsonl"):
            q = json.loads(l)
            Q[q["qid"]] = q
        e1["template"] = e1.qid.map(lambda x: Q[x]["template"])
        e1["set_id"] = e1.qid.map(lambda x: Q[x]["set_id"])
        e1["c"] = e1.qid.map(lambda x: Q[x]["n_present"])
        e1["y"] = e1.answer == "Yes"
        evid = {}
        if m2 is not None:
            m2["set_id"] = m2.qid.str.split("__").str[0]
            m2["tpl"] = m2.qid.str.split("__").str[1]
            for tpl, g in m2.groupby("tpl"):
                evid[f"M2[{tpl}]"] = dict(zip(g.set_id, g.m_list.map(lambda s: np.array(json.loads(s)))))
        if m1 is not None:
            s1 = m1.drop_duplicates("key").set_index("key")["m"]
            evid["M1"] = {}
            for q in Q.values():
                if q["template"] == "all":
                    ks = [f"{q['cls'].replace(' ', '_')}|{i}" for i in q["image_ids"]]
                    if all(k in s1.index for k in ks):
                        evid["M1"][q["set_id"]] = s1.loc[ks].to_numpy()
        for (tpl, N), g in e1.groupby(["template", "N"]):
            g = g.copy()
            bnd = np.array([boundary(tpl, c, N) for c in g.c])
            scores = {"base": g.m.to_numpy()}
            ops = ["most", "most_exact"] if tpl == "most" else [tpl]
            for ev, dct in evid.items():
                for op in ops:
                    if all(s in dct for s in g.set_id):
                        scores[f"{ev}·{op}"] = np.array([SYMB[op](dct[s], N, None) for s in g.set_id])
            y = g.y.to_numpy()
            for name, sc in scores.items():
                for subset, sel in [("all", np.ones(len(g), bool)), ("boundary", bnd)]:
                    yy, ss = y[sel], sc[sel]
                    if yy.all() or (~yy).all():
                        continue
                    idx = np.arange(sel.sum())
                    bs = [A.auc(ss[b][yy[b]], ss[b][~yy[b]]) for b in (RNG.choice(idx, len(idx)) for _ in range(300))
                          if yy[b].any() and (~yy[b]).any()]
                    rows.append(dict(model=model, template=tpl, N=N, subset=subset, method=name, n=int(sel.sum()),
                                     auc=A.auc(ss[yy], ss[~yy]), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                                     bal_acc_thr0=bal_acc(ss, yy)))
        # base 가 어떤 연산자로 합치나: softmin β + 순서통계량 r 별 R² (M1 증거)
        if "M1" in evid:
            for tpl, g in e1.groupby("template"):
                g = g[g.set_id.isin(evid["M1"])]
                E = [evid["M1"][s] for s in g.set_id]
                mv, Nv = g.m.to_numpy(), g.N.to_numpy()
                c_, best, lo, hi = A.fit_beta(mv, E, Nv, nboot=100)
                esd = float(np.concatenate(E).std())
                r = dict(model=model, template=tpl, n=len(g), beta=best, beta_lo=lo, beta_hi=hi, beta_norm=best * esd,
                         r2_best_beta=c_.max(), r2_mean=dict(zip(A.BETAS, c_))[0.0])
                # 상대 순위 q = (r-1)/(N-1) 별 R²: q=0 max, q=1 min
                for qrel in (0.0, 0.25, 0.5, 0.75, 1.0):
                    x = np.array([order_stat(e, int(round(qrel * (len(e) - 1))) + 1) for e in E])
                    r[f"r2_rank_q{qrel}"] = A.r2_with_N_intercept(mv, x, Nv)[0]
                rows_beta.append(r)
    T = {k: pd.DataFrame(v) for k, v in {"C_auc": rows, "C_beta": rows_beta}.items() if v}
    for k, v in T.items():
        v.to_csv(f"{out}/{k}.csv", index=False)
    if "C_auc" in T:
        X = T["C_auc"]
        print("\n== C: AUC (all items)\n", X[X.subset == "all"].pivot_table(
            index=["model", "template", "N"], columns="method", values="auc", sort=False).round(3).to_string())
        print("\n== C: AUC (boundary items c∈{k-1,k})\n", X[X.subset == "boundary"].pivot_table(
            index=["model", "template", "N"], columns="method", values="auc", sort=False).round(3).to_string())
    if "C_beta" in T:
        print("\n== C: base aggregation operator (M1 evidence)\n", T["C_beta"].round(3).to_string(index=False))
    return T


# ------------------------------------------------------------------ D: 그룹 테스팅
def eval_D(tags, out):
    rows = []
    for tag in tags:
        model = NAMES.get(tag, tag)
        D = read(f"{RAW}/D_group_{tag}.csv")
        if D is None:
            continue
        m2 = m2_forall(tag)
        D = D.set_index("qid")
        D["yes"] = D.qtype == "forall_yes"
        D["absent"] = D.absent_positions.map(loads).map(lambda a: a[0] if a else -1)
        if m2 is not None:
            mm = m2.set_index("qid")
            D["M2"] = mm.m_min.reindex(D.index)
            D["M2_loc"] = mm.m_list.reindex(D.index).map(lambda s: int(np.argmin(json.loads(s))) if isinstance(s, str) else -1)
        D["group"] = D.score_min
        D["base"] = D.m_full
        for N, g in D.groupby("N"):
            cols = [c for c in ("base", "M2", "group") if c in g and g[c].notna().all()]
            ci, diff = boot_auc(g, cols, unit="pair_id", ref="M2" if "M2" in cols else "base", nb=500)
            no = g[~g.yes]
            for c in cols:
                r = dict(model=model, N=N, method=c, n=len(g), auc=auc_of(g, c, g.yes.to_numpy()), lo=ci[c][0],
                         hi=ci[c][1], n_forward={"base": 1, "M2": N + 1, "group": g.n_forward.mean()}[c],
                         bal_acc_thr0=bal_acc(g[c], g.yes))
                if c == "group":
                    r["loc_acc"] = (no.located == no.absent).mean()
                if c == "M2":
                    r["loc_acc"] = (no.M2_loc == no.absent).mean()
                if c in diff:
                    r.update(d_vs_ref=diff[c][0], d_lo=diff[c][1], d_hi=diff[c][2], p=diff[c][3])
                rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T.to_csv(f"{out}/D_group.csv", index=False)
        print("\n== D: group testing vs M2\n", T.round(3).to_string(index=False))
    return T


# ------------------------------------------------------------------ E: 라벨 없는 임계값 (∃ 기준)
def fit_t_agree(score, ref_yes, cand):
    agree = [((score > t) == ref_yes).mean() for t in cand]
    return cand[int(np.argmax(agree))], max(agree)


def eval_E(tags, out):
    rows, rows_t = [], []
    for tag in tags:
        model = NAMES.get(tag, tag)
        m2 = m2_forall(tag)
        m2x = read(f"{RAW}/m2ex_post_{tag}.csv")
        e1 = read(f"{RAW}/e1_{tag}.csv")
        if m2 is None or m2x is None or e1 is None:
            continue
        m2 = m2[m2.N <= 8].reset_index(drop=True)
        ex = m2x.merge(e1[["qid", "m"]].rename(columns={"m": "m_base"}), on="qid")
        cand = np.quantile(np.concatenate([ex.m_max, m2.m_min]), np.linspace(0.001, 0.999, 999))
        preds = {"M2_thr0": (m2.m_min > 0).to_numpy()}
        pc = np.zeros(len(m2), bool)
        for c, g in ex.groupby("cls"):           # 같은 클래스 exists 문항 (라벨 미사용)
            t, ag = fit_t_agree(g.m_max.to_numpy(), (g.m_base > 0).to_numpy(), cand)
            sel = (m2.cls == c).to_numpy()
            pc[sel] = m2.m_min[sel] > t
            rows_t.append(dict(model=model, cls=c, t=t, agreement=ag, scope="per_class"))
        preds["M2_E_perclass(no label)"] = pc
        t, ag = fit_t_agree(ex.m_max.to_numpy(), (ex.m_base > 0).to_numpy(), cand)
        rows_t.append(dict(model=model, cls="all", t=t, agreement=ag, scope="pooled"))
        preds["M2_E_pooled(no label)"] = (m2.m_min > t).to_numpy()
        pn = np.zeros(len(m2), bool)                # 같은 클래스·같은 N
        for (c, N), g in ex.groupby(["cls", "N"]):
            t, _ = fit_t_agree(g.m_max.to_numpy(), (g.m_base > 0).to_numpy(), cand)
            sel = ((m2.cls == c) & (m2.N == N)).to_numpy()
            pn[sel] = m2.m_min[sel] > t
        preds["M2_E_perclassN(no label)"] = pn
        preds["M2_LOCO(label)"] = A.loco_threshold(m2, m2.m_min.to_numpy())
        sA, _ = A.calib_scores(m2.assign(m_list=m2.m_list.map(lambda s: np.array(json.loads(s)))), "m_list",
                               per_N=False)
        preds["M2_GMM(no label)"] = sA > 0
        base = e1[e1.qtype.str.startswith("forall")].set_index("qid").m.reindex(m2.qid).to_numpy()
        preds["base_thr0"] = base > 0
        # 후속: ∀ 이미지 묶음을 ∃ 문장으로 격리 판독 -> min (m2exq). 같은 per-class ∃ 보정 임계값도 적용
        xq = read(f"{RAW}/m2exq_post_{tag}.csv")
        if xq is not None:
            xs = xq.assign(qid=xq.qid.str.replace("__exq", "", regex=False)).set_index("qid").m_min.reindex(m2.qid)
            if xs.notna().all():
                xs = xs.to_numpy()
                preds["M2exq_min_thr0(no label)"] = xs > 0
                pc2 = np.zeros(len(m2), bool)
                for r in [r for r in rows_t if r["model"] == model and r["scope"] == "per_class"]:
                    sel = (m2.cls == r["cls"]).to_numpy()
                    pc2[sel] = xs[sel] > r["t"]
                preds["M2exq_min_E_perclass(no label)"] = pc2
                preds["M2exq_min_LOCO(label)"] = A.loco_threshold(m2, xs)
                ys = (m2.qtype == "forall_yes").to_numpy()
                rows_t.append(dict(model=model, cls="auc", t=A.auc(xs[ys], xs[~ys]), agreement=np.nan,
                                   scope="M2exq_min_AUC(N2-8)"))
        for name, p in preds.items():
            rows += [{**r, "model": model} for r in A.metr(m2, p, model, name)]
    T = pd.DataFrame(rows)
    if len(T):
        T.to_csv(f"{out}/E_calib.csv", index=False)
        pd.DataFrame(rows_t).to_csv(f"{out}/E_thresholds.csv", index=False)
        print("\n== E: balanced accuracy (∀, N 2-8)\n",
              T.pivot_table(index=["model", "method"], columns="N", values="bal_acc", sort=False).round(3).to_string())
        print("\n== E (all N) acc_yes / acc_no\n",
              T[T.N == "all"].set_index(["model", "method"])[["acc_yes", "acc_no", "bal_acc"]].round(3).to_string())
    return T


# ------------------------------------------------------------------ F: FOCUS 기준선
def eval_F(tags, out):
    rows = []
    for tag in tags:
        model = NAMES.get(tag, tag)
        m2 = m2_forall(tag)
        for f in sorted(glob.glob(f"{RAW}/F_focus_l*_{tag}.csv")):
            F = read(f)
            lam = F.lam.iloc[0]
            F = F.set_index("qid")
            F["yes"] = F.qtype == "forall_yes"
            ml = F.m_list.map(lambda s: np.array(json.loads(s)))
            F["FOCUS_min"] = ml.map(np.min)
            for a in (0.1, 0.4, 1.0):
                F[f"FOCUS_sum_a{a}"] = [np.sum(v - a * mn) for v, mn in zip(ml, F.m_noise)]
            F["base"] = F.m_full
            if m2 is not None:
                F["M2"] = m2.set_index("qid").m_min.reindex(F.index)
            cols = [c for c in ["base", "M2", "FOCUS_min", "FOCUS_sum_a0.1", "FOCUS_sum_a0.4", "FOCUS_sum_a1.0"]
                    if c in F and F[c].notna().all()]
            for Nsel in sorted(F.N.unique()) + ["2-8"]:
                g = F if Nsel == "2-8" else F[F.N == Nsel]
                ci, diff = boot_auc(g, cols, unit="pair_id", ref="M2" if "M2" in cols else "base",
                                    nb=500 if Nsel == "2-8" else 100)
                for c in cols:
                    r = dict(model=model, lam=lam, N=Nsel, method=c, auc=auc_of(g, c, g.yes.to_numpy()),
                             lo=ci[c][0], hi=ci[c][1], bal_acc_thr0=bal_acc(g[c], g.yes))
                    if c in diff:
                        r.update(d_vs_M2=diff[c][0], d_lo=diff[c][1], d_hi=diff[c][2], p_vs_M2=diff[c][3])
                    rows.append(r)
    T = pd.DataFrame(rows)
    if len(T):
        T.to_csv(f"{out}/F_focus.csv", index=False)
        print("\n== F: FOCUS vs M2 (∀ AUC)\n", T.pivot_table(index=["model", "lam", "method"], columns="N",
                                                              values="auc", sort=False).round(3).to_string())
        print(T[T.N == "2-8"].round(3).to_string(index=False))
    return T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", choices=list("BCDEF"))
    ap.add_argument("--tags", nargs="+", default=ORDER)
    ap.add_argument("--out_dir", default=OUT)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    for e in args.exps:
        {"B": eval_B, "C": eval_C, "D": eval_D, "E": eval_E, "F": eval_F}[e](args.tags, args.out_dir)


if __name__ == "__main__":
    main()
