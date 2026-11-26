"""논문용 그림 (results/multi/*.csv -> results/figs/multi_*.png).

    python -m src.plot_multi --in_dir results/multi --out_dir results/figs

색: 모델 = 고정 순서 범주형 팔레트(검증된 기본 팔레트 slot 1-5). 모든 그림 단일 y축.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "Qwen3-VL-8B", "InternVL3-8B", "LLaVA-OV-7B"]
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
COLOR = dict(zip(ORDER, SERIES))
QCOLOR = {"forall": SERIES[0], "exists": SERIES[1], "none": SERIES[6]}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def models_in(df):
    return [m for m in ORDER if m in set(df.model)]


def fig_auc(B, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, col, title in zip(axes, ["auc_forall", "auc_exists"], ["∀ “in all N images?”", "∃ “in any of the N images?”"]):
        for m in models_in(B):
            g = B[B.model == m].sort_values("N")
            ax.plot(g.N, g[col], marker="o", color=COLOR[m], label=m)
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(B.N.unique())); ax.set_xticklabels(sorted(B.N.unique()))
        ax.set_title(title, color=TEXT, fontsize=9); ax.set_xlabel("number of images N")
        ax.axhline(0.5, color=TEXT2, lw=0.8, ls=":")
    axes[0].set_ylabel("AUC (yes vs no items)")
    axes[1].legend(loc="lower left", fontsize=7.5)
    fig.savefig(out); plt.close(fig)


def fig_delta(A1, S, out):
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for m in models_in(A1):
        g = A1[A1.model == m].sort_values("N")
        g = g[g.delta > 0]
        ax.errorbar(g.N, g.delta, yerr=[g.delta - g.lo.clip(lower=1e-3), g.hi - g.delta], fmt="o", color=COLOR[m],
                    ms=4, lw=1, capsize=0)
        s = S[(S.model == m)].sort_values("Ns").iloc[-1]
        x = np.array([g.N.min(), g.N.max()], float)
        c = np.exp(np.mean(np.log(g.delta) + s.gamma * np.log(g.N)))
        ax.plot(x, c * x ** -s.gamma, color=COLOR[m], lw=1.5, label=f"{m}  γ={s.gamma:.2f}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    Ns = sorted(A1.N.unique()); ax.set_xticks(Ns); ax.set_xticklabels(Ns)
    ax.set_xlabel("N"); ax.set_ylabel("Δ = m(all present) − m(one absent)")
    ax.legend(fontsize=7, loc="lower left")
    fig.savefig(out); plt.close(fig)


def fig_beta(A2, out):
    ms = models_in(A2)
    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(ms) + 1.0))
    off = {"forall": -0.18, "exists": 0.0, "none": 0.18}
    for i, m in enumerate(ms):
        for q, dy in off.items():
            g = A2[(A2.model == m) & (A2.question == q)]
            if g.empty:
                continue
            base = g[g.template.isin(["base", "none"])]
            other = g[~g.template.isin(["base", "none", "small"])]
            ax.scatter(other.beta_norm, [i + dy] * len(other), s=14, color=QCOLOR[q], alpha=0.45, lw=0)
            for r in base.itertuples():
                ax.errorbar(r.beta_norm, i + dy, xerr=[[r.beta_norm - r.beta_norm_lo], [r.beta_norm_hi - r.beta_norm]],
                            fmt="o", color=QCOLOR[q], ms=6, lw=1.2, mec="white", mew=1)
    for q, lab in [("forall", "∀ (all)"), ("exists", "∃ (any)"), ("none", "¬∃ (none)")]:
        ax.scatter([], [], color=QCOLOR[q], label=lab, s=30)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms); ax.invert_yaxis()
    ax.set_xlabel("aggregation temperature β·sd(e)\n← max-like   ·   0 = mean   ·   min-like →")
    ax.legend(fontsize=7.5, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    fig.savefig(out); plt.close(fig)


def fig_lens(L, A2, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    sd = A2[A2.template == "base"].set_index(["model", "question"]).e_sd
    for ax, q in zip(axes, ["forall", "exists"]):
        for m in models_in(L):
            g = L[(L.model == m) & (L.question == q)].sort_values("rel_depth")
            if g.empty or (m, q) not in sd.index:
                continue
            # 판독이 아직 우연 수준(AUC<0.6)인 층의 β 는 의미가 없으므로 가림
            y = (g.beta * sd[(m, q)]).where(g.auc >= 0.6)
            ax.plot(g.rel_depth, y, color=COLOR[m], lw=1.5, marker="o", ms=2.5, label=m)
        ax.axhline(0, color=TEXT2, lw=0.8)
        ax.set_ylim(-12, 3)
        ax.set_title("∀ questions" if q == "forall" else "∃ questions", color=TEXT, fontsize=9)
        ax.set_xlabel("relative layer depth (logit lens)")
    axes[0].set_ylabel("best-fit β·sd(e)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.savefig(out); plt.close(fig)


def fig_methods(M, out):
    M = M[M.N.astype(str) == "all"]
    ms = models_in(M)
    meths = [k for k in ["base", "M0_neg", "M1_mean", "M2_post_min", "M1_min"] if k in set(M.method)]
    lab = {"base": "base", "M0_neg": "M0 negated q.", "M1_mean": "M1 per-image, mean",
           "M2_post_min": "M2 isolated readout, min", "M1_min": "M1 per-image, min (oracle-ish)"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.8 / len(meths)
    for j, k in enumerate(meths):
        for i, m in enumerate(ms):
            r = M[(M.model == m) & (M.method == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.bar(i + (j - (len(meths) - 1) / 2) * w, r.auc - 0.5, w * 0.9, bottom=0.5, color=SERIES[j],
                   label=lab[k] if i == 0 else None)
            ax.plot([i + (j - (len(meths) - 1) / 2) * w] * 2, [r.lo, r.hi], color=TEXT, lw=0.8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylim(0.5, 1.01)
    ax.set_ylabel("∀ AUC (all N)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(out); plt.close(fig)


def fig_e3(E, out):
    E = E[(E.N.astype(str) == "all") & (E.cond != "none_a1")]
    ms = models_in(E)
    conds = [("absent_a4", "absent image ×4"), ("absent_a0", "absent image ×0 (knockout)"),
             ("present_a4", "a present image ×4")]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    w = 0.8 / len(conds)
    for j, (c, lab) in enumerate(conds):
        for i, m in enumerate(ms):
            r = E[(E.model == m) & (E.cond == c)]
            if r.empty:
                continue
            ax.bar(i + (j - 1) * w, r.dm_over_sd.iloc[0], w * 0.9, color=SERIES[[0, 1, 2][j]], label=lab if i == 0 else None)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms)
    ax.set_ylabel("Δm / sd(m)  (+ = toward Yes)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(out); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/multi")
    ap.add_argument("--out_dir", default="results/figs")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    r = lambda k: pd.read_csv(f"{a.in_dir}/{k}.csv") if os.path.exists(f"{a.in_dir}/{k}.csv") else None  # noqa: E731
    B, A1, S, A2, M, L, E = (r(k) for k in ["behaviour", "a1", "a1s", "a2", "methods", "lens", "e3"])
    if B is not None:
        fig_auc(B, f"{a.out_dir}/multi_auc_vs_N.png")
    if A1 is not None:
        fig_delta(A1, S, f"{a.out_dir}/multi_delta_law.png")
    if A2 is not None:
        fig_beta(A2, f"{a.out_dir}/multi_beta_asymmetry.png")
    if L is not None and A2 is not None:
        fig_lens(L, A2, f"{a.out_dir}/multi_lens_beta.png")
    if M is not None:
        fig_methods(M, f"{a.out_dir}/multi_methods_auc.png")
    if E is not None:
        fig_e3(E, f"{a.out_dir}/multi_e3.png")
    print("figures written to", a.out_dir)


if __name__ == "__main__":
    main()
