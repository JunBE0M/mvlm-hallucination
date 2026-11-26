"""실험 기록 3 그림 (GPU 불필요, results/novelty/*.csv 만 읽는다).

    python -m src.novelty_figs   → results/figs/novelty_{B_auc,D_cost,E_balacc,F_focus}.png
(실험 A 그림 novelty_attn_*.png 는 `python -m src.attn_theory analyze` 가 만든다.)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

NOV, FIG = "results/novelty", "results/figs"
COL = {"base": "#8c8c8c", "student": "#1f77b4", "M2": "#d62728", "group": "#2ca02c", "FOCUS_min": "#ff7f0e"}


def _ints(d):
    d = d[d.N.astype(str).str.isdigit()].copy()
    d["N"] = d.N.astype(int)
    return d.sort_values("N")


def fig_B():
    d = pd.read_csv(f"{NOV}/B_auc.csv")
    d = _ints(d[(d.question == "forall") & (d.variant == "both_epoch1")])
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    for ax, m in zip(axes, models):
        for meth in ["base", "student", "M2"]:
            x = d[(d.model == m) & (d.method == meth)]
            ax.errorbar(x.N, x.auc, yerr=[x.auc - x.lo, x.hi - x.auc], marker="o", capsize=2, color=COL[meth],
                        label={"student": "student (1 forward)", "M2": "M2 (N+1 forwards)"}.get(meth, meth))
        ax.axvspan(10, 17, color="#eee", zorder=0)
        ax.set_title(m)
        ax.set_xlabel("N (shaded: not seen in training)")
        ax.set_xticks([2, 4, 6, 8, 12, 16])
    axes[0].set_ylabel("∀ AUC (held-out classes)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/novelty_B_auc.png", dpi=150)


def fig_D():
    d = pd.read_csv(f"{NOV}/D_group.csv")
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    for ax, m in zip(axes, models):
        x = d[d.model == m]
        for N, mk in zip([8, 12, 16], ["o", "s", "^"]):
            y = x[x.N.astype(int) == N].sort_values("n_forward")
            ax.plot(y.n_forward, y.auc, color="#bbb", lw=0.8, zorder=0)
            for _, r in y.iterrows():
                ax.scatter(r.n_forward, r.auc, color=COL[r.method], marker=mk, s=40,
                           label=f"{r.method}" if N == 8 else None)
            ax.annotate(f"N={N}", (y.n_forward.max(), y.auc[y.n_forward.idxmax()]), fontsize=7,
                        xytext=(3, 0), textcoords="offset points")
        ax.set_title(m)
        ax.set_xlabel("forwards per question")
    axes[0].set_ylabel("∀ AUC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/novelty_D_cost.png", dpi=150)


def fig_E():
    d = pd.read_csv(f"{NOV}/E_calib.csv")
    d = d[d.N.astype(str) == "all"]
    meths = ["base_thr0", "M2_thr0", "M2_GMM(no label)", "M2_E_perclass(no label)",
             "M2exq_min_thr0(no label)", "M2_LOCO(label)"]
    lab = {"base_thr0": "base, thr 0", "M2_thr0": "M2, thr 0", "M2_GMM(no label)": "M2, GMM",
           "M2_E_perclass(no label)": "M2, ∃-transfer thr", "M2exq_min_thr0(no label)": "∃-question readout + min, thr 0",
           "M2_LOCO(label)": "M2, LOCO (uses labels)"}
    p = d.pivot_table(index="model", columns="method", values="bal_acc")[[c for c in meths if c in d.method.unique()]]
    ax = p.rename(columns=lab).plot.bar(figsize=(10, 3.8), width=0.8)
    ax.set_ylim(0.45, 0.95)
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_ylabel("∀ balanced accuracy (N=2–8)")
    ax.set_xlabel("")
    ax.legend(fontsize=7, ncol=3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG}/novelty_E_balacc.png", dpi=150)


def fig_F():
    d = _ints(pd.read_csv(f"{NOV}/F_focus.csv"))
    models = list(dict.fromkeys(d.model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), sharey=True)
    for ax, m in zip(axes, models):
        x = d[d.model == m]
        for meth in ["base", "M2"]:
            y = x[(x.method == meth) & (x.lam == x.lam.min())]
            ax.plot(y.N, y.auc, marker="o", color=COL[meth], label=meth)
        for lam, ls in zip(sorted(x.lam.unique()), ["--", ":"]):
            y = x[(x.method == "FOCUS_min") & (x.lam == lam)]
            ax.plot(y.N, y.auc, marker="s", ls=ls, color=COL["FOCUS_min"], label=f"FOCUS (min), λ={lam:g}")
        ax.set_title(m)
        ax.set_xlabel("N")
        ax.set_xticks([2, 4, 6, 8])
    axes[0].set_ylabel("∀ AUC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/novelty_F_focus.png", dpi=150)


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_B, fig_D, fig_E, fig_F):
        f()
        print("done", f.__name__)


if __name__ == "__main__":
    main()
