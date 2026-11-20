"""E1/E2 결과 분석: 지표 계산 + 그래프.

    python -m src.analyze --e1 results/raw/e1_*.csv [--e2 results/raw/e2_*.csv] --out_dir results

출력 (results/ 에는 작은 csv만):
    summary_accuracy.csv  N·유형별 정확도, Yes 비율, False-Yes rate(정답 No 유형), 평균 m
    summary_delta.csv     forall 쌍 Δ = m(yes 묶음) − m(no 묶음): 평균, 95% bootstrap CI, Δ×N
    summary_sdt.csv       forall_yes=signal, forall_no=noise: hit, FA, d′, criterion c (log-linear 보정)
    summary_position.csv  홀로 다른 이미지 위치별 정확도 (forall_no: absent 위치, exists_yes: present 위치)
    summary_e2_layers.csv E2: 층별 absent/present 이미지가 받는 attention (헤드 평균), text/system 몫
    figs/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

LIST_COLS = ["image_ids", "present_positions", "absent_positions", "area_ratios", "n_image_tokens"]
QTYPE_ORDER = ["forall_yes", "forall_no", "exists_yes", "exists_no"]

# 색: 참조 팔레트(dataviz skill) 범주형 슬롯 고정 순서. present=slot1(blue), absent=slot2(orange)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ROLE_COLOR = {"present": SERIES[0], "absent": SERIES[1]}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]          # blue <-> gray <-> red


# ------------------------------------------------------------------ E1 loading
def load_results(paths) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        for c in LIST_COLS:
            if c in df:
                df[c] = df[c].apply(lambda s: json.loads(s) if isinstance(s, str) and s else [])
        df["source"] = str(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["N"] = df["N"].astype(int)
    df["yes"] = (df["pred"] == "Yes").astype(int)
    return df


# ------------------------------------------------------------------ E1 metrics
def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "N", "qtype"])
    out = g.agg(n=("qid", "size"), accuracy=("correct", "mean"), yes_rate=("yes", "mean"),
                mean_m=("m", "mean"), answer=("answer", "first")).reset_index()
    out["false_yes_rate"] = np.where(out["answer"] == "No", out["yes_rate"], np.nan)
    out["qtype"] = pd.Categorical(out["qtype"], QTYPE_ORDER, ordered=True)
    return out.drop(columns="answer").sort_values(["model", "qtype", "N"]).reset_index(drop=True)


def pair_deltas(df: pd.DataFrame) -> pd.DataFrame:
    f = df[df["qtype"].isin(["forall_yes", "forall_no"])]
    keys = ["model", "pair_id"]
    y = f[f["qtype"] == "forall_yes"].set_index(keys)
    n = f[f["qtype"] == "forall_no"].set_index(keys)
    common = y.index.intersection(n.index)
    out = pd.DataFrame({
        "cls": y.loc[common, "cls"], "N": y.loc[common, "N"],
        "m_yes": y.loc[common, "m"], "m_no": n.loc[common, "m"],
        "absent_pos": n.loc[common, "absent_positions"].apply(lambda a: a[0] if a else -1),
    }).reset_index()
    out["delta"] = out["m_yes"] - out["m_no"]
    return out


def bootstrap_ci(x, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def delta_summary(pairs: pd.DataFrame, n_boot=2000, seed=0) -> pd.DataFrame:
    rows = []
    for (model, N), g in pairs.groupby(["model", "N"]):
        lo, hi = bootstrap_ci(g["delta"], n_boot, seed=seed)
        mu = g["delta"].mean()
        rows.append(dict(model=model, N=N, n_pairs=len(g), delta_mean=mu, delta_ci_lo=lo, delta_ci_hi=hi,
                         delta_median=g["delta"].median(), delta_x_N=mu * N,
                         delta_x_N_ci_lo=lo * N, delta_x_N_ci_hi=hi * N,
                         frac_delta_pos=(g["delta"] > 0).mean()))
    return pd.DataFrame(rows)


def sdt(hits: int, n_signal: int, fas: int, n_noise: int) -> dict:
    """log-linear 보정 (Hautus 1995): (k + 0.5) / (n + 1). d′ = z(H) − z(F), c = −(z(H) + z(F)) / 2.
    c < 0 이면 Yes 쪽으로 치우친(liberal) 기준."""
    H = (hits + 0.5) / (n_signal + 1)
    F = (fas + 0.5) / (n_noise + 1)
    z = NormalDist().inv_cdf
    return dict(hit_rate=H, fa_rate=F, d_prime=z(H) - z(F), criterion=-(z(H) + z(F)) / 2,
                hit_rate_raw=hits / n_signal if n_signal else np.nan,
                fa_rate_raw=fas / n_noise if n_noise else np.nan)


def sdt_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, N), g in df.groupby(["model", "N"]):
        s, nz = g[g["qtype"] == "forall_yes"], g[g["qtype"] == "forall_no"]
        if len(s) == 0 or len(nz) == 0:
            continue
        rows.append(dict(model=model, N=N, n_signal=len(s), n_noise=len(nz),
                         **sdt(int(s["yes"].sum()), len(s), int(nz["yes"].sum()), len(nz))))
    return pd.DataFrame(rows)


def position_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["qtype"].isin(["forall_no", "exists_yes"])].copy()
    d["odd_pos"] = [(a if q == "forall_no" else p)[0]
                    for q, a, p in zip(d["qtype"], d["absent_positions"], d["present_positions"])]
    d["odd_pos_rel"] = d["odd_pos"] / (d["N"] - 1)
    return (d.groupby(["model", "qtype", "N", "odd_pos"])
             .agg(n=("qid", "size"), accuracy=("correct", "mean"), mean_m=("m", "mean"),
                  odd_pos_rel=("odd_pos_rel", "first"))
             .reset_index())


# ------------------------------------------------------------------ E2
def load_e2(meta: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for qid, path in zip(meta["qid"], meta["npz"]):
        with np.load(path) as z:
            out[qid] = {k: z[k] for k in z.files}
    return out


def e2_long(meta: pd.DataFrame, arrays: dict[str, dict]) -> pd.DataFrame:
    """질문×층×role(absent/present) 행. 값은 헤드 평균, role 내 이미지 평균."""
    rows = []
    for r in meta.itertuples(index=False):
        a = arrays[r.qid]
        s, m = a["img_sum"].mean(1), a["img_mean"].mean(1)          # [L, N]
        L = s.shape[0]
        for role, pos in (("absent", r.absent_positions), ("present", r.present_positions)):
            if not pos:
                continue
            for layer in range(L):
                rows.append((r.model, r.qid, r.N, r.qtype, layer, role,
                             float(s[layer, pos].mean()), float(m[layer, pos].mean()),
                             float(a["text_sum"][layer].mean()), float(a["sys_sum"][layer].mean())))
    return pd.DataFrame(rows, columns=["model", "qid", "N", "qtype", "layer", "role",
                                       "attn_sum", "attn_mean", "text_sum", "sys_sum"])


def e2_layer_summary(long: pd.DataFrame) -> pd.DataFrame:
    return (long.groupby(["model", "qtype", "N", "layer", "role"])
                .agg(n=("qid", "size"), attn_sum=("attn_sum", "mean"), attn_sum_sd=("attn_sum", "std"),
                     attn_mean=("attn_mean", "mean"), attn_mean_sd=("attn_mean", "std"),
                     text_sum=("text_sum", "mean"), sys_sum=("sys_sum", "mean"))
                .reset_index())


def e2_head_diff(meta: pd.DataFrame, arrays, metric="img_mean", qtype="forall_no") -> dict:
    """(model, N) -> [L, H] 평균 (absent − present 평균). absent·present 가 모두 있는 문항만."""
    out = {}
    m = meta[meta["qtype"] == qtype]
    for (model, N), g in m.groupby(["model", "N"]):
        diffs = []
        for r in g.itertuples(index=False):
            if not r.absent_positions or not r.present_positions:
                continue
            x = arrays[r.qid][metric]                                    # [L, H, N]
            diffs.append(x[:, :, r.absent_positions].mean(-1) - x[:, :, r.present_positions].mean(-1))
        if diffs:
            out[(model, int(N))] = np.mean(diffs, 0)
    return out


# ------------------------------------------------------------------ plotting
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6, "figure.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def plot_accuracy(acc: pd.DataFrame, figs: Path) -> list[Path]:
    plt = _plt()
    paths = []
    for model, g in acc.groupby("model"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for i, qt in enumerate(QTYPE_ORDER):
            d = g[g["qtype"] == qt].sort_values("N")
            if len(d):
                axes[0].plot(d["N"], d["accuracy"], "-o", color=SERIES[i], label=qt)
        for qt in ("forall_no", "exists_no"):
            d = g[g["qtype"] == qt].sort_values("N")
            if len(d):
                axes[1].plot(d["N"], d["false_yes_rate"], "-o", color=SERIES[QTYPE_ORDER.index(qt)], label=qt)
        axes[0].set(title="Accuracy", xlabel="N images", ylabel="accuracy", ylim=(-0.02, 1.02))
        axes[1].set(title="False-Yes rate (answer = No)", xlabel="N images", ylabel="P(pred = Yes)",
                    ylim=(-0.02, 1.02))
        for ax in axes:
            ax.set_xticks(sorted(g["N"].unique()))
            ax.legend(fontsize=8)
        fig.suptitle(model, x=0.01, ha="left", color=INK_MUTED, fontsize=9)
        paths.append(_save(fig, figs / f"accuracy_{_safe(model)}.png"))
    return paths


def plot_delta_sdt(delta: pd.DataFrame, sdt_df: pd.DataFrame, figs: Path) -> list[Path]:
    plt = _plt()
    paths = []
    for model in sorted(set(delta.get("model", [])) | set(sdt_df.get("model", []))):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3))
        d = delta[delta["model"] == model].sort_values("N") if len(delta) else delta
        s = sdt_df[sdt_df["model"] == model].sort_values("N") if len(sdt_df) else sdt_df
        if len(d):
            for ax, col, title in ((axes[0], "delta", "Δ = m(yes) − m(no)"), (axes[1], "delta_x_N", "Δ × N")):
                y = d["delta_mean"] if col == "delta" else d["delta_x_N"]
                lo = d["delta_ci_lo"] if col == "delta" else d["delta_x_N_ci_lo"]
                hi = d["delta_ci_hi"] if col == "delta" else d["delta_x_N_ci_hi"]
                ax.fill_between(d["N"], lo, hi, color=SERIES[0], alpha=0.15, linewidth=0)
                ax.plot(d["N"], y, "-o", color=SERIES[0])
                ax.axhline(0, color=INK_MUTED, linewidth=0.8)
                ax.set(title=title, xlabel="N images")
        if len(s):
            axes[2].plot(s["N"], s["d_prime"], "-o", color=SERIES[0])
            axes[2].set(title="d′ (forall_yes vs forall_no)", xlabel="N images")
            axes[3].plot(s["N"], s["criterion"], "-o", color=SERIES[0])
            axes[3].axhline(0, color=INK_MUTED, linewidth=0.8)
            axes[3].set(title="criterion c  (<0 = Yes bias)", xlabel="N images")
        for ax in axes:
            if len(d) or len(s):
                ax.set_xticks(sorted(set(d["N"] if len(d) else []) | set(s["N"] if len(s) else [])))
        fig.suptitle(f"{model}  (band = 95% bootstrap CI)", x=0.01, ha="left", color=INK_MUTED, fontsize=9)
        paths.append(_save(fig, figs / f"delta_sdt_{_safe(model)}.png"))
    return paths


def plot_position(pos: pd.DataFrame, figs: Path) -> list[Path]:
    plt = _plt()
    paths = []
    for model, g in pos.groupby("model"):
        Ns = sorted(g["N"].unique())
        fig, axes = plt.subplots(1, len(Ns), figsize=(3 * len(Ns), 2.8), sharey=True, squeeze=False)
        for ax, N in zip(axes[0], Ns):
            for i, qt in enumerate(("forall_no", "exists_yes")):
                d = g[(g["N"] == N) & (g["qtype"] == qt)].sort_values("odd_pos")
                if len(d):
                    ax.plot(d["odd_pos"], d["accuracy"], "-o", color=SERIES[QTYPE_ORDER.index(qt)],
                            label=qt if N == Ns[0] else None)
            ax.set(title=f"N = {N}", xlabel="odd image position", ylim=(-0.02, 1.02))
            ax.set_xticks(range(N))
        axes[0][0].set_ylabel("accuracy")
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{model}  position of the odd image (absent for forall_no, present for exists_yes)",
                     x=0.01, ha="left", color=INK_MUTED, fontsize=9)
        paths.append(_save(fig, figs / f"position_{_safe(model)}.png"))
    return paths


def plot_e2_layers(summ: pd.DataFrame, figs: Path, qtype="forall_no") -> list[Path]:
    plt = _plt()
    paths = []
    s = summ[summ["qtype"] == qtype]
    for model, g in s.groupby("model"):
        Ns = sorted(g["N"].unique())
        for metric, label in (("attn_sum", "attention mass per image (sum)"),
                              ("attn_mean", "attention per image token (mean)")):
            fig, axes = plt.subplots(1, len(Ns), figsize=(3.2 * len(Ns), 3), sharey=True, squeeze=False)
            for ax, N in zip(axes[0], Ns):
                for role in ("present", "absent"):
                    d = g[(g["N"] == N) & (g["role"] == role)].sort_values("layer")
                    ax.plot(d["layer"], d[metric], color=ROLE_COLOR[role],
                            label=f"{role} image" if N == Ns[0] else None)
                ax.set(title=f"N = {N}", xlabel="layer")
            axes[0][0].set_ylabel(label)
            axes[0][0].legend(fontsize=8)
            fig.suptitle(f"{model} · {qtype} · last-token attention (head mean; present = mean over present images)",
                         x=0.01, ha="left", color=INK_MUTED, fontsize=9)
            paths.append(_save(fig, figs / f"e2_layers_{metric}_{qtype}_{_safe(model)}.png"))
    return paths


def plot_e2_head_diff(diffs: dict, figs: Path, metric="img_mean") -> list[Path]:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    paths = []
    for model in sorted({k[0] for k in diffs}):
        Ns = sorted(N for (m, N) in diffs if m == model)
        mats = [diffs[(model, N)] for N in Ns]
        v = max(float(np.abs(x).max()) for x in mats) or 1.0
        fig, axes = plt.subplots(1, len(Ns), figsize=(2.8 * len(Ns) + 0.8, 4), squeeze=False)
        for ax, N, x in zip(axes[0], Ns, mats):
            im = ax.imshow(x, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
            ax.set(title=f"N = {N}", xlabel="head")
            ax.grid(False)
        axes[0][0].set_ylabel("layer")
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
        cb.set_label("absent − present  (red: absent gets more)")
        fig.suptitle(f"{model} · forall_no · {metric}", x=0.01, ha="left", color=INK_MUTED, fontsize=9)
        paths.append(_save(fig, figs / f"e2_headdiff_{metric}_{_safe(model)}.png"))
    return paths


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e1", nargs="*", default=[], help="E1 csv (여러 모델 가능)")
    ap.add_argument("--e2", nargs="*", default=[], help="E2 csv (attn_probe 출력)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args(argv)
    out, figs = Path(args.out_dir), Path(args.out_dir) / "figs"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    # E2 csv 도 logit 을 포함하므로 E1 이 없으면 E2 로 행동 지표 계산
    beh_paths = args.e1 or args.e2
    if beh_paths:
        df = load_results(beh_paths)
        acc, pairs = accuracy_table(df), pair_deltas(df)
        delta, sdt_df, pos = delta_summary(pairs, args.n_boot), sdt_table(df), position_table(df)
        for name, t in (("summary_accuracy", acc), ("summary_delta", delta),
                        ("summary_sdt", sdt_df), ("summary_position", pos)):
            t.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
            written.append(out / f"{name}.csv")
        written += plot_accuracy(acc, figs) + plot_delta_sdt(delta, sdt_df, figs) + plot_position(pos, figs)
        with pd.option_context("display.width", 160, "display.max_columns", 20):
            print(acc.to_string(index=False, float_format="%.3f"))
            print(delta[["model", "N", "n_pairs", "delta_mean", "delta_ci_lo", "delta_ci_hi", "delta_x_N"]]
                  .to_string(index=False, float_format="%.3f"))
            print(sdt_df[["model", "N", "hit_rate", "fa_rate", "d_prime", "criterion"]]
                  .to_string(index=False, float_format="%.3f"))

    if args.e2:
        meta = load_results(args.e2)
        arrays = load_e2(meta)
        summ = e2_layer_summary(e2_long(meta, arrays))
        summ.to_csv(out / "summary_e2_layers.csv", index=False, float_format="%.6g")
        written.append(out / "summary_e2_layers.csv")
        for qt in ("forall_no", "exists_yes"):
            if (summ["qtype"] == qt).any():
                written += plot_e2_layers(summ, figs, qt)
        for metric in ("img_mean", "img_sum"):
            written += plot_e2_head_diff(e2_head_diff(meta, arrays, metric), figs, metric)

    print("\nwrote:\n  " + "\n  ".join(map(str, written)))
    return written


if __name__ == "__main__":
    main()
