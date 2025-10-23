"""Generate RQ1/RQ2 figures and tables from per-student per-round logs.

This script:
  - Aggregates cohort-level per-round metrics from runs/<dataset>/<run>/<mode>_user_rounds.csv
  - RQ1: Computes per-round paired deltas (adaptive - baseline) for accuracy, dk, de
    - RQ2: Computes per-round retention ratios vs off_open adaptive for accuracy, knowledge, engagement
  - Saves figures to Paper/figures/*.pdf and tables to Paper/tables/*.tex

We intentionally do not read summary.csv and recompute from round-level data.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import comb

# ---------------- aesthetics and labels ----------------
# Use a clean, publication-friendly style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

MODE_LABELS: Dict[str, str] = {
    'adaptive': 'Adaptive',
    'linucb': 'LinUCB',
    'als': 'ALS',
    'fixed': 'Fixed',
}
MODE_COLORS: Dict[str, str] = {
    'adaptive': '#1f77b4',  # blue
    'linucb': '#ff7f0e',    # orange
    'als': '#2ca02c',       # green
    'fixed': '#d62728',     # red
}

DP_FRIENDLY: Dict[str, str] = {
    'standard_eps1_0': 'Standard (ε≈1.0)',
    'strong_eps0_5': 'Strong (ε≈0.5)',
    'locked_eps0_2': 'Locked (ε≈0.2)',
}
DP_COLORS: Dict[str, str] = {
    'standard_eps1_0': '#9467bd',  # purple
    'strong_eps0_5': '#8c564b',    # brown
    'locked_eps0_2': '#17becf',    # teal
}


DATASETS = ["oulad", "ednet"]
RUNS_RQ1 = ["off_open"]
RUNS_RQ2 = ["standard_eps1_0", "strong_eps0_5", "locked_eps0_2"]
MODES = ["adaptive", "linucb", "als", "fixed"]


def ensure_dirs() -> Tuple[Path, Path]:
    r"""Ensure output dirs align with the LaTeX manuscript include paths.

    We write directly into sac_paper/ACM_SAC_2026_Article_Template/Paper/{figures,tables}
    so that \includegraphics and \input statements in results.tex resolve to the
    freshly generated artifacts without manual copying.
    """
    paper_root = Path("sac_paper") / "ACM_SAC_2026_Article_Template" / "Paper"
    figs = paper_root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    tabs = paper_root / "tables"; tabs.mkdir(parents=True, exist_ok=True)
    return figs, tabs


def _normalize_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tel_shown" not in df.columns and "shown" in df.columns:
        df["tel_shown"] = df["shown"]
    if "tel_accepted" not in df.columns and "accepted" in df.columns:
        df["tel_accepted"] = df["accepted"]
    if "tel_correct" not in df.columns and "correct" in df.columns:
        df["tel_correct"] = df["correct"]
    if "tel_accept_rate" not in df.columns and {"tel_shown", "tel_accepted"}.issubset(df.columns):
        df["tel_accept_rate"] = df["tel_accepted"] / df["tel_shown"].clip(lower=1)
    for c in ["pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def aggregate_per_round(df_user: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-student rows into cohort-level per-round metrics."""
    df = _normalize_telemetry(df_user)
    grp = df.groupby("round", as_index=False).agg({
        "tel_shown": "sum",
        "tel_accepted": "sum",
        "tel_correct": "sum",
        "pre_knowledge": "mean",
        "post_knowledge": "mean",
        "pre_engagement": "mean",
        "post_engagement": "mean",
    })
    grp["accept_rate"] = grp["tel_accepted"] / grp["tel_shown"].clip(lower=1)
    grp["accuracy"] = np.where(grp["tel_accepted"] > 0, grp["tel_correct"] / grp["tel_accepted"], 0.0)
    grp["delta_knowledge"] = grp["post_knowledge"] - grp["pre_knowledge"]
    grp["delta_engagement"] = grp["post_engagement"] - grp["pre_engagement"]
    return grp


def load_user_rounds(dataset: str, run: str, mode: str) -> pd.DataFrame:
    """Load per-student per-round rows (raw), normalized to expected telemetry columns."""
    path = Path("runs") / dataset / run / f"{mode}_user_rounds.csv"
    if not path.exists():
        return pd.DataFrame()
    df_user = pd.read_csv(path)
    return _normalize_telemetry(df_user)


def load_per_round(dataset: str, run: str, mode: str) -> pd.DataFrame:
    path = Path("runs") / dataset / run / f"{mode}_user_rounds.csv"
    if not path.exists():
        return pd.DataFrame()
    df_user = pd.read_csv(path)
    df = aggregate_per_round(df_user)
    df["run"] = run
    df["mode"] = mode
    return df


def rq1_deltas_from_rounds(dataset: str) -> pd.DataFrame:
    """Compute per-round deltas (adaptive - baseline) for off_open."""
    run = "off_open"
    base = load_per_round(dataset, run, "adaptive")
    if base.empty:
        return pd.DataFrame()
    base = base.set_index("round")
    parts = []
    for m in ["linucb", "als", "fixed"]:
        other = load_per_round(dataset, run, m)
        if other.empty:
            continue
        other = other.set_index("round")
        inter = base[["accuracy", "delta_knowledge", "delta_engagement"]].join(
            other[["accuracy", "delta_knowledge", "delta_engagement"]],
            lsuffix="_adapt", rsuffix=f"_{m}", how="inner",
        )
        if inter.empty:
            continue
        tmp = inter.reset_index()
        tmp["baseline"] = m
        tmp["dataset"] = dataset
        tmp["d_acc"] = tmp["accuracy_adapt"] - tmp[f"accuracy_{m}"]
        tmp["d_dk"] = tmp["delta_knowledge_adapt"] - tmp[f"delta_knowledge_{m}"]
        tmp["d_de"] = tmp["delta_engagement_adapt"] - tmp[f"delta_engagement_{m}"]
        tmp["run"] = run
        parts.append(tmp[["dataset", "run", "baseline", "round", "d_acc", "d_dk", "d_de"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def rq2_retention_from_rounds(dataset: str) -> pd.DataFrame:
    base = load_per_round(dataset, "off_open", "adaptive").set_index("round")
    if base.empty:
        return pd.DataFrame()
    out_parts = []
    for rn in RUNS_RQ2:
        cur = load_per_round(dataset, rn, "adaptive").set_index("round")
        if cur.empty:
            continue
        inter = cur[["accuracy", "post_knowledge", "post_engagement", "delta_knowledge", "delta_engagement"]].join(
            base[["accuracy", "post_knowledge", "post_engagement", "delta_knowledge", "delta_engagement"]],
            rsuffix="_base", how="inner",
        )
        if inter.empty:
            continue
        inter = inter.reset_index()
        inter["dataset"] = dataset
        inter["run"] = rn

        def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
            return np.where((b != 0) & np.isfinite(b), a / b, np.nan)
        inter["ret_acc"] = safe_ratio(inter["accuracy"], inter["accuracy_base"])
        inter["ret_k"] = safe_ratio(inter["post_knowledge"], inter["post_knowledge_base"])
        inter["ret_e"] = safe_ratio(inter["post_engagement"], inter["post_engagement_base"])
        inter["ret_dk"] = safe_ratio(inter["delta_knowledge"], inter["delta_knowledge_base"])
        inter["ret_de"] = safe_ratio(inter["delta_engagement"], inter["delta_engagement_base"])
        inter["diff_dk"] = inter["delta_knowledge"] - inter["delta_knowledge_base"]
        inter["diff_de"] = inter["delta_engagement"] - inter["delta_engagement_base"]
        out_parts.append(inter[[
            "dataset", "run", "round",
            "ret_acc", "ret_k", "ret_e", "ret_dk", "ret_de",
            "diff_dk", "diff_de",
        ]])
    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()


def _beautify_axes(ax):
    ax.grid(True, axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def plot_rq1_lines(dataset: str, figs_dir: Path) -> None:
    run = "off_open"
    # Plot mean ± SD accuracy over rounds computed across users
    stats = {}
    for m in MODES:
        du = load_user_rounds(dataset, run, m)
        if du.empty:
            continue
        acc_user = np.where(du["tel_accepted"] > 0, du["tel_correct"] / du["tel_accepted"], np.nan)
        acc = pd.DataFrame({"round": du["round"].values, "acc_user": acc_user})
        agg = acc.groupby("round")["acc_user"].agg(["mean", "std"]).reset_index()
        stats[m] = agg

    if not stats:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for m, df in stats.items():
        label = MODE_LABELS.get(m, m)
        color = MODE_COLORS.get(m, None)
        df = df.sort_values("round")
        ax.plot(df["round"], df["mean"], label=label, color=color, linewidth=2.0)
        ax.fill_between(df["round"], df["mean"] - df["std"], df["mean"] + df["std"],
                        color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (mean ± SD)")
    ax.set_title(f"{dataset.upper()}: Accuracy per round by mode")
    _beautify_axes(ax)
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    base = figs_dir / f"rq1_accuracy_lines_{dataset}"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png")
    plt.close(fig)


def plot_rq1_delta_boxes(dataset: str, figs_dir: Path, df_delta: pd.DataFrame) -> None:
    if df_delta.empty:
        return
    order = ["linucb", "als", "fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharex=False)
    metrics = [("d_acc", "Δ Accuracy"), ("d_dk", "Δ Knowledge Δk"), ("d_de", "Δ Engagement Δe")]
    for ax, (col, title) in zip(axes, metrics):
        data = [df_delta[df_delta["baseline"] == b][col].dropna().values for b in order]
        # boxplot with notches and colored boxes
        bp = ax.boxplot(data, tick_labels=[MODE_LABELS.get(b, b) for b in order], showfliers=False,
                        notch=True, patch_artist=True)
        # color boxes to match baseline theme (reusing mode colors)
        for patch, b in zip(bp['boxes'], order):
            patch.set_facecolor(MODE_COLORS.get(b, '#BBBBBB'))
            patch.set_alpha(0.6)
        # overlay mean ± SD as errorbars for each baseline
        means = [float(np.nanmean(vals)) if len(vals) else np.nan for vals in data]
        sds   = [float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0 for vals in data]
        xs = np.arange(1, len(order) + 1)
        ax.errorbar(xs, means, yerr=sds, fmt='o', color='#333333', ecolor='#333333',
                    elinewidth=1, capsize=3, capthick=1, markersize=4, alpha=0.9)
        ax.axhline(0.0, color="#888", lw=1, ls="--")
        ax.set_title(title)
        if col == "d_acc":
            ax.set_ylabel("Adaptive − Baseline")
        _beautify_axes(ax)
    fig.suptitle(f"{dataset.upper()}: RQ1 — Per-round deltas (Adaptive − Baseline)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    base = figs_dir / f"rq1_deltas_box_{dataset}"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png")
    plt.close(fig)


def plot_rq2_retention_lines(dataset: str, figs_dir: Path, df_ret: pd.DataFrame) -> None:
    """Plot retention as mean ± SD across users by joining DP and Non-Private per user."""
    base_u = load_user_rounds(dataset, "off_open", "adaptive")
    if base_u.empty:
        return

    def per_user_retention(dp_run: str) -> pd.DataFrame:
        cur = load_user_rounds(dataset, dp_run, "adaptive")
        if cur.empty:
            return pd.DataFrame()
        joined = pd.merge(cur, base_u, on=["user_id", "round"], suffixes=("", "_base"), how="inner")
        acc_dp = np.where(joined["tel_accepted"] > 0, joined["tel_correct"] / joined["tel_accepted"], np.nan)
        acc_base = np.where(joined["tel_accepted_base"] > 0, joined["tel_correct_base"] / joined["tel_accepted_base"], np.nan)
        def safe_ratio(a, b):
            return np.where((b != 0) & np.isfinite(b), a / b, np.nan)
        joined = joined.assign(
            ret_acc=safe_ratio(acc_dp, acc_base),
            ret_k=safe_ratio(joined["post_knowledge"], joined["post_knowledge_base"]),
            ret_e=safe_ratio(joined["post_engagement"], joined["post_engagement_base"]),
        )
        out = joined.groupby("round")[ ["ret_acc", "ret_k", "ret_e"] ].agg(["mean", "std"]).reset_index()
        out.columns = [c if isinstance(c, str) else f"{c[0]}_{c[1]}" for c in out.columns.values]
        return out

    stats = {rn: per_user_retention(rn) for rn in RUNS_RQ2}
    stats = {k: v for k, v in stats.items() if not v.empty}
    if not stats:
        return

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    metrics = [
        ("ret_acc", "Accuracy retention (mean ± SD)"),
        ("ret_k", "Knowledge retention (mean ± SD)"),
        ("ret_e", "Engagement retention (mean ± SD)"),
    ]
    for ax, (base_col, ylabel) in zip(axes, metrics):
        for rn, df in stats.items():
            color = DP_COLORS.get(rn, None)
            label = DP_FRIENDLY.get(rn, rn)
            # ensure 'round' is a column
            if "round" not in df.columns:
                df = df.reset_index(drop=False)
                if "round" not in df.columns and "index" in df.columns:
                    df = df.rename(columns={"index": "round"})
            d = df.sort_values("round")
            ax.plot(d["round"], d[f"{base_col}_mean"], label=label, color=color, linewidth=2.0)
            ax.fill_between(d["round"], d[f"{base_col}_mean"] - d[f"{base_col}_std"], d[f"{base_col}_mean"] + d[f"{base_col}_std"],
                            color=color, alpha=0.15, linewidth=0)
        ax.axhline(1.0, color="#888", lw=1, ls="--")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.5, 1.2)
        _beautify_axes(ax)
    axes[-1].set_xlabel("Round")
    axes[0].legend(ncol=3, frameon=False)
    fig.suptitle(f"{dataset.upper()}: RQ2 — Retention vs Non-Private Adaptive")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    base = figs_dir / f"rq2_retention_lines_{dataset}"
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png")
    plt.close(fig)


def _sign_test_p_two_sided(d: np.ndarray) -> float:
    """Two-sided sign test p-value for paired differences d.
    Treat zeros as ties and drop them; exact binomial test under p0=0.5.
    """
    d = d[np.isfinite(d)]
    d = d[d != 0]
    n = d.size
    if n == 0:
        return np.nan
    k = int(np.sum(d > 0))
    # two-sided p-value: 2 * min( P[X<=k], P[X>=k] ) for X~Bin(n,0.5)
    # compute tail probabilities exactly via combinations
    def binom_p_leq(x):
        return sum(comb(n, i) for i in range(0, x + 1)) / (2 ** n)
    def binom_p_geq(x):
        return sum(comb(n, i) for i in range(x, n + 1)) / (2 ** n)
    p = 2.0 * min(binom_p_leq(k), binom_p_geq(k))
    return min(p, 1.0)


def _effect_sizes(d: np.ndarray) -> Tuple[float, float]:
    """Return (cohen_d_paired, cliffs_delta) for paired differences d.
    Cliff's delta computed with a fast approximation using ranks (fallback exact for small n).
    """
    d = d[np.isfinite(d)]
    if d.size == 0:
        return (np.nan, np.nan)
    # Cohen's d for paired differences: mean(d)/std(d)
    sd = np.nanstd(d, ddof=1)
    cohen_d = (np.nanmean(d) / sd) if sd > 0 else np.nan
    # Cliff's delta: (num_pos - num_neg) / Npairs; here pairs are differences vs 0
    # Reduce to sign: pos vs neg
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    cliffs = ((pos - neg) / n) if n > 0 else np.nan
    return (cohen_d, cliffs)


def make_rq1_significance_table(tabs_dir: Path) -> None:
    """Compute per-dataset, per-baseline significance for RQ1 deltas using per-user joins.
    Outputs Paper/tables/rq1_deltas_significance.tex with p-values and effect sizes.
    """
    rows = []
    for dataset in DATASETS:
        run = "off_open"
        base = load_user_rounds(dataset, run, "adaptive")
        if base.empty:
            continue
        for m in ["linucb", "als", "fixed"]:
            other = load_user_rounds(dataset, run, m)
            if other.empty:
                continue
            joined = pd.merge(
                base[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]],
                other[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]],
                on=["user_id", "round"], suffixes=("_a", f"_{m}"), how="inner",
            )
            if joined.empty:
                continue
            acc_a = np.where(joined["tel_accepted_a"] > 0, joined["tel_correct_a"] / joined["tel_accepted_a"], np.nan)
            acc_b = np.where(joined[f"tel_accepted_{m}"] > 0, joined[f"tel_correct_{m}"] / joined[f"tel_accepted_{m}"], np.nan)
            d_acc = acc_a - acc_b
            d_dk = (joined["post_knowledge_a"] - joined["pre_knowledge_a"]) - (joined[f"post_knowledge_{m}"] - joined[f"pre_knowledge_{m}"])
            d_de = (joined["post_engagement_a"] - joined["pre_engagement_a"]) - (joined[f"post_engagement_{m}"] - joined[f"pre_engagement_{m}"])
            for name, d in [("ΔAcc", d_acc), ("Δk", d_dk), ("Δe", d_de)]:
                d = d.astype(float)
                p = _sign_test_p_two_sided(d)
                d_cohen, d_cliff = _effect_sizes(d)
                rows.append({
                    "Dataset": dataset.upper(),
                    "Baseline": m,
                    "Metric": name,
                    "Mean": float(np.nanmean(d)),
                    "SD": float(np.nanstd(d, ddof=1)),
                    "Sign p": p,
                    "Cohen d": d_cohen,
                    "Cliff Δ": d_cliff,
                    "Npairs": int(np.isfinite(d).sum()),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = tabs_dir / "rq1_deltas_significance.tex"
    with out.open("w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def make_rq2_significance_table(tabs_dir: Path) -> None:
    """Compute per-dataset, per-DP-run significance for retention vs 1.0 and DP-off deltas using per-user joins.
    Outputs Paper/tables/rq2_retention_significance.tex.
    """
    rows = []
    for dataset in DATASETS:
        base = load_user_rounds(dataset, "off_open", "adaptive")
        if base.empty:
            continue
        for rn in RUNS_RQ2:
            cur = load_user_rounds(dataset, rn, "adaptive")
            if cur.empty:
                continue
            joined = pd.merge(cur, base, on=["user_id", "round"], suffixes=("", "_base"), how="inner")
            if joined.empty:
                continue
            acc_dp = np.where(joined["tel_accepted"] > 0, joined["tel_correct"] / joined["tel_accepted"], np.nan)
            acc_base = np.where(joined["tel_accepted_base"] > 0, joined["tel_correct_base"] / joined["tel_accepted_base"], np.nan)
            def safe_ratio(a, b):
                return np.where((b != 0) & np.isfinite(b), a / b, np.nan)
            ret_acc = safe_ratio(acc_dp, acc_base)
            ret_k = safe_ratio(joined["post_knowledge"], joined["post_knowledge_base"])
            ret_e = safe_ratio(joined["post_engagement"], joined["post_engagement_base"])
            # test deviation from 1.0 via sign test on (ret-1)
            for name, r in [("Acc Ret-1", ret_acc - 1.0), ("K Ret-1", ret_k - 1.0), ("E Ret-1", ret_e - 1.0)]:
                p = _sign_test_p_two_sided(r.astype(float))
                d_cohen, d_cliff = _effect_sizes(r.astype(float))
                rows.append({
                    "Dataset": dataset.upper(),
                    "Run": rn,
                    "Metric": name,
                    "Mean": float(np.nanmean(r)),
                    "SD": float(np.nanstd(r, ddof=1)),
                    "Sign p": p,
                    "Cohen d": d_cohen,
                    "Cliff Δ": d_cliff,
                    "Npairs": int(np.isfinite(r).sum()),
                })
            # DP-off deltas
            dk_dp = joined["post_knowledge"] - joined["pre_knowledge"]
            dk_base = joined["post_knowledge_base"] - joined["pre_knowledge_base"]
            de_dp = joined["post_engagement"] - joined["pre_engagement"]
            de_base = joined["post_engagement_base"] - joined["pre_engagement_base"]
            for name, d in [("Δk (DP-off)", (dk_dp - dk_base)), ("Δe (DP-off)", (de_dp - de_base))]:
                d = d.astype(float)
                p = _sign_test_p_two_sided(d)
                d_cohen, d_cliff = _effect_sizes(d)
                rows.append({
                    "Dataset": dataset.upper(),
                    "Run": rn,
                    "Metric": name,
                    "Mean": float(np.nanmean(d)),
                    "SD": float(np.nanstd(d, ddof=1)),
                    "Sign p": p,
                    "Cohen d": d_cohen,
                    "Cliff Δ": d_cliff,
                    "Npairs": int(np.isfinite(d).sum()),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = tabs_dir / "rq2_retention_significance.tex"
    with out.open("w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def make_rq1_table(df_delta_all: pd.DataFrame, tabs_dir: Path) -> None:
    if df_delta_all.empty:
        return
    rows = []
    for dataset in sorted(df_delta_all["dataset"].unique()):
        for base in ["linucb", "als", "fixed"]:
            d = df_delta_all[(df_delta_all["dataset"] == dataset) & (df_delta_all["baseline"] == base)]
            if d.empty:
                continue
            rows.append({
                "Dataset": dataset.upper(),
                "Baseline": base,
                "Median ΔAcc": np.nanmedian(d["d_acc"]),
                "IQR ΔAcc": np.nanpercentile(d["d_acc"], 75) - np.nanpercentile(d["d_acc"], 25),
                "Mean ΔAcc": np.nanmean(d["d_acc"]),
                "SD ΔAcc": np.nanstd(d["d_acc"], ddof=1),
                "Median Δk": np.nanmedian(d["d_dk"]),
                "IQR Δk": np.nanpercentile(d["d_dk"], 75) - np.nanpercentile(d["d_dk"], 25),
                "Mean Δk": np.nanmean(d["d_dk"]),
                "SD Δk": np.nanstd(d["d_dk"], ddof=1),
                "Median Δe": np.nanmedian(d["d_de"]),
                "IQR Δe": np.nanpercentile(d["d_de"], 75) - np.nanpercentile(d["d_de"], 25),
                "Mean Δe": np.nanmean(d["d_de"]),
                "SD Δe": np.nanstd(d["d_de"], ddof=1),
                "Rounds": int(d.shape[0]),
            })
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq1_deltas_summary.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def stratify_rq1_by_student_method(figs_dir: Path, tabs_dir: Path) -> None:
    """Build RQ1 delta summaries split by student model (student_method)."""
    rows = []
    for dataset in DATASETS:
        run = "off_open"
        base = load_user_rounds(dataset, run, "adaptive")
        if base.empty or "student_method" not in base.columns:
            continue
        for m in ["linucb", "als", "fixed"]:
            other = load_user_rounds(dataset, run, m)
            if other.empty:
                continue
            joined = pd.merge(
                base[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement", "student_method"]],
                other[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]],
                on=["user_id", "round"], suffixes=("_a", f"_{m}"), how="inner",
            )
            if joined.empty:
                continue
            acc_a = np.where(joined["tel_accepted_a"] > 0, joined["tel_correct_a"] / joined["tel_accepted_a"], np.nan)
            acc_b = np.where(joined[f"tel_accepted_{m}"] > 0, joined[f"tel_correct_{m}"] / joined[f"tel_accepted_{m}"], np.nan)
            joined = joined.assign(
                d_acc=acc_a - acc_b,
                d_dk=(joined["post_knowledge_a"] - joined["pre_knowledge_a"]) - (joined[f"post_knowledge_{m}"] - joined[f"pre_knowledge_{m}"]),
                d_de=(joined["post_engagement_a"] - joined["pre_engagement_a"]) - (joined[f"post_engagement_{m}"] - joined[f"pre_engagement_{m}"]),
            )
            for sm, grp in joined.groupby("student_method"):
                rows.append({
                    "Dataset": dataset.upper(),
                    "Baseline": m,
                    "StudentModel": sm,
                    "Median ΔAcc": np.nanmedian(grp["d_acc"]),
                    "Mean ΔAcc": np.nanmean(grp["d_acc"]),
                    "SD ΔAcc": np.nanstd(grp["d_acc"], ddof=1),
                    "Median Δk": np.nanmedian(grp["d_dk"]),
                    "Mean Δk": np.nanmean(grp["d_dk"]),
                    "SD Δk": np.nanstd(grp["d_dk"], ddof=1),
                    "Median Δe": np.nanmedian(grp["d_de"]),
                    "Mean Δe": np.nanmean(grp["d_de"]),
                    "SD Δe": np.nanstd(grp["d_de"], ddof=1),
                    "Pairs": int(grp.shape[0]),
                })
    if not rows:
        return
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq1_deltas_by_student_model.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def stratify_rq1_by_profile(figs_dir: Path, tabs_dir: Path) -> None:
    """Build RQ1 delta summaries split by learner profile (e.g., struggling, advancing)."""
    rows = []
    for dataset in DATASETS:
        run = "off_open"
        base = load_user_rounds(dataset, run, "adaptive")
        if base.empty or "profile" not in base.columns:
            continue
        for m in ["linucb", "als", "fixed"]:
            other = load_user_rounds(dataset, run, m)
            if other.empty:
                continue
            joined = pd.merge(
                base[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement", "profile"]],
                other[["user_id", "round", "tel_accepted", "tel_correct", "pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]],
                on=["user_id", "round"], suffixes=("_a", f"_{m}"), how="inner",
            )
            if joined.empty:
                continue
            acc_a = np.where(joined["tel_accepted_a"] > 0, joined["tel_correct_a"] / joined["tel_accepted_a"], np.nan)
            acc_b = np.where(joined[f"tel_accepted_{m}"] > 0, joined[f"tel_correct_{m}"] / joined[f"tel_accepted_{m}"], np.nan)
            joined = joined.assign(
                d_acc=acc_a - acc_b,
                d_dk=(joined["post_knowledge_a"] - joined["pre_knowledge_a"]) - (joined[f"post_knowledge_{m}"] - joined[f"pre_knowledge_{m}"]),
                d_de=(joined["post_engagement_a"] - joined["pre_engagement_a"]) - (joined[f"post_engagement_{m}"] - joined[f"pre_engagement_{m}"]),
            )
            for prof, grp in joined.groupby("profile"):
                rows.append({
                    "Dataset": dataset.upper(),
                    "Baseline": m,
                    "Profile": prof,
                    "Median ΔAcc": np.nanmedian(grp["d_acc"]),
                    "Mean ΔAcc": np.nanmean(grp["d_acc"]),
                    "SD ΔAcc": np.nanstd(grp["d_acc"], ddof=1),
                    "Median Δk": np.nanmedian(grp["d_dk"]),
                    "Mean Δk": np.nanmean(grp["d_dk"]),
                    "SD Δk": np.nanstd(grp["d_dk"], ddof=1),
                    "Median Δe": np.nanmedian(grp["d_de"]),
                    "Mean Δe": np.nanmean(grp["d_de"]),
                    "SD Δe": np.nanstd(grp["d_de"], ddof=1),
                    "Pairs": int(grp.shape[0]),
                })
    if not rows:
        return
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq1_deltas_by_profile.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def stratify_rq2_by_profile(figs_dir: Path, tabs_dir: Path) -> None:
    """Build RQ2 retention summaries split by learner profile for Adaptive."""
    rows = []
    for dataset in DATASETS:
        base = load_user_rounds(dataset, "off_open", "adaptive")
        if base.empty or "profile" not in base.columns:
            continue
        for rn in RUNS_RQ2:
            cur = load_user_rounds(dataset, rn, "adaptive")
            if cur.empty:
                continue
            joined = pd.merge(cur, base, on=["user_id", "round"], suffixes=("", "_base"), how="inner")
            if joined.empty:
                continue
            acc_dp = np.where(joined["tel_accepted"] > 0, joined["tel_correct"] / joined["tel_accepted"], np.nan)
            acc_base = np.where(joined["tel_accepted_base"] > 0, joined["tel_correct_base"] / joined["tel_accepted_base"], np.nan)
            def safe_ratio(a, b):
                return np.where((b != 0) & np.isfinite(b), a / b, np.nan)
            # derive deltas from pre/post to avoid dependency on prior aggregation
            dk_dp = joined["post_knowledge"] - joined["pre_knowledge"]
            dk_base = joined["post_knowledge_base"] - joined["pre_knowledge_base"]
            de_dp = joined["post_engagement"] - joined["pre_engagement"]
            de_base = joined["post_engagement_base"] - joined["pre_engagement_base"]
            joined = joined.assign(
                ret_acc=safe_ratio(acc_dp, acc_base),
                ret_k=safe_ratio(joined["post_knowledge"], joined["post_knowledge_base"]),
                ret_e=safe_ratio(joined["post_engagement"], joined["post_engagement_base"]),
                diff_dk=dk_dp - dk_base,
                diff_de=de_dp - de_base,
            )
            for prof, grp in joined.groupby("profile"):
                rows.append({
                    "Dataset": dataset.upper(),
                    "Run": rn,
                    "Profile": prof,
                    "Median Acc Ret": np.nanmedian(grp["ret_acc"]),
                    "Mean Acc Ret": np.nanmean(grp["ret_acc"]),
                    "SD Acc Ret": np.nanstd(grp["ret_acc"], ddof=1),
                    "Median K Ret": np.nanmedian(grp["ret_k"]),
                    "Mean K Ret": np.nanmean(grp["ret_k"]),
                    "SD K Ret": np.nanstd(grp["ret_k"], ddof=1),
                    "Median E Ret": np.nanmedian(grp["ret_e"]),
                    "Mean E Ret": np.nanmean(grp["ret_e"]),
                    "SD E Ret": np.nanstd(grp["ret_e"], ddof=1),
                    "Median Δk (DP-off)": np.nanmedian(grp["diff_dk"]),
                    "Mean Δk (DP-off)": np.nanmean(grp["diff_dk"]),
                    "SD Δk (DP-off)": np.nanstd(grp["diff_dk"], ddof=1),
                    "Median Δe (DP-off)": np.nanmedian(grp["diff_de"]),
                    "Mean Δe (DP-off)": np.nanmean(grp["diff_de"]),
                    "SD Δe (DP-off)": np.nanstd(grp["diff_de"], ddof=1),
                    "Pairs": int(grp.shape[0]),
                })
    if not rows:
        return
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq2_retention_by_profile.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def stratify_rq2_by_student_method(figs_dir: Path, tabs_dir: Path) -> None:
    """Build RQ2 retention summaries split by student model for Adaptive."""
    rows = []
    for dataset in DATASETS:
        base = load_user_rounds(dataset, "off_open", "adaptive")
        if base.empty or "student_method" not in base.columns:
            continue
        for rn in RUNS_RQ2:
            cur = load_user_rounds(dataset, rn, "adaptive")
            if cur.empty:
                continue
            joined = pd.merge(cur, base, on=["user_id", "round"], suffixes=("", "_base"), how="inner")
            if joined.empty:
                continue
            acc_dp = np.where(joined["tel_accepted"] > 0, joined["tel_correct"] / joined["tel_accepted"], np.nan)
            acc_base = np.where(joined["tel_accepted_base"] > 0, joined["tel_correct_base"] / joined["tel_accepted_base"], np.nan)
            def safe_ratio(a, b):
                return np.where((b != 0) & np.isfinite(b), a / b, np.nan)
            dk_dp = joined["post_knowledge"] - joined["pre_knowledge"]
            dk_base = joined["post_knowledge_base"] - joined["pre_knowledge_base"]
            de_dp = joined["post_engagement"] - joined["pre_engagement"]
            de_base = joined["post_engagement_base"] - joined["pre_engagement_base"]
            joined = joined.assign(
                ret_acc=safe_ratio(acc_dp, acc_base),
                ret_k=safe_ratio(joined["post_knowledge"], joined["post_knowledge_base"]),
                ret_e=safe_ratio(joined["post_engagement"], joined["post_engagement_base"]),
                diff_dk=dk_dp - dk_base,
                diff_de=de_dp - de_base,
            )
            for sm, grp in joined.groupby("student_method"):
                rows.append({
                    "Dataset": dataset.upper(),
                    "Run": rn,
                    "StudentModel": sm,
                    "Median Acc Ret": np.nanmedian(grp["ret_acc"]),
                    "Mean Acc Ret": np.nanmean(grp["ret_acc"]),
                    "SD Acc Ret": np.nanstd(grp["ret_acc"], ddof=1),
                    "Median K Ret": np.nanmedian(grp["ret_k"]),
                    "Mean K Ret": np.nanmean(grp["ret_k"]),
                    "SD K Ret": np.nanstd(grp["ret_k"], ddof=1),
                    "Median E Ret": np.nanmedian(grp["ret_e"]),
                    "Mean E Ret": np.nanmean(grp["ret_e"]),
                    "SD E Ret": np.nanstd(grp["ret_e"], ddof=1),
                    "Median Δk (DP-off)": np.nanmedian(grp["diff_dk"]),
                    "Mean Δk (DP-off)": np.nanmean(grp["diff_dk"]),
                    "SD Δk (DP-off)": np.nanstd(grp["diff_dk"], ddof=1),
                    "Median Δe (DP-off)": np.nanmedian(grp["diff_de"]),
                    "Mean Δe (DP-off)": np.nanmean(grp["diff_de"]),
                    "SD Δe (DP-off)": np.nanstd(grp["diff_de"], ddof=1),
                    "Pairs": int(grp.shape[0]),
                })
    if not rows:
        return
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq2_retention_by_student_model.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def make_rq2_table(df_ret_all: pd.DataFrame, tabs_dir: Path) -> None:
    if df_ret_all.empty:
        return
    rows = []
    for dataset in sorted(df_ret_all["dataset"].unique()):
        for rn in RUNS_RQ2:
            d = df_ret_all[(df_ret_all["dataset"] == dataset) & (df_ret_all["run"] == rn)]
            if d.empty:
                continue
            rows.append({
                "Dataset": dataset.upper(),
                "Run": rn,
                "Median Acc Ret": np.nanmedian(d["ret_acc"]),
                "Mean Acc Ret": np.nanmean(d["ret_acc"]),
                "SD Acc Ret": np.nanstd(d["ret_acc"], ddof=1),
                "Median K Ret": np.nanmedian(d["ret_k"]),
                "Mean K Ret": np.nanmean(d["ret_k"]),
                "SD K Ret": np.nanstd(d["ret_k"], ddof=1),
                "Median E Ret": np.nanmedian(d["ret_e"]),
                "Mean E Ret": np.nanmean(d["ret_e"]),
                "SD E Ret": np.nanstd(d["ret_e"], ddof=1),
                "Median Δk (DP-off)": np.nanmedian(d["diff_dk"]),
                "Mean Δk (DP-off)": np.nanmean(d["diff_dk"]),
                "SD Δk (DP-off)": np.nanstd(d["diff_dk"], ddof=1),
                "Median Δe (DP-off)": np.nanmedian(d["diff_de"]),
                "Mean Δe (DP-off)": np.nanmean(d["diff_de"]),
                "SD Δe (DP-off)": np.nanstd(d["diff_de"], ddof=1),
                "Rounds": int(d.shape[0]),
            })
    df_table = pd.DataFrame(rows)
    out = tabs_dir / "rq2_retention_summary.tex"
    with out.open("w") as f:
        f.write(df_table.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    figs_dir, tabs_dir = ensure_dirs()

    # Compute RQ1 deltas and RQ2 retention from scratch
    df_delta_all = []
    df_ret_all = []
    for ds in DATASETS:
        # RQ1 figures
        plot_rq1_lines(ds, figs_dir)
        df_delta = rq1_deltas_from_rounds(ds)
        if not df_delta.empty:
            plot_rq1_delta_boxes(ds, figs_dir, df_delta)
            df_delta_all.append(df_delta)

        # RQ2 figures
        df_ret = rq2_retention_from_rounds(ds)
        if not df_ret.empty:
            plot_rq2_retention_lines(ds, figs_dir, df_ret)
            df_ret_all.append(df_ret)

    # Tables
    df_delta_all = pd.concat(df_delta_all, ignore_index=True) if df_delta_all else pd.DataFrame()
    df_ret_all = pd.concat(df_ret_all, ignore_index=True) if df_ret_all else pd.DataFrame()
    make_rq1_table(df_delta_all, tabs_dir)
    make_rq2_table(df_ret_all, tabs_dir)

    # Stratified appendices
    stratify_rq1_by_student_method(figs_dir, tabs_dir)
    stratify_rq1_by_profile(figs_dir, tabs_dir)
    stratify_rq2_by_profile(figs_dir, tabs_dir)
    stratify_rq2_by_student_method(figs_dir, tabs_dir)

    # Significance appendices
    make_rq1_significance_table(tabs_dir)
    make_rq2_significance_table(tabs_dir)

    # Also write out the raw computed frames for traceability
    if not df_delta_all.empty:
        df_delta_all.to_csv(figs_dir.parent / "_rq1_deltas_from_rounds.csv", index=False)
    if not df_ret_all.empty:
        df_ret_all.to_csv(figs_dir.parent / "_rq2_retention_from_rounds.csv", index=False)


if __name__ == "__main__":
    main()
