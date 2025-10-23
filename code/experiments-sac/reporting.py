# src/orchid_ranker/experiments/reporting.py
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- flatteners (match the JSONL schema emitted by your orchestrator) ----------

def flatten_user_records_full(records: List[dict]) -> pd.DataFrame:
    """
    Extract rich per-student, per-round rows from orchestrator memory logger.
    Requires the orchestrator to log 'type'='user_round' with the fields added earlier.
    """
    rows = []
    for rec in records:
        if rec.get("type") != "user_round":
            continue
        r = {
            "round": rec.get("round"),
            "mode": rec.get("mode"),
            "user_id": rec.get("user_id"),
            "student_method": (rec.get("student_method") or "unknown").lower(),
            "profile": rec.get("profile"),
            # knobs
            "knob_top_k": rec.get("knobs", {}).get("top_k"),
            "knob_zpd_margin": rec.get("knobs", {}).get("zpd_margin"),
            "knob_mmr_lambda": rec.get("knobs", {}).get("mmr_lambda"),
            "knob_novelty": rec.get("knobs", {}).get("novelty_bonus"),
            # telemetry
            "telemetry_shown": rec.get("telemetry", {}).get("shown"),
            "telemetry_accepted": rec.get("telemetry", {}).get("accepted"),
            "telemetry_correct": rec.get("telemetry", {}).get("correct"),
            "telemetry_accept_rate": rec.get("telemetry", {}).get("accept_rate"),
            "telemetry_accept_at4": rec.get("telemetry", {}).get("accept_at4"),
            "telemetry_dwell_s": rec.get("telemetry", {}).get("dwell_s"),
            "telemetry_latency_s": rec.get("telemetry", {}).get("latency_s"),
            "telemetry_novelty_rate": rec.get("telemetry", {}).get("novelty_rate"),
            "telemetry_serendipity": rec.get("telemetry", {}).get("serendipity"),
            # state snapshots
            "state_pre_knowledge": rec.get("state_estimator", {}).get("pre", {}).get("knowledge"),
            "state_pre_engagement": rec.get("state_estimator", {}).get("pre", {}).get("engagement"),
            "state_pre_trust": rec.get("state_estimator", {}).get("pre", {}).get("trust"),
            "state_pre_fatigue": rec.get("state_estimator", {}).get("pre", {}).get("fatigue"),
            "state_post_knowledge": rec.get("state_estimator", {}).get("post", {}).get("knowledge"),
            "state_post_engagement": rec.get("state_estimator", {}).get("post", {}).get("engagement"),
            "state_post_trust": rec.get("state_estimator", {}).get("post", {}).get("trust"),
            "state_post_fatigue": rec.get("state_estimator", {}).get("post", {}).get("fatigue"),
        }
        rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["round", "student_method", "user_id"]).reset_index(drop=True)
    return df


def flatten_round_records(records: List[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        if rec.get("type") != "round_summary":
            continue
        m = rec.get("metrics", {})
        rows.append({
            "round": rec.get("round"),
            "mode": rec.get("mode"),
            "shown": m.get("shown"),
            "accepted": m.get("accepted"),
            "correct": m.get("correct"),
            "accept_rate": m.get("accept_rate"),
            "accuracy": m.get("accuracy"),
            "accept_at4": m.get("accept_at4"),
            "novelty_rate": m.get("novelty_rate"),
            "serendipity": m.get("serendipity"),
            "mean_knowledge": m.get("mean_knowledge"),
            "mean_fatigue": m.get("mean_fatigue"),
            "mean_engagement": m.get("mean_engagement"),
            "mean_trust": m.get("mean_trust"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("round").reset_index(drop=True)
    return df


# ---------- (1) Save a CSV per student & learning model ----------

def save_per_student_csvs(
    df_user: pd.DataFrame,
    out_dir: str | Path,
    experiment_name: str,
) -> List[Path]:
    """
    Writes one CSV per (student_method, user_id) with all their rounds for this experiment.
    Returns list of written paths.
    """
    out_dir = Path(out_dir) / "per_student_csv" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if df_user.empty:
        return paths

    for (method, uid), sub in df_user.groupby(["student_method", "user_id"]):
        fname = f"{experiment_name}__method-{method}__user-{uid}.csv"
        path = out_dir / fname
        # Keep a stable column order
        cols_order = [
            "round", "mode", "student_method", "user_id", "profile",
            "knob_top_k", "knob_zpd_margin", "knob_mmr_lambda", "knob_novelty",
            "telemetry_shown", "telemetry_accepted", "telemetry_correct",
            "telemetry_accept_rate", "telemetry_accept_at4",
            "telemetry_novelty_rate", "telemetry_serendipity",
            "telemetry_dwell_s", "telemetry_latency_s",
            "state_pre_knowledge", "state_pre_engagement", "state_pre_trust", "state_pre_fatigue",
            "state_post_knowledge", "state_post_engagement", "state_post_trust", "state_post_fatigue",
        ]
        keep = [c for c in cols_order if c in sub.columns] + [c for c in sub.columns if c not in cols_order]
        sub[keep].sort_values("round").to_csv(path, index=False)
        paths.append(path)
    return paths


# ---------- small plotting helpers ----------

def _lineplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path):
    """
    Minimal Matplotlib line plot: one line per 'hue' category.
    """
    plt.figure(figsize=(7.5, 4.5))
    for key, grp in df.groupby(hue):
        grp = grp.sort_values(x)
        plt.plot(grp[x].values, grp[y].values, marker="o", label=str(key))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- (2) Adaptive: one plot per metric (avg of students for each learning model) ----------

def plot_adaptive_learning_model_averages(
    df_user: pd.DataFrame,
    out_dir: str | Path,
    experiment_name: str,
    metrics: Iterable[str] | None = None,
) -> List[Path]:
    """
    For ADAPTIVE runs: averages over students WITHIN each learning model per round,
    then plots one figure per metric with a line per learning model.
    """
    out_dir = Path(out_dir) / "plots_adaptive" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if df_user.empty:
        return []

    # default metrics: post-states and key telemetry
    if metrics is None:
        metrics = [
            "state_post_knowledge",
            "state_post_engagement",
            "state_post_trust",
            "state_post_fatigue",
            "telemetry_accept_rate",
            "telemetry_accept_at4",
            "telemetry_novelty_rate",
            "telemetry_serendipity",
        ]

    paths = []
    # compute mean over students per (student_method, round)
    g = df_user.groupby(["student_method", "round"])
    means = g[metrics].mean().reset_index()

    for m in metrics:
        if m not in means.columns:
            continue
        out_path = out_dir / f"{experiment_name}__adaptive_mean_by_model__{m}.png"
        _lineplot(means, x="round", y=m, hue="student_method",
                  title=f"{experiment_name} | Adaptive | {m} (mean across students)", out_path=out_path)
        paths.append(out_path)
    return paths


# ---------- (3) Compare modes (adaptive vs baselines): averaged over students & learning models ----------

def plot_mode_comparison(
    per_mode_user_frames: Dict[str, pd.DataFrame],
    out_dir: str | Path,
    experiment_name: str,
    metrics: Iterable[str] | None = None,
) -> List[Path]:
    """
    per_mode_user_frames: dict like {"adaptive": df_user_adap, "random": df_user_rand, ...}
    For each metric, aggregate to (mode, round) mean across all users & models, then plot a line per mode.
    """
    out_dir = Path(out_dir) / "plots_mode_compare" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if metrics is None:
        metrics = [
            "state_post_knowledge",
            "state_post_engagement",
            "state_post_trust",
            "state_post_fatigue",
            "telemetry_accept_rate",
            "telemetry_accept_at4",
            "telemetry_novelty_rate",
            "telemetry_serendipity",
        ]

    # build a unified frame: (mode, round, metric columns)
    frames = []
    for mode, dfu in per_mode_user_frames.items():
        if dfu is None or dfu.empty:
            continue
        agg = dfu.groupby("round")[list(metrics)].mean().reset_index()
        agg.insert(0, "mode", mode)
        frames.append(agg)
    if not frames:
        return paths
    df_all = pd.concat(frames, ignore_index=True)

    for m in metrics:
        if m not in df_all.columns:
            continue
        out_path = out_dir / f"{experiment_name}__mode_compare__{m}.png"
        _lineplot(df_all, x="round", y=m, hue="mode",
                  title=f"{experiment_name} | Mode comparison | {m} (mean across users & models)",
                  out_path=out_path)
        paths.append(out_path)
    return paths


# ---------- Convenience: single entry point per experiment ----------

def export_experiment_artifacts(
    *,
    experiment_name: str,
    records: List[dict],
    out_dir: str | Path = "reports",
    is_adaptive: bool = True,
) -> Dict[str, List[Path] | pd.DataFrame]:
    """
    One-stop export for a SINGLE experiment (one mode).
    - Writes per-student CSVs.
    - If adaptive, writes plots per metric averaged by model.
    Returns the core dataframes & written paths.
    """
    df_user = flatten_user_records_full(records)
    df_round = flatten_round_records(records)

    paths_csv = save_per_student_csvs(df_user, out_dir, experiment_name)
    paths_plots = []
    if is_adaptive:
        paths_plots = plot_adaptive_learning_model_averages(df_user, out_dir, experiment_name)

    return {
        "df_user": df_user,
        "df_round": df_round,
        "csv_paths": paths_csv,
        "plot_paths": paths_plots,
    }


def export_mode_comparison(
    *,
    experiment_name: str,
    mode_to_records: Dict[str, List[dict]],
    out_dir: str | Path = "reports",
) -> Dict[str, List[Path] | pd.DataFrame]:
    """
    Multi-experiment export (adaptive + any baselines).
    mode_to_records: {"adaptive": recsA, "random": recsR, "popularity": recsP, ...}
    - Produces per-mode user DataFrames
    - Plots a line per mode for each metric averaged across users & models.
    """
    mode_to_df_user: Dict[str, pd.DataFrame] = {}
    for mode, recs in mode_to_records.items():
        mode_to_df_user[mode] = flatten_user_records_full(recs)

    paths = plot_mode_comparison(mode_to_df_user, out_dir, experiment_name)
    return {
        "mode_to_df_user": mode_to_df_user,
        "plot_paths": paths,
    }
