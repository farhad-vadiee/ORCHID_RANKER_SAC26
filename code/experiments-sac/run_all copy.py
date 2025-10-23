"""Batch runner for Orchid Ranker experiments on EdNet and OULAD."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

from orchid_ranker.dp import get_dp_config
from orchid_ranker.experiments import RankingExperiment
from orchid_ranker.visualization import (
    plot_round_comparison,
    plot_knowledge_trajectory,
    plot_metric_trajectory,
    plot_metric_grid,
)

COHORT_SIZE = 2
STUDENT_METHODS = ["irt", "mirt", "zpd", "contextual_zpd"]
STUDENT_METHODS = ["irt","zpd"]
INITIAL_PROFILES = [
    {"name": "struggling", "knowledge": 0.2, "fatigue": 0.7, "engagement": 0.4, "trust": 0.3},
    # {"name": "steady", "knowledge": 0.5, "fatigue": 0.5, "engagement": 0.6, "trust": 0.5},
    # {"name": "advancing", "knowledge": 0.7, "fatigue": 0.3, "engagement": 0.8, "trust": 0.7},
    # {"name": "high_flyer", "knowledge": 0.9, "fatigue": 0.2, "engagement": 0.9, "trust": 0.8},
]

ADAPTIVE_BASE_PARAMS = {
    "hidden": 96,
    "emb_dim": 48,
    "adapter_slots": 512,
    "kl_beta": 0.02,
    "blend_increment": 0.16,
    "teacher_ema": 0.9,
    "entropy_lambda": 0.08,
    "info_gain_lambda": 0.12,
    "linucb_alpha": 1.6,
    "use_linucb": False,
    "use_bootts": False,
    "ts_heads": 16,
    "ts_alpha": 0.95,
    "zpd_margin" : 0.14
}

DATASET_ADJUSTMENTS = {
    "ednet": {"blend_increment": 0.18, "entropy_lambda": 0.09, "info_gain_lambda": 0.14},
    "oulad": {"blend_increment": 0.16},
}

DP_STRENGTH_BASE = {
    "none": {},
    "standard": {"entropy_lambda": 0.11, "info_gain_lambda": 0.20},
    "locked": {"entropy_lambda": 0.11, "info_gain_lambda": 0.20},
    "strong": {"hidden": 128, "emb_dim": 64, "kl_beta": 0.015, "entropy_lambda": 0.12, "info_gain_lambda": 0.24},
}

DP_POLICY_SCALE = {
    "none": {"linucb_alpha": 1.6, "ts_alpha": 0.95},
    "standard": {"linucb_alpha": 1.85, "ts_alpha": 1.05},
    "locked": {"linucb_alpha": 1.85, "ts_alpha": 1.05},
    "strong": {"linucb_alpha": 2.2, "ts_alpha": 1.15},
}


def _infer_dp_strength(run: Dict) -> str:
    name = str(run.get("name", "")).lower()
    if "strong" in name:
        return "strong"
    if "locked" in name:
        return "locked"
    dp_cfg = run.get("dp", {})
    if isinstance(dp_cfg, dict) and dp_cfg.get("enabled"):
        return "standard"
    return "none"


def _resolve_adaptive_policy(dataset: str, run: Dict) -> str:
    policy = str(run.get("adaptive_policy", "auto")).lower()
    if policy == "auto":
        dataset = dataset.lower()
        if dataset == "ednet":
            return "linucb"
        return "bootts"
    if policy in {"linucb", "bootts", "hybrid"}:
        return policy
    raise ValueError(f"Unknown adaptive_policy '{policy}'")


def _build_adaptive_overrides(dataset: str, run: Dict, dp_cfg: Dict) -> Dict[str, object]:
    dataset_key = dataset.lower()
    dp_strength = run.get("dp_strength") or _infer_dp_strength(run)
    if dp_strength in (None, "none") and isinstance(dp_cfg, dict) and dp_cfg.get("enabled"):
        dp_strength = "standard"
    dp_strength = str(dp_strength or "none").lower()
    policy = _resolve_adaptive_policy(dataset_key, run)

    cfg: Dict[str, object] = dict(ADAPTIVE_BASE_PARAMS)
    cfg.update(DATASET_ADJUSTMENTS.get(dataset_key, {}))
    cfg.update(DP_STRENGTH_BASE.get(dp_strength, {}))

    policy_scale = DP_POLICY_SCALE.get(dp_strength, DP_POLICY_SCALE["none"])

    use_linucb = policy in {"linucb", "hybrid"}
    use_bootts = policy in {"bootts", "hybrid"}
    cfg["use_linucb"] = use_linucb
    cfg["use_bootts"] = use_bootts
    if use_linucb:
        cfg["linucb_alpha"] = policy_scale["linucb_alpha"]
    if use_bootts:
        cfg["ts_alpha"] = policy_scale["ts_alpha"]
        cfg["ts_heads"] = max(12, int(cfg.get("ts_heads", 16)))

    extra = run.get("adaptive_overrides")
    if extra:
        cfg.update(extra)

    # Pull selected config_overrides into the policy cfg
    co = run.get("config_overrides") or {}

    # propagate zpd_bounds into the policy
    if "zpd_bounds" in co:
        zb = co["zpd_bounds"]
        if isinstance(zb, (list, tuple)) and len(zb) == 2:
            cfg["zpd_bounds"] = (float(zb[0]), float(zb[1]))

    # allow run-level zpd_margin to override base/default
    if "zpd_margin" in co:
        cfg["zpd_margin"] = float(co["zpd_margin"])
    # if margin not set but bounds provided, derive a reasonable default
    if "zpd_margin" not in cfg and "zpd_bounds" in cfg:
        lo, hi = cfg["zpd_bounds"]
        cfg["zpd_margin"] = max(1e-6, 0.5 * (hi - lo))

    return cfg


def _diagnose_baseline_similarity(rounds_all: pd.DataFrame) -> List[str]:
    diagnostics: List[str] = []
    if rounds_all.empty or "mode" not in rounds_all.columns:
        return diagnostics
    baseline = rounds_all[rounds_all["mode"] != "adaptive"].copy()
    if baseline.empty:
        return diagnostics

    for metric in ("mean_knowledge", "mean_engagement"):
        if metric not in baseline.columns:
            continue
        final = (
            baseline.sort_values("round")
            .groupby("mode")[metric]
            .last()
            .dropna()
        )
        if final.empty or len(final) <= 1:
            continue
        spread = float(final.max() - final.min())
        if spread <= 1e-3:
            per_round_spread = (
                baseline.groupby("round")[metric]
                .apply(lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0)
            )
            avg_spread = float(per_round_spread.mean()) if not per_round_spread.empty else 0.0
            diagnostics.append(
                (
                    f"{metric.replace('_', ' ').title()} converged within {spread:.4f} across "
                    "baseline policies. The simulator tends to drive learners toward a similar "
                    f"equilibrium (average per-round spread {avg_spread:.4f}). Consider inspecting "
                    "accept_rate or novelty differences for policy separation."
                )
            )
    return diagnostics
WARM_START_CFG = {"enabled": True, "epochs": 3, "batch_size": 256, "max_batches": 320}

BASE_CONFIG_OVERRIDES = {
    "policy_gain": 1.6,
    "alpha_bounds": (0.05, 0.9),
    "k_bounds": (2, 8),
    "zpd_bounds": (0.06, 0.22),
    "zpd_margin": 0.1,
}
DP_CONFIG_OVERRIDES = {
    "policy_gain": 1.85,
    "alpha_bounds": (0.05, 0.9),
    "k_bounds": (2, 9),
    "zpd_bounds": (0.05, 0.24),
}
DP_CONFIG_OVERRIDES_STRONG = {
    **DP_CONFIG_OVERRIDES,
    "policy_gain": 2.1,
    "zpd_bounds": (0.05, 0.26),
}

EDNET_RUNS = [
    {
        "name": "open_nodp",
        "save_dir": "runs/rq1-ednet-n",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "linucb",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "off_open",
        "save_dir": "runs/privacy-ednet-n/off",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "linucb",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "standard_eps1_0",
        "save_dir": "runs/privacy-ednet-n/standard",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 1.2, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES, "per_round_eps_target": 0.03},
        "adaptive_policy": "bootts",
        "dp_strength": "standard",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "strong_eps0_5",
        "save_dir": "runs/privac-ednet-ny/strong",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 2.0, "sample_rate": 0.015, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES_STRONG, "per_round_eps_target": 0.02},
        "adaptive_policy": "bootts",
        "dp_strength": "strong",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "locked_eps0_2",
        "save_dir": "runs/privacy-ednet-n/locked",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 3.0, "sample_rate": 0.01, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES, "per_round_eps_target": 0.01, "min_candidates": 200},
        "adaptive_policy": "bootts",
        "dp_strength": "locked",
        "warm_start": WARM_START_CFG,
    },
]

OULAD_RUNS = [
    {
        "name": "open_nodp",
        "save_dir": "runs/rq1-oulad-n",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "off_open",
        "save_dir": "runs/privacy/off",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "standard_eps1_0",
        "save_dir": "runs/privacy/standard",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 1.2, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES, "per_round_eps_target": 0.03},
        "adaptive_policy": "bootts",
        "dp_strength": "standard",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "strong_eps0_5",
        "save_dir": "runs/privacy/strong",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 2.0, "sample_rate": 0.015, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES_STRONG, "per_round_eps_target": 0.02},
        "adaptive_policy": "bootts",
        "dp_strength": "strong",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "locked_eps0_2",
        "save_dir": "runs/privacy/locked",
        "modes": ["adaptive", "linucb", "als", "fixed"],
        "dp": {"enabled": True, "noise_multiplier": 3.0, "sample_rate": 0.01, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**DP_CONFIG_OVERRIDES, "per_round_eps_target": 0.01, "min_candidates": 200},
        "adaptive_policy": "bootts",
        "dp_strength": "locked",
        "warm_start": WARM_START_CFG,
    },
]

PLANS: List[Dict] = [
    {"name": "ednet", "config": "configs/ednet.yaml", "dataset": "ednet", "cohort_size": COHORT_SIZE, "runs": EDNET_RUNS},
    {"name": "oulad", "config": "configs/oulad.yaml", "dataset": "oulad", "cohort_size": COHORT_SIZE, "runs": OULAD_RUNS},
]


def run_plan(plan: Dict) -> None:
    runner = RankingExperiment(
        plan["config"],
        dataset=plan["dataset"],
        cohort_size=plan.get("cohort_size", COHORT_SIZE),
        seed=42,
        student_methods=STUDENT_METHODS,
        initial_profiles=INITIAL_PROFILES,
    )
    for run in plan["runs"]:
        save_dir = Path(run["save_dir"]) / run["name"]
        save_dir.mkdir(parents=True, exist_ok=True)
        dp_cfg = run["dp"]
        if isinstance(dp_cfg, str):
            dp_cfg = get_dp_config(dp_cfg)
        dp_cfg = dict(dp_cfg)  # avoid mutating shared presets
        summary_rows: List[Dict] = []
        round_frames: List[pd.DataFrame] = []
        diagnostics: List[str] = []
        for mode in run["modes"]:
            print(f"Running {plan['dataset']} / {run['name']} / {mode}")
            adaptive_kwargs = None
            if mode == "adaptive":
                adaptive_kwargs = _build_adaptive_overrides(plan["dataset"], run, dp_cfg)
            print(f"adaptive_kwargs={adaptive_kwargs}")
            result = runner.run(
                mode,
                dp_enabled=dp_cfg.get("enabled", False),
                dp_params=dict(dp_cfg),
                config_overrides=run.get("config_overrides"),
                log_path=str(save_dir / f"{mode}.jsonl"),
                adaptive_kwargs=adaptive_kwargs,
                warm_start=run.get("warm_start", WARM_START_CFG),
            )
            df_user = result["user_rounds"]

            print(result)
            exit(0)
            summary = asdict(result["summary"])
            summary.update({
                "dataset": plan["dataset"],
                "experiment": run["name"],
                "mode": mode,
            })
            summary_rows.append(summary)
            round_path = save_dir / f"{mode}_rounds.csv"
            df_round = result["round_metrics"].copy()
            df_round.to_csv(round_path, index=False)
            if not df_round.empty:
                if "mode" not in df_round.columns:
                    df_round["mode"] = mode
                round_frames.append(df_round)
        pd.DataFrame(summary_rows).to_csv(save_dir / "summary.csv", index=False)
        if round_frames:
            fig_dir = save_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            rounds_all = pd.concat(round_frames, ignore_index=True).sort_values("round").reset_index(drop=True)
            diagnostics.extend(_diagnose_baseline_similarity(rounds_all))
            ax_metrics = plot_round_comparison(rounds_all, metrics=["accuracy", "accept_rate", "novelty_rate", "serendipity", "mean_knowledge", "mean_engagement"])
            ax_metrics.figure.tight_layout()
            ax_metrics.figure.savefig(fig_dir / "round_metrics.png", dpi=150)
            plt.close(ax_metrics.figure)

            if {"round", "mean_knowledge", "mode"}.issubset(rounds_all.columns):
                ax_knowledge = plot_knowledge_trajectory(rounds_all)
                ax_knowledge.figure.tight_layout()
                ax_knowledge.figure.savefig(fig_dir / "knowledge_trajectory.png", dpi=150)
                plt.close(ax_knowledge.figure)

            metric_specs = [
                ("mean_engagement", "Engagement", "engagement_trajectory.png"),
                ("accuracy", "Accuracy", "accuracy_trajectory.png"),
                ("accept_rate", "Acceptance Rate", "accept_rate_trajectory.png"),
                ("novelty_rate", "Novelty Rate", "novelty_trajectory.png"),
                ("serendipity", "Serendipity", "serendipity_trajectory.png"),
            ]

            for metric, label, filename in metric_specs:
                try:
                    ax_metric = plot_metric_trajectory(
                        rounds_all,
                        metric,
                        ylabel=label,
                        title=f"{label} trajectory",
                    )
                except KeyError:
                    continue
                ax_metric.figure.tight_layout()
                ax_metric.figure.savefig(fig_dir / filename, dpi=150)
                plt.close(ax_metric.figure)

            try:
                grid_fig = plot_metric_grid(
                    rounds_all,
                    metrics=[
                        "accuracy",
                        "mean_knowledge",
                        "mean_engagement",
                        "accept_rate",
                        "novelty_rate",
                        "serendipity",
                    ],
                )
                grid_fig.savefig(fig_dir / "metrics_grid.png", dpi=150)
                plt.close(grid_fig)
            except KeyError:
                pass

        if diagnostics:
            diag_path = save_dir / "diagnostics.txt"
            diag_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
            for msg in diagnostics:
                print(f"[diagnostic] {run['name']}: {msg}")


def main() -> None:
    for plan in PLANS:
        run_plan(plan)


if __name__ == "__main__":
    main()