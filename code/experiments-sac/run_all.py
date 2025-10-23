"""Batch runner for Orchid Ranker experiments on EdNet and OULAD with summaries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from orchid_ranker.dp import get_dp_config
from orchid_ranker.experiments import RankingExperiment

# ------------------ experiment knobs ------------------

COHORT_SIZE = 16
STUDENT_METHODS = ["irt", "mirt", "zpd", "contextual_zpd"]
INITIAL_PROFILES = [
    {"name": "struggling", "knowledge": 0.2, "fatigue": 0.7, "engagement": 0.4, "trust": 0.3},
    {"name": "steady", "knowledge": 0.5, "fatigue": 0.5, "engagement": 0.6, "trust": 0.5},
    {"name": "advancing", "knowledge": 0.7, "fatigue": 0.3, "engagement": 0.8, "trust": 0.7},
    {"name": "high_flyer", "knowledge": 0.9, "fatigue": 0.2, "engagement": 0.9, "trust": 0.8},
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
    "zpd_margin": 0.14,
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

# ------------------ runs / plans ------------------

# COMMON_MODES = ["adaptive", "linucb", "als", "fixed", "popularity", "random"]
COMMON_MODES = ["adaptive", "linucb", "als", "fixed"]

EDNET_RUNS = [
    {
        "name": "open_nodp",
        "save_dir": "runs/ednet",
        "modes": COMMON_MODES,
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "linucb",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "off_open",
        "save_dir": "runs/ednet",
        "modes": COMMON_MODES,
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "linucb",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "standard_eps1_0",
        "save_dir": "runs/ednet",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 1.2, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.03, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "strong_eps0_5",
        "save_dir": "runs/ednet",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 2.0, "sample_rate": 0.015, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.02, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "locked_eps0_2",
        "save_dir": "runs/ednet",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 3.0, "sample_rate": 0.01, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.01, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
]

OULAD_RUNS = [
    {
        "name": "open_nodp",
        "save_dir": "runs/oulad",
        "modes": COMMON_MODES,
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "off_open",
        "save_dir": "runs/oulad",
        "modes": COMMON_MODES,
        "dp": "off",
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "standard_eps1_0",
        "save_dir": "runs/oulad",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 1.2, "sample_rate": 0.02, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.03, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "strong_eps0_5",
        "save_dir": "runs/oulad",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 2.0, "sample_rate": 0.015, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.02, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
    {
        "name": "locked_eps0_2",
        "save_dir": "runs/oulad",
        "modes": COMMON_MODES,
        "dp": {"enabled": True, "noise_multiplier": 3.0, "sample_rate": 0.01, "delta": 1e-5, "max_grad_norm": 1.0},
        "config_overrides": {**BASE_CONFIG_OVERRIDES, "per_round_eps_target": 0.01, "deterministic_pool": True, "persistent_pool": True, "pool_seed": 4242},
        "adaptive_policy": "bootts",
        "dp_strength": "none",
        "warm_start": WARM_START_CFG,
    },
]


PLANS: List[Dict] = [
    {"name": "ednet", "config": "configs/ednet.yaml", "dataset": "ednet",
     "cohort_size": COHORT_SIZE, "runs": EDNET_RUNS},
    # {"name": "oulad", "config": "configs/oulad.yaml", "dataset": "oulad",
    #  "cohort_size": COHORT_SIZE, "runs": OULAD_RUNS},
]

# ------------------ helpers ------------------

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

    co = run.get("config_overrides") or {}
    if "zpd_bounds" in co:
        zb = co["zpd_bounds"]
        if isinstance(zb, (list, tuple)) and len(zb) == 2:
            cfg["zpd_bounds"] = (float(zb[0]), float(zb[1]))
    if "zpd_margin" in co:
        cfg["zpd_margin"] = float(co["zpd_margin"])
    if "zpd_margin" not in cfg and "zpd_bounds" in cfg:
        lo, hi = cfg["zpd_bounds"]
        cfg["zpd_margin"] = max(1e-6, 0.5 * (hi - lo))

    return cfg

def _safe_cols(df: pd.DataFrame, desired: List[str]) -> List[str]:
    have = set(df.columns)
    return [c for c in desired if c in have]

def _ensure_rates(df_user: pd.DataFrame) -> pd.DataFrame:
    df = df_user.copy()

    # normalize common column aliases -> tel_*
    alias_telemetry = {
        "shown": "tel_shown",
        "accepted": "tel_accepted",
        "correct": "tel_correct",
        "accept_rate": "tel_accept_rate",
    }
    for old, new in alias_telemetry.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # derive accept_rate if not present
    if "tel_accept_rate" not in df.columns and {"tel_accepted", "tel_shown"}.issubset(df.columns):
        df["tel_accept_rate"] = df["tel_accepted"] / df["tel_shown"].clip(lower=1)

    # ensure optional telemetry columns exist (fill with NaN if missing)
    for col in ["novelty_rate", "serendipity", "accept_at4", "dwell_s", "latency_s"]:
        if col not in df.columns:
            df[col] = np.nan

    # ensure pre/post exist (fill with NaN if missing)
    for col in ["pre_knowledge", "post_knowledge", "pre_engagement", "post_engagement"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def export_mode_user_csv(
    df_user: pd.DataFrame,
    out_dir: Path,
    mode: str,
    user_id_to_name: Optional[Dict[int, str]] = None,
) -> Path:
    """
    Write one CSV for this mode with per-student, per-round rows.
    Columns include student identity, round, telemetry, knobs, and pre/post state.
    """
    df = _ensure_rates(df_user)

    # ensure user_name (prefer provided mapping)
    if "user_name" not in df.columns:
        if user_id_to_name:
            df["user_name"] = df["user_id"].map(lambda x: user_id_to_name.get(int(x), f"user_{int(x)}"))
        else:
            df["user_name"] = "user_" + df["user_id"].astype(str)

    alias_knobs = {"top_k": "knob_top_k", "zpd_margin": "knob_zpd_margin", "mmr_lambda": "knob_mmr_lambda", "novelty_bonus": "knob_novelty_bonus"}
    for old, new in alias_knobs.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    first_cols = [
        "round", "mode",
        "user_id", "user_name",
        "student_method", "profile",
    ]
    telemetry_cols = [
        "tel_shown", "tel_accepted", "tel_correct",
        "tel_accept_rate", "accept_at4", "novelty_rate", "serendipity",
        "dwell_s", "latency_s",
    ]
    knob_cols = ["knob_top_k", "knob_zpd_margin", "knob_mmr_lambda", "knob_novelty_bonus"]
    pre_cols = ["pre_knowledge", "pre_engagement", "pre_trust", "pre_fatigue"]
    post_cols = ["post_knowledge", "post_engagement", "post_trust", "post_fatigue"]

    ordered = (
        _safe_cols(df, first_cols)
        + _safe_cols(df, telemetry_cols)
        + _safe_cols(df, knob_cols)
        + _safe_cols(df, pre_cols)
        + _safe_cols(df, post_cols)
    )
    ordered += [c for c in df.columns if c not in ordered]

    df = df.sort_values(["user_name", "student_method", "round"]).reset_index(drop=True)[ordered]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_user_rounds.csv"
    df.to_csv(out_path, index=False)
    return out_path

# ------------------ summaries and deltas ------------------

def summarize_mode(df_user_rounds: pd.DataFrame, mode: str) -> Dict[str, float]:
    df = _ensure_rates(df_user_rounds)

    # build aggregation spec dynamically
    agg_spec = {
        "tel_correct": "mean",
        "tel_accept_rate": "mean",
        "pre_knowledge": "first",
        "post_knowledge": "last",
        "pre_engagement": "first",
        "post_engagement": "last",
    }
    if "novelty_rate" in df.columns:
        agg_spec["novelty_rate"] = "mean"
    if "serendipity" in df.columns:
        agg_spec["serendipity"] = "mean"

    grp = df.groupby("user_id", as_index=False).agg(agg_spec)

    # derive deltas (will be NaN-safe)
    grp["delta_knowledge"] = grp["post_knowledge"] - grp["pre_knowledge"]
    grp["delta_engagement"] = grp["post_engagement"] - grp["pre_engagement"]

    out = {
        "mode": mode,
        "mean_accuracy": float(grp["tel_correct"].mean()) if "tel_correct" in grp else np.nan,
        "mean_accept_rate": float(grp["tel_accept_rate"].mean()) if "tel_accept_rate" in grp else np.nan,
        "delta_knowledge": float(grp["delta_knowledge"].mean()),
        "delta_engagement": float(grp["delta_engagement"].mean()),
        "novelty_rate": float(grp["novelty_rate"].mean()) if "novelty_rate" in grp else np.nan,
        "serendipity": float(grp["serendipity"].mean()) if "serendipity" in grp else np.nan,
        "n_users": int(grp.shape[0]),
    }
    return out


def paired_delta(adapt_df: pd.DataFrame, base_df: pd.DataFrame, base_mode: str) -> Dict[str, float]:
    a = _ensure_rates(adapt_df).groupby("user_id").agg({"tel_correct":"mean","tel_accept_rate":"mean"})
    b = _ensure_rates(base_df).groupby("user_id").agg({"tel_correct":"mean","tel_accept_rate":"mean"})
    m = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if m.empty:
        return {"baseline": base_mode, "paired_delta_accuracy": np.nan, "paired_delta_accept_rate": np.nan, "paired_n": 0}
    return {
        "baseline": base_mode,
        "paired_delta_accuracy": float((m["tel_correct_a"] - m["tel_correct_b"]).mean()),
        "paired_delta_accept_rate": float((m["tel_accept_rate_a"] - m["tel_accept_rate_b"]).mean()),
        "paired_n": int(m.shape[0]),
    }

def write_retention_if_possible(dataset_root: Path, run_name: str) -> None:
    """
    If this run is DP-enabled, compare adaptive vs adaptive in 'off_open' (or 'open_nodp') and write retention.csv.
    """
    this_run = dataset_root / run_name
    this_adapt = this_run / "adaptive_user_rounds.csv"
    if not this_adapt.exists():
        return

    # find reference off run
    off_candidates = ["off_open", "open_nodp"]
    ref_adapt = None
    for off in off_candidates:
        cand = dataset_root / off / "adaptive_user_rounds.csv"
        if cand.exists():
            ref_adapt = cand
            break
    if ref_adapt is None:
        return

    df_dp = pd.read_csv(this_adapt)
    df_off = pd.read_csv(ref_adapt)
    summ_dp = summarize_mode(df_dp, "adaptive_dp")
    summ_off = summarize_mode(df_off, "adaptive_off")

    def safe_ratio(a: float, b: float) -> float:
        if b == 0 or not np.isfinite(b):
            return np.nan
        return float(a / b)

    retention = pd.DataFrame([{
        "metric": "mean_accuracy",
        "ratio": safe_ratio(summ_dp["mean_accuracy"], summ_off["mean_accuracy"]),
    }, {
        "metric": "mean_accept_rate",
        "ratio": safe_ratio(summ_dp["mean_accept_rate"], summ_off["mean_accept_rate"]),
    }, {
        "metric": "delta_knowledge",
        "ratio": safe_ratio(summ_dp["delta_knowledge"], summ_off["delta_knowledge"]),
    }])
    retention.to_csv(this_run / "retention.csv", index=False)

# ------------------ runner ------------------

def run_plan(plan: Dict) -> None:
    runner = RankingExperiment(
        plan["config"],
        dataset=plan["dataset"],
        cohort_size=plan.get("cohort_size", COHORT_SIZE),
        seed=42,
        student_methods=STUDENT_METHODS,
        initial_profiles=INITIAL_PROFILES,
        assignment_mode="cartesian",
    )

    for run in plan["runs"]:
        # directory shape: runs/<dataset>/<experiment_name>/
        root_dir = Path(run["save_dir"]) / run["name"]
        root_dir.mkdir(parents=True, exist_ok=True)

        # dp config (accepts preset string or dict)
        dp_cfg = run["dp"]
        if isinstance(dp_cfg, str):
            dp_cfg = get_dp_config(dp_cfg)
        dp_cfg = dict(dp_cfg)  # avoid mutating shared presets

        per_mode_paths: Dict[str, Path] = {}
        per_mode_frames: Dict[str, pd.DataFrame] = {}

        for mode in run["modes"]:
            print(f"Running {plan['dataset']} / {run['name']} / {mode}")
            adaptive_kwargs = None
            if mode == "adaptive":
                adaptive_kwargs = _build_adaptive_overrides(plan["dataset"], run, dp_cfg)

            # run experiment
            result = runner.run(
                mode,
                dp_enabled=dp_cfg.get("enabled", False),
                dp_params=dict(dp_cfg),
                config_overrides=run.get("config_overrides"),
                log_path=str(root_dir / f"{mode}.jsonl"),  # JSONL for provenance
                adaptive_kwargs=adaptive_kwargs,
                warm_start=run.get("warm_start", WARM_START_CFG),
            )

            # per-student, per-round CSV (primary artifact)
            df_user = result["user_rounds"]
            csv_path = export_mode_user_csv(
                df_user=df_user,
                out_dir=root_dir,
                mode=mode,
                user_id_to_name=None,
            )
            per_mode_paths[mode] = csv_path
            per_mode_frames[mode] = df_user
            print(f"[saved] {csv_path}")

        # --------- write summary.csv ---------
        summaries = []
        for mode, df in per_mode_frames.items():
            summaries.append(summarize_mode(df, mode))
        pd.DataFrame(summaries).to_csv(root_dir / "summary.csv", index=False)

        # --------- write deltas.csv (adaptive vs each baseline) ---------
        deltas = []
        if "adaptive" in per_mode_frames:
            adapt_df = per_mode_frames["adaptive"]
            for base in [m for m in per_mode_frames.keys() if m != "adaptive"]:
                deltas.append(paired_delta(adapt_df, per_mode_frames[base], base_mode=base))
        pd.DataFrame(deltas).to_csv(root_dir / "deltas.csv", index=False)

        # --------- write retention.csv (DP runs only, if reference off run exists) ---------
        if isinstance(dp_cfg, dict) and dp_cfg.get("enabled", False):
            write_retention_if_possible(Path(run["save_dir"]), run["name"])

def main() -> None:
    for plan in PLANS:
        run_plan(plan)

if __name__ == "__main__":
    main()
