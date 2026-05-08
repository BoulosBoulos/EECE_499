"""Phase 29.5 — Synthesize 60 fake training runs to exercise the analysis stats path.

Decision E (per SPEC_PHASE_2_VERIFICATION_GATE):
  10 seeds x 6 methods x 1 cell (scenario=1a, maneuver=stem_right, intent_on=False).

Per-method final_collision_rate distributions (mean +/- std across 10 seeds):
  drppo        : 0.20 +/- 0.05   (baseline)
  hjb_aux      : 0.10 +/- 0.03   should beat drppo  (d ~ -2.4, sig)
  soft_hjb_aux : 0.08 +/- 0.04   should beat drppo  (d ~ -2.6, sig)
  eikonal_aux  : 0.18 +/- 0.05   should NOT sig     (d ~ -0.4, ns)
  cbf_aux      : 0.05 +/- 0.02   should beat drppo  (d ~ -3.9, sig)
  fusion_aux   : 0.04 +/- 0.02   should beat drppo  (d ~ -4.2, sig)

Synthetic seed: 99999 (fixed for reproducibility).
"""
from __future__ import annotations
import os, json, sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd

SYNTHETIC_RNG_SEED = 99999
SEEDS = [42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555]
METHOD_SPECS = {
    "drppo":         {"collision_mean": 0.20, "collision_std": 0.05},
    "hjb_aux":       {"collision_mean": 0.10, "collision_std": 0.03},
    "soft_hjb_aux":  {"collision_mean": 0.08, "collision_std": 0.04},
    "eikonal_aux":   {"collision_mean": 0.18, "collision_std": 0.05},
    "cbf_aux":       {"collision_mean": 0.05, "collision_std": 0.02},
    "fusion_aux":    {"collision_mean": 0.04, "collision_std": 0.02},
}

# Required by analysis/loader._flatten_config — provide all 24 UNIFORM_CONFIG_KEYS.
UNIFORM_CONFIG_DEFAULTS = {
    "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "n_epochs_per_update": 8, "batch_size": 128, "n_steps": 4096,
    "policy_hidden_size": 256, "policy_n_layers": 3,
    "gru_hidden_size": 256, "gru_n_layers": 1,
    "alpha_cbf": None, "tau_soft": None, "w_fail": None, "barrier_offset": None,
    "lambda_residual": None, "lambda_distill": None, "lambda_actor_kl": None,
    "collocation_size": None,
    "w_optimality": None, "w_safety": None,
}

METHOD_OVERRIDES = {
    "hjb_aux":      {"lambda_residual": 0.2, "lambda_distill": 0.25, "collocation_size": 256},
    "soft_hjb_aux": {"lambda_residual": 0.2, "lambda_distill": 0.25, "tau_soft": 1.0,
                     "lambda_actor_kl": 0.1, "collocation_size": 256},
    "eikonal_aux":  {"lambda_residual": 0.2, "lambda_distill": 0.25, "w_fail": 50.0,
                     "collocation_size": 256},
    "cbf_aux":      {"lambda_residual": 0.2, "lambda_distill": 0.25, "alpha_cbf": 1.0,
                     "barrier_offset": 10.0, "collocation_size": 256},
    "fusion_aux":   {"lambda_residual": 0.2, "lambda_distill": 0.25, "tau_soft": 1.0,
                     "alpha_cbf": 1.0, "barrier_offset": 10.0, "lambda_actor_kl": 0.1,
                     "w_optimality": 0.125, "w_safety": 0.125, "collocation_size": 256},
}


def _residual_pair(method: str, rng: np.random.Generator) -> tuple[float, float]:
    """Per-method residual values that pass quality.py's _check_method_specific.

    drppo                  : both = 0
    hjb_aux / soft_hjb_aux : opt > 0, saf = 0
    eikonal_aux / cbf_aux  : opt = 0, saf > 0
    fusion_aux             : both > 0
    """
    if method == "drppo":
        return 0.0, 0.0
    if method in ("hjb_aux", "soft_hjb_aux"):
        return float(rng.uniform(0.05, 0.5)), 0.0
    if method in ("eikonal_aux", "cbf_aux"):
        return 0.0, float(rng.uniform(0.05, 0.5))
    if method == "fusion_aux":
        return float(rng.uniform(0.05, 0.5)), float(rng.uniform(0.05, 0.5))
    return 0.0, 0.0


def _action_dist_row(rng: np.random.Generator) -> dict:
    """Five action probabilities that sum to exactly 1.0 (within ~1e-12)."""
    raw = rng.dirichlet(np.array([2.0, 4.0, 2.0, 8.0, 1.0]))  # bias toward 'go'
    return {
        "action_dist_stop":  float(raw[0]),
        "action_dist_creep": float(raw[1]),
        "action_dist_yield": float(raw[2]),
        "action_dist_go":    float(raw[3]),
        "action_dist_abort": float(raw[4]),
    }


def _make_metrics_df(rng: np.random.Generator, target_collision_rate: float, method: str) -> pd.DataFrame:
    """Five iterations whose final-window (10%) gives the controlled rate.
    n_window = max(1, 5*0.10) = 1, so the LAST row alone determines final metrics.
    """
    n_eps_per_iter = 100
    rows = []
    for it_idx, total_steps in enumerate([4096, 8192, 12288, 16384, 20480], start=1):
        if it_idx < 5:
            warmup_rate = float(rng.uniform(0.30, 0.50))
            r = warmup_rate + (target_collision_rate - warmup_rate) * (it_idx / 5.0)
        else:
            r = target_collision_rate
        n_coll = int(round(r * n_eps_per_iter))
        n_succ = int(round(0.50 * n_eps_per_iter))
        n_succ = max(0, min(n_succ, n_eps_per_iter - n_coll))
        n_timeout = max(0, n_eps_per_iter - n_coll - n_succ)
        opt, saf = _residual_pair(method, rng)
        adist = _action_dist_row(rng)
        rows.append({
            "iteration": it_idx,
            "total_steps": total_steps,
            "wall_time_seconds": float(it_idx) * 60.0,
            "iter_time_seconds": 60.0,
            "env_step_time_seconds": 30.0,
            "learn_step_time_seconds": 25.0,
            "residual_compute_time_seconds": 5.0,
            "L_total": float(rng.uniform(1.0, 5.0)),
            "L_policy": float(rng.uniform(0.1, 1.0)),
            "L_value": float(rng.uniform(0.1, 1.0)),
            "L_entropy": float(rng.uniform(-0.2, -0.01)),
            "L_residual_optimality": opt,
            "L_residual_safety":     saf,
            "L_distill":             float(rng.uniform(0.05, 0.5)),
            "mean_reward": float(rng.uniform(-50, 50)),
            "mean_episode_length": float(rng.uniform(20, 120)),
            "n_episodes": n_eps_per_iter,
            "n_collisions": n_coll,
            "n_successes": n_succ,
            "n_timeouts": n_timeout,
            "n_aborts": 0,
            **adist,
        })
    return pd.DataFrame(rows)


def _make_meta(method: str, seed: int, target_collision_rate: float, run_dir: Path) -> dict:
    cfg = dict(UNIFORM_CONFIG_DEFAULTS)
    cfg.update(METHOD_OVERRIDES.get(method, {}))
    return {
        "run_id": f"SYN_1a_stem_right_{method}_nointent_s{seed}",
        "start_time_iso": "2026-05-04T00:00:00",
        "end_time_iso":   "2026-05-04T00:05:00",
        "wall_time_seconds": 300.0,
        "method": method,
        "scenario": "1a",
        "ego_maneuver": "stem_right",
        "seed": int(seed),
        "intent_on": False,
        "total_steps_target": 20000,
        "total_steps_actual": 20480,
        "convergence_reason": "synthetic",
        "git_commit": "synthetic",
        "git_branch": "synthetic",
        "git_dirty": False,
        "hostname": "synthetic-host",
        "device": "cpu",
        "torch_version": "2.11.0",
        "python_version": "3.12.3",
        "config": cfg,
        "result_summary": {
            "final_collision_rate": float(target_collision_rate),
            "final_success_rate": 0.50,
            "final_mean_reward": 0.0,
        },
    }


def synthesize(output_dir: str) -> dict:
    rng_master = np.random.default_rng(SYNTHETIC_RNG_SEED)
    out_root = Path(output_dir)
    tier1_dir = out_root / "tier1"
    tier1_dir.mkdir(parents=True, exist_ok=True)

    summary = {"runs": []}
    for method, specs in METHOD_SPECS.items():
        # Per-method, deterministically force the 10-seed sample to match the
        # Decision E (mean, std) EXACTLY. We sample from N(0,1), z-score it
        # (so sample mean=0, sample std=1), then scale -> N(method_mean, method_std).
        # This eliminates sampling-variability deviations from Decision E that
        # would push borderline cases (e.g. eikonal d ~ -0.4) over the
        # significance line by chance.
        z = rng_master.normal(0.0, 1.0, size=len(SEEDS))
        z = (z - z.mean()) / z.std(ddof=1)
        rates = specs["collision_mean"] + specs["collision_std"] * z
        rates = np.clip(rates, 0.0, 1.0)
        for seed, target_rate in zip(SEEDS, rates):
            run_id = f"SYN_1a_stem_right_{method}_nointent_s{seed}"
            run_dir = tier1_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            # Per-run seeded RNG so synthesis is fully reproducible.
            run_rng = np.random.default_rng(SYNTHETIC_RNG_SEED ^ (seed * 100003) ^ (hash(method) & 0xFFFF))
            metrics = _make_metrics_df(run_rng, float(target_rate), method)
            metrics.to_csv(run_dir / "metrics.csv", index=False)
            meta = _make_meta(method, int(seed), float(target_rate), run_dir)
            with open(run_dir / "meta.json", "w") as fh:
                json.dump(meta, fh, indent=2)
            summary["runs"].append({
                "method": method, "seed": int(seed),
                "target_collision_rate": float(target_rate),
                "run_dir": str(run_dir),
            })
    summary["n_runs"] = len(summary["runs"])
    summary["output_dir"] = str(out_root)
    return summary


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_60run"
    s = synthesize(out_dir)
    print(json.dumps({"n_runs": s["n_runs"], "output_dir": s["output_dir"]}, indent=2))
