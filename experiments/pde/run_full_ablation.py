"""Launch the full ablation grid as parallel subprocesses.

Tiers:
    1: Main comparison (600 runs) — 12 combos x 5 methods x 5 seeds x 2 intent
    2: Lambda sensitivity (48) + Occlusion ablation (60) = 108 runs
    3: State ablation (60) + Behavioral robustness (60) + Dense scenarios (45) = 165 runs
    supp: Full 42-combo table with best method (126 runs)
    all: Tiers 1 + 2 + 3 (873 runs, excludes supp)

Usage:
    python experiments/pde/run_full_ablation.py --tier 1 --max_parallel 32
    python experiments/pde/run_full_ablation.py --tier 2 --max_parallel 16
    python experiments/pde/run_full_ablation.py --tier all --dry_run
"""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import os
from itertools import product

# Phase 1F: tier configurations are sourced from config_frozen_v1.yaml so the
# orchestrator and the rest of the codebase share a single canonical config.
from config_loader import (  # noqa: E402
    check_config_lock,
    get_config as _get_frozen_config,
    get_tier_config as _get_tier_cfg,
)

_FROZEN = _get_frozen_config()

# ── Tier 1: Main comparison (sourced from config_frozen_v1.yaml :: tier1) ─
_TIER1_CFG = _get_tier_cfg("tier1")
TIER1_COMBOS = [(c["scenario"], c["maneuver"]) for c in _TIER1_CFG["combos"]]
TIER1_METHODS = list(_TIER1_CFG["methods"])
TIER1_SEEDS = list(_TIER1_CFG["seeds"])
TIER1_INTENTS = list(_TIER1_CFG["intents"])
TIER1_TOTAL_STEPS = int(_TIER1_CFG["total_steps"])

# ── Tier 2 configuration (Phase 1D, YAML-sourced) ─────────────────────────
# Three sub-grids: 2a (lambda sweep, 600 jobs), 2b (occlusion sweep, 400),
# 2c (fusion weight sweep, 160). Total 1,160 training jobs.
_TIER2_CFG = _get_tier_cfg("tier2")
DEFAULT_TOTAL_STEPS = int(_TIER2_CFG["total_steps"])  # post-Phase-3 calibrated

TIER2_PDE_METHODS = list(_TIER2_CFG["shared"]["methods"])
TIER2_MANEUVERS = list(_TIER2_CFG["shared"]["maneuvers"])
TIER2_SEEDS = list(_TIER2_CFG["shared"]["seeds"])

# Sub-grid 2a — lambda residual sweep
_T2A = _TIER2_CFG["sub_grid_2a"]
TIER2A_SCENARIOS = list(_T2A["scenarios"])
TIER2A_LAMBDA_VALUES = list(_T2A["lambda_values"])

# Sub-grid 2b — occlusion sweep
_T2B = _TIER2_CFG["sub_grid_2b"]
TIER2B_SCENARIOS = list(_T2B["scenarios"])
TIER2B_OCCLUSION_VALUES = list(_T2B["occlusion_values"])

# Sub-grid 2c — fusion weight sweep
_T2C = _TIER2_CFG["sub_grid_2c"]
TIER2C_SCENARIOS = list(_T2C["scenarios"])
TIER2C_FUSION_WEIGHTS = [tuple(w) for w in _T2C["fusion_weights"]]

# ── Tier 3 (sourced from config_frozen_v1.yaml :: tier3) ───────────────────
_TIER3_CFG = _get_tier_cfg("tier3")

_T3S = _TIER3_CFG["state_ablation"]
TIER3_STATE_COMBOS = [(c["scenario"], c["maneuver"]) for c in _T3S["combos"]]
TIER3_STATE_METHODS = list(_T3S["methods"])
TIER3_STATE_SEEDS = list(_T3S["seeds"])

_T3B = _TIER3_CFG["behavioral_robustness"]
TIER3_BEHAV_COMBOS = [(c["scenario"], c["maneuver"]) for c in _T3B["combos"]]
TIER3_BEHAV_METHODS = list(_T3B["methods"])
TIER3_BEHAV_SEEDS = list(_T3B["seeds"])

_T3D = _TIER3_CFG["dense_scenarios"]
TIER3_DENSE_SCENARIOS = list(_T3D["scenarios"])
TIER3_DENSE_METHODS = list(_T3D["methods"])
TIER3_DENSE_SEEDS = list(_T3D["seeds"])

# ── Supplementary (YAML-sourced) ───────────────────────────────────────────
_SUPP_CFG = _get_tier_cfg("supplementary")
ALL_SCENARIOS = list(_SUPP_CFG["scenarios"])
ALL_MANEUVERS = list(_SUPP_CFG["maneuvers"])
SUPP_SEEDS = list(_SUPP_CFG["seeds"])
SUPP_BEST_METHOD = str(_SUPP_CFG["method"])

# ── Tier 4: Held-out evaluation (YAML-sourced) ────────────────────────────
_TIER4_CFG = _get_tier_cfg("tier4")
TIER4_HELDOUT_CONFIGS = [
    {
        "name": ho["name"],
        "source_tier": ho["source_tier"],
        "eval_overrides": dict(ho["eval_overrides"]),
    }
    for ho in _TIER4_CFG["holdout_configs"]
]
TIER4_N_EVAL_EPISODES = int(_TIER4_CFG["n_eval_episodes"])
TIER4_EVAL_SEED_OFFSETS = list(_TIER4_CFG["eval_seed_offsets"])


def generate_tier4_jobs(total_steps: int = 50000) -> list[dict]:
    """Generate held-out eval-only jobs from existing checkpoints."""
    import glob as _glob
    jobs = []
    base_dir = "results/ablation"
    METHODS_SET = ["soft_hjb_aux", "hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]

    for ho_cfg in TIER4_HELDOUT_CONFIGS:
        ho_name = ho_cfg["name"]
        source_dir = os.path.join(base_dir, ho_cfg["source_tier"])
        if not os.path.isdir(source_dir):
            continue
        ckpt_paths = _glob.glob(os.path.join(source_dir, "*", "model_*.pt"))
        for ckpt_path in ckpt_paths:
            ckpt_file = os.path.basename(ckpt_path)
            ckpt_dir = os.path.basename(os.path.dirname(ckpt_path))
            # Skip intermediate checkpoints (contain "_step")
            if "_step" in ckpt_file:
                continue
            # Parse method from filename
            name = ckpt_file.replace("model_", "").replace(".pt", "")
            method = None
            for m in sorted(METHODS_SET, key=len, reverse=True):
                if name.startswith(m + "_"):
                    method = m
                    rest = name[len(m) + 1:]
                    break
            if method is None:
                continue
            # Parse scenario
            scenario = None
            for s in ["4_dense", "3_dense", "2_dense", "1a", "1b", "1c", "1d", "2", "3", "4"]:
                if rest.startswith(s + "_"):
                    scenario = s
                    maneuver = rest[len(s) + 1:]
                    break
                elif rest == s:
                    scenario = s
                    maneuver = "stem_right"
                    break
            if scenario is None:
                continue
            # Parse seed
            seed = None
            for part in ckpt_dir.split("_"):
                if part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])
                    break
            if seed is None:
                continue

            eval_out_dir = os.path.join(base_dir, f"tier4_{ho_name}",
                                        f"{scenario}_{maneuver}_{method}_s{seed}")
            eval_cmd = _build_eval_cmd(
                method, scenario, maneuver, seed, eval_out_dir,
                n_eval_episodes=TIER4_N_EVAL_EPISODES,
                no_buildings=ho_cfg["eval_overrides"].get("no_buildings", False),
                style_filter=ho_cfg["eval_overrides"].get("style_filter", None),
                state_ablation=ho_cfg["eval_overrides"].get("state_ablation", None),
                use_intent=("_intent_" in ckpt_dir
                            and "_nointent_" not in ckpt_dir),
            )
            # Override checkpoint path to point to source
            if "--checkpoint" in eval_cmd:
                ckpt_idx = eval_cmd.index("--checkpoint") + 1
                eval_cmd[ckpt_idx] = ckpt_path

            jobs.append({
                "cmd_train": None,
                "cmd_eval": eval_cmd,
                "tag": f"T4_{ho_name}_{scenario}_{maneuver}_{method}_s{seed}",
                "tier": 4,
                "method": method,
                "scenario": scenario,
                "ego_maneuver": maneuver,
                "seed": seed,
                "intent_on": False,
            })
    return jobs


def manifest_sort_key(job: dict) -> tuple:
    """Deterministic sort key for the multi-machine Tier 1 launch.

    Imported by experiments/pde/preview_tier_1_split.py — both scripts MUST
    use the same key so slice boundaries on different machines correspond
    to identical job sets. Changing the key ordering breaks reproducibility
    across rentals.

    Tuple shape: (method, scenario, ego_maneuver, seed, intent_on, tier_label).
    `tier_label` is derived as `tier_<tier>` (matches the post-run annotation
    written into meta.json by `_annotate_meta_with_tier`).
    """
    return (
        str(job.get("method", "")),
        str(job.get("scenario", "")),
        str(job.get("ego_maneuver", "")),
        int(job.get("seed", 0) or 0),
        bool(job.get("intent_on", False)),
        f"tier_{job.get('tier', '')}",
    )


def _partition_balance_jobs(jobs: list[dict]) -> list[dict]:
    """Interleave rule_based jobs evenly through the trainable jobs so any
    contiguous slice of the manifest sees a proportional rule_based share.

    Motivation: rule_based jobs are eval-only (~30 min each); trainable jobs
    are training+eval (~5–6 h each at 400 k steps). Plain alphabetical sort
    by `manifest_sort_key` clusters rule_based after the trainable methods,
    which on a 4-way slice means M1/M2 get all-trainable workloads (~30 h)
    and M3/M4 get rule_based-heavy workloads (~22–25 h). Balancing the slices
    cuts the M1/M2 wall-time bottleneck while keeping the total compute the
    same.

    Algorithm: deterministic stride placement.
      1. Partition: trainable = method != "rule_based"; rule_based otherwise.
      2. Sort each partition by `manifest_sort_key` (alphabetical determinism
         within each kind).
      3. Compute, for each rule_based index i ∈ [0, R), the global position
         `pos = ((2*i + 1) * total) // (2 * R)` — i.e., centered on the
         (i+0.5)/R fractile, rounded down. Integer arithmetic so the result
         is bit-identical across machines and Python versions.
      4. Should two positions collide (mathematically can't happen for the
         canonical Tier 1 case T:R = 1440:240 = 6:1, but covered defensively
         for arbitrary ratios), the colliding entry spills to the next
         vacant slot, wrapping at `total`.
      5. Fill the global ordering: rule_based at their stride positions,
         trainable filling the remaining positions in their sorted order.

    For T:R = 1440:240 (Tier 1) the rule_based jobs land at positions
    {3, 10, 17, ..., 1676} — exactly every 7th. A 4-way 420-job slice
    therefore receives exactly 60 rule_based + 360 trainable.

    The output is the canonical "global ordering" used for slicing across
    machines. Within-slice deterministic ordering (alphabetical by
    `manifest_sort_key`) is applied separately by the orchestrator AFTER
    slicing and by the preview's per-slice display logic.
    """
    trainable = sorted(
        [j for j in jobs if j.get("method") != "rule_based"],
        key=manifest_sort_key,
    )
    rule_based = sorted(
        [j for j in jobs if j.get("method") == "rule_based"],
        key=manifest_sort_key,
    )
    T, R = len(trainable), len(rule_based)
    if R == 0:
        return trainable
    if T == 0:
        return rule_based
    total = T + R
    # Compute deterministic stride positions for the rule_based partition.
    rb_positions = [None] * R
    used = set()
    for i in range(R):
        pos = ((2 * i + 1) * total) // (2 * R)
        # Defensive collision spill (should be a no-op for T:R = 1440:240).
        while pos in used:
            pos = (pos + 1) % total
        used.add(pos)
        rb_positions[i] = pos
    # Order the rule_based assignments by ascending position so the overall
    # slice retains the rule_based partition's alphabetical stride.
    rb_pairs = sorted(zip(rb_positions, range(R)))
    out: list[dict] = [None] * total  # type: ignore[assignment]
    for pos, rb_idx in rb_pairs:
        out[pos] = rule_based[rb_idx]
    train_iter = iter(trainable)
    for i in range(total):
        if out[i] is None:
            out[i] = next(train_iter)
    return out


def _annotate_meta_with_tier(out_dir: str, tier_label: str) -> None:
    """Post-run annotation: write tier_label into the per-job meta.json.

    Called after each subprocess returns (success or failure) so analysis
    can group runs by tier without parsing output_dir paths. Silently no-ops
    if meta.json is absent (e.g. job crashed before write_meta_start ran).
    """
    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        meta["tier_label"] = tier_label
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
    except (OSError, ValueError):
        pass


def _build_train_cmd(method, scenario, maneuver, seed, out_dir,
                     total_steps=50000, use_intent=False, lambda_aux=None,
                     no_buildings=False, style_filter=None, state_ablation=None):
    """Build the command list for a single training run. Returns None for rule_based."""
    if method == "rule_based":
        return None  # no training needed
    if method == "drppo":
        script = "experiments/pde/train_drppo_baseline.py"
    else:
        script = f"experiments/pde/train_{method}.py"
    cmd = [
        "python3", script,
        "--scenario", scenario,
        "--ego_maneuver", maneuver,
        "--seed", str(seed),
        "--out_dir", out_dir,
        "--total_steps", str(total_steps),
    ]
    if use_intent:
        cmd.append("--use_intent")
    if lambda_aux is not None and method != "drppo":
        cmd.extend(["--lambda_aux", str(lambda_aux)])
    if no_buildings:
        cmd.append("--no_buildings")
    if style_filter is not None:
        cmd.extend(["--style_filter", style_filter])
    if state_ablation is not None:
        cmd.extend(["--state_ablation", state_ablation])
    return cmd


def _build_eval_cmd(method, scenario, maneuver, train_seed, out_dir,
                    n_eval_episodes=100, no_buildings=False,
                    style_filter=None, state_ablation=None,
                    use_intent=False):
    """Build eval command for a trained checkpoint.

    `use_intent` must match the value the matching train command was launched
    with — intent-trained policies expect obs_dim=165 (135 base + 30 intent
    block); evaluating them without --use_intent crashes the policy network
    with a 165-vs-135 input dim mismatch.
    """
    ckpt_path = os.path.join(out_dir, f"model_{method}_{scenario}_{maneuver}.pt")
    eval_seeds = [train_seed + off for off in TIER4_EVAL_SEED_OFFSETS]
    cmd = [
        "python3", "experiments/pde/eval.py",
        "--method", method,
        "--checkpoint", ckpt_path,
        "--scenario", scenario,
        "--ego_maneuver", maneuver,
        "--episodes", str(n_eval_episodes),
        "--seeds", *[str(s) for s in eval_seeds],
        "--out_dir", out_dir,
        "--save_failures", "--max_failures", "5",
    ]
    if use_intent:
        cmd.append("--use_intent")
    if no_buildings:
        cmd.append("--no_buildings")
    if style_filter:
        cmd.extend(["--style_filter", style_filter])
    if state_ablation:
        cmd.extend(["--state_ablation", state_ablation])
    return cmd


# ── Tier 2 sub-grid generators (Phase 1D) ──────────────────────────────


def _tier2_method_specific_args(method: str) -> list[str]:
    """Method-specific default args to pass explicitly so meta.json records them.

    Fusion gets the union of Soft-HJB and CBF defaults plus its weights;
    single-PDE methods get only the args their script accepts.
    """
    if method == "fusion_aux":
        return [
            "--w_optimality", "1.0",
            "--w_safety", "1.0",
            "--alpha_cbf", "1.0",
            "--barrier_offset", "10.0",
            "--tau_soft", "1.0",
            "--lambda_actor_kl", "0.1",
        ]
    if method == "soft_hjb_aux":
        return ["--tau_soft", "1.0", "--lambda_actor_kl", "0.1"]
    if method == "cbf_aux":
        return ["--alpha_cbf", "1.0", "--barrier_offset", "10.0"]
    if method == "eikonal_aux":
        return ["--w_fail", "50.0"]
    return []


def _generate_tier2a_jobs(total_steps: int, output_root: str) -> list[dict]:
    """Sub-grid 2a — lambda residual sweep (600 jobs)."""
    jobs = []
    for method in TIER2_PDE_METHODS:
        for scenario in TIER2A_SCENARIOS:
            for maneuver in TIER2_MANEUVERS:
                for seed in TIER2_SEEDS:
                    for lam in TIER2A_LAMBDA_VALUES:
                        tag = f"T2a_{scenario}_{maneuver}_{method}_lam{lam}_s{seed}"
                        out_dir = os.path.join(
                            output_root, "tier2", "2a_lambda_sweep", tag,
                        )
                        cmd = [
                            "python3", f"experiments/pde/train_{method}.py",
                            "--scenario", scenario,
                            "--ego_maneuver", maneuver,
                            "--seed", str(seed),
                            "--lambda_residual", str(lam),
                            "--output_dir", out_dir,
                            "--total_steps", str(total_steps),
                        ]
                        cmd.extend(_tier2_method_specific_args(method))
                        jobs.append({
                            "tag": tag,
                            "tier": 2,
                            "subgrid": "2a",
                            "method": method,
                            "scenario": scenario,
                            "ego_maneuver": maneuver,
                            "seed": seed,
                            "intent_on": False,
                            "cmd_train": cmd,
                            "cmd_eval": _build_eval_cmd(
                                method, scenario, maneuver, seed, out_dir,
                                use_intent=False,
                            ),
                        })
    return jobs


def _generate_tier2b_jobs(total_steps: int, output_root: str) -> list[dict]:
    """Sub-grid 2b — occlusion sweep (400 jobs)."""
    jobs = []
    for method in TIER2_PDE_METHODS:
        for scenario in TIER2B_SCENARIOS:
            for maneuver in TIER2_MANEUVERS:
                for seed in TIER2_SEEDS:
                    for occ in TIER2B_OCCLUSION_VALUES:
                        occ_str = "ON" if occ == "on" else "OFF"
                        tag = f"T2b_{scenario}_{maneuver}_{method}_occ{occ_str}_s{seed}"
                        out_dir = os.path.join(
                            output_root, "tier2", "2b_occlusion_sweep", tag,
                        )
                        cmd = [
                            "python3", f"experiments/pde/train_{method}.py",
                            "--scenario", scenario,
                            "--ego_maneuver", maneuver,
                            "--seed", str(seed),
                            "--lambda_residual", "0.2",
                            "--output_dir", out_dir,
                            "--total_steps", str(total_steps),
                        ]
                        if occ == "off":
                            cmd.append("--no_buildings")
                        cmd.extend(_tier2_method_specific_args(method))
                        jobs.append({
                            "tag": tag,
                            "tier": 2,
                            "subgrid": "2b",
                            "method": method,
                            "scenario": scenario,
                            "ego_maneuver": maneuver,
                            "seed": seed,
                            "intent_on": False,
                            "cmd_train": cmd,
                            "cmd_eval": _build_eval_cmd(
                                method, scenario, maneuver, seed, out_dir,
                                no_buildings=(occ == "off"),
                                use_intent=False,
                            ),
                        })
    return jobs


def _generate_tier2c_jobs(total_steps: int, output_root: str) -> list[dict]:
    """Sub-grid 2c — fusion weight sweep (160 jobs)."""
    jobs = []
    method = "fusion_aux"
    for scenario in TIER2C_SCENARIOS:
        for maneuver in TIER2_MANEUVERS:
            for seed in TIER2_SEEDS:
                for (w_o, w_s) in TIER2C_FUSION_WEIGHTS:
                    tag = (
                        f"T2c_{scenario}_{maneuver}_fusion_aux"
                        f"_w{w_o}_{w_s}_s{seed}"
                    )
                    out_dir = os.path.join(
                        output_root, "tier2", "2c_fusion_weights", tag,
                    )
                    cmd = [
                        "python3", f"experiments/pde/train_{method}.py",
                        "--scenario", scenario,
                        "--ego_maneuver", maneuver,
                        "--seed", str(seed),
                        "--w_optimality", str(w_o),
                        "--w_safety", str(w_s),
                        "--lambda_residual", "0.2",
                        "--alpha_cbf", "1.0",
                        "--barrier_offset", "10.0",
                        "--tau_soft", "1.0",
                        "--lambda_actor_kl", "0.1",
                        "--output_dir", out_dir,
                        "--total_steps", str(total_steps),
                    ]
                    jobs.append({
                        "tag": tag,
                        "tier": 2,
                        "subgrid": "2c",
                        "method": method,
                        "scenario": scenario,
                        "ego_maneuver": maneuver,
                        "seed": seed,
                        "intent_on": False,
                        "cmd_train": cmd,
                        "cmd_eval": _build_eval_cmd(
                            method, scenario, maneuver, seed, out_dir,
                            use_intent=False,
                        ),
                    })
    return jobs


def _generate_tier2_jobs(
    total_steps: int, output_root: str, subgrid: str | None = None,
) -> list[dict]:
    """Generate all Tier 2 jobs (2a + 2b + 2c) or only the requested sub-grid."""
    if subgrid == "2a":
        return _generate_tier2a_jobs(total_steps, output_root)
    if subgrid == "2b":
        return _generate_tier2b_jobs(total_steps, output_root)
    if subgrid == "2c":
        return _generate_tier2c_jobs(total_steps, output_root)
    jobs = []
    jobs.extend(_generate_tier2a_jobs(total_steps, output_root))
    jobs.extend(_generate_tier2b_jobs(total_steps, output_root))
    jobs.extend(_generate_tier2c_jobs(total_steps, output_root))
    return jobs


def generate_jobs(
    tier: str, total_steps: int = 50000, subgrid: str | None = None,
    output_root: str | None = None,
) -> list[dict]:
    """Generate all jobs for the given tier."""
    jobs = []
    base_dir = output_root if output_root is not None else "results/ablation"

    # ── TIER 1: Main comparison ─────────────────────────────────────────
    if tier in ("1", "all"):
        for (scen, man), method, seed, intent in product(
            TIER1_COMBOS, TIER1_METHODS, TIER1_SEEDS, TIER1_INTENTS
        ):
            intent_tag = "intent" if intent else "nointent"
            out_dir = os.path.join(base_dir, "tier1",
                                   f"{scen}_{man}_{method}_{intent_tag}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, use_intent=intent),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            use_intent=intent),
                "tag": f"T1_{scen}_{man}_{method}_{intent_tag}_s{seed}",
                "tier": 1,
                "method": method,
                "scenario": scen,
                "ego_maneuver": man,
                "seed": seed,
                "intent_on": bool(intent),
            })

    # ── TIER 2 (Phase 1D): 2a lambda + 2b occlusion + 2c fusion weights ─
    if tier in ("2", "all"):
        # Tier 2 uses the calibrated post-Phase-3 step count (200000) by
        # default; honor explicit --total_steps override if user set a
        # value other than the legacy 50000 default.
        ts_tier2 = total_steps if total_steps != 50000 else DEFAULT_TOTAL_STEPS
        jobs.extend(_generate_tier2_jobs(ts_tier2, base_dir, subgrid=subgrid))

    # ── TIER 3: State ablation, behavioral robustness, dense ────────────
    if tier in ("3", "all"):
        # State ablation
        for (scen, man), method, seed in product(
            TIER3_STATE_COMBOS, TIER3_STATE_METHODS, TIER3_STATE_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_state",
                                   f"{scen}_{man}_{method}_novis_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, state_ablation="no_visibility"),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            state_ablation="no_visibility",
                                            use_intent=False),
                "tag": f"T3S_{scen}_{method}_novis_s{seed}",
                "tier": 3,
                "method": method,
                "scenario": scen,
                "ego_maneuver": man,
                "seed": seed,
                "intent_on": False,
            })

        # Behavioral robustness (train on nominal styles)
        for (scen, man), method, seed in product(
            TIER3_BEHAV_COMBOS, TIER3_BEHAV_METHODS, TIER3_BEHAV_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_behav",
                                   f"{scen}_{man}_{method}_nominal_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, man, seed, out_dir,
                                              total_steps, style_filter="nominal"),
                "cmd_eval": _build_eval_cmd(method, scen, man, seed, out_dir,
                                            style_filter="nominal",
                                            use_intent=False),
                "tag": f"T3B_{scen}_{method}_nominal_s{seed}",
                "tier": 3,
                "method": method,
                "scenario": scen,
                "ego_maneuver": man,
                "seed": seed,
                "intent_on": False,
            })

        # Dense scenarios
        for scen, method, seed in product(
            TIER3_DENSE_SCENARIOS, TIER3_DENSE_METHODS, TIER3_DENSE_SEEDS
        ):
            out_dir = os.path.join(base_dir, "tier3_dense",
                                   f"{scen}_stem_right_{method}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(method, scen, "stem_right", seed,
                                              out_dir, total_steps),
                "cmd_eval": _build_eval_cmd(method, scen, "stem_right", seed, out_dir,
                                            use_intent=False),
                "tag": f"T3D_{scen}_{method}_s{seed}",
                "tier": 3,
                "method": method,
                "scenario": scen,
                "ego_maneuver": "stem_right",
                "seed": seed,
                "intent_on": False,
            })

    # ── SUPPLEMENTARY: Full 42-combo table ──────────────────────────────
    if tier == "supp":
        best_method = SUPP_BEST_METHOD  # sourced from config_frozen_v1.yaml :: supplementary.method
        for scen, man, seed in product(ALL_SCENARIOS, ALL_MANEUVERS, SUPP_SEEDS):
            out_dir = os.path.join(base_dir, "supplementary",
                                   f"{scen}_{man}_{best_method}_s{seed}")
            jobs.append({
                "cmd_train": _build_train_cmd(best_method, scen, man, seed,
                                              out_dir, total_steps),
                "cmd_eval": _build_eval_cmd(best_method, scen, man, seed, out_dir,
                                            use_intent=False),
                "tag": f"SUP_{scen}_{man}_s{seed}",
                "tier": "supp",
                "method": best_method,
                "scenario": scen,
                "ego_maneuver": man,
                "seed": seed,
                "intent_on": False,
            })

    # ── TIER 4: Held-out eval (eval-only on existing checkpoints) ────────
    if tier in ("4",):
        jobs.extend(generate_tier4_jobs(total_steps))

    return jobs


def _apply_filters(jobs: list[dict], args) -> list[dict]:
    """Apply CLI filters to the generated job list (Phase 2 / Step 2)."""
    filtered = []
    for job in jobs:
        if args.seeds is not None and job.get("seed") not in args.seeds:
            continue
        if args.methods is not None and job.get("method") not in args.methods:
            continue
        if args.scenarios is not None and job.get("scenario") not in args.scenarios:
            continue
        if args.maneuvers is not None and job.get("ego_maneuver") not in args.maneuvers:
            continue
        if args.intents is not None:
            intent_strs = [s.lower() for s in args.intents]
            if str(job.get("intent_on", False)).lower() not in intent_strs:
                continue
        if not args.include_rule_based and job.get("method") == "rule_based":
            continue
        filtered.append(job)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Launch parallel ablation jobs")
    parser.add_argument("--tier", default="1", choices=["1", "2", "3", "4", "all", "supp"])
    parser.add_argument("--max_parallel", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--dry_run", action="store_true", help="Print jobs without running")
    parser.add_argument(
        "--subgrid",
        choices=["2a", "2b", "2c"],
        default=None,
        help="Run only a specific Tier 2 sub-grid. Requires --tier 2.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Filter to specific seeds. If omitted, uses all tier seeds.")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Filter to specific methods. If omitted, uses all tier methods.")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Filter to specific scenarios. If omitted, uses all tier scenarios.")
    parser.add_argument("--maneuvers", type=str, nargs="+", default=None,
                        help="Filter to specific ego maneuvers. If omitted, uses all tier maneuvers.")
    parser.add_argument("--intents", type=str, nargs="+", default=None,
                        choices=["true", "false", "True", "False"],
                        help="Filter to specific intent settings. Values: true/false. If omitted, uses all.")
    # rule_based is eval-only (no training compute), provides the heuristic
    # baseline reviewers compare PDE methods against. Tier 1's spec target is
    # 1,680 = 1,440 trainable + 240 rule_based eval-only; excluding rule_based
    # by default historically masked the spec count and risked a launch missing
    # the baseline. Default flipped to True; pass --no-include_rule_based to
    # opt out (e.g. for trainable-only smoke filters).
    parser.add_argument(
        "--include_rule_based",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include rule_based eval-only baseline (default: True). "
             "--no-include_rule_based excludes it (e.g. for trainable-only smoke).",
    )
    parser.add_argument("--output_root", type=str, default=None,
                        help="Override base output dir (default: results/ablation). Phase 2 smoke tests "
                        "use /tmp/phase2_smoke for transient outputs.")
    # ── Multi-machine Tier 1 launch (SPEC_TIER_1_MULTI_MACHINE_LAUNCH) ───
    # Each rental runs a deterministic slice of the full sorted manifest.
    # job_index_start/end indices are computed locally via
    # `experiments/pde/preview_tier_1_split.py` using the SAME sort key
    # (manifest_sort_key) — diverging keys would mean different machines
    # produce overlapping or gap-bearing slices.
    parser.add_argument(
        "--job_index_start", type=int, default=0,
        help="Start index (inclusive) into the sorted manifest. "
             "Default 0 (run from the beginning).",
    )
    parser.add_argument(
        "--job_index_end", type=int, default=None,
        help="End index (exclusive) into the sorted manifest. "
             "Default: len(manifest) after filtering.",
    )
    parser.add_argument(
        "--machine_id", type=str, default="local",
        help="Tag results under results/tier_1_machine_<id>/ when not "
             "'local'. Used during multi-machine Tier 1 to isolate per-"
             "rental outputs prior to aggregate_tier_1_results.py merge.",
    )
    args = parser.parse_args()

    # Multi-machine: when machine_id is set and the user did not explicitly
    # override --output_root, write to results/tier_1_machine_<id>/ so each
    # rental's outputs land in a non-colliding subtree. The job_dir paths
    # are constructed downstream by joining base_dir with `tier1/<tag>`.
    if args.machine_id != "local" and args.output_root is None:
        args.output_root = f"results/tier_1_machine_{args.machine_id}"

    if args.subgrid is not None and args.tier not in ("2", "all"):
        parser.error("--subgrid requires --tier 2 (or --tier all).")

    # Phase 1F: warn (and prompt before launch) if config_frozen_v1.yaml has
    # been modified since the lock was generated. Dry-runs only print the
    # warning; actual launches require explicit confirmation.
    lock_status = check_config_lock(strict=False)
    if not lock_status["matches"] and lock_status.get("warning"):
        print(lock_status["warning"])
        if not args.dry_run:
            confirm = input("Proceed anyway? [y/N]: ")
            if confirm.strip().lower() != "y":
                sys.exit(1)

    jobs = generate_jobs(
        args.tier, args.total_steps, subgrid=args.subgrid,
        output_root=args.output_root,
    )

    # Phase 2 / Step 2: apply CLI filter flags
    pre_filter_count = len(jobs)
    jobs = _apply_filters(jobs, args)
    if pre_filter_count != len(jobs):
        print(f"Filtered {pre_filter_count} -> {len(jobs)} jobs "
              f"(seeds={args.seeds}, methods={args.methods}, "
              f"scenarios={args.scenarios}, maneuvers={args.maneuvers}, "
              f"intents={args.intents}, include_rule_based={args.include_rule_based})")

    # Multi-machine: produce a balanced global ordering (rule_based
    # interleaved among trainable jobs at deterministic stride positions)
    # so contiguous slices [start, end) on different rentals each get a
    # proportional rule_based share. The same helper is imported by
    # `preview_tier_1_split.py` so slice indices match across scripts.
    jobs = _partition_balance_jobs(jobs)

    total_after_filter = len(jobs)
    slice_start = max(0, int(args.job_index_start))
    slice_end = total_after_filter if args.job_index_end is None else min(
        int(args.job_index_end), total_after_filter,
    )
    if slice_end < slice_start:
        parser.error(
            f"--job_index_end ({slice_end}) must be >= --job_index_start "
            f"({slice_start})"
        )
    if slice_start != 0 or slice_end != total_after_filter:
        jobs = jobs[slice_start:slice_end]
        n_rb = sum(1 for j in jobs if j.get("method") == "rule_based")
        print(
            f"[machine={args.machine_id}] running jobs [{slice_start}:{slice_end}]"
            f" of {total_after_filter} total ({len(jobs)} jobs in slice: "
            f"{len(jobs) - n_rb} trainable + {n_rb} rule_based)"
        )
    else:
        n_rb = sum(1 for j in jobs if j.get("method") == "rule_based")
        print(
            f"[machine={args.machine_id}] running full manifest "
            f"({total_after_filter} jobs: {total_after_filter - n_rb} "
            f"trainable + {n_rb} rule_based)"
        )

    # Within-slice deterministic ordering: re-sort by manifest_sort_key so
    # this rental launches jobs in alphabetical order regardless of where
    # they sat in the global balanced ordering. Identical-content slices
    # produce identical run logs across re-launches.
    jobs = sorted(jobs, key=manifest_sort_key)

    # Print tier breakdown
    tier_counts = {}
    for j in jobs:
        t = j["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"Generated {len(jobs)} jobs for tier '{args.tier}':")
    for t, c in sorted(tier_counts.items(), key=lambda x: str(x[0])):
        print(f"  Tier {t}: {c} jobs")

    if args.dry_run:
        # Phase 1D: emit per-subgrid section headers + final summary for Tier 2
        # so verification can parse the structure deterministically.
        is_tier2_view = (args.tier in ("2", "all")) and any(
            j.get("subgrid") in ("2a", "2b", "2c") for j in jobs
        )
        subgrid_titles = {
            "2a": "lambda sweep",
            "2b": "occlusion sweep",
            "2c": "fusion weight sweep",
        }
        if is_tier2_view:
            print("[DRY_RUN] Tier 2 — full ablation grid")
            print("[DRY_RUN]")
            counts_by_sg = {"2a": 0, "2b": 0, "2c": 0}
            for j in jobs:
                if j.get("subgrid") in counts_by_sg:
                    counts_by_sg[j["subgrid"]] += 1

        emitted_subgrid_header = set()
        for j in jobs:
            sg = j.get("subgrid")
            if is_tier2_view and sg in subgrid_titles and sg not in emitted_subgrid_header:
                title = subgrid_titles[sg]
                print(
                    f"[DRY_RUN] Sub-grid {sg} ({title}): "
                    f"{counts_by_sg[sg]} jobs"
                )
                emitted_subgrid_header.add(sg)
            print(f"  {j['tag']}:")
            if j["cmd_train"] is not None:
                print(f"    TRAIN: {' '.join(j['cmd_train'])}")
            else:
                print(f"    TRAIN: (none — rule-based, eval only)")
            print(f"    EVAL:  {' '.join(j['cmd_eval'])}")

        if is_tier2_view:
            tier2_total = sum(counts_by_sg.values())
            print("[DRY_RUN]")
            print(f"[DRY_RUN] Total Tier 2 jobs: {tier2_total}")
            print(f"[DRY_RUN]   2a (lambda sweep):    {counts_by_sg['2a']}")
            print(f"[DRY_RUN]   2b (occlusion sweep): {counts_by_sg['2b']}")
            print(f"[DRY_RUN]   2c (fusion weights):  {counts_by_sg['2c']}")
        return

    active = []
    completed = 0
    failed = 0
    start = time.time()

    for job in jobs:
        while len(active) >= args.max_parallel:
            time.sleep(5)
            still_active = []
            for proc, tag, log_f, meta_dir, tier_label in active:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, tag, log_f, meta_dir, tier_label))
                elif ret == 0:
                    log_f.close()
                    _annotate_meta_with_tier(meta_dir, tier_label)
                    completed += 1
                    elapsed = time.time() - start
                    print(f"  [OK] {tag} ({completed}/{len(jobs)}, {elapsed/60:.0f}m elapsed)")
                else:
                    log_f.close()
                    _annotate_meta_with_tier(meta_dir, tier_label)
                    failed += 1
                    print(f"  [FAIL] {tag} (exit {ret})")
            active = still_active

        # Resolve out_dir from the correct command (cmd_train is None for
        # eval-only jobs: rule_based and Tier 4 held-out evaluations)
        if job["cmd_train"] is None:
            out_dir_idx = job["cmd_eval"].index("--out_dir") + 1
            job_out_dir = job["cmd_eval"][out_dir_idx]
        else:
            _dir_flag = "--out_dir" if "--out_dir" in job["cmd_train"] else "--output_dir"
            out_dir_idx = job["cmd_train"].index(_dir_flag) + 1
            job_out_dir = job["cmd_train"][out_dir_idx]

        # Skip if done; re-run eval only if checkpoint exists but eval was killed.
        if job["cmd_train"] is not None:
            import glob as _glob
            _done = _glob.glob(os.path.join(job_out_dir, f"*_step{args.total_steps}.pt"))
            _eval_done = os.path.exists(os.path.join(job_out_dir, "eval_metrics.csv"))
            if _done and _eval_done:
                completed += 1
                print(f"  [SKIP] {job['tag']} (step{args.total_steps} checkpoint found)")
                continue
            elif _done and not _eval_done:
                # Training complete but eval was killed — re-run eval only
                os.makedirs(job_out_dir, exist_ok=True)
                eval_str = " ".join(f"'{c}'" for c in job["cmd_eval"])
                log_f = open(os.path.join(job_out_dir, "stdout.log"), "a")
                proc = subprocess.Popen(f"({eval_str})", stdout=log_f, stderr=subprocess.STDOUT, shell=True)
                active.append((proc, job["tag"], log_f, job_out_dir, f"tier_{job['tier']}"))
                print(f"  [EVAL-ONLY] {job['tag']} (pid {proc.pid})")
                continue
        elif os.path.exists(os.path.join(job_out_dir, "eval_metrics.csv")):
            completed += 1
            print(f"  [SKIP] {job['tag']} (eval_metrics.csv found)")
            continue

        os.makedirs(job_out_dir, exist_ok=True)
        log_path = os.path.join(job_out_dir, "stdout.log")
        # Chain train then eval in one shell command
        eval_str = " ".join(f"'{c}'" for c in job["cmd_eval"])
        if job["cmd_train"] is None:
            # Rule-based: eval only, no training
            shell_cmd = f"({eval_str})"
        else:
            train_str = " ".join(f"'{c}'" for c in job["cmd_train"])
            shell_cmd = f"({train_str} && echo '=== EVAL ===' && {eval_str})"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(shell_cmd, stdout=log_f, stderr=subprocess.STDOUT, shell=True)
        tier_label = f"tier_{job['tier']}"
        active.append((proc, job["tag"], log_f, job_out_dir, tier_label))
        print(f"  [START] {job['tag']} (pid {proc.pid})")

    for proc, tag, log_f, meta_dir, tier_label in active:
        proc.wait()
        log_f.close()
        _annotate_meta_with_tier(meta_dir, tier_label)
        if proc.returncode == 0:
            completed += 1
        else:
            failed += 1
            print(f"  [FAIL] {tag} (exit {proc.returncode})")

    elapsed = time.time() - start
    print(f"\nDone: {completed} completed, {failed} failed, {elapsed/3600:.1f}h total")


if __name__ == "__main__":
    main()
