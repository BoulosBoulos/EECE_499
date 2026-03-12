"""Ablation study with plug-and-play PINN variants, safety filter, and residual toggles.

Plug-and-play variants (each independently toggled):
  nopinn          - baseline: no physics loss, no filter
  pinn_critic     - Design A: physics loss on critic only
  pinn_actor      - Design B: physics loss on actor only
  pinn_both       - Designs A+B: physics loss on both
  pinn_ego        - Design A + L_ego dynamics prediction
  pinn_no_ttc     - Design A without TTC residual
  pinn_no_stop    - Design A without stopping-distance residual
  pinn_no_fric    - Design A without friction-circle residual
  safety_filter   - no physics loss, but safety filter overrides actions
  pinn_critic_sf  - Design A + safety filter

Multi-seed: runs each (scenario, variant, lambda) with multiple seeds.
Sensitivity: sweeps lambda_physics_critic across specified values.
Incremental CSV: results written after each run (not only at the end).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]

VARIANTS = [
    # (name, pinn_placement, use_l_ego, use_safety_filter, use_ttc, use_stop, use_fric)
    ("nopinn",         "none",   False, False, True,  True,  True),
    ("pinn_critic",    "critic", False, False, True,  True,  True),
    ("pinn_actor",     "actor",  False, False, True,  True,  True),
    ("pinn_both",      "both",   False, False, True,  True,  True),
    ("pinn_ego",       "critic", True,  False, True,  True,  True),
    ("pinn_no_ttc",    "critic", False, False, False, True,  True),
    ("pinn_no_stop",   "critic", False, False, True,  False, True),
    ("pinn_no_fric",   "critic", False, False, True,  True,  False),
    ("safety_filter",  "none",   False, True,  True,  True,  True),
    ("pinn_critic_sf", "critic", False, True,  True,  True,  True),
]

DEFAULT_SEEDS = [42, 123, 456, 789, 999]


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _append_csv(path: str, row: dict, fieldnames: list[str]):
    """Append a single row to CSV, creating header if file is new."""
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


EVAL_FIELDS = [
    "scenario", "variant", "pinn_placement", "use_l_ego", "use_safety_filter",
    "lambda_phys", "seed", "eval_mode",
    "mean_return", "std_return", "collision_rate", "pothole_hits_mean",
    "mean_ttc", "min_ttc",
]

TRAIN_LOG_FIELDS = [
    "scenario", "variant", "lambda_phys", "seed", "step",
    "actor_loss", "vf_loss", "entropy", "total_loss",
    "l_physics", "l_actor_physics", "l_ego",
    "viol_ttc_rate", "viol_stop_rate", "viol_fric_rate",
    "viol_ttc_mag", "viol_stop_mag", "viol_fric_mag",
]


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    from experiments.run_train import collect_rollouts as _c
    return _c(env, policy, n_steps, gamma, gae_lambda)


def train_one(
    env, scenario: str, variant_name: str,
    pinn_placement: str, use_l_ego: bool, use_safety_filter: bool,
    use_ttc: bool, use_stop: bool, use_fric: bool,
    total_steps: int, out_dir: str, device: str, seed: int,
    lambda_phys: float = 0.5,
    train_csv: str | None = None,
) -> str:
    """Train one run. Returns checkpoint path. Writes train log incrementally."""
    from models.drppo import DRPPO

    np.random.seed(seed)
    if torch:
        torch.manual_seed(seed)

    cfg = _load_config("configs/algo/default.yaml")
    res_cfg = _load_config("configs/residuals/default.yaml")
    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(
        obs_dim=obs_dim, n_actions=5,
        lr=float(cfg.get("lr", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_range=float(cfg.get("clip_range", 0.2)),
        pinn_placement=pinn_placement,
        use_l_ego=use_l_ego,
        use_safety_filter=use_safety_filter,
        use_ttc=use_ttc, use_stop=use_stop, use_fric=use_fric,
        lambda_physics_critic=lambda_phys,
        lambda_physics_actor=float(res_cfg.get("lambda_physics_actor", 0.1)),
        lambda_physics_ttc=float(res_cfg.get("lambda_physics_ttc", 1.0)),
        lambda_physics_stop=float(res_cfg.get("lambda_physics_stop", 1.0)),
        lambda_physics_fric=float(res_cfg.get("lambda_physics_fric", 1.0)),
        lambda_physics_ego=float(res_cfg.get("lambda_physics_ego", 0.1)),
        physics_ttc_thr=float(res_cfg.get("physics_ttc_thr", 3.0)),
        physics_tau=float(res_cfg.get("physics_tau", 0.5)),
        a_max=float(res_cfg.get("a_max", 5.0)),
        mu=float(res_cfg.get("mu", 0.8)),
        g=float(res_cfg.get("g", 9.81)),
        hidden_dim=int(cfg.get("gru_hidden", 128)),
        device=device,
        dt=float(res_cfg.get("dt", 0.1)),
        a_go=float(res_cfg.get("a_go", 2.0)),
    )
    n_steps = int(cfg.get("n_steps", 256))
    batch_size = int(cfg.get("batch_size", 64))
    n_epochs = int(cfg.get("n_epochs", 5))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))

    step = 0
    while step < total_steps:
        obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra, hidden_arr = \
            collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
        step += n_steps
        metrics = {}
        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) == 0:
                    continue
                ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                     for k, v in extra.items()} if extra else None
                h = hidden_arr[idx] if hidden_arr is not None else None
                metrics = policy.train_step(
                    obs_arr[idx], actions_arr[idx], log_probs_arr[idx],
                    returns_arr[idx], advantages_arr[idx],
                    hiddens=h, extra=ex,
                )

        if train_csv:
            row = {"scenario": scenario, "variant": variant_name,
                   "lambda_phys": lambda_phys, "seed": seed, "step": step}
            row.update(metrics)
            _append_csv(train_csv, row, TRAIN_LOG_FIELDS)

    ckpt = os.path.join(out_dir, f"ablation_{scenario}_{variant_name}_lp{lambda_phys}_s{seed}.pt")
    policy.save(ckpt)
    return ckpt


def eval_one(
    env, checkpoint: str, n_episodes: int, device: str, seed: int,
    deterministic: bool = True,
    pinn_placement: str = "none",
    use_safety_filter: bool = False,
) -> dict:
    from models.drppo import DRPPO

    np.random.seed(seed)
    obs_dim = int(env.observation_space.shape[0])
    policy = DRPPO(
        obs_dim=obs_dim, n_actions=5,
        pinn_placement=pinn_placement,
        use_safety_filter=use_safety_filter,
        device=device,
    )
    policy.load(checkpoint)

    returns, coll_eps, pothole_hits, ttc_means, ttc_mins = [], [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset_hidden()
        r_tot, coll, pot = 0, 0, 0
        ttc_list = []
        for _ in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            r_tot += r
            if info.get("collision", False):
                coll += 1
            if info.get("in_pothole", False):
                pot += 1
            ttc_list.append(info.get("ttc_min", 10.0))
            if term or trunc:
                break
        returns.append(r_tot)
        coll_eps.append(1 if coll > 0 else 0)
        pothole_hits.append(pot)
        ttc_arr = np.array(ttc_list)
        ttc_means.append(float(np.mean(ttc_arr)))
        ttc_mins.append(float(np.min(ttc_arr)))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "collision_rate": float(np.mean(coll_eps)),
        "pothole_hits_mean": float(np.mean(pothole_hits)),
        "mean_ttc": float(np.mean(ttc_means)),
        "min_ttc": float(np.mean(ttc_mins)),
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation + sensitivity study with plug-and-play variants")
    parser.add_argument("--out_dir", default="results/ablation")
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variant names to run (default: all)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--lambda_phys", type=float, nargs="+", default=[0.5],
                        help="lambda_physics sweep (e.g. 0.001 0.01 0.05 0.1 0.2 0.5 1.0)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--reward_config", default=None)
    parser.add_argument("--eval_stochastic", action="store_true",
                        help="Run both deterministic and stochastic evaluation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    from env.sumo_env import SumoEnv

    variant_names = args.variants or [v[0] for v in VARIANTS]
    active_variants = [v for v in VARIANTS if v[0] in variant_names]

    results_csv = os.path.join(args.out_dir, "ablation_results.csv")
    train_csv = os.path.join(args.out_dir, "ablation_train_log.csv")

    for scenario in args.scenarios:
        env = SumoEnv(scenario_name=scenario, use_intent=args.use_intent,
                      reward_config=args.reward_config)

        for vname, placement, use_l_ego, use_sf, use_ttc, use_stop, use_fric in active_variants:
            has_pinn = placement != "none"
            lp_values = args.lambda_phys if has_pinn else [0.0]

            for lp in lp_values:
                seed_results = []
                for seed in args.seeds:
                    run_tag = f"{scenario}_{vname}_lp{lp}_s{seed}"
                    ckpt = os.path.join(args.out_dir, f"ablation_{run_tag}.pt")

                    if not args.skip_train:
                        print(f"Training {run_tag}...")
                        ckpt = train_one(
                            env, scenario, vname,
                            pinn_placement=placement, use_l_ego=use_l_ego,
                            use_safety_filter=use_sf,
                            use_ttc=use_ttc, use_stop=use_stop, use_fric=use_fric,
                            total_steps=args.total_steps, out_dir=args.out_dir,
                            device=device, seed=seed, lambda_phys=lp,
                            train_csv=train_csv,
                        )

                    if os.path.isfile(ckpt):
                        eval_modes = ["deterministic"]
                        if args.eval_stochastic:
                            eval_modes.append("stochastic")

                        for mode in eval_modes:
                            det = mode == "deterministic"
                            print(f"Eval {run_tag} [{mode}]...")
                            m = eval_one(env, ckpt, args.eval_episodes, device, seed,
                                         deterministic=det,
                                         pinn_placement=placement,
                                         use_safety_filter=use_sf)
                            seed_results.append(m)

                            row = {
                                "scenario": scenario, "variant": vname,
                                "pinn_placement": placement, "use_l_ego": use_l_ego,
                                "use_safety_filter": use_sf,
                                "lambda_phys": lp, "seed": seed,
                                "eval_mode": mode, **m,
                            }
                            _append_csv(results_csv, row, EVAL_FIELDS)

                if seed_results:
                    agg_return = np.mean([r["mean_return"] for r in seed_results])
                    agg_std = np.std([r["mean_return"] for r in seed_results])
                    agg_coll = np.mean([r["collision_rate"] for r in seed_results])
                    agg_ttc = np.mean([r["mean_ttc"] for r in seed_results])
                    print(f"  [{vname} lp={lp}] {len(seed_results)} evals: "
                          f"return={agg_return:.2f}+/-{agg_std:.2f} "
                          f"coll={agg_coll:.3f} ttc={agg_ttc:.2f}")

        env.close()

    print(f"\nAblation done. Results: {results_csv}")
    print(f"Training log: {train_csv}")


if __name__ == "__main__":
    main()
