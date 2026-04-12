"""Curriculum training for the interaction benchmark.

Trains the policy in stages:
  Stage 1: scenarios 1a, 1b, 1c  (single-actor conflicts)
  Stage 2: scenario 2             (car + pedestrian)
  Stage 3: scenario 3             (car + ped + moto)
  Stage 4: scenario 4             (full stress test)

Each stage loads the checkpoint from the previous stage as warm-start.
Scenario **1d** (pothole-only) is not included; train it with ``run_train.py``
if needed.
"""

from __future__ import annotations

import argparse
import csv
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

from env.sumo_env_interaction import InteractionEnv, N_ACTIONS
from experiments.interaction.metrics_util import episode_event_rates
from experiments.interaction.run_train import collect_rollouts


def _load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


CURRICULUM_STAGES = [
    {"name": "stage1_single", "scenarios": ["1a", "1b", "1c"], "steps_per_scenario": 15000},
    {"name": "stage2_mixed",  "scenarios": ["2"],              "steps_per_scenario": 20000},
    {"name": "stage3_multi",  "scenarios": ["3"],              "steps_per_scenario": 20000},
    {"name": "stage4_stress", "scenarios": ["4"],              "steps_per_scenario": 20000},
]


def main():
    parser = argparse.ArgumentParser(description="Curriculum training on interaction benchmark")
    parser.add_argument("--config", default="configs/algo/default.yaml")
    parser.add_argument("--out_dir", default="results/interaction/curriculum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pinn_placement", default="none",
                        choices=["critic", "actor", "both", "none"])
    parser.add_argument("--total_multiplier", type=float, default=1.0,
                        help="Scale all per-stage step counts")
    args = parser.parse_args()

    np.random.seed(args.seed)
    if torch:
        torch.manual_seed(args.seed)

    cfg = _load_yaml(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    lr = float(cfg.get("lr", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_range = float(cfg.get("clip_range", 0.2))
    n_steps = int(cfg.get("n_steps", 2048))
    batch_size = int(cfg.get("batch_size", 128))
    n_epochs = int(cfg.get("n_epochs", 8))
    hidden_dim = int(cfg.get("gru_hidden", 128))

    tag = args.pinn_placement if args.pinn_placement != "none" else "nopinn"
    from models.drppo import DRPPO
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    # Determine obs_dim from a probe env
    probe = InteractionEnv(scenario_name="1a", seed=args.seed)
    obs_dim = probe._state_dim
    probe.close()

    policy = DRPPO(
        obs_dim=obs_dim, n_actions=N_ACTIONS,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range,
        pinn_placement=args.pinn_placement,
        hidden_dim=hidden_dim, device=device,
    )

    log_path = os.path.join(args.out_dir, f"curriculum_{tag}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "stage", "scenario", "iteration", "total_env_steps",
            "mean_return", "success_rate", "collision_rate", "row_violation_rate",
        ])

    total_steps_global = 0

    for stage in CURRICULUM_STAGES:
        stage_name = stage["name"]
        scenarios = stage["scenarios"]
        budget = int(stage["steps_per_scenario"] * args.total_multiplier)

        print(f"\n{'='*60}")
        print(f"Curriculum {stage_name}: scenarios {scenarios}, {budget} steps each")
        print(f"{'='*60}")

        for sc in scenarios:
            env = InteractionEnv(scenario_name=sc, seed=args.seed)
            env_steps = 0
            iteration = 0

            while env_steps < budget:
                iteration += 1
                rollout = collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
                obs_arr, act_arr, lp_arr, returns, advantages, extra, hid_arr, infos = rollout
                env_steps += len(obs_arr)
                total_steps_global += len(obs_arr)

                ep_returns, n_eps = [], 0
                cur_ret = 0.0
                for i, info in enumerate(infos):
                    cur_ret += info.get("reward", 0)
                    done = (i < len(infos) - 1 and
                            infos[i+1].get("step", 1) <= info.get("step", 0))
                    if done or i == len(infos) - 1:
                        ep_returns.append(cur_ret)
                        n_eps += 1
                        cur_ret = 0.0

                n_eps = max(n_eps, 1)
                evr, _ = episode_event_rates(infos)
                successes_r, collisions_r, row_viols_r = (
                    evr["success"], evr["collision"], evr["row_violation"])
                N = len(obs_arr)
                indices = np.arange(N)
                for _ in range(n_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, N, batch_size):
                        mb = indices[start:start + batch_size]
                        policy.train_step(
                            obs_arr[mb], act_arr[mb], lp_arr[mb],
                            returns[mb], advantages[mb], hid_arr[mb],
                            extra={k: v[mb] for k, v in extra.items()
                                   if isinstance(v, np.ndarray) and len(v) == N
                                   } if extra else None,
                        )

                mean_ret = np.mean(ep_returns) if ep_returns else 0
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        stage_name, sc, iteration, total_steps_global,
                        f"{mean_ret:.2f}",
                        f"{successes_r:.3f}",
                        f"{collisions_r:.3f}",
                        f"{row_viols_r:.3f}",
                    ])
                print(f"  [{stage_name}/{sc}] iter={iteration} steps={env_steps}/{budget} "
                      f"ret={mean_ret:.1f} succ={successes_r:.2f} coll={collisions_r:.2f}")

            env.close()

        # Save checkpoint after each stage
        ckpt = os.path.join(args.out_dir, f"model_curriculum_{tag}_{stage_name}.pt")
        policy.save(ckpt)
        print(f"  Saved stage checkpoint: {ckpt}")

    # Final checkpoint
    final = os.path.join(args.out_dir, f"model_curriculum_{tag}_final.pt")
    policy.save(final)
    print(f"\nCurriculum training complete. Final model: {final}")


if __name__ == "__main__":
    main()
