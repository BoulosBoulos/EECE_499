"""Visualize the interaction benchmark with SUMO GUI.

Shows template, dominant actor, ETA gaps, phase, actions in the console
while running the GUI.
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from env.sumo_env_interaction import InteractionEnv, ACTION_NAMES, N_ACTIONS


def run_episode(env, policy_fn, max_steps: int = 120):
    obs, info = env.reset()
    template = info.get("template_family", "?")
    print(f"  Template: {template}")
    log = []

    for step in range(max_steps):
        action = policy_fn(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        events = info.get("events", {})

        row = {
            "step": step,
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < N_ACTIONS else "?",
            "reward": round(reward, 3),
            "phase": info.get("phase", ""),
            "template": template,
            "ego_speed": round(info.get("ego_speed", 0), 2),
            "n_actors": info.get("n_actors", 0),
            "sim_time": round(info.get("sim_time", 0), 2),
        }
        for k, v in events.items():
            row[f"ev_{k}"] = int(v)

        actors = info.get("actors", [])
        for i, ag in enumerate(actors[:3]):
            p = ag.get("p", [0, 0])
            row[f"ag{i}_type"] = ag.get("actor_type", "?")
            row[f"ag{i}_x"] = round(float(p[0]), 1) if hasattr(p, '__len__') else 0
            row[f"ag{i}_y"] = round(float(p[1]), 1) if hasattr(p, '__len__') else 0
            row[f"ag{i}_v"] = round(ag.get("v", 0), 2)
            row[f"ag{i}_eta"] = round(ag.get("eta_enter", 99), 2)
            row[f"ag{i}_committed"] = int(ag.get("committed", 0))
            row[f"ag{i}_yielding"] = int(ag.get("yielding", 0))

        log.append(row)

        active_events = [k for k, v in events.items() if v]
        ev_str = f" [{', '.join(active_events)}]" if active_events else ""
        print(f"    t={step:3d} {ACTION_NAMES[action]:6s} v={row['ego_speed']:5.2f} "
              f"phase={row['phase']:10s}{ev_str}")

        if term or trunc:
            outcome = "SUCCESS" if events.get("success") else (
                "COLLISION" if events.get("collision") else (
                    "DEADLOCK" if events.get("deadlock") else "TRUNCATED"))
            print(f"  >> {outcome} after {step + 1} steps, return={sum(r['reward'] for r in log):.1f}")
            break

    return log


def main():
    parser = argparse.ArgumentParser(description="Visualize interaction benchmark")
    parser.add_argument("--scenario", default="1a",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--policy", choices=["rule", "go", "random", "checkpoint"],
                        default="rule")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/interaction")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    env = InteractionEnv(
        scenario_name=args.scenario, use_gui=args.gui, seed=args.seed,
    )

    if args.policy == "checkpoint" and args.checkpoint:
        from models.drppo import DRPPO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        obs_dim = env._state_dim
        policy_net = DRPPO(obs_dim=obs_dim, n_actions=N_ACTIONS,
                           use_pinn=False, device=device)
        policy_net.load(args.checkpoint)

        def policy_fn(obs, info):
            return policy_net.get_action(obs, deterministic=True)[0]
    elif args.policy == "rule":
        from experiments.interaction.rule_baseline import RuleBasedPolicy
        rule = RuleBasedPolicy()

        def policy_fn(obs, info):
            return rule.act(obs, info)
    elif args.policy == "go":
        def policy_fn(obs, info):
            return 3
    else:
        def policy_fn(obs, info):
            return np.random.randint(0, N_ACTIONS)

    try:
        for ep in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {ep} — Scenario {args.scenario}")
            print(f"{'='*60}")
            log = run_episode(env, policy_fn)

            if log:
                csv_path = os.path.join(
                    args.out_dir,
                    f"interaction_viz_{args.scenario}_ep{ep}.csv",
                )
                keys = sorted(set().union(*(r.keys() for r in log)))
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(log)
    finally:
        env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
