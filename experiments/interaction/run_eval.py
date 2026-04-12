"""Evaluate trained models on the interaction benchmark with full metrics."""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from env.sumo_env_interaction import InteractionEnv, ACTION_NAMES, N_ACTIONS


def _run_single_episode(env: InteractionEnv, policy_fn, max_steps: int = 200):
    obs, info = env.reset()
    ep_ret = 0.0
    ep_events: dict = {}
    last_t = 0
    for t in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        ep_ret += reward
        ep_events = info.get("events", {}) or {}
        last_t = t
        if term or trunc:
            break
    return ep_ret, last_t + 1, info.get("ego_progress", 0), ep_events, info


METRICS_HEADER = [
    "scenario", "variant", "seed", "eval_mode",
    "mean_return", "std_return",
    "success_rate", "collision_rate", "row_violation_rate",
    "conflict_intrusion_rate", "forced_brake_rate",
    "deadlock_rate", "unnecessary_wait_rate",
    "mean_steps", "mean_progress",
]


def evaluate(
    env: InteractionEnv,
    policy_fn,
    episodes: int = 50,
) -> dict:
    """Run episodes and compute aggregate metrics."""
    returns, steps, progresses = [], [], []
    events_counts = {
        "success": 0, "collision": 0, "row_violation": 0,
        "conflict_intrusion": 0, "forced_other_brake": 0,
        "deadlock": 0, "unnecessary_wait": 0,
    }
    template_results = {}

    for ep in range(episodes):
        ep_ret, nstep, prog, ep_events, info = _run_single_episode(env, policy_fn)
        returns.append(ep_ret)
        steps.append(nstep)
        progresses.append(prog)

        for k in events_counts:
            if ep_events.get(k):
                events_counts[k] += 1

        tfam = info.get("template_family", "unknown")
        if tfam not in template_results:
            template_results[tfam] = {"n": 0, "success": 0, "collision": 0,
                                       "row_violation": 0, "returns": []}
        template_results[tfam]["n"] += 1
        template_results[tfam]["returns"].append(ep_ret)
        if ep_events.get("success"):
            template_results[tfam]["success"] += 1
        if ep_events.get("collision"):
            template_results[tfam]["collision"] += 1
        if ep_events.get("row_violation"):
            template_results[tfam]["row_violation"] += 1

    n = max(episodes, 1)
    metrics = {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "success_rate": events_counts["success"] / n,
        "collision_rate": events_counts["collision"] / n,
        "row_violation_rate": events_counts["row_violation"] / n,
        "conflict_intrusion_rate": events_counts["conflict_intrusion"] / n,
        "forced_brake_rate": events_counts["forced_other_brake"] / n,
        "deadlock_rate": events_counts["deadlock"] / n,
        "unnecessary_wait_rate": events_counts["unnecessary_wait"] / n,
        "mean_steps": np.mean(steps),
        "mean_progress": np.mean(progresses),
        "template_results": template_results,
    }
    return metrics


def evaluate_manifest_rows(
    policy_fn,
    rows: list,
    use_gui: bool = False,
) -> dict:
    """Replay evaluation episodes in manifest order (same RNG stream as generator)."""
    returns, steps, progresses = [], [], []
    events_counts = {
        "success": 0, "collision": 0, "row_violation": 0,
        "conflict_intrusion": 0, "forced_other_brake": 0,
        "deadlock": 0, "unnecessary_wait": 0,
    }
    template_results: dict = {}
    env = None
    prev_key = None

    for row in rows:
        key = (row["scenario"], row["seed"])
        if key != prev_key:
            if env is not None:
                env.close()
            env = InteractionEnv(
                scenario_name=row["scenario"],
                seed=row["seed"],
                use_gui=use_gui,
            )
            prev_key = key

        ep_ret, nstep, prog, ep_events, info = _run_single_episode(env, policy_fn)
        returns.append(ep_ret)
        steps.append(nstep)
        progresses.append(prog)

        for k in events_counts:
            if ep_events.get(k):
                events_counts[k] += 1

        tfam = info.get("template_family", "unknown")
        if tfam not in template_results:
            template_results[tfam] = {"n": 0, "success": 0, "collision": 0,
                                      "row_violation": 0, "returns": []}
        template_results[tfam]["n"] += 1
        template_results[tfam]["returns"].append(ep_ret)
        if ep_events.get("success"):
            template_results[tfam]["success"] += 1
        if ep_events.get("collision"):
            template_results[tfam]["collision"] += 1
        if ep_events.get("row_violation"):
            template_results[tfam]["row_violation"] += 1

    if env is not None:
        env.close()

    n = max(len(rows), 1)
    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "success_rate": events_counts["success"] / n,
        "collision_rate": events_counts["collision"] / n,
        "row_violation_rate": events_counts["row_violation"] / n,
        "conflict_intrusion_rate": events_counts["conflict_intrusion"] / n,
        "forced_brake_rate": events_counts["forced_other_brake"] / n,
        "deadlock_rate": events_counts["deadlock"] / n,
        "unnecessary_wait_rate": events_counts["unnecessary_wait"] / n,
        "mean_steps": np.mean(steps),
        "mean_progress": np.mean(progresses),
        "template_results": template_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on interaction benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", default="1a",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--variant", default="nopinn")
    parser.add_argument("--out_dir", default="results/interaction")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON manifest; evaluates rows matching --scenario in order",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_csv = os.path.join(args.out_dir, "interaction_eval_results.csv")
    write_header = not os.path.isfile(results_csv)

    from models.drppo import DRPPO
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    manifest_rows = None
    if args.manifest:
        from experiments.interaction.eval_manifest import load_manifest

        manifest_rows = [
            r for r in load_manifest(args.manifest)
            if r["scenario"] == args.scenario
        ]
        if not manifest_rows:
            raise SystemExit(
                f"No manifest entries for scenario {args.scenario!r} in {args.manifest}"
            )

    seeds_loop = [manifest_rows[0]["seed"]] if manifest_rows else args.seeds

    for seed in seeds_loop:
        probe = InteractionEnv(scenario_name=args.scenario, seed=seed, use_gui=args.gui)
        obs_dim = probe._state_dim
        probe.close()

        policy_net = DRPPO(obs_dim=obs_dim, n_actions=N_ACTIONS,
                           use_pinn=False, device=device)
        policy_net.load(args.checkpoint)

        if args.deterministic:
            def policy_fn(obs):
                return policy_net.get_action(obs, deterministic=True)[0]
        else:
            def policy_fn(obs):
                return policy_net.get_action(obs)[0]

        mode = "deterministic" if args.deterministic else "stochastic"
        if manifest_rows:
            metrics = evaluate_manifest_rows(
                policy_fn, manifest_rows, use_gui=args.gui,
            )
        else:
            env = InteractionEnv(
                scenario_name=args.scenario, seed=seed, use_gui=args.gui,
            )
            metrics = evaluate(env, policy_fn, episodes=args.episodes)
            env.close()

        row = {
            "scenario": args.scenario, "variant": args.variant,
            "seed": seed, "eval_mode": mode,
        }
        for k in METRICS_HEADER[4:]:
            row[k] = f"{metrics.get(k, 0):.4f}"

        with open(results_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_HEADER, extrasaction="ignore")
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)

        print(f"[{args.variant}] scenario={args.scenario} seed={seed} "
              f"ret={metrics['mean_return']:.1f} "
              f"succ={metrics['success_rate']:.2f} "
              f"coll={metrics['collision_rate']:.2f} "
              f"row={metrics['row_violation_rate']:.2f}")

        # Per-template breakdown
        for tfam, tr in metrics.get("template_results", {}).items():
            n = tr["n"]
            if n > 0:
                print(f"    {tfam}: n={n} succ={tr['success'] / n:.2f} "
                      f"coll={tr['collision'] / n:.2f} "
                      f"ret={np.mean(tr['returns']):.1f}")


if __name__ == "__main__":
    main()
