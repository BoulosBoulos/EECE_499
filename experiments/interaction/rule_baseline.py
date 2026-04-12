"""Rule-based gap-acceptance policy for the interaction benchmark.

This is a hand-coded baseline that uses explicit gap-acceptance logic.
It should:
  - Perform well on clear-priority templates.
  - Struggle on ambiguous and violation templates.
  - Serve as a sanity check for the benchmark difficulty.
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np

from env.sumo_env_interaction import InteractionEnv, ACTION_NAMES, N_ACTIONS


class RuleBasedPolicy:
    """Gap-acceptance heuristic that reads the conflict-centric state."""

    def __init__(self, safe_gap_s: float = 2.5, creep_dist_m: float = 12.0):
        self.safe_gap_s = safe_gap_s
        self.creep_dist_m = creep_dist_m
        self._yield_ticks = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        """Return action index given the raw state vector and info dict."""
        actors = info.get("actors", [])
        ego_speed = info.get("ego_speed", 0.0)
        phase = info.get("phase", "approach")

        if phase == "clearing":
            self._yield_ticks = 0
            return 3  # GO

        if phase == "committed":
            self._yield_ticks = 0
            return 3  # GO once committed

        # Find the dominant actor (lowest absolute eta, closest to conflict)
        dom_eta = 999.0
        dom_committed = False
        dom_yielding = False
        dom_priority = 0.5
        dom_dist = 999.0

        for ag in actors:
            eta = ag.get("eta_enter", 999)
            if abs(eta) < abs(dom_eta):
                dom_eta = eta
                dom_committed = bool(ag.get("committed", False))
                dom_yielding = bool(ag.get("yielding", False))
                dom_priority = ag.get("legal_priority", 0.5)
                p = ag.get("p", [0, 0])
                dom_dist = float(np.linalg.norm(np.array(p)))

        if not actors:
            self._yield_ticks = 0
            return 3  # GO if no actors

        # Actor is far away (ETA large) -- just go
        if dom_eta > self.safe_gap_s * 2:
            self._yield_ticks = 0
            return 3  # GO

        # Actor has priority and is close
        if dom_priority <= 0.3:
            if dom_committed and dom_dist < 15.0:
                self._yield_ticks += 1
                if self._yield_ticks > 20:
                    return 1  # CREEP if waited too long (actor may have cleared)
                return 0  # STOP for committed priority actor nearby
            if dom_committed and dom_dist >= 15.0:
                self._yield_ticks = 0
                return 3  # GO -- committed actor is far, safe to proceed
            if dom_eta < self.safe_gap_s and not dom_yielding:
                self._yield_ticks += 1
                if self._yield_ticks > 25:
                    return 1  # CREEP after long wait
                return 2  # YIELD
            if dom_yielding:
                self._yield_ticks = 0
                return 3  # GO when actor yields
            self._yield_ticks += 1
            if self._yield_ticks > 30:
                return 1  # CREEP to avoid infinite yielding
            return 2  # YIELD by default when actor has priority

        # Ego has priority or ambiguous
        if dom_committed and dom_eta < 1.5:
            self._yield_ticks = 0
            return 4  # ABORT - actor committed despite ego priority
        if dom_eta > self.safe_gap_s:
            self._yield_ticks = 0
            return 3  # GO - safe gap
        if dom_yielding:
            self._yield_ticks = 0
            return 3  # GO - actor yielding

        # Ambiguous: creep to gather info
        return 1  # CREEP


def run_eval(
    scenario: str,
    episodes: int = 50,
    seed: int = 42,
    use_gui: bool = False,
    out_dir: str = "results/interaction",
) -> dict:
    """Run rule-based policy on the interaction benchmark and return metrics."""
    rng = np.random.RandomState(seed)
    env = InteractionEnv(scenario_name=scenario, use_gui=use_gui, seed=seed)
    policy = RuleBasedPolicy()

    metrics = {
        "success": 0, "collision": 0, "row_violation": 0,
        "conflict_intrusion": 0, "forced_brake": 0, "deadlock": 0,
        "unnecessary_wait_steps": 0, "total_steps": 0,
        "total_return": 0.0, "episode_returns": [],
    }
    logs = []

    try:
        for ep in range(episodes):
            obs, info = env.reset()
            ep_return = 0.0
            for step in range(200):
                action = policy.act(obs, info)
                obs, reward, term, trunc, info = env.step(action)
                ep_return += reward
                events = info.get("events", {})

                logs.append({
                    "episode": ep, "step": step,
                    "action": action, "action_name": ACTION_NAMES[action],
                    "reward": reward, "phase": info.get("phase", ""),
                    "template": info.get("template_family", ""),
                    "ego_speed": info.get("ego_speed", 0),
                    "n_actors": info.get("n_actors", 0),
                    **{f"ev_{k}": int(v) for k, v in events.items()},
                })

                if term or trunc:
                    if events.get("success"):
                        metrics["success"] += 1
                    if events.get("collision"):
                        metrics["collision"] += 1
                    if events.get("row_violation"):
                        metrics["row_violation"] += 1
                    if events.get("deadlock"):
                        metrics["deadlock"] += 1
                    break

            metrics["total_return"] += ep_return
            metrics["episode_returns"].append(ep_return)
            metrics["total_steps"] += step + 1
    finally:
        env.close()

    n = max(episodes, 1)
    metrics["success_rate"] = metrics["success"] / n
    metrics["collision_rate"] = metrics["collision"] / n
    metrics["row_violation_rate"] = metrics["row_violation"] / n
    metrics["deadlock_rate"] = metrics["deadlock"] / n
    metrics["mean_return"] = metrics["total_return"] / n
    metrics["mean_steps"] = metrics["total_steps"] / n

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"rule_baseline_{scenario}_s{seed}.csv")
    if logs:
        keys = sorted(set().union(*(r.keys() for r in logs)))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(logs)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Rule-based baseline for interaction benchmark")
    parser.add_argument("--scenario", default="1a",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out_dir", default="results/interaction")
    args = parser.parse_args()

    metrics = run_eval(
        scenario=args.scenario, episodes=args.episodes,
        seed=args.seed, use_gui=args.gui, out_dir=args.out_dir,
    )
    print(f"\n{'='*50}")
    print(f"Rule Baseline — Scenario {args.scenario}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k not in ("episode_returns",):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
