"""Verify conflict-guaranteed spawning works for all maneuvers.

Runs N episodes per (scenario, maneuver) with a competent-proxy policy.
Reports both near-miss rate (15m) and interaction rate (30m).
Target: interaction rate >= 70% for all combos.

Usage:
    python experiments/pde/verify_conflicts.py --episodes 50
    python experiments/pde/verify_conflicts.py --episodes 30 --scenarios 1a 1b 2 --maneuvers stem_right stem_left
"""

import argparse
import numpy as np

ALL_MANEUVERS = ["stem_right", "stem_left", "right_left",
                 "right_stem", "left_right", "left_stem"]
SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]


def verify(scenario, maneuver, n_episodes,
           near_miss_thr=15.0, interaction_thr=30.0):
    """Verify spawning produces both near-misses and interactions.

    Returns (near_miss_rate, interaction_rate).
    """
    from env.sumo_env import SumoEnv, N_ACTIONS
    env = SumoEnv(scenario_name=scenario, ego_maneuver=maneuver)
    near_miss_count = 0
    interaction_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        had_near_miss = False
        had_interaction = False
        for _ in range(300):
            # Competent-proxy policy: mostly GO with some CREEP
            action = np.random.choice(N_ACTIONS, p=[0.05, 0.15, 0.10, 0.65, 0.05])
            obs, r, term, trunc, info = env.step(action)
            nearest = info.get("nearest_agent_dist", float("inf"))
            if nearest < near_miss_thr:
                had_near_miss = True
            if nearest < interaction_thr:
                had_interaction = True
            if term or trunc:
                break
        if had_near_miss:
            near_miss_count += 1
        if had_interaction:
            interaction_count += 1

    env.close()
    return (near_miss_count / n_episodes, interaction_count / n_episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=["1a", "1b", "2"])
    parser.add_argument("--maneuvers", nargs="+", default=ALL_MANEUVERS)
    args = parser.parse_args()

    print(f"{'Scenario':<10} {'Maneuver':<15} {'Near-miss':>12} {'Interaction':>14} {'Status':>8}")
    print("-" * 62)

    all_pass = True
    for scen in args.scenarios:
        for man in args.maneuvers:
            near_miss, interaction = verify(scen, man, args.episodes)
            status = "OK" if interaction >= 0.70 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"{scen:<10} {man:<15} {near_miss:>11.1%} {interaction:>13.1%} {status:>8}")

    print()
    print(f"{'ALL PASS' if all_pass else 'SOME FAILED'} (target: interaction rate >= 70%)")
    print("Near-miss column is diagnostic; interaction is the paper-aligned metric.")


if __name__ == "__main__":
    main()
