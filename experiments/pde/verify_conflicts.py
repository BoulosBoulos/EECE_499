"""Verify conflict-guaranteed spawning works for all maneuvers.

Runs N episodes per (scenario, maneuver) with a random policy.
Reports the fraction of episodes where an agent comes within 15m of ego.
Target: >= 70% for all combos.

Usage:
    python experiments/pde/verify_conflicts.py --episodes 50
    python experiments/pde/verify_conflicts.py --episodes 20 --scenarios 1a 2 --maneuvers stem_right stem_left
"""

import argparse
import numpy as np

ALL_MANEUVERS = ["stem_right", "stem_left", "right_left",
                 "right_stem", "left_right", "left_stem"]
SCENARIOS = ["1a", "1b", "1c", "1d", "2", "3", "4"]


def verify(scenario, maneuver, n_episodes, proximity_thr=15.0):
    from env.sumo_env import SumoEnv, N_ACTIONS
    env = SumoEnv(scenario_name=scenario, ego_maneuver=maneuver)
    conflict_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        had_conflict = False
        for _ in range(300):
            action = np.random.randint(N_ACTIONS)
            obs, r, term, trunc, info = env.step(action)
            nearest = info.get("nearest_agent_dist", float("inf"))
            if nearest < proximity_thr:
                had_conflict = True
            if term or trunc:
                break
        if had_conflict:
            conflict_count += 1

    env.close()
    rate = conflict_count / n_episodes
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--scenarios", nargs="+", default=["1a", "1b", "2"])
    parser.add_argument("--maneuvers", nargs="+", default=ALL_MANEUVERS)
    args = parser.parse_args()

    print(f"{'Scenario':<10} {'Maneuver':<15} {'Conflict Rate':>15} {'Status':>8}")
    print("-" * 50)

    all_pass = True
    for scen in args.scenarios:
        for man in args.maneuvers:
            rate = verify(scen, man, args.episodes)
            status = "OK" if rate >= 0.70 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"{scen:<10} {man:<15} {rate:>14.1%} {status:>8}")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED -- adjust spawn timing'}")


if __name__ == "__main__":
    main()
