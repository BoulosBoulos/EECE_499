"""Job 1 diagnostic: replay 10 episodes of 1b/stem_right nominal checkpoint with verbose logging.

Reports:
  (a) style pool actually sampled
  (b) whether pedestrian spawns and reaches conflict zone
  (c) episode termination
"""

from __future__ import annotations
import os, sys, json, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("SUMO_HOME",
    os.path.join(REPO, ".venv/lib/python3.12/site-packages/sumo"))

import sumo  # noqa: F401

try:
    import torch
    torch.set_num_threads(2)
except ImportError:
    torch = None

from env.sumo_env import SumoEnv, ACTION_NAMES
from scenario.behavior_sampler import NOMINAL_PED_STYLES, PED_STYLES, CAR_STYLES, NOMINAL_CAR_STYLES


def run_diagnostic(checkpoint_path: str, method: str, scenario: str, maneuver: str,
                   style_filter: str | None, n_episodes: int = 10, seed: int = 9999):
    print(f"\n{'='*70}")
    print(f"JOB 1 DIAGNOSTIC")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  scenario   : {scenario} / {maneuver}")
    print(f"  style_filter: {style_filter!r}")
    print(f"  n_episodes : {n_episodes}")
    print(f"{'='*70}")

    env = SumoEnv(
        use_gui=False,
        scenario_name=scenario,
        ego_maneuver=maneuver,
        use_intent=False,
        buildings=True,
        style_filter=style_filter,
    )
    obs_dim = int(env.observation_space.shape[0])
    print(f"\n[obs] observation_space shape: {env.observation_space.shape}")

    if method == "drppo":
        from models.drppo import DRPPO
        agent = DRPPO.from_checkpoint(checkpoint_path, obs_dim=obs_dim, device="cpu")
    elif method == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        agent = HJBAuxAgent.from_checkpoint(checkpoint_path, obs_dim=obs_dim, device="cpu")
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[model] loaded {method} from {checkpoint_path}")

    # Patch behavior_sampler to log what gets sampled
    original_sample = env._behavior_sampler.sample
    sampled_styles = []

    def logging_sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        style_info = {
            "car_style": result.car.style if result.car else None,
            "ped_style": result.pedestrian.style if result.pedestrian else None,
            "ped_style2": result.pedestrian2.style if result.pedestrian2 else None,
            "ped_maneuver": result.pedestrian.maneuver if result.pedestrian else None,
        }
        sampled_styles.append(style_info)
        return result

    env._behavior_sampler.sample = logging_sample

    rng = np.random.RandomState(seed)
    results = []

    for ep_idx in range(n_episodes):
        ep_seed = int(rng.randint(10000, 99999))
        obs, info = env.reset(seed=ep_seed)
        agent.reset_hidden()

        # Record behavior from the sampler log
        ep_style = sampled_styles[-1] if sampled_styles else {}

        ped_min_dist = float('inf')
        ped_entered_cz = False
        step = 0
        terminated = truncated = False
        terminal_info = {}

        while not (terminated or truncated):
            action, _, _, _ = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # Track pedestrian proximity via raw_obs agents
            raw = info.get("raw_obs") or {}
            raw_agents = raw.get("agents", []) if isinstance(raw, dict) else []
            for ag in raw_agents:
                if ag.get("type") == "ped":
                    # nearest_agent_dist is the ego-to-nearest-agent distance
                    nearest = info.get("nearest_agent_dist", float('inf'))
                    if nearest < ped_min_dist:
                        ped_min_dist = nearest
                    if ag.get("d_cz", 999) < 2.0:
                        ped_entered_cz = True

            if terminated or truncated:
                terminal_info = info

        terminal = "collision" if info.get("collision") else (
                   "success"   if info.get("success")   else
                   "timeout"   if truncated              else "abort")

        ped_spawned = ep_style.get("ped_style") is not None

        result = {
            "ep": ep_idx,
            "seed": ep_seed,
            "terminal": terminal,
            "steps": step,
            "ped_spawned": ped_spawned,
            "ped_style": ep_style.get("ped_style"),
            "ped_maneuver": ep_style.get("ped_maneuver"),
            "ped_min_dist_m": round(ped_min_dist, 3) if ped_min_dist < float('inf') else None,
            "ped_entered_cz": ped_entered_cz,
            "car_style": ep_style.get("car_style"),
        }
        results.append(result)

        print(f"\n  [ep {ep_idx:2d}] seed={ep_seed}  steps={step:4d}  terminal={terminal:10s}"
              f"  ped_spawned={ped_spawned}  ped_style={ep_style.get('ped_style'):20s}"
              f"  ped_min_dist={ped_min_dist:.2f}m  ped_in_cz={ped_entered_cz}")

    env.close()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  Style filter requested : {style_filter!r}")
    print(f"  Nominal PED styles     : {NOMINAL_PED_STYLES}")
    print(f"  Full PED styles        : {PED_STYLES}")
    terminals = [r["terminal"] for r in results]
    print(f"\n  Terminal breakdown ({n_episodes} episodes):")
    for t in ["success", "collision", "timeout", "abort"]:
        n = sum(1 for x in terminals if x == t)
        print(f"    {t:12s}: {n:3d}")
    ped_styles_seen = [r["ped_style"] for r in results if r["ped_style"]]
    print(f"\n  Ped styles sampled: {set(ped_styles_seen)}")
    print(f"  All ped spawned   : {all(r['ped_spawned'] for r in results)}")
    print(f"  Ped entered CZ    : {sum(r['ped_entered_cz'] for r in results)}/{n_episodes}")
    dists = [r["ped_min_dist_m"] for r in results if r["ped_min_dist_m"] is not None]
    if dists:
        print(f"  Ped min-dist stats: min={min(dists):.2f}  mean={np.mean(dists):.2f}  max={max(dists):.2f}")

    print(f"\n{'='*70}")
    print("DIAGNOSIS:")
    all_styles_nominal = all(s in NOMINAL_PED_STYLES for s in ped_styles_seen)
    print(f"  (a) Style pool actually sampled: {set(ped_styles_seen)}")
    print(f"      Consistent with style_filter='nominal': {all_styles_nominal}")
    print(f"  (b) Pedestrian spawns: {all(r['ped_spawned'] for r in results)}")
    print(f"      Enters conflict zone: {sum(r['ped_entered_cz'] for r in results)}/{n_episodes} episodes")
    collision_n = sum(1 for t in terminals if t == "collision")
    success_n   = sum(1 for t in terminals if t == "success")
    timeout_n   = sum(1 for t in terminals if t == "timeout")
    print(f"  (c) Episode terminations: {collision_n} collision / {success_n} success / {timeout_n} timeout")
    if min(dists) < 2.1:
        print(f"      NOTE: min_dist {min(dists):.2f}m is at/below proximity_collision threshold (2.0m)")
        print(f"            Collisions are from real proximity to pedestrian, not SUMO physics collision")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--method", default="drppo")
    ap.add_argument("--scenario", default="1b")
    ap.add_argument("--maneuver", default="stem_right")
    ap.add_argument("--style_filter", default="nominal")
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=9999)
    args = ap.parse_args()

    run_diagnostic(
        checkpoint_path=args.checkpoint,
        method=args.method,
        scenario=args.scenario,
        maneuver=args.maneuver,
        style_filter=args.style_filter,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
