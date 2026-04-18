"""Visualize PDE-family trained models in SUMO GUI with action overlays."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

from env.sumo_env import SumoEnv, ACTION_NAMES

try:
    import traci
except ImportError:
    traci = None


def _update_visibility_overlay(env, ego_pos, step):
    """Update visibility overlay POIs in SUMO GUI."""
    if traci is None:
        return
    for i in range(20):
        try:
            traci.poi.remove(f"vis_{i}")
        except Exception:
            pass
    cz_center = np.array([0.0, 0.0])
    cz_radius = 12.0
    n_samples = 20
    for i in range(n_samples):
        angle = 2 * np.pi * i / n_samples
        radius = cz_radius * (0.3 + 0.7 * (i % 3) / 2)
        sample = cz_center + radius * np.array([np.cos(angle), np.sin(angle)])
        visible = True
        for occ in env._occlusion_polygons:
            if env._line_intersects_polygon(ego_pos, sample, occ["corners"]):
                visible = False
                break
        color = (0, 200, 0, 180) if visible else (200, 0, 0, 180)
        try:
            traci.poi.add(f"vis_{i}", sample[0], sample[1],
                          color=color, layer=8, imgWidth=1.5, imgHeight=1.5)
        except Exception:
            pass


def run_episode(env, policy_fn, max_steps: int = 500, show_visibility: bool = False):
    obs, info = env.reset()
    log = []
    for step in range(max_steps):
        action = policy_fn(obs)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        ego = info.get("raw_obs", {}).get("ego", {})
        agents = info.get("raw_obs", {}).get("agents", [])
        row = {
            "step": step,
            "ego_x": ego.get("p", [0, 0])[0] if isinstance(ego.get("p"), (list, np.ndarray)) else 0,
            "ego_y": ego.get("p", [0, 0])[1] if isinstance(ego.get("p"), (list, np.ndarray)) else 0,
            "ego_v": ego.get("v", 0),
            "ego_a": ego.get("a", 0),
            "action": action,
            "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "?",
            "reward": reward,
            "ttc_min": info.get("ttc_min", float("nan")),
            "collision": 1 if info.get("collision", False) else 0,
            "nearest_agent_dist": info.get("nearest_agent_dist", float("inf")),
            "n_agents": len(agents),
        }
        for i, ag in enumerate(agents[:5]):
            p = ag.get("p", [0, 0])
            row[f"ag{i}_x"] = p[0] if isinstance(p, (list, np.ndarray)) else 0
            row[f"ag{i}_y"] = p[1] if isinstance(p, (list, np.ndarray)) else 0
            row[f"ag{i}_v"] = ag.get("v", 0)
            row[f"ag{i}_type"] = ag.get("type", "?")
        if show_visibility:
            ego_pos = np.array(ego.get("p", [0, 0]))
            _update_visibility_overlay(env, ego_pos, step)
        log.append(row)
        if done:
            break
    return log


def main():
    parser = argparse.ArgumentParser(description="Visualize PDE-family model in SUMO")
    parser.add_argument("--checkpoint", required=True, help="Path to PDE-family checkpoint")
    parser.add_argument("--method", required=True,
                        choices=["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out_dir", default="results/pde")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4", "2_dense", "3_dense", "4_dense"])
    parser.add_argument("--ego_maneuver", default="stem_right",
                        choices=["stem_right", "stem_left", "right_left",
                                 "right_stem", "left_right", "left_stem"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_buildings", action="store_true",
                        help="Disable static occlusion buildings (full visibility)")
    parser.add_argument("--style_filter", default=None, choices=["nominal", "adversarial"],
                        help="Filter agent behavioral styles for robustness ablation")
    parser.add_argument("--state_ablation", default=None, choices=["no_visibility"],
                        help="State ablation: remove specific feature groups")
    parser.add_argument("--show_visibility", action="store_true",
                        help="Show visibility overlay (green=visible, red=occluded) in GUI")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import torch
    except ImportError:
        torch = None
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    env_for_dim = SumoEnv(scenario_name=args.scenario, ego_maneuver=args.ego_maneuver, use_intent=args.use_intent,
                          buildings=not args.no_buildings, style_filter=args.style_filter,
                          state_ablation=args.state_ablation)
    obs_dim = int(env_for_dim.observation_space.shape[0])

    if args.method == "hjb_aux":
        from models.pde.hjb_aux_agent import HJBAuxAgent
        policy = HJBAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "soft_hjb_aux":
        from models.pde.soft_hjb_aux_agent import SoftHJBAuxAgent
        policy = SoftHJBAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "eikonal_aux":
        from models.pde.eikonal_aux_agent import EikonalAuxAgent
        policy = EikonalAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "cbf_aux":
        from models.pde.cbf_aux_agent import CBFAuxAgent
        policy = CBFAuxAgent(obs_dim=obs_dim, device=device)
    elif args.method == "drppo":
        from models.drppo import DRPPO
        policy = DRPPO(obs_dim=obs_dim, device=device)
    policy.load(args.checkpoint)

    def policy_fn(obs):
        return policy.get_action(obs, deterministic=True)[0]

    env = SumoEnv(use_gui=args.gui, scenario_name=args.scenario, ego_maneuver=args.ego_maneuver, use_intent=args.use_intent,
                  buildings=not args.no_buildings, style_filter=args.style_filter,
                  state_ablation=args.state_ablation)
    try:
        for ep in range(args.episodes):
            policy.reset_hidden()
            log = run_episode(env, policy_fn, show_visibility=args.show_visibility)
            if not log:
                continue
            csv_path = os.path.join(args.out_dir, f"pde_trajectory_{args.method}_{args.ego_maneuver}_ep{ep}.csv")
            all_keys = sorted(set().union(*(set(r.keys()) for r in log)))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(log)
            ret = sum(r["reward"] for r in log)
            colls = sum(r["collision"] for r in log)
            print(f"Ep{ep}: {len(log)} steps, return={ret:.1f}, collisions={colls}, saved {csv_path}")
    finally:
        env.close()
    print("Done.")


if __name__ == "__main__":
    main()
