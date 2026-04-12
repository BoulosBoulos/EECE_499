"""Train DRPPO variants on the interaction benchmark."""

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

from env.sumo_env_interaction import InteractionEnv, ACTION_NAMES, N_ACTIONS
from experiments.interaction.metrics_util import episode_event_rates
from state.interaction_xi import xi_from_interaction_obs


def _load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    """Collect rollouts with PINN-aligned extra dict for physics-informed training."""
    obs, info = env.reset()
    policy.reset_hidden()

    obs_list, actions_list, rewards_list, dones_list = [], [], [], []
    log_probs_list, values_list, hidden_list = [], [], []
    infos_list = []
    terminated_list = []
    pre_step_infos = []
    post_step_infos = []
    collision_events = []
    success_terminal_list = []
    collision_terminal_list = []

    pre_step_info = info

    for _ in range(n_steps):
        pre_step_infos.append(pre_step_info)

        h = policy._hidden
        if h is not None:
            hidden_list.append(h.detach().cpu().squeeze(1).numpy())
        else:
            hidden_list.append(np.zeros((1, policy.policy.hidden_dim), dtype=np.float32))

        action, _, log_prob, value = policy.get_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        obs_list.append(obs)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        terminated_list.append(term)
        log_probs_list.append(log_prob)
        values_list.append(value)
        infos_list.append(info)
        post_step_infos.append(info)
        ev_post = info.get("events", {}) or {}
        collision_events.append(bool(ev_post.get("collision")))
        success_terminal_list.append(
            bool(term and ev_post.get("success") and not ev_post.get("collision"))
        )
        collision_terminal_list.append(bool(term and ev_post.get("collision")))

        obs = next_obs
        pre_step_info = info
        if done:
            obs, info = env.reset()
            pre_step_info = info
            policy.reset_hidden()

    obs_arr = np.array(obs_list, dtype=np.float32)
    actions_arr = np.array(actions_list, dtype=np.int64)
    rewards_arr = np.array(rewards_list, dtype=np.float32)
    dones_arr = np.array(dones_list, dtype=np.float32)
    terminated_arr = np.array(terminated_list, dtype=np.float32)
    log_probs_arr = np.array(log_probs_list, dtype=np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    hidden_arr = np.array(hidden_list, dtype=np.float32)

    advantages = np.zeros_like(rewards_arr)
    last_gae = 0.0
    for t in reversed(range(len(rewards_arr))):
        if t == len(rewards_arr) - 1 or terminated_arr[t]:
            next_val = 0.0
            next_nt = 0.0
        else:
            next_val = values_arr[t + 1]
            next_nt = 1.0
        delta = rewards_arr[t] + gamma * next_val - values_arr[t]
        if dones_arr[t]:
            last_gae = 0.0
        last_gae = delta + gamma * gae_lambda * next_nt * last_gae
        advantages[t] = last_gae

    returns = advantages + values_arr
    std = advantages.std()
    if std > 1e-8:
        advantages = (advantages - advantages.mean()) / std

    # Build PINN-aligned extra dict from pre-step info
    N = len(pre_step_infos)
    extra = None
    if N > 0:
        ego_speeds = np.zeros(N, dtype=np.float32)
        ego_accels = np.zeros(N, dtype=np.float32)
        d_conflict = np.zeros(N, dtype=np.float32)
        dom_etas = np.zeros(N, dtype=np.float32)

        for i, psi in enumerate(pre_step_infos):
            ego_speeds[i] = float(psi.get("ego_speed", 0.0))
            ego_accels[i] = float(psi.get("ego_accel", 0.0))
            d_conflict[i] = float(psi.get("d_conflict_entry", 0.0))
            actors = psi.get("actors", [])
            if actors:
                dom = min(actors, key=lambda a: abs(a.get("eta_enter", 99)))
                dom_etas[i] = dom.get("eta_enter", 10.0)

        # Approximate physics quantities for PINN compatibility
        ttc_min = np.clip(dom_etas, 0.1, 10.0)
        xi_stack = np.stack([xi_from_interaction_obs(o) for o in obs_list], axis=0).astype(
            np.float32
        )
        extra = {
            "v": ego_speeds,
            "a_lon": ego_accels,
            "d_cz": d_conflict,
            "ttc_min": ttc_min,
            "kappa": np.zeros(N, dtype=np.float32),
            "collision_events": np.array(collision_events, dtype=bool),
            "xi_curr": xi_stack,
            "success_terminal": np.array(success_terminal_list, dtype=bool),
            "collision_terminal": np.array(collision_terminal_list, dtype=bool),
        }

    return obs_arr, actions_arr, log_probs_arr, returns, advantages, extra, hidden_arr, infos_list


def main():
    parser = argparse.ArgumentParser(description="Train on interaction benchmark")
    parser.add_argument("--config", default="configs/algo/default.yaml")
    parser.add_argument("--scenario", default="1a",
                        choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--out_dir", default="results/interaction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pinn_placement", default="none",
                        choices=["critic", "actor", "both", "none"])
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    if torch:
        torch.manual_seed(args.seed)

    cfg = _load_yaml(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    env = InteractionEnv(
        scenario_name=args.scenario, use_gui=args.gui, seed=args.seed,
    )
    obs_dim = env._state_dim
    n_actions = N_ACTIONS

    lr = float(cfg.get("lr", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_range = float(cfg.get("clip_range", 0.2))
    n_steps = int(cfg.get("n_steps", 2048))
    batch_size = int(cfg.get("batch_size", 128))
    n_epochs = int(cfg.get("n_epochs", 8))
    hidden_dim = int(cfg.get("gru_hidden", 128))

    from models.drppo import DRPPO
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    policy = DRPPO(
        obs_dim=obs_dim, n_actions=n_actions,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda,
        clip_range=clip_range,
        pinn_placement=args.pinn_placement,
        hidden_dim=hidden_dim, device=device,
    )

    tag = args.pinn_placement if args.pinn_placement != "none" else "nopinn"
    log_path = os.path.join(args.out_dir, f"train_interaction_{tag}_{args.scenario}.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "iteration", "total_env_steps", "mean_return", "mean_ep_len",
            "collision_rate", "success_rate", "row_violation_rate",
            "actor_loss", "vf_loss", "entropy",
        ])

    total_env_steps = 0
    iteration = 0

    while total_env_steps < args.total_steps:
        iteration += 1
        rollout = collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
        obs_arr, actions_arr, lp_arr, returns, advantages, extra, hidden_arr, infos = rollout
        total_env_steps += len(obs_arr)

        ep_returns, ep_lens = [], []
        current_return, current_len = 0.0, 0
        for i, info in enumerate(infos):
            current_return += info.get("reward", 0)
            current_len += 1
            done = (i < len(infos) - 1 and infos[i + 1].get("step", 1) <= info.get("step", 0))
            if done or i == len(infos) - 1:
                ep_returns.append(current_return)
                ep_lens.append(current_len)
                current_return, current_len = 0.0, 0

        ev_rates, _n_eps_ev = episode_event_rates(infos)
        collisions_r = ev_rates["collision"]
        successes_r = ev_rates["success"]
        row_violations_r = ev_rates["row_violation"]

        n_eps = max(len(ep_returns), 1)
        mean_ret = np.mean(ep_returns) if ep_returns else 0
        mean_len = np.mean(ep_lens) if ep_lens else 0

        N = len(obs_arr)
        indices = np.arange(N)
        stats_agg = {"actor_loss": 0, "vf_loss": 0, "entropy": 0}
        n_updates = 0
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                end = start + batch_size
                mb = indices[start:end]
                ex = None
                if extra is not None:
                    ex = {
                        k: (v[mb] if isinstance(v, np.ndarray) and len(v) == N else v)
                        for k, v in extra.items()
                    }
                stats = policy.train_step(
                    obs_arr[mb], actions_arr[mb], lp_arr[mb],
                    returns[mb], advantages[mb], hidden_arr[mb],
                    extra=ex,
                )
                for k in stats_agg:
                    stats_agg[k] += stats.get(k, 0)
                n_updates += 1

        for k in stats_agg:
            stats_agg[k] /= max(n_updates, 1)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                iteration, total_env_steps, f"{mean_ret:.2f}", f"{mean_len:.1f}",
                f"{collisions_r:.3f}", f"{successes_r:.3f}",
                f"{row_violations_r:.3f}",
                f"{stats_agg['actor_loss']:.4f}",
                f"{stats_agg['vf_loss']:.4f}",
                f"{stats_agg['entropy']:.4f}",
            ])
        print(f"[iter {iteration}] steps={total_env_steps} ret={mean_ret:.1f} "
              f"succ={successes_r:.2f} coll={collisions_r:.2f} "
              f"row_viol={row_violations_r:.2f}")

    ckpt = os.path.join(args.out_dir, f"model_interaction_{tag}_{args.scenario}.pt")
    policy.save(ckpt)
    print(f"Saved checkpoint: {ckpt}")
    env.close()


if __name__ == "__main__":
    main()
