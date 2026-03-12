"""Train PINN-augmented vs traditional PPO with proper recurrence and collision tracking.
See docs/RUNNING.md and docs/FRAMEWORK.md."""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    """Collect trajectories with correct timestep alignment for PINN.

    Key fix: physics quantities (d_cz, v, kappa, a_lon, ttc_min) are stored
    from the PRE-STEP state, aligned with obs_list[t] = s_t.
    Previously they came from env.step() info (= s_{t+1}), causing misalignment.
    """
    obs, info = env.reset()
    policy.reset_hidden()
    obs_list, actions_list, rewards_list, dones_list = [], [], [], []
    log_probs_list, values_list = [], []
    hidden_list = []
    pre_step_infos = []
    post_step_infos = []
    collision_events = []
    behavior_labels = []

    pre_step_info = info
    terminated_list = []
    truncated_bootstrap = {}

    for step_i in range(n_steps):
        pre_step_infos.append(pre_step_info)

        h = policy._hidden
        if h is not None:
            hidden_list.append(h.detach().cpu().squeeze(1).numpy())
        else:
            hidden_list.append(np.zeros((1, policy.policy.hidden_dim), dtype=np.float32))

        action, new_hidden, log_prob, value = policy.get_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        obs_list.append(obs)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        terminated_list.append(term)
        log_probs_list.append(log_prob)
        values_list.append(value)
        post_step_infos.append(info)
        collision_events.append(info.get("collision", False))

        if trunc and not term:
            with torch.no_grad():
                o_t = torch.FloatTensor(next_obs).unsqueeze(0).unsqueeze(0)
                if torch is not None:
                    o_t = o_t.to(policy.device if hasattr(policy, 'device') else 'cpu')
                truncated_bootstrap[step_i] = policy.policy.get_value(o_t, policy._hidden).item()

        obs = next_obs
        pre_step_info = info
        if done:
            bcfg = info.get("behavior")
            if bcfg:
                behavior_labels.append({
                    "car_intent": bcfg.car_intent_label if bcfg.car else -1,
                    "car_style": bcfg.car_style_label if bcfg.car else -1,
                    "ped_intent": bcfg.ped_intent_label if bcfg.pedestrian else -1,
                    "ped_style": bcfg.ped_style_label if bcfg.pedestrian else -1,
                    "moto_intent": bcfg.moto_intent_label if bcfg.motorcycle else -1,
                    "moto_style": bcfg.moto_style_label if bcfg.motorcycle else -1,
                })
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

    # Bootstrap for the final step if rollout ends mid-episode
    last_idx = len(dones_arr) - 1
    if last_idx >= 0 and not dones_arr[last_idx]:
        with torch.no_grad():
            o_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
            if torch is not None:
                o_t = o_t.to(policy.device if hasattr(policy, 'device') else 'cpu')
            truncated_bootstrap[last_idx] = policy.policy.get_value(o_t, policy._hidden).item()

    # GAE with correct handling of all episode boundaries:
    #   terminated step: future value = 0 (true terminal)
    #   truncated step:  future value = V(s') from truncated_bootstrap
    #   mid-episode:     future value = values_arr[t+1]
    #   rollout end:     future value = V(s_T) from truncated_bootstrap (if not done)
    advantages = np.zeros_like(rewards_arr, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards_arr))):
        if t == len(rewards_arr) - 1:
            if t in truncated_bootstrap:
                next_val = truncated_bootstrap[t]
                next_non_term = 1.0
            elif terminated_arr[t]:
                next_val = 0.0
                next_non_term = 0.0
            else:
                next_val = 0.0
                next_non_term = 0.0
        else:
            if t in truncated_bootstrap:
                next_val = truncated_bootstrap[t]
                next_non_term = 1.0
            elif terminated_arr[t]:
                next_val = 0.0
                next_non_term = 0.0
            else:
                next_val = values_arr[t + 1]
                next_non_term = 1.0

        delta = rewards_arr[t] + gamma * next_val - values_arr[t]
        if dones_arr[t]:
            last_gae = 0.0
        last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
        advantages[t] = last_gae
    returns = advantages + values_arr
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - advantages.mean()) / adv_std

    # Build PINN extra from PRE-STEP info (aligned with obs_list[t] = s_t)
    extra = None
    N = len(pre_step_infos)
    if N > 0 and "raw_obs" in pre_step_infos[0]:
        built = [r.get("built", {}) for r in pre_step_infos]
        d_cz = [b.get("s_geom", np.zeros(12))[1] for b in built]
        v = [b.get("s_ego", np.zeros(6))[0] for b in built]
        ttc_min = [r.get("ttc_min", 10.0) for r in pre_step_infos]
        ego = [r.get("raw_obs", {}).get("ego", {}) for r in pre_step_infos]
        kappa = [r.get("raw_obs", {}).get("geom", {}).get("kappa", 0.0) for r in pre_step_infos]
        a_lon = [e.get("a", 0.0) for e in ego]
        extra = {
            "d_cz": np.array(d_cz, dtype=np.float32),
            "v": np.array(v, dtype=np.float32),
            "ttc_min": np.array(ttc_min, dtype=np.float32),
            "kappa": np.array(kappa, dtype=np.float32),
            "a_lon": np.array(a_lon, dtype=np.float32),
            "collision_events": np.array(collision_events, dtype=bool),
        }

        # L_ego: transition (s_t, a_t) -> s_{t+1}
        # pre_step_infos[t] has s_t, post_step_infos[t] has s_{t+1}
        ego_x_prev = np.zeros(N, dtype=np.float32)
        ego_y_prev = np.zeros(N, dtype=np.float32)
        ego_psi_prev = np.zeros(N, dtype=np.float32)
        ego_v_prev = np.zeros(N, dtype=np.float32)
        ego_x_next = np.zeros(N, dtype=np.float32)
        ego_y_next = np.zeros(N, dtype=np.float32)
        ego_psi_next = np.zeros(N, dtype=np.float32)
        ego_v_next = np.zeros(N, dtype=np.float32)
        ego_action_prev = np.zeros(N, dtype=np.int64)
        ego_valid = np.zeros(N, dtype=bool)
        for i in range(N - 1):
            if dones_arr[i]:
                continue
            e_curr = pre_step_infos[i].get("raw_obs", {}).get("ego", {})
            e_next = post_step_infos[i].get("raw_obs", {}).get("ego", {})
            p_curr = np.array(e_curr.get("p", [0, 0]), dtype=np.float32)
            p_next = np.array(e_next.get("p", [0, 0]), dtype=np.float32)
            ego_x_prev[i], ego_y_prev[i] = p_curr[0], p_curr[1]
            ego_psi_prev[i] = float(e_curr.get("psi", 0))
            ego_v_prev[i] = float(e_curr.get("v", 0))
            ego_x_next[i], ego_y_next[i] = p_next[0], p_next[1]
            ego_psi_next[i] = float(e_next.get("psi", 0))
            ego_v_next[i] = float(e_next.get("v", 0))
            ego_action_prev[i] = actions_arr[i]
            ego_valid[i] = True
        extra.update({
            "ego_x_prev": ego_x_prev, "ego_y_prev": ego_y_prev,
            "ego_psi_prev": ego_psi_prev, "ego_v_prev": ego_v_prev,
            "ego_x_next": ego_x_next, "ego_y_next": ego_y_next,
            "ego_psi_next": ego_psi_next, "ego_v_next": ego_v_next,
            "ego_action_prev": ego_action_prev, "ego_valid": ego_valid,
        })
        if behavior_labels:
            extra["behavior_labels"] = behavior_labels

    return obs_arr, actions_arr, log_probs_arr, returns, advantages, extra, hidden_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/algo/default.yaml")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--sumo_gui", action="store_true")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--use_intent", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pinn_placement", default=None, choices=["critic", "actor", "both", "none"],
                        help="Override PINN placement (default: critic for pinn, none for nopinn)")
    parser.add_argument("--use_safety_filter", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        if torch:
            torch.manual_seed(args.seed)

    cfg = _load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    from env.sumo_env import SumoEnv
    env_kwargs = {"use_gui": args.sumo_gui, "scenario_name": args.scenario, "use_intent": args.use_intent}
    obs_dim = int(SumoEnv(**env_kwargs).observation_space.shape[0])
    n_actions = 5
    lr = float(cfg.get("lr", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_range = float(cfg.get("clip_range", 0.2))
    n_steps = int(cfg.get("n_steps", 4096))
    batch_size = int(cfg.get("batch_size", 128))
    n_epochs = int(cfg.get("n_epochs", 8))
    hidden_dim = int(cfg.get("gru_hidden", 128))

    res_cfg = _load_config("configs/residuals/default.yaml")

    variants = []
    if args.pinn_placement is not None:
        placement = args.pinn_placement
        tag = placement if placement != "none" else "nopinn"
        variants = [(tag, placement)]
    elif args.no_compare:
        placement = "critic" if cfg.get("use_pinn", True) else "none"
        tag = "pinn" if placement != "none" else "nopinn"
        variants = [(tag, placement)]
    else:
        variants = [("pinn", "critic"), ("nopinn", "none")]

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    for name, placement in variants:
        env = SumoEnv(**env_kwargs)
        from models.drppo import DRPPO
        policy = DRPPO(
            obs_dim=obs_dim, n_actions=n_actions,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range,
            pinn_placement=placement,
            use_safety_filter=args.use_safety_filter,
            lambda_physics_critic=float(res_cfg.get("lambda_physics_critic", 0.5)),
            lambda_physics_actor=float(res_cfg.get("lambda_physics_actor", 0.1)),
            lambda_physics_ttc=float(res_cfg.get("lambda_physics_ttc", 1.0)),
            lambda_physics_stop=float(res_cfg.get("lambda_physics_stop", 1.0)),
            lambda_physics_fric=float(res_cfg.get("lambda_physics_fric", 1.0)),
            physics_ttc_thr=float(res_cfg.get("physics_ttc_thr", 3.0)),
            physics_tau=float(res_cfg.get("physics_tau", 0.5)),
            a_max=float(res_cfg.get("a_max", 5.0)),
            mu=float(res_cfg.get("mu", 0.8)),
            g=float(res_cfg.get("g", 9.81)),
            hidden_dim=hidden_dim,
            device=device,
            dt=float(res_cfg.get("dt", 0.1)),
            a_go=float(res_cfg.get("a_go", 2.0)),
        )

        csv_path = os.path.join(args.out_dir, f"train_{name}_{args.scenario}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
                "collision_count", "collision_rate", "mean_ttc", "min_ttc", "entropy",
                "l_physics", "l_actor_physics", "l_ego",
                "viol_ttc_rate", "viol_stop_rate", "viol_fric_rate",
                "viol_ttc_mag", "viol_stop_mag", "viol_fric_mag",
            ])

        step = 0
        ep_returns = []

        while step < args.total_steps:
            obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra, hidden_arr = \
                collect_rollouts(env, policy, n_steps, gamma, gae_lambda)
            step += n_steps

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

            rewards = []
            ttc_list = []
            coll_count = 0
            obs, _ = env.reset()
            policy.reset_hidden()
            for _ in range(500):
                action, _, _, _ = policy.get_action(obs)
                obs, r, term, trunc, info = env.step(action)
                rewards.append(r)
                ttc_list.append(info.get("ttc_min", 10.0))
                if info.get("collision", False):
                    coll_count += 1
                if term or trunc:
                    break
            ep_returns.append(sum(rewards))
            ttc_arr = np.array(ttc_list)
            mean_ttc = float(np.mean(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
            min_ttc = float(np.min(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
            coll_rate = coll_count / max(len(rewards), 1)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step, np.mean(ep_returns[-10:]) if ep_returns else 0,
                    len(rewards), metrics.get("actor_loss", 0), metrics.get("vf_loss", 0),
                    coll_count, coll_rate, mean_ttc, min_ttc, metrics.get("entropy", 0),
                    metrics.get("l_physics", 0), metrics.get("l_actor_physics", 0),
                    metrics.get("l_ego", 0),
                    metrics.get("viol_ttc_rate", 0), metrics.get("viol_stop_rate", 0),
                    metrics.get("viol_fric_rate", 0),
                    metrics.get("viol_ttc_mag", 0), metrics.get("viol_stop_mag", 0),
                    metrics.get("viol_fric_mag", 0),
                ])
            if step % 5000 == 0:
                print(f"[{name}] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                      f"coll={coll_count} ttc={mean_ttc:.2f}")

        ckpt_path = os.path.join(args.out_dir, f"model_{name}_{args.scenario}.pt")
        policy.save(ckpt_path)
        print(f"Saved {name} to {ckpt_path}")
        env.close()

    print("Training complete. Metrics in", args.out_dir)


if __name__ == "__main__":
    main()
