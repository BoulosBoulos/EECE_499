"""Train PINN-augmented vs traditional PPO. See docs/RUNNING.md."""

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
    """Collect trajectories and compute GAE advantages."""
    obs, _ = env.reset()
    obs_list, actions_list, rewards_list, dones_list = [], [], [], []
    log_probs_list, values_list = [], []
    raw_list = []

    for _ in range(n_steps):
        action, _, log_prob, value = policy.get_action(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        obs_list.append(obs)
        actions_list.append(action)
        rewards_list.append(reward)
        dones_list.append(done)
        log_probs_list.append(log_prob)
        values_list.append(value)
        raw_list.append(info)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    obs_arr = np.array(obs_list, dtype=np.float32)
    actions_arr = np.array(actions_list, dtype=np.int64)
    rewards_arr = np.array(rewards_list, dtype=np.float32)
    dones_arr = np.array(dones_list, dtype=np.float32)
    log_probs_arr = np.array(log_probs_list, dtype=np.float32)
    values_arr = np.array(values_list, dtype=np.float32)

    # GAE
    advantages = np.zeros_like(rewards_arr, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards_arr))):
        next_non_term = 1.0 - dones_arr[t]
        next_val = values_arr[t + 1] * next_non_term if t + 1 < len(rewards_arr) else 0.0
        delta = rewards_arr[t] + gamma * next_val - values_arr[t]
        last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
        advantages[t] = last_gae
    returns = advantages + values_arr
    advantages = np.array(advantages, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    if adv_std > 1e-8:
        advantages = (advantages - adv_mean) / adv_std

    # Build PINN extra from raw_list
    extra = None
    if raw_list and "raw_obs" in raw_list[0]:
        built = [r.get("built", {}) for r in raw_list]
        d_cz = [b.get("s_geom", np.zeros(12))[1] for b in built]
        v = [b.get("s_ego", np.zeros(6))[0] for b in built]
        ttc_min = [r.get("ttc_min", 10.0) for r in raw_list]
        ego = [r.get("raw_obs", {}).get("ego", {}) for r in raw_list]
        kappa = [r.get("raw_obs", {}).get("geom", {}).get("kappa", 0.0) for r in raw_list]
        a_lon = [e.get("a", 0.0) for e in ego]
        extra = {
            "d_cz": np.array(d_cz, dtype=np.float32),
            "v": np.array(v, dtype=np.float32),
            "v_all": np.array(v, dtype=np.float32),
            "ttc_min": np.array(ttc_min, dtype=np.float32),
            "kappa": np.array(kappa, dtype=np.float32),
            "a_lon": np.array(a_lon, dtype=np.float32),
        }

    return obs_arr, actions_arr, log_probs_arr, returns, advantages, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/algo/default.yaml")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--total_steps", type=int, default=50_000)
    parser.add_argument("--no-compare", action="store_true", help="Train only one variant (default: train both PINN and non-PINN)")
    parser.add_argument("--sumo_gui", action="store_true", help="Use sumo-gui to watch training")
    parser.add_argument("--scenario", default="1a", choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="SUMO scenario (1a=car, 1b=ped, 1c=moto, 1d=pothole, 2-4=composite)")
    parser.add_argument("--use_intent", action="store_true", help="Use intent LSTM in state")
    args = parser.parse_args()

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
    n_steps = int(cfg.get("n_steps", 256))
    batch_size = int(cfg.get("batch_size", 64))
    n_epochs = int(cfg.get("n_epochs", 5))

    res_cfg = _load_config("configs/residuals/default.yaml")
    lambda_ego = float(res_cfg.get("lambda_ego", 0.5))
    lambda_stop = float(res_cfg.get("lambda_stop", 0.5))
    lambda_fric = float(res_cfg.get("lambda_fric", 0.3))
    lambda_risk = float(res_cfg.get("lambda_risk", 0.5))

    variants = []
    if not args.no_compare:
        variants = [("pinn", True), ("nopinn", False)]
    else:
        use_pinn = cfg.get("use_pinn", True)
        variants = [("pinn" if use_pinn else "nopinn", use_pinn)]

    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    for name, use_pinn in variants:
        env = SumoEnv(**env_kwargs)
        policy = __import__("models.drppo", fromlist=["DRPPO"]).DRPPO(
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range, use_pinn=use_pinn,
            lambda_ego=lambda_ego, lambda_stop=lambda_stop,
            lambda_fric=lambda_fric, lambda_risk=lambda_risk,
            device=device,
        )

        csv_path = os.path.join(args.out_dir, f"train_{name}_{args.scenario}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "episode_return", "episode_len", "actor_loss", "vf_loss",
                "collision_steps", "collision_rate", "mean_ttc", "min_ttc",
            ])

        step = 0
        ep_returns = []
        ep_lens = []
        ep_ret = 0
        ep_len = 0

        while step < args.total_steps:
            obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra = collect_rollouts(
                env, policy, n_steps, gamma, gae_lambda
            )
            step += n_steps

            # PPO epochs
            n_batches = (len(obs_arr) + batch_size - 1) // batch_size
            for _ in range(n_epochs):
                perm = np.random.permutation(len(obs_arr))
                for start in range(0, len(obs_arr), batch_size):
                    idx = perm[start : start + batch_size]
                    o = obs_arr[idx]
                    a = actions_arr[idx]
                    lp = log_probs_arr[idx]
                    ret = returns_arr[idx]
                    adv = advantages_arr[idx]
                    ex = {k: (v[idx] if isinstance(v, np.ndarray) else np.array(v)[idx]) for k, v in extra.items()} if extra else None
                    metrics = policy.train_step(o, a, lp, ret, adv, ex)

            # Probe rollout for episodic metrics
            rewards = []
            ttc_list = []
            coll_steps = 0
            obs, _ = env.reset()
            for _ in range(500):
                action, _, _, _ = policy.get_action(obs)
                obs, r, term, trunc, info = env.step(action)
                rewards.append(r)
                ttc_list.append(info.get("ttc_min", 10.0))
                if r < -5:
                    coll_steps += 1
                if term or trunc:
                    break
            ep_returns.append(sum(rewards))
            ep_lens.append(len(rewards))
            ttc_arr = np.array(ttc_list)
            mean_ttc = float(np.mean(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
            min_ttc = float(np.min(ttc_arr)) if len(ttc_arr) > 0 else float("nan")
            coll_rate = coll_steps / len(rewards) if rewards else 0.0

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    np.mean(ep_returns[-10:]) if ep_returns else 0,
                    np.mean(ep_lens[-10:]) if ep_lens else 0,
                    metrics.get("actor_loss", 0),
                    metrics.get("vf_loss", 0),
                    coll_steps,
                    coll_rate,
                    mean_ttc,
                    min_ttc,
                ])
            if step % 5000 == 0:
                print(f"[{name}] step={step} ret={np.mean(ep_returns[-10:]):.2f} "
                      f"coll={coll_steps} ttc={mean_ttc:.2f}")

        ckpt_path = os.path.join(args.out_dir, f"model_{name}_{args.scenario}.pt")
        policy.save(ckpt_path)
        print(f"Saved {name} to {ckpt_path}")
        if hasattr(env, "close"):
            env.close()

    print("Training complete. Metrics in", args.out_dir)


if __name__ == "__main__":
    main()
