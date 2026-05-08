"""Rollout collector for PDE-family methods.

Same as legacy collect_rollouts but also builds xi_curr, xi_next,
terminal labels for PDE critic training.
"""

from __future__ import annotations
import time
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from models.pde.state_builder import ReducedPDEState


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    """Collect rollouts with PDE state arrays.

    Returns the same tuple as legacy collect_rollouts plus PDE extras in extra dict.
    """
    pde_builder = ReducedPDEState()

    obs, info = env.reset()
    policy.reset_hidden()
    obs_list, actions_list, rewards_list, dones_list = [], [], [], []
    log_probs_list, values_list = [], []
    hidden_list = []
    pre_step_infos = []
    post_step_infos = []
    collision_events = []
    behavior_labels = []

    xi_curr_list = []
    xi_next_list = []
    success_terminal_list = []
    collision_terminal_list = []

    pre_step_info = info
    terminated_list = []
    truncated_bootstrap = {}

    # Phase 1A logging instrumentation (per-iteration aggregates).
    env_step_time_total = 0.0
    action_counts = [0, 0, 0, 0, 0]
    n_episodes = 0
    n_collisions = 0
    n_successes = 0
    n_timeouts = 0
    n_aborts = 0
    episode_lengths = []
    current_ep_len = 0
    current_ep_steps = []
    completed_collision_episodes = []

    for step_i in range(n_steps):
        pre_step_infos.append(pre_step_info)

        h = policy._hidden
        if h is not None:
            hidden_list.append(h.detach().cpu().squeeze(1).numpy())
        else:
            hidden_list.append(np.zeros((1, policy.policy.hidden_dim), dtype=np.float32))

        built_pre = pre_step_info.get("built", {})
        xi_pre = pde_builder.build(built_pre, pre_step_info)
        xi_curr_list.append(xi_pre)

        action, new_hidden, log_prob, value = policy.get_action(obs)
        env_step_t0 = time.time()
        next_obs, reward, term, trunc, info = env.step(action)
        env_step_time_total += time.time() - env_step_t0
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

        # Track per-iteration logging aggregates.
        a_idx = int(action)
        if 0 <= a_idx < len(action_counts):
            action_counts[a_idx] += 1

        # Build a per-step trajectory row for the current episode buffer.
        ego_raw = info.get("raw_obs", {}).get("ego", {}) if isinstance(info.get("raw_obs"), dict) else {}
        ego_p = ego_raw.get("p", (0.0, 0.0))
        try:
            ego_x = float(ego_p[0])
            ego_y = float(ego_p[1])
        except (TypeError, IndexError):
            ego_x = ego_y = 0.0
        n_agents_now = len(info.get("raw_obs", {}).get("agents", []) or []) if isinstance(info.get("raw_obs"), dict) else 0
        current_ep_steps.append({
            "step": current_ep_len,
            "ego_x": ego_x,
            "ego_y": ego_y,
            "ego_psi": float(ego_raw.get("psi", 0.0)),
            "ego_v": float(ego_raw.get("v", 0.0)),
            "ego_a": float(ego_raw.get("a", 0.0)),
            "action": a_idx,
            "reward": float(reward),
            "min_ttc": float(info.get("ttc_min", float("inf"))),
            "n_agents": int(n_agents_now),
            "collision_agent_id": "none",
            "terminal_flag": int(bool(done) and bool(info.get("collision", False))),
        })
        current_ep_len += 1

        built_post = info.get("built", {})
        xi_post = pde_builder.build(built_post, info)
        xi_next_list.append(xi_post)

        is_success = term and not info.get("collision", False)
        is_collision = term and info.get("collision", False)
        success_terminal_list.append(is_success)
        collision_terminal_list.append(is_collision)

        if trunc and not term:
            with torch.no_grad():
                o_t = torch.FloatTensor(next_obs).unsqueeze(0).unsqueeze(0)
                if torch is not None:
                    o_t = o_t.to(policy.device if hasattr(policy, 'device') else 'cpu')
                truncated_bootstrap[step_i] = policy.policy.get_value(o_t, policy._hidden).item()

        obs = next_obs
        pre_step_info = info
        if done:
            n_episodes += 1
            episode_lengths.append(current_ep_len)
            if info.get("collision"):
                n_collisions += 1
                completed_collision_episodes.append({
                    "steps": list(current_ep_steps),
                    "terminal_step": current_ep_len - 1,
                    "collision_agent_id": str(info.get("collision_agent_id", "unknown")),
                })
            elif info.get("success"):
                n_successes += 1
            elif info.get("timeout"):
                n_timeouts += 1
            elif info.get("aborted"):
                n_aborts += 1
            else:
                # fall back: treat truncation without explicit timeout as timeout
                if trunc and not term:
                    n_timeouts += 1

            current_ep_len = 0
            current_ep_steps = []

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

    last_idx = len(dones_arr) - 1
    if last_idx >= 0 and not dones_arr[last_idx]:
        with torch.no_grad():
            o_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
            if torch is not None:
                o_t = o_t.to(policy.device if hasattr(policy, 'device') else 'cpu')
            truncated_bootstrap[last_idx] = policy.policy.get_value(o_t, policy._hidden).item()

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

    extra = {
        "xi_curr": np.array(xi_curr_list, dtype=np.float32),
        "xi_next": np.array(xi_next_list, dtype=np.float32),
        "success_terminal": np.array(success_terminal_list, dtype=bool),
        "collision_terminal": np.array(collision_terminal_list, dtype=bool),
        "done_arr": dones_arr,
        "collision_events": np.array(collision_events, dtype=bool),
    }

    N = len(pre_step_infos)
    if N > 0 and "raw_obs" in pre_step_infos[0]:
        built = [r.get("built", {}) for r in pre_step_infos]
        d_cz = [b.get("s_geom", np.zeros(12))[1] for b in built]
        v = [b.get("s_ego", np.zeros(6))[0] for b in built]
        ttc_min = [r.get("ttc_min", 10.0) for r in pre_step_infos]
        ego = [r.get("raw_obs", {}).get("ego", {}) for r in pre_step_infos]
        kappa = [r.get("raw_obs", {}).get("geom", {}).get("kappa", 0.0) for r in pre_step_infos]
        a_lon = [e.get("a", 0.0) for e in ego]
        extra.update({
            "d_cz": np.array(d_cz, dtype=np.float32),
            "v": np.array(v, dtype=np.float32),
            "ttc_min": np.array(ttc_min, dtype=np.float32),
            "kappa": np.array(kappa, dtype=np.float32),
            "a_lon": np.array(a_lon, dtype=np.float32),
        })

    if behavior_labels:
        extra["behavior_labels"] = behavior_labels

    # Phase 1A iteration-level aggregates.
    extra["env_step_time"] = float(env_step_time_total)
    extra["action_counts"] = list(action_counts)
    extra["n_episodes"] = int(n_episodes)
    extra["n_collisions"] = int(n_collisions)
    extra["n_successes"] = int(n_successes)
    extra["n_timeouts"] = int(n_timeouts)
    extra["n_aborts"] = int(n_aborts)
    extra["mean_episode_length"] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    extra["completed_collision_episodes"] = completed_collision_episodes

    return obs_arr, actions_arr, log_probs_arr, returns, advantages, extra, hidden_arr
