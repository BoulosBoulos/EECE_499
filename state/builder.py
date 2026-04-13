"""State builder: raw perception -> structured state per System_Architecture.txt."""

from __future__ import annotations

import numpy as np
from typing import Any

# Optional YAML
try:
    import yaml
except ImportError:
    yaml = None


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _rot2d(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def _wrap(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class StateBuilder:
    """Build full continuous state: ego, geom, vis, per-agent features."""

    def __init__(self, config_path: str | None = None):
        cfg = _load_config(config_path)
        self.top_n = cfg.get("top_n_agents", 5)
        self.eps = cfg.get("epsilon", 1e-6)
        self.t_h = cfg.get("t_h_cpa", 3.0)
        self.d_safe = cfg.get("d_safe", 2.0)

    def build(self, raw_obs: dict[str, Any], prev_ego: dict | None = None) -> dict[str, np.ndarray]:
        """
        Build s_t = [s_ego, s_geom, s_vis, f^1, ..., f^N].
        raw_obs: ego (p, psi, v, a, psi_dot), agents, geom, vis.
        prev_ego: {a, psi_dot} for jerk/yaw-accel terms.
        """
        ego = raw_obs.get("ego", {})
        agents = raw_obs.get("agents", [])
        geom = raw_obs.get("geom", {})
        vis = raw_obs.get("vis", {})

        # 2.1 Ego state
        v_e = float(ego.get("v", 0.0))
        a_e = float(ego.get("a", 0.0))
        psi_dot_e = float(ego.get("psi_dot", 0.0))
        psi_e = float(ego.get("psi", 0.0))
        a_prev = float(prev_ego["a"]) if prev_ego else a_e
        psi_dot_prev = float(prev_ego["psi_dot"]) if prev_ego else psi_dot_e
        s_ego = np.array([
            v_e, a_e, psi_dot_e, psi_e,
            a_e - a_prev, psi_dot_e - psi_dot_prev
        ], dtype=np.float32)

        # 2.2 Route & intersection geometry
        d_stop = float(geom.get("d_stop", 0.0))
        d_cz = float(geom.get("d_cz", 0.0))
        d_exit = float(geom.get("d_exit", 0.0))
        kappa = float(geom.get("kappa", 0.0))
        e_y = float(geom.get("e_y", 0.0))
        e_psi = float(geom.get("e_psi", 0.0))
        # NOTE: g_turn is always [0,0,1] (right turn) and w_lane is always 3.5
        # for this fixed T-intersection geometry. These are retained for
        # architectural generality but provide no information to the policy.
        w_lane = float(geom.get("w_lane", 3.5))
        g_turn = np.array(geom.get("g_turn", [0, 1, 0]), dtype=np.float32)  # one-hot left/straight/right
        rho = np.array(geom.get("rho", [1, 0]), dtype=np.float32)  # ROW context
        s_geom = np.concatenate([[d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane], g_turn, rho])

        # 2.3 Visibility
        alpha_cz = float(vis.get("alpha_cz", 1.0))
        alpha_cross = float(vis.get("alpha_cross", 1.0))
        d_occ = float(vis.get("d_occ", 1e6))
        dt_seen = float(vis.get("dt_seen", 0.0))
        sigma_percep = float(vis.get("sigma_percep", 0.0))
        n_occ = float(vis.get("n_occ", 0.0))
        s_vis = np.array([alpha_cz, alpha_cross, d_occ, dt_seen, sigma_percep, n_occ], dtype=np.float32)

        # 2.4 Per-agent: select top-N by time-to-conflict / distance, compute features
        p_e = np.array(ego.get("p", [0, 0]))
        v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])
        d_cz_e = d_cz
        tau_e = d_cz_e / (max(v_e, self.eps))

        agent_features = []
        agent_list = list(agents)
        # Score: smaller tau (earlier conflict) and closer distance -> higher relevance
        scores = []
        for i, ag in enumerate(agent_list):
            p_i = np.array(ag.get("p", [0, 0]))
            psi_i = float(ag.get("psi", 0.0))
            v_i = float(ag.get("v", 0.0))
            d_cz_i = float(ag.get("d_cz", 1e6))
            tau_i = d_cz_i / (max(v_i, self.eps))
            dist = np.linalg.norm(p_i - p_e) + self.eps
            scores.append((tau_i, dist, i))
        scores.sort(key=lambda x: (x[0], x[1]))
        indices = [x[2] for x in scores[: self.top_n]]

        R = _rot2d(-psi_e)
        for idx in indices:
            ag = agent_list[idx]
            p_i = np.array(ag.get("p", [0, 0]))
            psi_i = float(ag.get("psi", 0.0))
            v_i = float(ag.get("v", 0.0))
            a_i = float(ag.get("a", 0.0))
            c_i = ag.get("type", "veh")
            nu_i = float(ag.get("nu", 1.0))  # visibility
            sigma_i = float(ag.get("sigma", 0.0))
            d_cz_i = float(ag.get("d_cz", 1e6))
            d_exit_i = float(ag.get("d_exit", 1e6))
            chi_i = float(ag.get("chi", 0.0))
            pi_row_i = float(ag.get("pi_row", 0.0))

            dp = p_i - p_e
            delta_xy = R @ dp
            v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])
            delta_v = R @ (v_i_vec - v_e_vec)
            delta_psi = _wrap(psi_i - psi_e)

            tau_i = d_cz_i / (max(v_i, self.eps))
            delta_tau_i = tau_i - tau_e

            t_cpa = np.clip(
                -np.dot(delta_xy, delta_v) / (np.dot(delta_v, delta_v) + self.eps),
                0.0, self.t_h
            )
            p_cpa = delta_xy + t_cpa * delta_v
            d_cpa = np.linalg.norm(p_cpa)
            TTC_i = max(d_cpa - self.d_safe, 0.0) / (np.linalg.norm(delta_v) + self.eps)

            type_onehot = {"veh": [1, 0, 0], "ped": [0, 1, 0], "cyc": [0, 0, 1]}.get(c_i, [1, 0, 0])
            mask_i = 1.0

            f_i = [
                delta_xy[0], delta_xy[1], delta_v[0], delta_v[1], delta_psi,
                v_i, a_i, d_cz_i, d_exit_i, tau_i, delta_tau_i,
                t_cpa, d_cpa, TTC_i,
                chi_i, pi_row_i, nu_i, sigma_i,
                *type_onehot, mask_i
            ]
            agent_features.append(np.array(f_i, dtype=np.float32))

        # Pad if fewer than top_n
        n_agent_feat = 22
        while len(agent_features) < self.top_n:
            agent_features.append(np.zeros(n_agent_feat, dtype=np.float32))
        agent_features = agent_features[: self.top_n]
        f_all = np.stack(agent_features, axis=0)

        # 2.5 Final state
        s_t = np.concatenate([s_ego, s_geom, s_vis, f_all.flatten()])
        return {
            "state": s_t,
            "s_ego": s_ego,
            "s_geom": s_geom,
            "s_vis": s_vis,
            "f_agents": f_all,
            "n_agents": min(len(agent_list), self.top_n),
            "raw_obs": raw_obs,
        }
