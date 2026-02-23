"""Synthetic T-intersection environment for training without SUMO."""

from __future__ import annotations

import numpy as np
from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from state.builder import StateBuilder


ACTION_NAMES = ["STOP", "CREEP", "YIELD", "GO", "ABORT"]
N_ACTIONS = 5


def _make_gym():
    if gym is None:
        return object
    return gym.Env


class TIntersectionEnv(_make_gym()):
    """
    Synthetic T-intersection: ego approaches stop line, cross-traffic agents.
    Produces raw_obs compatible with StateBuilder; returns flat state for policy.
    """

    def __init__(
        self,
        max_steps: int = 200,
        dt: float = 0.1,
        state_config: str | None = None,
    ):
        if gym is not None:
            super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.state_builder = StateBuilder(state_config)

        self._ego = None
        self._agents = None
        self._step_count = 0
        self._prev_ego = None

        # State dim: built from state_builder
        # Approximate: s_ego(6) + s_geom(12) + s_vis(6) + top_n * 23
        self._state_dim = 6 + 12 + 6 + 5 * 22  # 134 for top_n=5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._state_dim,), dtype=np.float32
        ) if spaces else None
        self.action_space = spaces.Discrete(N_ACTIONS) if spaces else None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        self._ego = {
            "p": np.array([0.0, 0.0]),
            "psi": 0.0,
            "v": 2.0,
            "a": 0.0,
            "psi_dot": 0.0,
        }
        self._prev_ego = {"a": 0.0, "psi_dot": 0.0}
        self._spawn_agents()
        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        self._prev_ego = {"a": self._ego["a"], "psi_dot": self._ego["psi_dot"]}
        return built["state"].astype(np.float32), {"raw_obs": raw, "built": built}

    def _spawn_agents(self):
        n = np.random.randint(1, 4)
        self._agents = []
        for _ in range(n):
            side = np.random.choice(["left", "right"])
            y = 15.0 if side == "left" else -15.0
            x = np.random.uniform(5, 25)
            v = np.random.uniform(0.5, 3.0)
            self._agents.append({
                "p": np.array([x, y]),
                "psi": np.pi / 2 if side == "left" else -np.pi / 2,
                "v": v, "a": 0.0, "type": "veh",
                "nu": 1.0, "sigma": 0.1,
                "d_cz": max(0, x - 10),
                "d_exit": max(0, x - 5),
                "chi": 1.0 if 5 < x < 20 else 0.0,
                "pi_row": 0.5,
            })

    def _get_raw_obs(self) -> dict:
        ego_x, ego_y = self._ego["p"]
        d_stop = max(0, 20 - ego_x)
        d_cz = max(0, 15 - ego_x)
        d_exit = max(0, 25 - ego_x)
        geom = {
            "d_stop": d_stop, "d_cz": d_cz, "d_exit": d_exit,
            "kappa": 0.0, "e_y": 0.0, "e_psi": 0.0,
            "w_lane": 3.5,
            "g_turn": [0, 1, 0],
            "rho": [1, 0],
        }
        vis = {
            "alpha_cz": 1.0, "alpha_cross": 1.0,
            "d_occ": 50.0, "dt_seen": 0.0,
            "sigma_percep": 0.05, "n_occ": 0.0,
        }
        agents = []
        for ag in self._agents:
            ax, ay = ag["p"]
            dx_cz = max(0, abs(ay) - 5) if abs(ay) < 15 else 20
            agents.append({
                **ag,
                "d_cz": dx_cz,
                "d_exit": max(0, dx_cz - 10),
            })
        return {"ego": self._ego, "agents": agents, "geom": geom, "vis": vis}

    def _compute_ttc_min(self) -> float:
        ttc = 1e6
        for ag in self._agents:
            dp = ag["p"] - self._ego["p"]
            v_e = self._ego["v"] * np.array([np.cos(self._ego["psi"]), np.sin(self._ego["psi"])])
            v_i = ag["v"] * np.array([np.cos(ag["psi"]), np.sin(ag["psi"])])
            dv = v_i - v_e
            dv_norm = np.linalg.norm(dv) + 1e-6
            t_cpa = np.clip(-np.dot(dp, dv) / (dv_norm ** 2), 0, 3.0)
            p_cpa = dp + t_cpa * dv
            d_cpa = np.linalg.norm(p_cpa)
            ttc_i = max(d_cpa - 2.0, 0) / dv_norm
            ttc = min(ttc, ttc_i)
        return ttc if ttc < 1e5 else 10.0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Map action to acceleration
        if action == 0:
            a_cmd = -5.0
        elif action == 1:
            a_cmd = 0.5 if self._ego["v"] < 1.0 else -0.2
        elif action == 2:
            a_cmd = -0.5
        elif action == 3:
            a_cmd = 1.0
        else:
            a_cmd = -5.0

        v = self._ego["v"]
        psi = self._ego["psi"]
        p = self._ego["p"].copy()
        v_new = np.clip(v + a_cmd * self.dt, 0, 10)
        p_new = p + v * np.array([np.cos(psi), np.sin(psi)]) * self.dt
        self._prev_ego = {"a": self._ego["a"], "psi_dot": self._ego["psi_dot"]}
        self._ego["p"] = p_new
        self._ego["v"] = v_new
        self._ego["a"] = a_cmd

        for ag in self._agents:
            vi = ag["v"]
            psi_i = ag["psi"]
            ag["p"] = ag["p"] + vi * np.array([np.cos(psi_i), np.sin(psi_i)]) * self.dt

        raw = self._get_raw_obs()
        built = self.state_builder.build(raw, self._prev_ego)
        self._prev_ego = {"a": self._ego["a"], "psi_dot": self._ego["psi_dot"]}

        ttc_min = self._compute_ttc_min()
        prog = (p_new[0] - p[0])
        r = 1.0 * prog - 0.1 * self.dt
        if ttc_min < 3.0:
            r -= 1.0
        for ag in self._agents:
            if np.linalg.norm(ag["p"] - self._ego["p"]) < 2.0:
                r -= 10.0

        self._step_count += 1
        done = self._step_count >= self.max_steps or p_new[0] > 30
        return (
            built["state"].astype(np.float32),
            float(r),
            False,
            done,
            {"raw_obs": raw, "built": built, "ttc_min": ttc_min},
        )
