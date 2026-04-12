"""Reduced PDE state builder: extracts physically meaningful features for HJB/Soft-HJB critics."""

from __future__ import annotations
import numpy as np

XI_DIM = 79

IDX_EGO = slice(0, 8)
IDX_VIS = slice(8, 12)
IDX_AGENT1 = slice(12, 34)
IDX_AGENT2 = slice(34, 56)
IDX_AGENT3 = slice(56, 78)
IDX_POTHOLE = 78

IDX_V = 0
IDX_A = 1
IDX_PSI_DOT = 2
IDX_D_STOP = 3
IDX_D_CZ = 4
IDX_D_EXIT = 5
IDX_KAPPA = 6
IDX_TTC_MIN = 7

N_AGENT_FEAT = 22
N_AGENTS_PDE = 3


class ReducedPDEState:
    """Build xi from state builder output and env info dict.

    The PDE state only includes features that are dynamic and physically
    meaningful. Constants (e_y, e_psi, w_lane, g_turn, rho, sigma_percep,
    n_occ) are excluded.
    """

    def __init__(self, n_agents: int = N_AGENTS_PDE):
        self.n_agents = n_agents

    def build(self, built: dict, info: dict) -> np.ndarray:
        """Build reduced state xi from state builder output and env info.

        Args:
            built: dict from StateBuilder.build() with keys s_ego, s_geom, s_vis, f_agents
            info: dict from env.step() or env.reset() with keys ttc_min, raw_obs
        Returns:
            xi: np.ndarray of shape (XI_DIM,)
        """
        s_ego = built["s_ego"]
        s_geom = built["s_geom"]
        s_vis = built["s_vis"]
        f_agents = built["f_agents"]
        raw = info.get("raw_obs", {})

        xi = np.zeros(XI_DIM, dtype=np.float32)

        xi[IDX_V] = float(s_ego[0])
        xi[IDX_A] = float(s_ego[1])
        xi[IDX_PSI_DOT] = float(s_ego[2])
        xi[IDX_D_STOP] = float(s_geom[0])
        xi[IDX_D_CZ] = float(s_geom[1])
        xi[IDX_D_EXIT] = float(s_geom[2])
        xi[IDX_KAPPA] = float(s_geom[3])
        xi[IDX_TTC_MIN] = float(info.get("ttc_min", 10.0))

        xi[8] = float(s_vis[0])   # alpha_cz
        xi[9] = float(s_vis[1])   # alpha_cross
        xi[10] = float(s_vis[2])  # d_occ
        xi[11] = float(s_vis[3])  # dt_seen

        for i in range(min(self.n_agents, f_agents.shape[0])):
            start = 12 + i * N_AGENT_FEAT
            xi[start:start + N_AGENT_FEAT] = f_agents[i].astype(np.float32)

        xi[IDX_POTHOLE] = float(raw.get("d_pothole", 100.0))

        return xi

    def build_batch(self, built_list: list[dict], info_list: list[dict]) -> np.ndarray:
        """Build batch of xi states."""
        return np.stack([self.build(b, i) for b, i in zip(built_list, info_list)])
