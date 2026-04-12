"""Behavioral local dynamics: mode-conditioned one-step update for PDE state xi.

Each discrete action (STOP=0, CREEP=1, YIELD=2, GO=3, ABORT=4) maps to a
nominal longitudinal acceleration. The dynamics propagate ego kinematics,
path distances, agent relative positions, and derived conflict metrics.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from models.pde.state_builder import (
    XI_DIM, N_AGENT_FEAT, N_AGENTS_PDE,
    IDX_V, IDX_A, IDX_PSI_DOT, IDX_D_STOP, IDX_D_CZ, IDX_D_EXIT,
    IDX_KAPPA, IDX_TTC_MIN, IDX_POTHOLE,
)

NOMINAL_ACCEL = {
    0: "stop",
    1: "creep",
    2: "yield",
    3: "go",
    4: "abort",
}


class BehavioralDynamics:
    """Differentiable one-step dynamics for the reduced PDE state."""

    def __init__(self, dt: float = 0.1, a_brake: float = 5.0, a_abort: float = 8.0,
                 a_go: float = 2.0, v_creep: float = 1.0, v_max: float = 13.89,
                 L: float = 2.5, eps: float = 1e-6, d_safe: float = 2.0,
                 t_h: float = 3.0):
        self.dt = dt
        self.a_brake = a_brake
        self.a_abort = a_abort
        self.a_go = a_go
        self.v_creep = v_creep
        self.v_max = v_max
        self.L = L
        self.eps = eps
        self.d_safe = d_safe
        self.t_h = t_h

    def _nominal_accel(self, v: torch.Tensor, action: int) -> torch.Tensor:
        if action == 0:    # STOP
            return torch.full_like(v, -self.a_brake)
        elif action == 1:  # CREEP
            return torch.clamp(self.v_creep - v, -0.5, 0.5)
        elif action == 2:  # YIELD
            return torch.full_like(v, -0.5)
        elif action == 3:  # GO
            return torch.full_like(v, self.a_go)
        else:              # ABORT
            return torch.full_like(v, -self.a_abort)

    def one_step(self, xi: torch.Tensor, action: int) -> torch.Tensor:
        """Propagate xi by one timestep under the given action.

        Args:
            xi: (batch, XI_DIM) or (XI_DIM,) tensor
            action: int in {0,1,2,3,4}
        Returns:
            xi_next: same shape as xi
        """
        squeeze = xi.dim() == 1
        if squeeze:
            xi = xi.unsqueeze(0)
        B = xi.shape[0]
        xi_next = xi.clone()

        v = xi[:, IDX_V]
        a_nom = self._nominal_accel(v, action)
        v_new = torch.clamp(v + a_nom * self.dt, min=0.0, max=self.v_max)
        delta_s = 0.5 * (v + v_new) * self.dt

        xi_next[:, IDX_V] = v_new
        xi_next[:, IDX_A] = a_nom
        xi_next[:, IDX_D_STOP] = torch.clamp(xi[:, IDX_D_STOP] - delta_s, min=0.0)
        xi_next[:, IDX_D_CZ] = torch.clamp(xi[:, IDX_D_CZ] - delta_s, min=0.0)
        xi_next[:, IDX_D_EXIT] = torch.clamp(xi[:, IDX_D_EXIT] - delta_s, min=0.0)

        kappa = xi[:, IDX_KAPPA]
        delta_nom = torch.atan(self.L * kappa)
        psi_dot_new = (v_new / self.L) * torch.tan(delta_nom.clamp(-0.5, 0.5))
        xi_next[:, IDX_PSI_DOT] = psi_dot_new

        xi_next[:, IDX_POTHOLE] = torch.clamp(xi[:, IDX_POTHOLE] - delta_s, min=0.0)

        ttc_min_new = torch.full((B,), 10.0, device=xi.device, dtype=xi.dtype)
        for ag_idx in range(N_AGENTS_PDE):
            start = 12 + ag_idx * N_AGENT_FEAT
            mask = xi[:, start + 21]
            if mask.sum() < 0.5:
                continue

            dx = xi[:, start + 0]
            dy = xi[:, start + 1]
            dvx = xi[:, start + 2]
            dvy = xi[:, start + 3]
            v_i = xi[:, start + 5]
            d_cz_i = xi[:, start + 7]
            d_exit_i = xi[:, start + 8]

            dx_new = dx + dvx * self.dt
            dy_new = dy + dvy * self.dt
            d_cz_i_new = torch.clamp(d_cz_i - v_i * self.dt, min=0.0)
            d_exit_i_new = torch.clamp(d_exit_i - v_i * self.dt, min=0.0)

            xi_next[:, start + 0] = dx_new
            xi_next[:, start + 1] = dy_new
            xi_next[:, start + 7] = d_cz_i_new
            xi_next[:, start + 8] = d_exit_i_new

            tau_e = xi_next[:, IDX_D_CZ] / (v_new + self.eps)
            tau_i = d_cz_i_new / (v_i + self.eps)
            xi_next[:, start + 9] = tau_i
            xi_next[:, start + 10] = tau_i - tau_e

            dv_sq = dvx ** 2 + dvy ** 2 + self.eps
            t_cpa = torch.clamp(-(dx_new * dvx + dy_new * dvy) / dv_sq, 0.0, self.t_h)
            px_cpa = dx_new + t_cpa * dvx
            py_cpa = dy_new + t_cpa * dvy
            d_cpa = torch.sqrt(px_cpa ** 2 + py_cpa ** 2 + self.eps)
            dv_norm = torch.sqrt(dv_sq)
            ttc_i = torch.clamp(d_cpa - self.d_safe, min=0.0) / dv_norm

            xi_next[:, start + 11] = t_cpa
            xi_next[:, start + 12] = d_cpa
            xi_next[:, start + 13] = ttc_i

            ttc_min_new = torch.where(
                (mask > 0.5) & (ttc_i < ttc_min_new),
                ttc_i, ttc_min_new
            )

        xi_next[:, IDX_TTC_MIN] = ttc_min_new

        if squeeze:
            xi_next = xi_next.squeeze(0)
        return xi_next

    def drift(self, xi: torch.Tensor, action: int) -> torch.Tensor:
        """Compute drift f_a(xi) = (F_a(xi) - xi) / dt."""
        return (self.one_step(xi, action) - xi) / self.dt

    def all_action_drifts(self, xi: torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute drifts for all 5 actions."""
        return {a: self.drift(xi, a) for a in range(5)}

    def all_action_next_states(self, xi: torch.Tensor) -> dict[int, torch.Tensor]:
        """Compute next states for all 5 actions."""
        return {a: self.one_step(xi, a) for a in range(5)}
