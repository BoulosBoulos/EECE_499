"""Physics predictors: bicycle kinematics, stop distance, friction, risk envelopes."""

from __future__ import annotations

import torch
import torch.nn as nn
import math
from typing import Optional


def _rotate_2d(x: torch.Tensor, y: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = torch.cos(psi)
    s = torch.sin(psi)
    return x * c - y * s, x * s + y * c


class PhysicsPredictor:
    """
    Lightweight physics for residuals and safety metrics.
    - Bicycle kinematics: X_dot=v*cos(psi), Y_dot=v*sin(psi), psi_dot=v/L*tan(delta), v_dot=a
    - STOP: a=-a_brake, delta=0
    - CREEP: regulate toward v_creep
    - YIELD: a≈0, delta=0
    - GO: a=a_go, delta from curvature
    - ABORT: a=-a_brake, delta stay in lane
    """

    def __init__(
        self,
        dt: float = 0.1,
        L: float = 2.5,
        a_brake: float = 5.0,
        v_creep: float = 1.0,
        a_go: float = 2.0,
        a_max: float = 5.0,
        mu: float = 0.8,
        g: float = 9.81,
    ):
        self.dt = dt
        self.L = L
        self.a_brake = a_brake
        self.v_creep = v_creep
        self.a_go = a_go
        self.a_max = a_max
        self.mu = mu
        self.g = g

    def action_to_control(self, action: int) -> tuple[float, float]:
        """Map discrete action to (a, delta). action: 0=STOP, 1=CREEP, 2=YIELD, 3=GO, 4=ABORT."""
        if action == 0:  # STOP
            return -self.a_brake, 0.0
        if action == 1:  # CREEP
            return 0.5, 0.0  # gentle accel toward v_creep
        if action == 2:  # YIELD
            return 0.0, 0.0
        if action == 3:  # GO
            return self.a_go, 0.0
        if action == 4:  # ABORT
            return -self.a_brake, 0.0
        return 0.0, 0.0

    def one_step_euler(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        psi: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step Euler: x,y,psi,v -> x',y',psi',v'."""
        v = v.clamp(min=1e-4)
        x_new = x + v * torch.cos(psi) * self.dt
        y_new = y + v * torch.sin(psi) * self.dt
        psi_new = psi + (v / self.L) * torch.tan(delta.clamp(-0.5, 0.5)) * self.dt
        v_new = (v + a * self.dt).clamp(min=0.0)
        return x_new, y_new, psi_new, v_new

    def stopping_distance(self, v: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """d_stop = v*tau + v^2/(2*a_max)."""
        return v * tau + (v ** 2) / (2 * self.a_max)

    def lateral_accel(self, v: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """a_lat = v^2 * kappa (R ≈ 1/|kappa|)."""
        return (v ** 2) * torch.abs(kappa)


def compute_L_ego(
    x_next: torch.Tensor, y_next: torch.Tensor, psi_next: torch.Tensor, v_next: torch.Tensor,
    x_pred: torch.Tensor, y_pred: torch.Tensor, psi_pred: torch.Tensor, v_pred: torch.Tensor,
) -> torch.Tensor:
    """L_ego = E[|x_{t+1} - hat{x}_{t+1}(a_t)|^2]."""
    return (
        (x_next - x_pred).pow(2).mean() +
        (y_next - y_pred).pow(2).mean() +
        (psi_next - psi_pred).pow(2).mean() +
        (v_next - v_pred).pow(2).mean()
    )


def compute_L_stop(
    d_cz: torch.Tensor,
    v: torch.Tensor,
    a_max: float = 5.0,
    tau: float = 0.5,
) -> torch.Tensor:
    """L_stop = E[ReLU(-m_stop)^2], m_stop = d_cz - d_stop(v)."""
    d_stop = v * tau + (v ** 2) / (2 * a_max)
    m_stop = d_cz - d_stop
    return torch.relu(-m_stop).pow(2).mean()


def compute_L_fric(
    v: torch.Tensor,
    kappa: torch.Tensor,
    a_lon: torch.Tensor,
    mu: float = 0.8,
    g: float = 9.81,
) -> torch.Tensor:
    """L_fric = E[ReLU(a_lat^2 + a_lon^2 - (mu*g)^2)^2]."""
    a_lat = (v ** 2) * torch.abs(kappa)
    constraint = a_lat.pow(2) + a_lon.pow(2) - (mu * g) ** 2
    return torch.relu(constraint).pow(2).mean()


def compute_risk_from_state(
    s_agents: torch.Tensor,
    d_cz_ego: torch.Tensor,
    v_ego: torch.Tensor,
    a_ego: torch.Tensor,
    action: torch.Tensor,
    ttc_min: torch.Tensor,
) -> torch.Tensor:
    """
    Conservative risk surrogate: TTC-based and distance-based.
    s_agents has per-agent features including TTC, d_cpa, chi, pi_row.
    """
    # Simple risk: low TTC and close agents
    risk = torch.relu(3.0 - ttc_min)  # hinge when TTC < 3s
    return risk.mean()
