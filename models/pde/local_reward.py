"""Local reward surrogate for PDE-based critics.

Mirrors the env reward structure but operates on the reduced PDE state xi.
Does NOT include collision penalty -- collisions are handled via terminal
boundary conditions in the PDE loss.
"""

from __future__ import annotations
import torch
from models.pde.state_builder import IDX_V, IDX_TTC_MIN, IDX_POTHOLE
from models.pde.dynamics import BehavioralDynamics


def local_reward(
    xi: torch.Tensor,
    action: int,
    dynamics: BehavioralDynamics,
    w_prog: float = 1.0,
    w_time: float = -0.1,
    w_risk: float = -3.0,
    w_pothole: float = -5.0,
    ttc_thr: float = 3.0,
    pothole_thr: float = 1.0,
) -> torch.Tensor:
    """Compute one-step surrogate reward r(xi, a).

    Args:
        xi: (batch, XI_DIM) or (XI_DIM,) reduced PDE state
        action: int action index
        dynamics: BehavioralDynamics instance
        w_prog, w_time, w_risk, w_pothole, ttc_thr, pothole_thr: reward params
    Returns:
        reward: (batch,) or scalar tensor
    """
    squeeze = xi.dim() == 1
    if squeeze:
        xi = xi.unsqueeze(0)

    v = xi[:, IDX_V]
    a_nom = dynamics._nominal_accel(v, action)
    v_new = torch.clamp(v + a_nom * dynamics.dt, min=0.0, max=dynamics.v_max)
    progress = 0.5 * (v + v_new) * dynamics.dt

    xi_next = dynamics.one_step(xi, action)
    ttc_next = xi_next[:, IDX_TTC_MIN]
    d_pot_next = xi_next[:, IDX_POTHOLE]

    r = w_prog * progress + w_time * dynamics.dt
    r = r + w_risk * (ttc_next < ttc_thr).float()
    r = r + w_pothole * (d_pot_next < pothole_thr).float()

    if squeeze:
        r = r.squeeze(0)
    return r


def local_reward_all_actions(
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    **kwargs,
) -> dict[int, torch.Tensor]:
    """Compute reward for all 5 actions."""
    return {a: local_reward(xi, a, dynamics, **kwargs) for a in range(5)}
