"""Local reward surrogate for PDE-based critics.

Mirrors the env reward structure but operates on the reduced PDE state xi.
Does NOT include collision penalty -- collisions are handled via terminal
boundary conditions in the PDE loss.
"""

from __future__ import annotations
import torch
from models.pde.state_builder import IDX_V, IDX_TTC_MIN, IDX_POTHOLE, IDX_D_CZ
from models.pde.dynamics import BehavioralDynamics


def local_reward_from_next(
    xi: torch.Tensor,
    action: int,
    xi_next: torch.Tensor,
    dynamics: BehavioralDynamics,
    w_prog: float = 1.0,
    w_time: float = -0.1,
    w_risk: float = -3.0,
    w_pothole: float = -5.0,
    w_abort_comfort: float = -0.5,
    w_rule: float = -2.0,
    ttc_thr: float = 3.0,
) -> torch.Tensor:
    """Compute one-step surrogate reward using a pre-computed xi_next.

    Aligned with the env reward in sumo_env.py (Option A: no prev_action,
    no action-switching penalty).  Terminal rewards (collision/success) are
    handled via boundary conditions in the PDE loss, NOT here.

    Args:
        xi: (batch, XI_DIM) or (XI_DIM,) reduced PDE state
        action: int action index
        xi_next: (batch, XI_DIM) or (XI_DIM,) next PDE state
        dynamics: BehavioralDynamics instance
    Returns:
        reward: (batch,) or scalar tensor
    """
    squeeze = xi.dim() == 1
    if squeeze:
        xi = xi.unsqueeze(0)
        xi_next = xi_next.unsqueeze(0)

    v = xi[:, IDX_V]
    ttc_next = xi_next[:, IDX_TTC_MIN]
    d_pot_next = xi_next[:, IDX_POTHOLE]
    d_cz = xi[:, IDX_D_CZ]

    # Progress: v * dt  (matches env exactly)
    progress = v * dynamics.dt

    # Time penalty
    r = w_prog * progress + w_time * dynamics.dt

    # Risk: sharp sigmoid (temp=0.1)
    r = r + w_risk * torch.sigmoid((ttc_thr - ttc_next) / 0.1)

    # Pothole: sharp sigmoid (temp=0.05)
    r = r + w_pothole * torch.sigmoid((1.0 - d_pot_next) / 0.05)

    # Abort comfort penalty
    if action == 4:
        r = r + w_abort_comfort

    # ROW proxy: penalise entering CZ at speed when TTC is low
    row_proxy = (
        torch.sigmoid((ttc_thr - ttc_next) / 0.1)
        * torch.sigmoid((3.0 - d_cz) / 0.5)
        * torch.sigmoid((v - 1.0) / 0.3)
    )
    r = r + w_rule * row_proxy

    if squeeze:
        r = r.squeeze(0)
    return r


def local_reward(
    xi: torch.Tensor,
    action: int,
    dynamics: BehavioralDynamics,
    **kwargs,
) -> torch.Tensor:
    """Compute one-step surrogate reward r(xi, a).

    Delegates to local_reward_from_next after computing xi_next via
    dynamics.one_step.
    """
    squeeze = xi.dim() == 1
    if squeeze:
        xi = xi.unsqueeze(0)

    xi_next = dynamics.one_step(xi, action)
    r = local_reward_from_next(xi, action, xi_next, dynamics, **kwargs)

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
