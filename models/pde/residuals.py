"""PDE residual computations for HJB, Soft-HJB, Eikonal, and CBF critics.

Hard-HJB residual: rho = U(xi)*ln(gamma) + max_a [r(xi,a) + gamma * grad_U^T * (F_a(xi) - xi)]
Soft-HJB residual: rho = U(xi)*ln(gamma) + tau * logsumexp([r(xi,a) + gamma * grad_U^T * (F_a(xi) - xi)] / tau)
Eikonal residual:  rho = ||grad_U||^2 - c(xi)^2
CBF-PDE residual:  rho = ReLU(-max_a [grad_U^T * (F_a(xi) - xi) + alpha * U(xi)])

All require autograd for nabla_xi U.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
from models.pde.dynamics import BehavioralDynamics
from models.pde.local_reward import local_reward, local_reward_from_next
from models.pde.state_builder import IDX_V, IDX_D_CZ, IDX_TTC_MIN


def _compute_grad_U(U_net: nn.Module, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute U(xi) and grad_xi U(xi) using autograd."""
    xi_req = xi.detach().requires_grad_(True)
    U_val = U_net(xi_req)
    grad_U = torch.autograd.grad(
        U_val.sum(), xi_req, create_graph=True, retain_graph=True
    )[0]
    return U_val, grad_U


def pde_q_values(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    gamma: float = 0.99,
    reward_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDE-based Q-values for all actions.

    q_a = r(xi, a) + gamma * grad_U(xi)^T * (F_a(xi) - xi)

    Args:
        U_net: auxiliary critic network xi -> scalar
        xi: (batch, XI_DIM)
        dynamics: BehavioralDynamics
        gamma: discount factor
        reward_kwargs: extra kwargs for local_reward
    Returns:
        q_all: (batch, 5) Q-values per action
        U_val: (batch,) value function output
    """
    rk = reward_kwargs or {}
    U_val, grad_U = _compute_grad_U(U_net, xi)

    q_list = []
    for a in range(5):
        xi_next = dynamics.one_step(xi, a)
        delta_xi = xi_next - xi
        r_a = local_reward_from_next(xi, a, xi_next, dynamics, **rk)
        advection = (grad_U * delta_xi).sum(dim=-1)
        q_a = r_a + gamma * advection
        q_list.append(q_a)

    q_all = torch.stack(q_list, dim=-1)
    return q_all, U_val


def hjb_residual(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    gamma: float = 0.99,
    reward_kwargs: dict | None = None,
) -> torch.Tensor:
    """Compute hard-HJB residual: should be zero at optimality.

    rho = U(xi) * ln(gamma) + max_a q_a
    """
    q_all, U_val = pde_q_values(U_net, xi, dynamics, gamma, reward_kwargs)
    max_q = q_all.max(dim=-1).values
    rho = U_val * math.log(gamma) + max_q
    return rho


def soft_hjb_residual(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    gamma: float = 0.99,
    tau: float = 0.1,
    reward_kwargs: dict | None = None,
) -> torch.Tensor:
    """Compute soft-HJB residual: entropy-regularized version.

    rho = U(xi) * ln(gamma) + tau * logsumexp(q_a / tau)
    """
    q_all, U_val = pde_q_values(U_net, xi, dynamics, gamma, reward_kwargs)
    soft_max = tau * torch.logsumexp(q_all / tau, dim=-1)
    rho = U_val * math.log(gamma) + soft_max
    return rho


def soft_policy_from_q(q_all: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Compute soft policy pi_soft(a|xi) = softmax(q_a / tau).

    Args:
        q_all: (batch, 5) Q-values
        tau: temperature
    Returns:
        pi_soft: (batch, 5) probability distribution
    """
    return torch.softmax(q_all / tau, dim=-1)


def eikonal_residual(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    v_min: float = 0.5,
    ttc_thr: float = 3.0,
    m_stop_ref: float = 5.0,
) -> torch.Tensor:
    """Eikonal PDE residual: ||nabla_xi U||^2 - c(xi)^2.

    c(xi) = 1 / v_eff(xi) where v_eff is the maximum safe speed achievable
    by any action. An action is "safe" if the post-action TTC remains above
    ttc_thr. This grounds the Eikonal constraint in the actual dynamics
    and action space rather than hand-crafted heuristic features.
    """
    U_val, grad_U = _compute_grad_U(U_net, xi)
    grad_norm_sq = (grad_U ** 2).sum(dim=-1)

    alpha_cz = xi[:, 8]  # visibility of conflict zone

    # Compute maximum safe speed from dynamics
    safe_speeds = []
    for a in range(5):
        xi_next = dynamics.one_step(xi, a)
        v_next = xi_next[:, IDX_V]
        ttc_next = xi_next[:, IDX_TTC_MIN]
        # Smooth safety indicator: 1.0 if ttc >> ttc_thr, 0.0 if ttc << ttc_thr
        safe_weight = torch.sigmoid((ttc_next - ttc_thr) / 0.5)
        safe_speeds.append(v_next * safe_weight)

    # v_eff = max safe speed across all actions
    v_eff_dynamics = torch.stack(safe_speeds, dim=0).max(dim=0).values

    # Visibility modulation: can't go fast if you can't see
    vis_factor = alpha_cz.clamp(min=0.1)
    v_eff = (v_eff_dynamics * vis_factor).clamp(min=v_min)

    c_sq = (1.0 / v_eff) ** 2

    rho = grad_norm_sq - c_sq
    return rho


def cbf_residual(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    alpha_cbf: float = 1.0,
    cbf_safe_offset: float = 10.0,
) -> torch.Tensor:
    """CBF-PDE residual: ReLU(-max_a [h_dot(xi,a) + alpha*h(xi)]) where h(xi) = U(xi) + offset.

    The barrier function h(xi) = U(xi) + cbf_safe_offset shifts the zero-level
    set so that the safe/unsafe boundary occurs at U = -cbf_safe_offset,
    which is the midpoint between typical safe returns and the collision
    penalty. This ensures:
      - Success states (U ~ 0): h > 0, safely inside the safe set
      - Collision states (U ~ -20): h < 0, in the unsafe set
      - Safety boundary (U = -10): h = 0, meaningful transition point

    Since the offset is constant, grad_h = grad_U -- only the barrier value
    changes, not the gradient computation.
    """
    U_val, grad_U = _compute_grad_U(U_net, xi)

    # Shifted barrier: h(xi) = U(xi) + offset
    h_val = U_val + cbf_safe_offset

    # For each action, compute h_dot = grad_U^T * delta_xi (grad_h = grad_U since offset is const)
    h_dot_list = []
    for a in range(5):
        delta_xi_a = dynamics.one_step(xi, a) - xi
        h_dot_a = (grad_U * delta_xi_a).sum(dim=-1)
        h_dot_list.append(h_dot_a + alpha_cbf * h_val)  # h_dot + alpha*h (shifted h)

    # max over actions: if ANY action satisfies the CBF condition, residual = 0
    h_dot_all = torch.stack(h_dot_list, dim=-1)
    max_h_dot = h_dot_all.max(dim=-1).values

    # Penalize when no action can maintain safety
    rho = torch.relu(-max_h_dot)
    return rho
