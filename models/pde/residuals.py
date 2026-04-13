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

    c(xi) = 1 / v_eff(xi) where v_eff incorporates safety margins.
    """
    U_val, grad_U = _compute_grad_U(U_net, xi)
    grad_norm_sq = (grad_U ** 2).sum(dim=-1)

    # Compute effective safe speed
    v = xi[:, IDX_V].clamp(min=1e-6)
    d_cz = xi[:, IDX_D_CZ]
    ttc_min = xi[:, IDX_TTC_MIN]
    alpha_cz = xi[:, 8]  # visibility of conflict zone

    # Stopping margin: m_stop = d_cz - d_stop(v)
    d_stop_v = v * 0.5 + v ** 2 / (2 * 5.0)  # tau=0.5, a_max=5.0
    m_stop = (d_cz - d_stop_v).clamp(min=0.0)

    # Safety multiplier sigma in (0, 1]
    sigma_stop = torch.sigmoid((m_stop - 1.0) / 1.0)     # smooth ramp around m_stop=1
    sigma_ttc = torch.sigmoid((ttc_min - ttc_thr) / 0.5)  # smooth ramp around TTC threshold
    sigma_vis = alpha_cz.clamp(min=0.1)                    # visibility factor
    sigma_safe = sigma_stop * sigma_ttc * sigma_vis

    v_eff = (v * sigma_safe).clamp(min=v_min)
    c_sq = (1.0 / v_eff) ** 2

    rho = grad_norm_sq - c_sq
    return rho


def cbf_residual(
    U_net: nn.Module,
    xi: torch.Tensor,
    dynamics: BehavioralDynamics,
    alpha_cbf: float = 1.0,
) -> torch.Tensor:
    """CBF-PDE residual: ReLU(-max_a [h_dot(xi,a) + alpha*h(xi)]) where h = U.

    Uses U_phi itself as the barrier function. The CBF condition requires
    that for each state, there exists at least one action maintaining
    forward invariance of the safe superlevel set {xi : U(xi) >= 0}.
    """
    U_val, grad_U = _compute_grad_U(U_net, xi)

    # For each action, compute h_dot = grad_U^T * delta_xi_a (corrected state change)
    h_dot_list = []
    for a in range(5):
        delta_xi_a = dynamics.one_step(xi, a) - xi
        h_dot_a = (grad_U * delta_xi_a).sum(dim=-1)  # grad_U^T * (F_a - xi)
        h_dot_list.append(h_dot_a + alpha_cbf * U_val)  # h_dot + alpha*h

    # max over actions: if ANY action satisfies the CBF condition, residual = 0
    h_dot_all = torch.stack(h_dot_list, dim=-1)
    max_h_dot = h_dot_all.max(dim=-1).values

    # Penalize when no action can maintain safety
    rho = torch.relu(-max_h_dot)
    return rho
