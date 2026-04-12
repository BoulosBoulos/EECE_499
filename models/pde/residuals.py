"""PDE residual computations for HJB and Soft-HJB critics.

Hard-HJB residual: rho = U(xi) * ln(gamma) + max_a [r(xi,a) + grad_U . f_a(xi)]
Soft-HJB residual: rho = U(xi) * ln(gamma) + tau * logsumexp([r + grad_U . f_a] / tau)

Both require autograd for nabla_xi U.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
from models.pde.dynamics import BehavioralDynamics
from models.pde.local_reward import local_reward


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

    q_a = r(xi, a) + grad_U(xi)^T f_a(xi)

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
        r_a = local_reward(xi, a, dynamics, **rk)
        f_a = dynamics.drift(xi, a)
        advection = (grad_U * f_a).sum(dim=-1)
        q_a = r_a + advection
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
