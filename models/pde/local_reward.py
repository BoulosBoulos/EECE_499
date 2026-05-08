"""Local reward surrogate for PDE-based critics.

Mirrors the env reward structure but operates on the reduced PDE state xi.
Does NOT include terminal rewards (collision / success) -- those are handled
via boundary conditions in the PDE loss (each PDE agent anchors U at the
terminal target value at success/collision states).

Phase 31 Stage 1B fix: env w_success increased +10 -> +200. Matching change
applied to PDE agent boundary conditions: HJB / Soft-HJB / CBF / Eikonal /
Fusion now anchor U at +200 at success_terminal states (was 0). Collision
boundary unchanged at w_coll = -20. This keeps PDE residuals consistent
with the new env reward; without it, U(success)=0 in PDE but +200 in env,
breaking the surrogate's value-landscape approx.
See verification/phase31_investigation_2_dense.json for diagnosis.

Phase 31 Stage 1D fix: potential-based reward shaping per Ng, Harada, Russell
(ICML 1999, Theorem 1), finite-horizon form per Wiewiora (JAIR 2003).
F(s, a, s') = gamma_shaping * Phi(s') - Phi(s); Phi(s) = -d_route(s).
With gamma_shaping = 1 the cumulative shaping over an episode telescopes
EXACTLY to Phi(s_T) - Phi(s_0), regardless of whether s_T is terminal
or truncated; this eliminates the (1-gamma)*T*mean_d drift bias that
Stage 1C (gamma=0.99) allowed the agent to reward-hack
(+209k mean_reward on 0% success in 2_dense). Stage 1C also used
Euclidean distance, which let the policy hover near the exit; Stage 1D
uses route distance, forcing the agent to drive through the
intersection. Env reads d_route via SUMO's getDistanceRoad. PDE
surrogate uses xi[IDX_D_EXIT] (route distance to CZ exit) as the
closest analog in the reduced PDE state. The PDE residual functional
form is unchanged: rho = U * ln(gamma_agent) + max_a [r_shaped + grad(U) . f_a];
the auxiliary critic learns the shaped value function and the optimal
policy follows by Ng et al.'s policy-invariance theorem (the shaping
is independent of the agent's discount gamma_agent).
"""

from __future__ import annotations
import torch
from models.pde.state_builder import IDX_V, IDX_TTC_MIN, IDX_POTHOLE, IDX_D_CZ, IDX_D_EXIT
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
    # Phase 31 Stage 1D: shaping defaults match env reward cfg.
    # gamma_shaping=1.0 is the Wiewiora (JAIR 2003) finite-horizon form;
    # w_shaping is scaled to success_bonus / typical_initial_route_distance.
    gamma_shaping: float = 1.0,
    w_shaping: float = 3.0,
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

    # Risk: sharp sigmoid on (ttc_thr - ttc_next), gated by ego speed.
    # Phase 31 Stage 1 fix: this gating must match env/sumo_env.py:1308-1310
    # (speed-gate w_risk so a stopped ego near agents is not penalised).
    speed_gate = torch.sigmoid((v - 0.5) / 0.1)
    r = r + w_risk * torch.sigmoid((ttc_thr - ttc_next) / 0.1) * speed_gate

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

    # Phase 31 Stage 1D fix: potential-based shaping per Ng, Harada, Russell
    # (ICML 1999), finite-horizon form per Wiewiora (JAIR 2003).
    # gamma_shaping = 1 makes per-step F = -d_next + d_curr telescope to
    # d_route(s_0) - d_route(s_T) over any episode, with no drift bias.
    # PDE surrogate uses xi[IDX_D_EXIT] (route distance to CZ exit) as the
    # PDE-state proxy for the env's d_route. PDE residual is unchanged:
    # rho = U * ln(gamma_agent) + max_a [r_shaped + grad(U) . f_a].
    d_exit_curr = xi[:, IDX_D_EXIT]
    d_exit_next = xi_next[:, IDX_D_EXIT]
    shaping = gamma_shaping * (-d_exit_next) - (-d_exit_curr)
    r = r + w_shaping * shaping

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
