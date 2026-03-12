"""DRPPO: Recurrent PPO with plug-and-play physics augmentation.

Plug-and-play options (all toggled independently):
  - pinn_placement: "critic" | "actor" | "both" | "none"
  - use_safety_filter: physics-based action override (STOP when constraints violated)
  - Per-residual toggles: use_ttc, use_stop, use_fric
  - use_l_ego: ego dynamics prediction error term

Design A (critic): L = E[ V(s) * violation(s) ]
Design B (actor):  L = E[ violation(s).detach() * log_prob ]
Safety filter:     override action to STOP when d_cz < d_stop(v) or friction violated
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.physics import PhysicsPredictor


def _compute_physics_violations(
    ttc_min: torch.Tensor,
    d_cz: torch.Tensor,
    v: torch.Tensor,
    kappa: torch.Tensor,
    a_lon: torch.Tensor,
    ttc_thr: float = 3.0,
    tau: float = 0.5,
    a_max: float = 5.0,
    mu: float = 0.8,
    g: float = 9.81,
    use_ttc: bool = True,
    use_stop: bool = True,
    use_fric: bool = True,
    lambda_ttc: float = 1.0,
    lambda_stop: float = 1.0,
    lambda_fric: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute per-term physics violations and aggregate stats.
    Returns (total_violation_per_sample, stats_dict).
    """
    zero = torch.zeros_like(v)

    violation_ttc = F.relu(ttc_thr - ttc_min) if use_ttc else zero
    d_stop = v * tau + (v ** 2) / (2 * a_max)
    violation_stop = F.relu(-(d_cz - d_stop)) if use_stop else zero
    a_lat = (v ** 2) * torch.abs(kappa)
    constraint = a_lat.pow(2) + a_lon.pow(2) - (mu * g) ** 2
    violation_fric = F.relu(constraint) if use_fric else zero

    total = lambda_ttc * violation_ttc + lambda_stop * violation_stop + lambda_fric * violation_fric

    n = float(v.numel()) if v.numel() > 0 else 1.0
    stats = {
        "viol_ttc_rate": float((violation_ttc > 0).sum()) / n,
        "viol_stop_rate": float((violation_stop > 0).sum()) / n,
        "viol_fric_rate": float((violation_fric > 0).sum()) / n,
        "viol_ttc_mag": float(violation_ttc.mean()),
        "viol_stop_mag": float(violation_stop.mean()),
        "viol_fric_mag": float(violation_fric.mean()),
    }
    return total, stats


class RecurrentActorCritic(nn.Module):
    """GRU encoder + actor (categorical) + critic (value)."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_actions: int = 5, n_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        out, new_hidden = self.gru(obs, hidden)
        h = out[:, -1]
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return logits, value, log_prob, action, new_hidden

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor, hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        out, _ = self.gru(obs, hidden)
        h = out[:, -1]
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy

    def get_value(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        out, _ = self.gru(obs, hidden)
        return self.critic(out[:, -1]).squeeze(-1)

    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)


class DRPPO:
    """Recurrent PPO with plug-and-play physics augmentation.

    pinn_placement controls where physics losses are applied:
      "critic" - Design A: V(s) * violation (default)
      "actor"  - Design B: violation.detach() * log_prob
      "both"   - Both critic and actor
      "none"   - No physics loss (baseline)

    use_safety_filter overrides actions to STOP when d_cz < d_stop(v).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Plug-and-play PINN
        pinn_placement: str = "critic",
        use_l_ego: bool = False,
        use_safety_filter: bool = False,
        # Per-residual toggles
        use_ttc: bool = True,
        use_stop: bool = True,
        use_fric: bool = True,
        # Lambda weights
        lambda_physics_critic: float = 0.5,
        lambda_physics_actor: float = 0.1,
        lambda_physics_ttc: float = 1.0,
        lambda_physics_stop: float = 1.0,
        lambda_physics_fric: float = 1.0,
        lambda_physics_ego: float = 0.1,
        # Physics constants
        physics_ttc_thr: float = 3.0,
        physics_tau: float = 0.5,
        a_max: float = 5.0,
        mu: float = 0.8,
        g: float = 9.81,
        # Architecture
        hidden_dim: int = 128,
        device: str = "cpu",
        # Legacy backward compat (use_pinn=True/False maps to pinn_placement)
        use_pinn: bool | None = None,
        **_kwargs,
    ):
        if use_pinn is not None:
            pinn_placement = "critic" if use_pinn else "none"

        self.pinn_placement = pinn_placement
        self.use_l_ego = use_l_ego
        self.use_safety_filter = use_safety_filter
        self.use_ttc = use_ttc
        self.use_stop = use_stop
        self.use_fric = use_fric

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.lambda_physics_critic = lambda_physics_critic
        self.lambda_physics_actor = lambda_physics_actor
        self.lambda_physics_ego = lambda_physics_ego
        self.lambda_physics_ttc = lambda_physics_ttc
        self.lambda_physics_stop = lambda_physics_stop
        self.lambda_physics_fric = lambda_physics_fric
        self.physics_ttc_thr = physics_ttc_thr
        self.physics_tau = physics_tau
        self.a_max = a_max
        self.mu = mu
        self.g = g
        self.device = device

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.physics = PhysicsPredictor(
            dt=_kwargs.get("dt", 0.1),
            a_brake=a_max,
            a_go=_kwargs.get("a_go", 2.0),
            a_max=a_max,
            mu=mu,
            g=g,
        )
        self._hidden = None

    @property
    def use_pinn(self) -> bool:
        return self.pinn_placement != "none"

    def reset_hidden(self):
        self._hidden = self.policy.init_hidden(batch_size=1, device=self.device)

    def get_action(
        self, obs: np.ndarray, hidden: Optional[torch.Tensor] = None, deterministic: bool = False,
    ) -> tuple[int, Optional[torch.Tensor], float, float]:
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            h_in = hidden if hidden is not None else self._hidden
            logits, value, _lp, action, new_hidden = self.policy(o, h_in)
            self._hidden = new_hidden

            if deterministic:
                action = logits.argmax(dim=-1)

            # Safety filter: override to STOP when physics constraints violated.
            # When overriding, store log_prob=0.0 (probability 1.0) to signal
            # that this was a deterministic intervention, not a policy sample.
            # This prevents unstable PPO importance ratios from off-policy data.
            safety_override = False
            if self.use_safety_filter and len(obs) > 9:
                v_ego = float(obs[0])
                d_cz_obs = float(obs[7])
                kappa_obs = float(obs[9])
                a_lon_obs = float(obs[1])
                d_stop_req = v_ego * self.physics_tau + v_ego ** 2 / (2 * self.a_max)
                stop_violated = d_cz_obs > 0 and d_cz_obs < d_stop_req and v_ego > 0.5
                a_lat = v_ego ** 2 * abs(kappa_obs)
                fric_violated = a_lat ** 2 + a_lon_obs ** 2 > (self.mu * self.g) ** 2
                if stop_violated or fric_violated:
                    action = torch.tensor([0], device=self.device)
                    safety_override = True

            if safety_override:
                final_log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(logits=logits)
                final_log_prob = dist.log_prob(action).item()
            return action.item(), new_hidden, final_log_prob, value.item()

    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        hiddens: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> dict[str, float]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Use stored hidden states for correct GRU context during training
        hidden_t = None
        if hiddens is not None:
            hidden_t = torch.FloatTensor(hiddens).to(self.device)
            if hidden_t.dim() == 3:
                hidden_t = hidden_t.permute(1, 0, 2).contiguous()

        value, log_prob, entropy = self.policy.evaluate_actions(obs_t, actions_t, hidden_t)

        ratio = torch.exp(log_prob - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()
        vf_loss = F.mse_loss(value, returns_t)

        l_physics_val = 0.0
        l_actor_physics_val = 0.0
        l_ego_val = 0.0
        phys_stats = {}

        # Compute physics violations (shared between critic and actor losses)
        violation = None
        if self.pinn_placement != "none" and extra is not None:
            ttc_min = extra.get("ttc_min")
            d_cz = extra.get("d_cz")
            v = extra.get("v")
            kappa = extra.get("kappa")
            a_lon = extra.get("a_lon")
            if all(x is not None for x in [ttc_min, d_cz, v, kappa, a_lon]):
                ttc_t = torch.FloatTensor(ttc_min).to(self.device)
                d_cz_t = torch.FloatTensor(d_cz).to(self.device)
                v_t = torch.FloatTensor(v).to(self.device)
                kappa_t = torch.FloatTensor(kappa).to(self.device)
                a_lon_t = torch.FloatTensor(a_lon).to(self.device)
                violation, phys_stats = _compute_physics_violations(
                    ttc_t, d_cz_t, v_t, kappa_t, a_lon_t,
                    ttc_thr=self.physics_ttc_thr, tau=self.physics_tau,
                    a_max=self.a_max, mu=self.mu, g=self.g,
                    use_ttc=self.use_ttc, use_stop=self.use_stop, use_fric=self.use_fric,
                    lambda_ttc=self.lambda_physics_ttc,
                    lambda_stop=self.lambda_physics_stop,
                    lambda_fric=self.lambda_physics_fric,
                )

        # Design A: Critic PINN — penalize V(s) when physics violated
        if violation is not None and self.pinn_placement in ("critic", "both"):
            L_critic = (value * violation).mean()
            l_physics_val = L_critic.item()
            vf_loss = vf_loss + self.lambda_physics_critic * L_critic

        # Design B: Actor PINN — reduce probability of actions taken in violating states.
        # violation >= 0, log_prob <= 0.  Minimizing violation * log_prob pushes
        # log_prob more negative (lower probability) when violation is large.
        if violation is not None and self.pinn_placement in ("actor", "both"):
            L_actor_phys = (violation.detach() * log_prob).mean()
            l_actor_physics_val = L_actor_phys.item()
            actor_loss = actor_loss + self.lambda_physics_actor * L_actor_phys

        # L_ego: penalize V(s) when ego dynamics prediction error is high
        if self.use_l_ego and extra is not None:
            ego_valid = extra.get("ego_valid")
            ego_x_prev = extra.get("ego_x_prev")
            ego_x_next = extra.get("ego_x_next")
            ego_action_prev = extra.get("ego_action_prev")
            if all(x is not None for x in [ego_valid, ego_x_prev, ego_x_next, ego_action_prev]):
                valid = torch.BoolTensor(ego_valid).to(self.device)
                if valid.any():
                    x_p = torch.FloatTensor(extra["ego_x_prev"]).to(self.device)
                    y_p = torch.FloatTensor(extra["ego_y_prev"]).to(self.device)
                    psi_p = torch.FloatTensor(extra["ego_psi_prev"]).to(self.device)
                    v_p = torch.FloatTensor(extra["ego_v_prev"]).to(self.device)
                    x_n = torch.FloatTensor(extra["ego_x_next"]).to(self.device)
                    y_n = torch.FloatTensor(extra["ego_y_next"]).to(self.device)
                    psi_n = torch.FloatTensor(extra["ego_psi_next"]).to(self.device)
                    v_n = torch.FloatTensor(extra["ego_v_next"]).to(self.device)
                    a_prev = torch.LongTensor(ego_action_prev).to(self.device)
                    kappa_arr = extra.get("kappa")
                    a_flat, d_flat = [], []
                    for k in range(a_prev.shape[0]):
                        k_val = float(kappa_arr[k]) if kappa_arr is not None else 0.0
                        ak, dk = self.physics.action_to_control(a_prev[k].item(), kappa=k_val)
                        a_flat.append(ak)
                        d_flat.append(dk)
                    a_t = torch.FloatTensor(a_flat).to(self.device).unsqueeze(1)
                    d_t = torch.FloatTensor(d_flat).to(self.device).unsqueeze(1)
                    x_pred, y_pred, psi_pred, v_pred = self.physics.one_step_euler(
                        x_p.unsqueeze(1), y_p.unsqueeze(1), psi_p.unsqueeze(1),
                        v_p.clamp(min=1e-4).unsqueeze(1), a_t, d_t,
                    )
                    err = (
                        (x_n - x_pred.squeeze(1)).pow(2) + (y_n - y_pred.squeeze(1)).pow(2)
                        + (psi_n - psi_pred.squeeze(1)).pow(2) + (v_n - v_pred.squeeze(1)).pow(2)
                    )
                    err_masked = torch.where(valid, err, torch.zeros_like(err))
                    n_valid = valid.sum().float().clamp(min=1)
                    L_ego = (value * err_masked).sum() / n_valid
                    l_ego_val = L_ego.item()
                    vf_loss = vf_loss + self.lambda_physics_ego * L_ego

        loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        result = {
            "actor_loss": actor_loss.item(),
            "vf_loss": vf_loss.item() if isinstance(vf_loss, torch.Tensor) else vf_loss,
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            "l_physics": l_physics_val,
            "l_actor_physics": l_actor_physics_val,
            "l_ego": l_ego_val,
        }
        result.update(phys_stats)
        return result

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(data["policy"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
