"""DRPPO: Recurrent PPO with physics-informed critic (Design A).

Design A: L_physics_critic penalizes V(s) when physics is violated.
Gradients flow through the critic, improving safety-aware value estimates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any
from collections import deque

from models.physics import PhysicsPredictor


def _compute_physics_critic_loss(
    value: torch.Tensor,
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
    lambda_ttc: float = 1.0,
    lambda_stop: float = 1.0,
    lambda_fric: float = 1.0,
) -> torch.Tensor:
    """
    Design A: Penalize critic V(s) when physics is violated.
    L = E[ V(s) * (λ_ttc*violation_ttc + λ_stop*violation_stop + λ_fric*violation_fric) ]
    Gradients flow through value -> critic learns to assign lower V when unsafe.
    """
    # TTC violation: dangerous when ttc < threshold
    violation_ttc = F.relu(ttc_thr - ttc_min)

    # Stopping-distance violation: too close to conflict zone for current speed
    # d_stop(v) = v*tau + v^2/(2*a_max); violation when d_cz < d_stop
    d_stop = v * tau + (v ** 2) / (2 * a_max)
    violation_stop = F.relu(-(d_cz - d_stop))

    # Friction-circle violation: a_lat^2 + a_lon^2 > (mu*g)^2
    a_lat = (v ** 2) * torch.abs(kappa)
    constraint = a_lat.pow(2) + a_lon.pow(2) - (mu * g) ** 2
    violation_fric = F.relu(constraint)

    # Penalize high V(s) when any violation is positive; per-residual weights
    total_violation = (
        lambda_ttc * violation_ttc
        + lambda_stop * violation_stop
        + lambda_fric * violation_fric
    )
    return (value * total_violation).mean()


class RecurrentActorCritic(nn.Module):
    """GRU encoder + actor (categorical) + critic (value)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        n_actions: int = 5,
        n_layers: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (B, T, obs_dim) or (B, obs_dim) for single step
        Returns: logits, value, log_prob, action
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        out, new_hidden = self.gru(obs, hidden)
        h = out[:, -1]
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return logits, value, log_prob, action

    def get_value(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        _, h = self.gru(obs, hidden)
        return self.critic(h[-1]).squeeze(-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
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


class DRPPO:
    """Recurrent PPO with physics-informed critic (Design A)."""

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
        use_pinn: bool = True,
        use_l_ego: bool = False,
        lambda_physics_critic: float = 0.5,
        lambda_physics_ttc: float = 1.0,
        lambda_physics_stop: float = 1.0,
        lambda_physics_fric: float = 1.0,
        lambda_physics_ego: float = 0.1,
        physics_ttc_thr: float = 3.0,
        physics_tau: float = 0.5,
        a_max: float = 5.0,
        mu: float = 0.8,
        g: float = 9.81,
        lambda_ego: float = 0.5,
        lambda_stop: float = 0.5,
        lambda_fric: float = 0.3,
        lambda_risk: float = 0.5,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_pinn = use_pinn
        self.use_l_ego = use_l_ego
        self.lambda_physics_critic = lambda_physics_critic
        self.lambda_physics_ego = lambda_physics_ego
        self.lambda_physics_ttc = lambda_physics_ttc
        self.lambda_physics_stop = lambda_physics_stop
        self.lambda_physics_fric = lambda_physics_fric
        self.physics_ttc_thr = physics_ttc_thr
        self.physics_tau = physics_tau
        self.a_max = a_max
        self.mu = mu
        self.g = g
        self.lambda_ego = lambda_ego
        self.lambda_stop = lambda_stop
        self.lambda_fric = lambda_fric
        self.lambda_risk = lambda_risk
        self.device = device

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim,
            hidden_dim=128,
            n_actions=n_actions,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.physics = PhysicsPredictor(dt=0.1)

    def get_action(
        self,
        obs: np.ndarray,
        hidden: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[int, np.ndarray, float, float]:
        with torch.no_grad():
            o = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            logits, value, log_prob, action = self.policy(o, None)
            if deterministic:
                action = logits.argmax(dim=-1)
            return (
                action.item(),
                None,
                log_prob.item(),
                value.item(),
            )

    def train_step(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        extra: Optional[dict] = None,
    ) -> dict[str, float]:
        """Single PPO update. extra can contain physics rollout data for PINN residuals."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        value, log_prob, entropy = self.policy.evaluate_actions(obs_t, actions_t, None)
        ratio = torch.exp(log_prob - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        # Critic loss: RL + optional PINN residuals
        vf_loss = F.mse_loss(value, returns_t)

        if self.use_pinn and extra is not None:
            # Design A: Physics-informed critic - penalize V(s) when physics violated
            # Gradients flow through value -> critic learns safety-aware value estimates
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
                L_physics_critic = _compute_physics_critic_loss(
                    value, ttc_t, d_cz_t, v_t, kappa_t, a_lon_t,
                    ttc_thr=self.physics_ttc_thr,
                    tau=self.physics_tau,
                    a_max=self.a_max,
                    mu=self.mu,
                    g=self.g,
                    lambda_ttc=self.lambda_physics_ttc,
                    lambda_stop=self.lambda_physics_stop,
                    lambda_fric=self.lambda_physics_fric,
                )
                vf_loss = vf_loss + self.lambda_physics_critic * L_physics_critic

            # L_ego: penalize V(s_next) when ego dynamics prediction error is high
            if self.use_l_ego:
                ego_valid = extra.get("ego_valid")
                ego_x_prev = extra.get("ego_x_prev")
                ego_y_prev = extra.get("ego_y_prev")
                ego_psi_prev = extra.get("ego_psi_prev")
                ego_v_prev = extra.get("ego_v_prev")
                ego_x_next = extra.get("ego_x_next")
                ego_y_next = extra.get("ego_y_next")
                ego_psi_next = extra.get("ego_psi_next")
                ego_v_next = extra.get("ego_v_next")
                ego_action_prev = extra.get("ego_action_prev")
                if all(
                    x is not None
                    for x in [
                        ego_valid,
                        ego_x_prev,
                        ego_x_next,
                        ego_action_prev,
                    ]
                ):
                    valid = torch.BoolTensor(ego_valid).to(self.device)
                    if valid.any():
                        x_p = torch.FloatTensor(ego_x_prev).to(self.device)
                        y_p = torch.FloatTensor(ego_y_prev).to(self.device)
                        psi_p = torch.FloatTensor(ego_psi_prev).to(self.device)
                        v_p = torch.FloatTensor(ego_v_prev).to(self.device)
                        x_n = torch.FloatTensor(ego_x_next).to(self.device)
                        y_n = torch.FloatTensor(ego_y_next).to(self.device)
                        psi_n = torch.FloatTensor(ego_psi_next).to(self.device)
                        v_n = torch.FloatTensor(ego_v_next).to(self.device)
                        a_prev = torch.LongTensor(ego_action_prev).to(self.device)
                        a_flat, d_flat = [], []
                        for k in range(a_prev.shape[0]):
                            ak, dk = self.physics.action_to_control(a_prev[k].item())
                            a_flat.append(ak)
                            d_flat.append(dk)
                        a_t = torch.FloatTensor(a_flat).to(self.device).unsqueeze(1)
                        d_t = torch.FloatTensor(d_flat).to(self.device).unsqueeze(1)
                        x_pred, y_pred, psi_pred, v_pred = self.physics.one_step_euler(
                            x_p.unsqueeze(1), y_p.unsqueeze(1), psi_p.unsqueeze(1),
                            v_p.clamp(min=1e-4).unsqueeze(1), a_t, d_t,
                        )
                        x_pred = x_pred.squeeze(1)
                        y_pred = y_pred.squeeze(1)
                        psi_pred = psi_pred.squeeze(1)
                        v_pred = v_pred.squeeze(1)
                        err = (
                            (x_n - x_pred).pow(2) + (y_n - y_pred).pow(2)
                            + (psi_n - psi_pred).pow(2) + (v_n - v_pred).pow(2)
                        )
                        err_masked = torch.where(valid, err, torch.zeros_like(err))
                        n_valid = valid.sum().float().clamp(min=1)
                        L_ego_weighted = (value * err_masked).sum() / n_valid
                        vf_loss = vf_loss + self.lambda_physics_ego * self.lambda_physics_critic * L_ego_weighted

        loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "vf_loss": vf_loss.item() if isinstance(vf_loss, torch.Tensor) else vf_loss,
            "entropy": -entropy_loss.item(),
        }

    def save(self, path: str):
        torch.save({"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
