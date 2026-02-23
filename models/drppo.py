"""DRPPO: Recurrent PPO with optional physics-informed critic residuals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any
from collections import deque

from models.physics import PhysicsPredictor, compute_L_ego, compute_L_stop, compute_L_fric, compute_risk_from_state


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
    """Recurrent PPO with optional PINN critic residuals."""

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
            L_ego = torch.tensor(0.0, device=self.device)
            L_stop = torch.tensor(0.0, device=self.device)
            L_fric = torch.tensor(0.0, device=self.device)
            L_risk = torch.tensor(0.0, device=self.device)

            x_next = extra.get("x_next")
            if x_next is not None:
                x_pred = extra.get("x_pred")
                if x_pred is not None:
                    L_ego = compute_L_ego(
                        x_next[:, 0], x_next[:, 1], x_next[:, 2], x_next[:, 3],
                        x_pred[:, 0], x_pred[:, 1], x_pred[:, 2], x_pred[:, 3],
                    )

            d_cz = extra.get("d_cz")
            v = extra.get("v")
            if d_cz is not None and v is not None:
                d_cz_t = torch.FloatTensor(d_cz).to(self.device)
                v_t = torch.FloatTensor(v).to(self.device)
                L_stop = compute_L_stop(d_cz_t, v_t)

            v_all = extra.get("v_all")
            kappa = extra.get("kappa")
            a_lon = extra.get("a_lon")
            if v_all is not None and kappa is not None and a_lon is not None:
                v_t = torch.FloatTensor(v_all).to(self.device)
                kappa_t = torch.FloatTensor(kappa).to(self.device)
                a_lon_t = torch.FloatTensor(a_lon).to(self.device)
                L_fric = compute_L_fric(v_t, kappa_t, a_lon_t)

            ttc_min = extra.get("ttc_min")
            if ttc_min is not None:
                ttc_t = torch.FloatTensor(ttc_min).to(self.device)
                L_risk = torch.relu(3.0 - ttc_t).mean()

            vf_loss = (
                vf_loss
                + self.lambda_ego * L_ego
                + self.lambda_stop * L_stop
                + self.lambda_fric * L_fric
                + self.lambda_risk * L_risk
            )

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
