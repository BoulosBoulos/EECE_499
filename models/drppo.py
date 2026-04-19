"""DRPPO: Recurrent PPO baseline (no physics augmentation).

Used as the baseline comparison against PDE-augmented agents.
The heuristic physics methods (Design A/B, safety filter, L_ego)
have been removed. PDE methods are in models/pde/.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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
    """Recurrent PPO baseline (no physics augmentation).

    Used as the baseline comparison against PDE-augmented agents.
    Legacy physics parameters are accepted for backward compatibility
    with existing experiment scripts but are ignored.
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
        hidden_dim: int = 128,
        device: str = "cpu",
        # Legacy params kept for backward compatibility (ignored)
        pinn_placement: str = "none",
        use_l_ego: bool = False,
        use_safety_filter: bool = False,
        use_ttc: bool = True,
        use_stop: bool = True,
        use_fric: bool = True,
        lambda_physics_critic: float = 0.5,
        lambda_physics_actor: float = 0.1,
        lambda_physics_ttc: float = 1.0,
        lambda_physics_stop: float = 1.0,
        lambda_physics_fric: float = 1.0,
        lambda_physics_ego: float = 0.1,
        physics_ttc_thr: float = 3.0,
        physics_tau: float = 0.5,
        a_max: float = 5.0,
        mu: float = 0.8,
        g: float = 9.81,
        use_pinn: bool | None = None,
        **_kwargs,
    ):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self._hidden = None

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

            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(action).item()
            return action.item(), new_hidden, log_prob, value.item()

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

        loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "family": "drppo",
            "config": {
                "obs_dim": self.obs_dim,
                "hidden_dim": self.hidden_dim,
                "n_actions": self.n_actions,
                "lr": self.lr,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm,
            },
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        # Backward compatibility: old checkpoints lack "family" / "config"
        if "policy" not in data:
            raise ValueError(
                f"Checkpoint at {path} is missing the 'policy' key. "
                "Cannot load model weights."
            )
        if "family" in data and data["family"] != "drppo":
            raise ValueError(
                f"Checkpoint family is '{data['family']}', expected 'drppo'. "
                "Are you loading the wrong model type?"
            )
        if "config" in data:
            saved_obs = data["config"].get("obs_dim")
            if saved_obs is not None and saved_obs != self.obs_dim:
                raise ValueError(
                    f"obs_dim mismatch: checkpoint has {saved_obs}, "
                    f"but model was created with {self.obs_dim}."
                )
        self.policy.load_state_dict(data["policy"])
        if "optimizer" in data:
            self.optimizer.load_state_dict(data["optimizer"])
