"""Soft-HJB auxiliary critic training agent.

Like HJBAuxAgent but uses:
  - Soft-HJB residual (logsumexp instead of hard max)
  - Actor-alignment KL term: aligns the PPO actor to the soft policy
    induced by the PDE Q-values
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.drppo import RecurrentActorCritic
from models.pde.soft_hjb_aux_critic import SoftHJBAuxCritic
from models.pde.state_builder import ReducedPDEState, XI_DIM
from models.pde.dynamics import BehavioralDynamics
from models.pde.residuals import soft_hjb_residual, pde_q_values, soft_policy_from_q
from models.pde.collocation import sample_collocation
from models.pde.local_reward import local_reward
from models.pde.checkpointing import save_pde_checkpoint, load_pde_checkpoint


class SoftHJBAuxAgent:
    """Recurrent PPO + Soft-HJB auxiliary critic agent with actor alignment."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        lr: float = 3e-4,
        aux_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        lambda_anchor: float = 1.0,
        lambda_soft: float = 0.2,
        lambda_bc: float = 0.5,
        lambda_distill: float = 0.25,
        lambda_align: float = 0.05,
        tau_soft: float = 0.1,
        aux_hidden_dim: int = 256,
        collocation_ratio: float = 0.7,
        hidden_dim: int = 128,
        device: str = "cpu",
        w_coll: float = -20.0,
        reward_kwargs: dict | None = None,
    ):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lambda_anchor = lambda_anchor
        self.lambda_soft = lambda_soft
        self.lambda_bc = lambda_bc
        self.lambda_distill = lambda_distill
        self.lambda_align = lambda_align
        self.tau_soft = tau_soft
        self.collocation_ratio = collocation_ratio
        self.device = device
        self.w_coll = w_coll
        self.reward_kwargs = reward_kwargs or {}

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)
        self.aux_critic = SoftHJBAuxCritic(in_dim=XI_DIM, hidden_dim=aux_hidden_dim).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.aux_optimizer = torch.optim.Adam(self.aux_critic.parameters(), lr=aux_lr)

        self.dynamics = BehavioralDynamics()
        self.pde_state_builder = ReducedPDEState()
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

        soft_res_mean = 0.0
        anchor_loss_val = 0.0
        bc_loss_val = 0.0
        distill_loss_val = 0.0
        actor_align_kl_val = 0.0

        if extra is not None and "xi_curr" in extra:
            xi_np = extra["xi_curr"]
            xi_t = torch.FloatTensor(xi_np).to(self.device)

            xi_colloc = sample_collocation(xi_t, ratio_real=self.collocation_ratio)

            rho = soft_hjb_residual(self.aux_critic, xi_colloc, self.dynamics,
                                    gamma=self.gamma, tau=self.tau_soft,
                                    reward_kwargs=self.reward_kwargs)
            L_soft = (rho ** 2).mean()

            U_rollout = self.aux_critic(xi_t)
            L_anchor = F.mse_loss(U_rollout, returns_t[:len(U_rollout)])

            L_bc = torch.tensor(0.0, device=self.device)
            if "success_terminal" in extra:
                succ = torch.BoolTensor(extra["success_terminal"]).to(self.device)
                if succ.any():
                    succ_xi = xi_t[succ[:len(xi_t)]]
                    if len(succ_xi) > 0:
                        L_bc = L_bc + (self.aux_critic(succ_xi) ** 2).mean()
            if "collision_terminal" in extra:
                coll = torch.BoolTensor(extra["collision_terminal"]).to(self.device)
                if coll.any():
                    coll_xi = xi_t[coll[:len(xi_t)]]
                    if len(coll_xi) > 0:
                        L_bc = L_bc + ((self.aux_critic(coll_xi) - self.w_coll) ** 2).mean()

            aux_loss = (self.lambda_anchor * L_anchor +
                        self.lambda_soft * L_soft +
                        self.lambda_bc * L_bc)

            self.aux_optimizer.zero_grad()
            aux_loss.backward()
            nn.utils.clip_grad_norm_(self.aux_critic.parameters(), self.max_grad_norm)
            self.aux_optimizer.step()

            with torch.no_grad():
                U_distill = self.aux_critic(xi_t).detach()
            L_distill = F.mse_loss(value[:len(U_distill)], U_distill)
            vf_loss = vf_loss + self.lambda_distill * L_distill

            with torch.no_grad():
                q_all, _ = pde_q_values(self.aux_critic, xi_t, self.dynamics,
                                        gamma=self.gamma, reward_kwargs=self.reward_kwargs)
                pi_soft = soft_policy_from_q(q_all, tau=self.tau_soft)

            if obs_t.dim() == 2:
                obs_for_eval = obs_t.unsqueeze(1)
            else:
                obs_for_eval = obs_t
            out, _ = self.policy.gru(obs_for_eval, hidden_t)
            h = out[:, -1]
            actor_logits = self.policy.actor(h)
            pi_theta = torch.softmax(actor_logits, dim=-1)

            n_align = min(len(pi_soft), len(pi_theta))
            pi_s = pi_soft[:n_align].clamp(min=1e-8)
            pi_t = pi_theta[:n_align].clamp(min=1e-8)
            kl = (pi_s * (pi_s.log() - pi_t.log())).sum(dim=-1).mean()
            actor_loss = actor_loss + self.lambda_align * kl

            soft_res_mean = float(rho.detach().abs().mean())
            anchor_loss_val = float(L_anchor.detach())
            bc_loss_val = float(L_bc.detach())
            distill_loss_val = float(L_distill.detach())
            actor_align_kl_val = float(kl.detach())

        loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "vf_loss": float(vf_loss.item()) if isinstance(vf_loss, torch.Tensor) else vf_loss,
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            "soft_residual_mean": soft_res_mean,
            "anchor_loss": anchor_loss_val,
            "bc_loss": bc_loss_val,
            "distill_loss": distill_loss_val,
            "distill_gap": distill_loss_val,
            "actor_align_kl": actor_align_kl_val,
        }

    def save(self, path: str):
        save_pde_checkpoint(
            path=path,
            policy_state=self.policy.state_dict(),
            policy_optim_state=self.optimizer.state_dict(),
            aux_state=self.aux_critic.state_dict(),
            aux_optim_state=self.aux_optimizer.state_dict(),
            obs_dim=self.obs_dim,
            method="soft_hjb_aux",
            config={
                "lambda_anchor": self.lambda_anchor,
                "lambda_soft": self.lambda_soft,
                "lambda_bc": self.lambda_bc,
                "lambda_distill": self.lambda_distill,
                "lambda_align": self.lambda_align,
                "tau_soft": self.tau_soft,
                "gamma": self.gamma,
            },
        )

    def load(self, path: str):
        data = load_pde_checkpoint(path, device=self.device)
        self.policy.load_state_dict(data["policy"])
        if "policy_optimizer" in data:
            self.optimizer.load_state_dict(data["policy_optimizer"])
        self.aux_critic.load_state_dict(data["aux_critic"])
        if "aux_optimizer" in data:
            self.aux_optimizer.load_state_dict(data["aux_optimizer"])
