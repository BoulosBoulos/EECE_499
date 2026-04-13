"""Eikonal auxiliary critic training agent.

Combines standard PPO (recurrent actor-critic) with an auxiliary Eikonal critic
on the reduced PDE state. The auxiliary critic is trained with:
  - L_anchor: MSE to GAE returns
  - L_eik: Eikonal residual squared (||grad U||^2 - c(xi)^2)
  - L_bc: terminal boundary conditions
The PPO critic is distilled from the auxiliary critic via L_distill.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.drppo import RecurrentActorCritic
from models.pde.eikonal_aux_critic import EikonalAuxCritic
from models.pde.state_builder import ReducedPDEState, XI_DIM
from models.pde.dynamics import BehavioralDynamics
from models.pde.residuals import eikonal_residual
from models.pde.collocation import sample_collocation
from models.pde.checkpointing import save_pde_checkpoint, load_pde_checkpoint


class EikonalAuxAgent:
    """Recurrent PPO + Eikonal auxiliary critic agent."""

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
        lambda_eik: float = 0.2,
        lambda_bc: float = 0.5,
        lambda_distill: float = 0.25,
        aux_hidden_dim: int = 256,
        collocation_ratio: float = 0.7,
        hidden_dim: int = 128,
        device: str = "cpu",
        w_coll: float = -20.0,
        v_min: float = 0.5,
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
        self.lambda_eik = lambda_eik
        self.lambda_bc = lambda_bc
        self.lambda_distill = lambda_distill
        self.collocation_ratio = collocation_ratio
        self.device = device
        self.w_coll = w_coll
        self.v_min = v_min
        self.reward_kwargs = reward_kwargs or {}

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)
        self.aux_critic = EikonalAuxCritic(in_dim=XI_DIM, hidden_dim=aux_hidden_dim).to(device)

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

        eik_res_mean = 0.0
        anchor_loss_val = 0.0
        bc_loss_val = 0.0
        distill_loss_val = 0.0

        if extra is not None and "xi_curr" in extra:
            xi_np = extra["xi_curr"]
            xi_t = torch.FloatTensor(xi_np).to(self.device)

            xi_colloc = sample_collocation(xi_t, ratio_real=self.collocation_ratio)

            rho = eikonal_residual(self.aux_critic, xi_colloc, self.dynamics,
                                   v_min=self.v_min)
            L_eik = (rho ** 2).mean()

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
                        self.lambda_eik * L_eik +
                        self.lambda_bc * L_bc)

            self.aux_optimizer.zero_grad()
            aux_loss.backward()
            nn.utils.clip_grad_norm_(self.aux_critic.parameters(), self.max_grad_norm)
            self.aux_optimizer.step()

            with torch.no_grad():
                U_distill = self.aux_critic(xi_t).detach()
            L_distill = F.mse_loss(value[:len(U_distill)], U_distill)
            vf_loss = vf_loss + self.lambda_distill * L_distill

            eik_res_mean = float(rho.detach().abs().mean())
            anchor_loss_val = float(L_anchor.detach())
            bc_loss_val = float(L_bc.detach())
            distill_loss_val = float(L_distill.detach())

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
            "eikonal_residual_mean": eik_res_mean,
            "anchor_loss": anchor_loss_val,
            "bc_loss": bc_loss_val,
            "distill_loss": distill_loss_val,
            "distill_gap": distill_loss_val,
        }

    def save(self, path: str):
        save_pde_checkpoint(
            path=path,
            policy_state=self.policy.state_dict(),
            policy_optim_state=self.optimizer.state_dict(),
            aux_state=self.aux_critic.state_dict(),
            aux_optim_state=self.aux_optimizer.state_dict(),
            obs_dim=self.obs_dim,
            method="eikonal_aux",
            config={
                "lambda_anchor": self.lambda_anchor,
                "lambda_eik": self.lambda_eik,
                "lambda_bc": self.lambda_bc,
                "lambda_distill": self.lambda_distill,
                "v_min": self.v_min,
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
