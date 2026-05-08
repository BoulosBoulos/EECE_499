"""Fusion auxiliary critic agent: Soft-HJB optimality + CBF safety.

Phase 1C Option A architecture, with the Phase-3F pre-Tier-1 verification
fix applied (see verification/fusion_bug_artifact.md and the
fusion_fix_verification.json file). The pre-fix Decision F locked the
distill target to U_optimality only, which made U_safety architecturally
unused by the policy. The fix routes both critics into the policy via a
convex combination weighted by w_optimality / w_safety:

    U_optimality(xi)  ── Soft-HJB residual ── L_soft  ─┐ (aux_opt_optimality)
                                                       │
                                                       │  V_PPO  ← (w_o·U_opt + w_s·U_saf) / (w_o + w_s)
                                                       │  pi_eik ← softmax((w_o·q_opt + w_s·q_saf)/(w_o+w_s) / tau)
                                                       │
    U_safety(xi)      ── CBF residual ────── L_cbf  ───┘ (aux_opt_safety)

Two completely independent critic networks (no shared weights, no shared
gradients during the per-critic auxiliary update). The convex combination
is applied AFTER both critics have been updated; the combined target is
.detach()'d before entering the policy's value-head distillation and
actor-KL terms, so neither critic backpropagates from the policy's loss.

Three optimizers:
  - policy_optimizer   — PPO surrogate + entropy + value (vf_loss includes
                         distill_loss against the convex combination
                         (w_o·U_opt + w_s·U_saf).detach()) + actor-KL
                         against softmax((w_o·q_opt + w_s·q_saf).detach()/tau).
                         Owns ONLY policy params (actor + value head + GRU).
  - aux_opt_optimality — Soft-HJB residual + anchor + BC for U_optimality.
                         Owns ONLY U_optimality params.
  - aux_opt_safety     — CBF residual + anchor + BC for U_safety.
                         Owns ONLY U_safety params.

With w_optimality = w_safety = 1.0 (defaults), the convex combination is
the simple mean of the two critics. Tuning w_optimality / w_safety
without other code changes lets Tier 2 sweep the fusion-weight axis
(see config_frozen_v1.yaml::tier2.sub_grid_2c).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.drppo import RecurrentActorCritic
from models.pde.soft_hjb_aux_critic import SoftHJBAuxCritic
from models.pde.cbf_aux_critic import CBFAuxCritic
from models.pde.state_builder import ReducedPDEState, XI_DIM
from models.pde.dynamics import BehavioralDynamics
from models.pde.residuals import (
    soft_hjb_residual, cbf_residual, pde_q_values, soft_policy_from_q,
)
from models.pde.collocation import sample_collocation
from models.pde.checkpointing import (
    save_pde_checkpoint, load_pde_checkpoint, verify_arch, peek_checkpoint_arch,
)


class FusionAuxAgent:
    """Recurrent PPO + (Soft-HJB ⊕ CBF) fused auxiliary critics."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 5,
        # PPO core
        lr: float = 3e-4,
        aux_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Anchor / boundary (per-critic; same as single-method agents)
        lambda_anchor: float = 1.0,
        lambda_bc: float = 0.5,
        # Fusion-specific weights
        lambda_residual: float = 0.2,
        lambda_distill: float = 0.25,
        lambda_actor_kl: float = 0.1,
        w_optimality: float = 1.0,
        w_safety: float = 1.0,
        # Soft-HJB-specific
        tau_soft: float = 1.0,
        # CBF-specific
        alpha_cbf: float = 1.0,
        barrier_offset: float = 10.0,
        # Critic architecture
        aux_hidden_dim: int = 256,
        collocation_ratio: float = 0.7,
        # Policy architecture
        hidden_dim: int = 128,
        # Misc
        device: str = "cpu",
        w_coll: float = -20.0,
        seed: int = 42,
        reward_kwargs: dict | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.aux_hidden_dim = aux_hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.lambda_anchor = lambda_anchor
        self.lambda_bc = lambda_bc
        self.lambda_residual = lambda_residual
        self.lambda_distill = lambda_distill
        self.lambda_actor_kl = lambda_actor_kl
        self.w_optimality = w_optimality
        self.w_safety = w_safety
        self.tau_soft = tau_soft
        self.alpha_cbf = alpha_cbf
        self.barrier_offset = barrier_offset
        self.collocation_ratio = collocation_ratio
        self.device = device
        self.w_coll = w_coll
        self.seed = seed
        self.reward_kwargs = reward_kwargs or {}

        # ── Policy (DRPPO + GRU) ──────────────────────────────────────────
        torch.manual_seed(seed)
        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)

        # ── Optimality critic (Soft-HJB residual target) ──────────────────
        # Uses current seed.
        self.U_optimality = SoftHJBAuxCritic(
            in_dim=XI_DIM, hidden_dim=aux_hidden_dim,
        ).to(device)

        # ── Safety critic (CBF residual target) ───────────────────────────
        # Decision A: seed offset by 1 so U_safety starts at a different
        # parameter-space point from U_optimality.
        torch.manual_seed(seed + 1)
        self.U_safety = CBFAuxCritic(
            in_dim=XI_DIM, hidden_dim=aux_hidden_dim,
        ).to(device)

        # Restore RNG state for downstream determinism.
        torch.manual_seed(seed)

        # ── Three optimizers (cleanup post-Phase-1C trace review) ─────────
        # Each optimizer owns disjoint parameters; this matches the
        # single-method-PDE pattern (policy optimizer + per-critic
        # aux_optimizer) extended to two critics.
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr,
        )
        self.aux_opt_optimality = torch.optim.Adam(
            self.U_optimality.parameters(), lr=aux_lr,
        )
        self.aux_opt_safety = torch.optim.Adam(
            self.U_safety.parameters(), lr=aux_lr,
        )

        self.dynamics = BehavioralDynamics()
        self.pde_state_builder = ReducedPDEState()
        self._hidden = None

    # ---------------------------------------------------------------------
    # Inference helpers (match single-method agent API)
    # ---------------------------------------------------------------------
    def reset_hidden(self):
        self._hidden = self.policy.init_hidden(batch_size=1, device=self.device)

    def get_action(
        self, obs: np.ndarray, hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
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

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
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

        # ── PPO actor + value (vs returns) ────────────────────────────────
        value, log_prob, entropy = self.policy.evaluate_actions(
            obs_t, actions_t, hidden_t,
        )
        ratio = torch.exp(log_prob - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(
            ratio, 1 - self.clip_range, 1 + self.clip_range,
        ) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()
        vf_loss = F.mse_loss(value, returns_t)

        soft_res_mean = 0.0
        cbf_res_mean = 0.0
        anchor_opt_val = 0.0
        anchor_saf_val = 0.0
        bc_opt_val = 0.0
        bc_saf_val = 0.0
        distill_loss_val = 0.0
        distill_opt_val = 0.0
        distill_saf_val = 0.0
        actor_align_kl_val = 0.0

        if extra is not None and "xi_curr" in extra:
            xi_np = extra["xi_curr"]
            xi_t = torch.FloatTensor(xi_np).to(self.device)

            # Decision B/C: shared collocation set, applied to both critics.
            xi_colloc = sample_collocation(
                xi_t, ratio_real=self.collocation_ratio,
            )

            # ── Soft-HJB residual on U_optimality ─────────────────────────
            rho_soft = soft_hjb_residual(
                self.U_optimality, xi_colloc, self.dynamics,
                gamma=self.gamma, tau=self.tau_soft,
                reward_kwargs=self.reward_kwargs,
            )
            L_soft = (rho_soft ** 2).mean()

            # Anchor + BC for the optimality critic
            U_opt_rollout = self.U_optimality(xi_t)
            L_anchor_opt = F.mse_loss(
                U_opt_rollout, returns_t[:len(U_opt_rollout)],
            )
            L_bc_opt = torch.tensor(0.0, device=self.device)
            if "success_terminal" in extra:
                succ = torch.BoolTensor(extra["success_terminal"]).to(self.device)
                if succ.any():
                    succ_xi = xi_t[succ[:len(xi_t)]]
                    if len(succ_xi) > 0:
                        # Phase 31 Stage 1B fix: anchor U at +200 (env w_success)
                        # at success terminals to keep PDE residual consistent with env.
                        L_bc_opt = L_bc_opt + ((self.U_optimality(succ_xi) - 200.0) ** 2).mean()
            if "collision_terminal" in extra:
                coll = torch.BoolTensor(extra["collision_terminal"]).to(self.device)
                if coll.any():
                    coll_xi = xi_t[coll[:len(xi_t)]]
                    if len(coll_xi) > 0:
                        L_bc_opt = L_bc_opt + (
                            (self.U_optimality(coll_xi) - self.w_coll) ** 2
                        ).mean()

            aux_loss_opt = (
                self.lambda_anchor * L_anchor_opt
                + self.lambda_residual * self.w_optimality * L_soft
                + self.lambda_bc * L_bc_opt
            )
            self.aux_opt_optimality.zero_grad()
            aux_loss_opt.backward()
            nn.utils.clip_grad_norm_(
                self.U_optimality.parameters(), self.max_grad_norm,
            )
            self.aux_opt_optimality.step()

            # ── CBF residual on U_safety ──────────────────────────────────
            rho_cbf = cbf_residual(
                self.U_safety, xi_colloc, self.dynamics,
                alpha_cbf=self.alpha_cbf, cbf_safe_offset=self.barrier_offset,
            )
            L_cbf = (rho_cbf ** 2).mean()

            U_saf_rollout = self.U_safety(xi_t)
            L_anchor_saf = F.mse_loss(
                U_saf_rollout, returns_t[:len(U_saf_rollout)],
            )
            L_bc_saf = torch.tensor(0.0, device=self.device)
            if "success_terminal" in extra:
                succ = torch.BoolTensor(extra["success_terminal"]).to(self.device)
                if succ.any():
                    succ_xi = xi_t[succ[:len(xi_t)]]
                    if len(succ_xi) > 0:
                        # Phase 31 Stage 1B fix: anchor U at +200 (env w_success)
                        # at success terminals to keep PDE residual consistent with env.
                        L_bc_saf = L_bc_saf + ((self.U_safety(succ_xi) - 200.0) ** 2).mean()
            if "collision_terminal" in extra:
                coll = torch.BoolTensor(extra["collision_terminal"]).to(self.device)
                if coll.any():
                    coll_xi = xi_t[coll[:len(xi_t)]]
                    if len(coll_xi) > 0:
                        L_bc_saf = L_bc_saf + (
                            (self.U_safety(coll_xi) - self.w_coll) ** 2
                        ).mean()

            aux_loss_saf = (
                self.lambda_anchor * L_anchor_saf
                + self.lambda_residual * self.w_safety * L_cbf
                + self.lambda_bc * L_bc_saf
            )
            self.aux_opt_safety.zero_grad()
            aux_loss_saf.backward()
            nn.utils.clip_grad_norm_(
                self.U_safety.parameters(), self.max_grad_norm,
            )
            self.aux_opt_safety.step()

            # ── Distillation: V_PPO ← convex(U_optimality, U_safety).detach() ──
            # Phase-3F pre-Tier-1 fix: revised Decision F. Original locked the
            # distill target to U_optimality only, which made U_safety unused
            # by the policy. Restore the dual-critic dependency by distilling
            # V_PPO from a convex combination weighted by w_optimality /
            # w_safety (parameters were already exposed for this purpose).
            w_total = max(self.w_optimality + self.w_safety, 1e-9)
            with torch.no_grad():
                U_distill_opt = self.U_optimality(xi_t).detach()
                U_distill_saf = self.U_safety(xi_t).detach()
                U_distill = (
                    self.w_optimality * U_distill_opt
                    + self.w_safety * U_distill_saf
                ) / w_total
            L_distill = F.mse_loss(value[:len(U_distill)], U_distill)
            vf_loss = vf_loss + self.lambda_distill * L_distill
            # Diagnostic: how much of the distill target is each critic
            # contributing? Used by the post-fix smoke-test sanity check
            # to confirm the U_safety gradient path is flowing.
            with torch.no_grad():
                v_aligned = value[:len(U_distill)]
                L_distill_opt = F.mse_loss(v_aligned, U_distill_opt[:len(v_aligned)])
                L_distill_saf = F.mse_loss(v_aligned, U_distill_saf[:len(v_aligned)])
                distill_opt_val = float(L_distill_opt.detach())
                distill_saf_val = float(L_distill_saf.detach())

            # ── Actor-KL alignment (Decision D, dual-critic) ─────────────
            # pde_q_values uses autograd to compute grad_U; we detach the
            # outputs so the actor-KL doesn't backprop into either critic.
            # Use the same convex combination as the distillation target.
            q_opt, _ = pde_q_values(
                self.U_optimality, xi_t, self.dynamics,
                gamma=self.gamma, reward_kwargs=self.reward_kwargs,
            )
            q_saf, _ = pde_q_values(
                self.U_safety, xi_t, self.dynamics,
                gamma=self.gamma, reward_kwargs=self.reward_kwargs,
            )
            q_all = (
                self.w_optimality * q_opt.detach()
                + self.w_safety * q_saf.detach()
            ) / w_total
            pi_soft = soft_policy_from_q(q_all, tau=self.tau_soft).detach()

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
            actor_loss = actor_loss + self.lambda_actor_kl * kl

            soft_res_mean = float(rho_soft.detach().abs().mean())
            cbf_res_mean = float(rho_cbf.detach().abs().mean())
            anchor_opt_val = float(L_anchor_opt.detach())
            anchor_saf_val = float(L_anchor_saf.detach())
            bc_opt_val = float(L_bc_opt.detach())
            bc_saf_val = float(L_bc_saf.detach())
            distill_loss_val = float(L_distill.detach())
            actor_align_kl_val = float(kl.detach())
            # distill_opt_val and distill_saf_val already populated above

        # ── Policy optimizer step (PPO + distill + actor-KL) ─────────────
        loss = (
            actor_loss
            + self.ent_coef * entropy_loss
            + self.vf_coef * vf_loss
        )
        self.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm,
        )
        self.policy_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "vf_loss": (
                float(vf_loss.item()) if isinstance(vf_loss, torch.Tensor) else vf_loss
            ),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            # Decision J: log unweighted residuals so each PDE's progress is
            # tracked independently in metrics.csv.
            "soft_residual_mean": soft_res_mean,
            "cbf_residual_mean": cbf_res_mean,
            "anchor_loss_optimality": anchor_opt_val,
            "anchor_loss_safety": anchor_saf_val,
            "bc_loss_optimality": bc_opt_val,
            "bc_loss_safety": bc_saf_val,
            "distill_loss": distill_loss_val,
            "distill_gap": distill_loss_val,
            "distill_loss_optimality_component": distill_opt_val,
            "distill_loss_safety_component": distill_saf_val,
            "actor_align_kl": actor_align_kl_val,
        }

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def _arch_dict(self) -> dict:
        return {
            "obs_dim": int(self.obs_dim),
            "n_actions": int(self.n_actions),
            "hidden_dim": int(self.hidden_dim),
            "aux_hidden_dim": int(self.aux_hidden_dim),
        }

    def save(self, path: str):
        save_pde_checkpoint(
            path=path,
            policy_state=self.policy.state_dict(),
            policy_optim_state=self.policy_optimizer.state_dict(),
            aux_state={
                "U_optimality": self.U_optimality.state_dict(),
                "U_safety": self.U_safety.state_dict(),
            },
            aux_optim_state={
                "aux_opt_optimality": self.aux_opt_optimality.state_dict(),
                "aux_opt_safety": self.aux_opt_safety.state_dict(),
            },
            obs_dim=self.obs_dim,
            method="fusion_aux",
            arch=self._arch_dict(),
            config={
                "lambda_anchor": self.lambda_anchor,
                "lambda_bc": self.lambda_bc,
                "lambda_residual": self.lambda_residual,
                "lambda_distill": self.lambda_distill,
                "lambda_actor_kl": self.lambda_actor_kl,
                "w_optimality": self.w_optimality,
                "w_safety": self.w_safety,
                "tau_soft": self.tau_soft,
                "alpha_cbf": self.alpha_cbf,
                "barrier_offset": self.barrier_offset,
                "gamma": self.gamma,
            },
        )

    def load(self, path: str, strict_arch: bool = True):
        data = load_pde_checkpoint(path, device=self.device)
        verify_arch(data.get("arch"), self._arch_dict(), strict=strict_arch, ckpt_path=path)
        self.policy.load_state_dict(data["policy"])
        if "policy_optimizer" in data:
            self.policy_optimizer.load_state_dict(data["policy_optimizer"])
        aux = data.get("aux_critic", {})
        if isinstance(aux, dict) and "U_optimality" in aux:
            self.U_optimality.load_state_dict(aux["U_optimality"])
            self.U_safety.load_state_dict(aux["U_safety"])
        aux_opt = data.get("aux_optimizer", {})
        if isinstance(aux_opt, dict):
            if "aux_opt_optimality" in aux_opt:
                self.aux_opt_optimality.load_state_dict(aux_opt["aux_opt_optimality"])
            if "aux_opt_safety" in aux_opt:
                self.aux_opt_safety.load_state_dict(aux_opt["aux_opt_safety"])

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **overrides) -> "FusionAuxAgent":
        arch = peek_checkpoint_arch(path, device=device)
        ctor_kwargs = {k: arch[k] for k in ("obs_dim", "n_actions", "hidden_dim", "aux_hidden_dim")
                       if k in arch}
        ctor_kwargs.update(overrides)
        if "obs_dim" not in ctor_kwargs:
            raise ValueError(f"Checkpoint at {path} has no 'arch.obs_dim'; pass obs_dim= explicitly.")
        agent = cls(device=device, **ctor_kwargs)
        agent.load(path, strict_arch=True)
        return agent
