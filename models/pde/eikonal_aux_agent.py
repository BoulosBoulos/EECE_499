"""Phase 3F-A Eikonal auxiliary critic — time-of-arrival reformulation.

This file replaces the legacy value-function Eikonal critic. The auxiliary
critic T_phi : R^79 -> R_{>=0} now represents *expected time-of-arrival* to
the maneuver-exit goal set in units of macro-timesteps. See

  verification/phase3F_A_math_derivation.md

for the full derivation. The four-term auxiliary loss is

  L_aux = w_eik     * L_eik(phi)
        + w_bc      * L_bc(phi)
        + w_ground  * L_ground(phi)
        + w_distill * L_distill(theta;  stop_grad(T_phi))

  L_eik     = E_{xi ~ collocation}[ (||grad T_phi||^2 - c(xi)^2)^2 ]
  L_bc      = MSE( T_phi at success terminals, 0 )
            + MSE( T_phi at collision terminals, T_max )
  L_ground  = MSE( T_phi at rollout states, T_obs(s_t) )
  L_distill = beta_KL * KL( pi_theta || pi_eik ),
              pi_eik = softmax( stop_grad(A_eik) / tau ),
              A_eik(s,a) = T_phi(s) - T_phi(f_a(s))

The stop-gradient on T_phi inside L_distill is mandatory: it ensures the
actor's loss does not pollute T_phi's training signal. T_phi is governed
*only* by L_eik + L_bc + L_ground (cf. math derivation §6.3).
"""

from __future__ import annotations

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from models.drppo import RecurrentActorCritic
from models.pde.eikonal_aux_critic import EikonalAuxCritic
from models.pde.state_builder import ReducedPDEState, XI_DIM
from models.pde.dynamics import BehavioralDynamics, compute_one_step_toa_advantages
from models.pde.residuals import eikonal_residual_toa
from models.pde.collocation import sample_collocation
from models.pde.checkpointing import (
    save_pde_checkpoint, load_pde_checkpoint, verify_arch, peek_checkpoint_arch,
)


class EikonalAuxAgent:
    """Recurrent PPO + Phase 3F-A time-of-arrival Eikonal critic.

    The auxiliary critic T_phi(s) approximates expected time-of-arrival in
    macro-timesteps. Four-term loss (eikonal residual + boundary conditions
    + observed-time grounding + soft-KL policy distillation) is described
    in verification/phase3F_A_math_derivation.md.
    """

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
        # Phase 3F-A four-term loss weights (legacy fixed-weight knobs).
        # Step 7C: w_eik, w_bc, w_ground are SUPERSEDED by learnable σ params
        # below; they are still accepted for backward compatibility but no
        # longer enter the gradient. w_distill remains in effect (actor-side).
        w_eik: float = 1.0,
        w_bc: float = 0.5,
        w_ground: float = 1.0,
        w_distill: float = 0.5,
        beta_KL: float = 0.1,
        tau: float = 1.0,
        T_max: float = 100.0,
        # Step 7C/7D: Kendall et al. 2018 uncertainty weighting (Step 7D: BC/GROUND only).
        log_sigma_init: float = 0.0,        # σ_i = exp(log_sigma_init); =1 by default
        # Step 7D: Augmented Lagrangian for the L_eik constraint (Bertsekas 1996;
        # Lu et al. 2021). Replaces Step 7C log_sigma_eik. See math derivation
        # §10.2-10.3.
        alm_lambda_init: float = 0.0,       # initial Lagrange multiplier
        alm_mu_init: float = 1.0,           # initial penalty
        alm_mu_max: float = 1.0e4,          # upper bound on μ
        alm_alpha: float = 0.25,            # required improvement ratio per ALM epoch
        alm_beta: float = 5.0,              # geometric growth factor for μ
        alm_update_interval: int = 5000,    # training steps between λ, μ updates
        # Step 7C: terminal replay buffers (DeepXDE-style sparse-data PINN training).
        replay_buffer_size: int = 1000,
        replay_batch_size: int = 256,
        # Existing hyperparameters retained:
        aux_hidden_dim: int = 256,
        collocation_ratio: float = 0.7,
        hidden_dim: int = 128,
        v_min: float = 0.5,
        device: str = "cpu",
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

        # Phase 3F-A weights
        self.w_eik = w_eik
        self.w_bc = w_bc
        self.w_ground = w_ground
        self.w_distill = w_distill
        self.beta_KL = beta_KL
        self.tau = tau
        self.T_max = T_max

        self.collocation_ratio = collocation_ratio
        self.v_min = v_min
        self.device = device
        self.reward_kwargs = reward_kwargs or {}

        # Step 7C: replay-buffer hyperparameters
        self.replay_buffer_size = int(replay_buffer_size)
        self.replay_batch_size = int(replay_batch_size)

        self.policy = RecurrentActorCritic(
            obs_dim=obs_dim, hidden_dim=hidden_dim, n_actions=n_actions,
        ).to(device)
        self.aux_critic = EikonalAuxCritic(in_dim=XI_DIM, hidden_dim=aux_hidden_dim).to(device)

        # Step 7D: Kendall et al. 2018 learnable uncertainty parameters — RETAINED
        # for L_bc and L_ground (legitimate noisy supervised tasks). log_sigma_eik
        # is REMOVED; L_eik is now treated as a constraint via ALM (see below).
        self.log_sigma_bc = nn.Parameter(torch.full((), float(log_sigma_init), device=device))
        self.log_sigma_ground = nn.Parameter(torch.full((), float(log_sigma_init), device=device))

        # Step 7D: Augmented Lagrangian state for the L_eik constraint
        # (canonical Bertsekas 1996, §4.2 update rule). λ and μ are *scalars*
        # (not nn.Parameters) updated explicitly by the ALM rule, not by
        # gradient descent. See math derivation §10.2.
        self.alm_lambda: float = float(alm_lambda_init)
        self.alm_mu: float = float(alm_mu_init)
        self.alm_mu_init: float = float(alm_mu_init)
        self.alm_mu_max: float = float(alm_mu_max)
        self.alm_alpha: float = float(alm_alpha)
        self.alm_beta: float = float(alm_beta)
        self.alm_update_interval: int = int(alm_update_interval)
        # Per-step L_eik history; reset to [] after each ALM update.
        self.alm_l_eik_history: list[float] = []
        # Mean L_eik from the previous ALM epoch (for the μ update rule).
        self.alm_l_eik_prev_epoch: float | None = None
        # Step counter for triggering ALM updates externally; set by training loop.
        self.alm_last_update_step: int = 0
        # Diagnostic: per-update history of (step, λ, μ, mean L_eik).
        self.alm_history: list[dict] = []

        # Step 7C: terminal replay buffers (math derivation §9.4).
        # Stored as deques of (xi_np: ndarray, T_obs: float).
        self.success_buffer: deque = deque(maxlen=self.replay_buffer_size)
        self.collision_buffer: deque = deque(maxlen=self.replay_buffer_size)
        self.intermediate_buffer: deque = deque(maxlen=self.replay_buffer_size)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        # Step 7D: aux optimizer trains the critic network plus the two retained
        # log_σ parameters (BC, GROUND). The ALM scalars λ, μ are NOT optimizer
        # parameters — they are updated by the explicit Bertsekas 1996 rule via
        # update_alm_parameters(). See math derivation §10.2.
        self.aux_optimizer = torch.optim.Adam(
            list(self.aux_critic.parameters())
            + [self.log_sigma_bc, self.log_sigma_ground],
            lr=aux_lr,
        )

        self.dynamics = BehavioralDynamics()
        self.pde_state_builder = ReducedPDEState()
        self._hidden = None

    def reset_hidden(self):
        self._hidden = self.policy.init_hidden(batch_size=1, device=self.device)

    # ------------------------------------------------------------------
    # Step 7D: Augmented Lagrangian update (Bertsekas 1996, §4.2)
    # ------------------------------------------------------------------
    def update_alm_parameters(self, current_step: int) -> dict:
        """Apply the canonical Bertsekas 1996 ALM update.

        Multiplier (gradient-ascent on the dual):
            λ_{k+1} = λ_k + μ_k · g(φ_{k+1}),   g := mean L_eik over recent inner steps.
        Penalty (geometric increase if constraint not improving):
            μ_{k+1} = β·μ_k   if g_{k+1} > α·g_k,
                     μ_k     otherwise,
            capped at μ_max.

        Called externally from the training loop every ``alm_update_interval``
        steps. No-op if no L_eik history has been accumulated since the last
        call (e.g., very first iteration before any train_step has run).

        Returns a diagnostic dict with the updated values.
        """
        if not self.alm_l_eik_history:
            return {"alm_lambda": self.alm_lambda, "alm_mu": self.alm_mu,
                    "mean_L_eik_epoch": None, "step": current_step,
                    "no_history": True}
        mean_l_eik = float(np.mean(self.alm_l_eik_history))
        # λ update: dual ascent
        self.alm_lambda = float(self.alm_lambda + self.alm_mu * mean_l_eik)
        # μ update: increase if constraint not improving by factor α
        mu_grew = False
        if self.alm_l_eik_prev_epoch is not None:
            if mean_l_eik > self.alm_alpha * self.alm_l_eik_prev_epoch:
                self.alm_mu = float(min(self.alm_beta * self.alm_mu, self.alm_mu_max))
                mu_grew = True
        self.alm_l_eik_prev_epoch = mean_l_eik
        # Reset history for the next epoch
        self.alm_l_eik_history = []
        self.alm_last_update_step = int(current_step)
        record = {
            "step": int(current_step),
            "alm_lambda": float(self.alm_lambda),
            "alm_mu": float(self.alm_mu),
            "mean_L_eik_epoch": float(mean_l_eik),
            "mu_grew": bool(mu_grew),
            "mu_at_max": bool(self.alm_mu >= self.alm_mu_max),
        }
        self.alm_history.append(record)
        return record

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

    # ------------------------------------------------------------------
    # Step 7C: terminal replay-buffer machinery
    # ------------------------------------------------------------------
    def update_replay_buffers(self, extra: dict, xi_t: torch.Tensor) -> None:
        """Walk current rollout segments and append to replay buffers.

        Phase 3F-A Step 7C: see math derivation §9.4. For each completed
        trajectory in the current minibatch:
          - success → append final state to ``success_buffer`` with T_obs=0;
            append intermediate states to ``intermediate_buffer`` with
            T_obs=K-t (countdown).
          - collision → append every state in the segment to
            ``collision_buffer`` with T_obs=T_max.
          - timeout → skip (T_obs undefined per §3.1 / §6.1).
        Buffers are FIFO with capacity self.replay_buffer_size; oldest
        entries are evicted when full.
        """
        n = xi_t.shape[0]
        succ = np.asarray(extra.get("success_terminal", []), dtype=bool)
        coll = np.asarray(extra.get("collision_terminal", []), dtype=bool)
        done = np.asarray(extra.get("done", []), dtype=bool)
        # Pad / truncate masks to length n
        def _fix(arr):
            if len(arr) < n:
                return np.concatenate([arr, np.zeros(n - len(arr), dtype=bool)])
            return arr[:n]
        succ = _fix(succ); coll = _fix(coll); done = _fix(done)
        terminal_step = succ | coll | done
        xi_np = xi_t.detach().cpu().numpy() if isinstance(xi_t, torch.Tensor) else np.asarray(xi_t)
        seg_start = 0
        for i in range(n):
            if not terminal_step[i] and i != n - 1:
                continue
            seg_end = i
            if succ[seg_end]:
                self.success_buffer.append((xi_np[seg_end].astype(np.float32, copy=True), 0.0))
                # Intermediate states get countdown targets
                for t in range(seg_start, seg_end):
                    self.intermediate_buffer.append(
                        (xi_np[t].astype(np.float32, copy=True), float(seg_end - t))
                    )
            elif coll[seg_end]:
                # Every state in collision segment anchors at T_max
                for t in range(seg_start, seg_end + 1):
                    self.collision_buffer.append(
                        (xi_np[t].astype(np.float32, copy=True), float(self.T_max))
                    )
            # timeouts: T_obs undefined, skip
            seg_start = seg_end + 1

    def _sample_bc_data(
        self, batch_size: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Sample success and collision terminal states from replay buffers."""
        if batch_size is None:
            batch_size = self.replay_batch_size
        n_per_side = batch_size // 2

        succ_states = None
        if len(self.success_buffer) > 0:
            k = min(n_per_side, len(self.success_buffer))
            sampled = random.sample(list(self.success_buffer), k)
            succ_states = torch.from_numpy(np.stack([s for s, _ in sampled])).to(self.device)

        coll_states = None
        if len(self.collision_buffer) > 0:
            k = min(n_per_side, len(self.collision_buffer))
            sampled = random.sample(list(self.collision_buffer), k)
            coll_states = torch.from_numpy(np.stack([s for s, _ in sampled])).to(self.device)

        return succ_states, coll_states

    def _sample_ground_data(
        self, batch_size: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Sample (state, T_obs) pairs from intermediate + collision buffers.

        intermediate buffer supplies T_obs=K-t targets from successful
        trajectories; collision buffer supplies T_obs=T_max targets.
        """
        if batch_size is None:
            batch_size = self.replay_batch_size
        n_per_side = batch_size // 2

        states_list: list[np.ndarray] = []
        targets_list: list[float] = []
        if len(self.intermediate_buffer) > 0:
            k = min(n_per_side, len(self.intermediate_buffer))
            sampled = random.sample(list(self.intermediate_buffer), k)
            for s, t in sampled:
                states_list.append(s); targets_list.append(t)
        if len(self.collision_buffer) > 0:
            k = min(n_per_side, len(self.collision_buffer))
            sampled = random.sample(list(self.collision_buffer), k)
            for s, t in sampled:
                states_list.append(s); targets_list.append(t)
        if not states_list:
            return None, None
        states = torch.from_numpy(np.stack(states_list)).to(self.device)
        targets = torch.tensor(targets_list, device=self.device, dtype=torch.float32)
        return states, targets

    @staticmethod
    def _compute_grounding_data(
        extra: dict, xi_t: torch.Tensor, T_max: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build (states, T_obs) pairs for L_ground.

        For each state in the rollout buffer, T_obs(s_t) is determined by the
        TERMINATION CAUSE of the trajectory s_t belongs to:
          - success-terminated: T_obs(s_t) = (t_succ - t)  in macro-steps
          - collision-terminated: T_obs(s_t) = T_max for every t in the trajectory
          - timeout-terminated: T_obs(s_t) = undefined  (excluded entirely)

        Implementation note: ``extra`` provides per-step boolean masks for the
        rollout buffer as well as ``episode_id`` per step (segment id). We
        compute T_obs by walking the rollout buffer in chronological order and
        accumulating per-segment data, then assigning T_obs values based on
        the terminal state's cause.

        Returns
        -------
        states  : (N_grounding, XI_DIM) tensor or None if no eligible data
        targets : (N_grounding,) tensor of T_obs values, or None
        """
        # Per-step boolean masks. Either of them may be missing if the rollout
        # had no terminal of that kind; treat as all-False in that case.
        n = xi_t.shape[0]
        succ_mask = extra.get("success_terminal")
        coll_mask = extra.get("collision_terminal")
        # episode_id (per-step segment id) lets us identify which trajectory
        # each rollout step belongs to. Trajectories within a rollout are
        # delineated by transitions in episode_id.
        ep_id = extra.get("episode_id")
        # Some training scripts may provide a flat "done" array; we accept that
        # as fallback to delineate trajectories (any True indicates segment end).
        done_mask = extra.get("done")

        if succ_mask is None and coll_mask is None:
            return None, None

        succ_arr = np.asarray(succ_mask, dtype=bool) if succ_mask is not None else np.zeros(n, dtype=bool)
        coll_arr = np.asarray(coll_mask, dtype=bool) if coll_mask is not None else np.zeros(n, dtype=bool)
        if len(succ_arr) < n:
            succ_arr = np.concatenate([succ_arr, np.zeros(n - len(succ_arr), dtype=bool)])
        if len(coll_arr) < n:
            coll_arr = np.concatenate([coll_arr, np.zeros(n - len(coll_arr), dtype=bool)])
        succ_arr = succ_arr[:n]
        coll_arr = coll_arr[:n]
        # Segment delimiter: any-True in (success | collision | done)
        terminal_step = succ_arr | coll_arr
        if done_mask is not None:
            done_arr = np.asarray(done_mask, dtype=bool)
            if len(done_arr) < n:
                done_arr = np.concatenate([done_arr, np.zeros(n - len(done_arr), dtype=bool)])
            terminal_step = terminal_step | done_arr[:n]

        # Walk the rollout in order; collect per-segment indices and decide
        # T_obs for every step in segment when the segment ENDS at success/coll.
        states_list: list[torch.Tensor] = []
        targets_list: list[float] = []
        seg_start = 0
        for i in range(n):
            if not terminal_step[i] and i != n - 1:
                continue
            seg_end = i  # inclusive
            if succ_arr[seg_end]:
                # Success-terminated: T_obs(s_t) = (seg_end - t) for t in [seg_start, seg_end]
                seg_states = xi_t[seg_start:seg_end + 1]
                seg_targets = torch.arange(
                    seg_end - seg_start, -1, -1, device=xi_t.device, dtype=torch.float32,
                )
                states_list.append(seg_states)
                targets_list.append(seg_targets)
            elif coll_arr[seg_end]:
                # Collision-terminated: every step in the segment gets T_max
                seg_states = xi_t[seg_start:seg_end + 1]
                seg_targets = torch.full(
                    (seg_end - seg_start + 1,), T_max,
                    device=xi_t.device, dtype=torch.float32,
                )
                states_list.append(seg_states)
                targets_list.append(seg_targets)
            # else: segment ended without success/coll = timeout. Exclude
            # entirely (T_obs undefined).
            seg_start = seg_end + 1

        if not states_list:
            return None, None

        states = torch.cat(states_list, dim=0)
        # targets_list now contains tensors for each segment; concat them
        targets_t: list[torch.Tensor] = []
        for x in targets_list:
            if isinstance(x, torch.Tensor):
                targets_t.append(x)
            else:
                # Robustness fallback: should not occur given construction above
                targets_t.append(torch.tensor(x, device=xi_t.device, dtype=torch.float32))
        targets = torch.cat(targets_t, dim=0)
        return states, targets

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

        # Standard PPO step.
        value, log_prob, entropy = self.policy.evaluate_actions(obs_t, actions_t, hidden_t)
        ratio = torch.exp(log_prob - old_log_probs_t)
        surr1 = ratio * advantages_t
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_t
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()
        vf_loss = F.mse_loss(value, returns_t)

        # Default zeros (in case there is no PDE data this iter).
        L_eik_val = 0.0
        L_bc_val = 0.0
        L_bc_raw_val = 0.0
        L_ground_val = 0.0
        L_ground_raw_val = 0.0
        L_distill_val = 0.0
        mean_rho_val = 0.0
        mean_T_succ = 0.0
        mean_T_coll = 0.0
        n_grounding_points = 0
        # Step 7D: σ values for retained KGC tasks + ALM diagnostics.
        sigma_eik_val = 0.0  # removed in 7D — kept in dict for back-compat (=0)
        sigma_bc_val = float(torch.exp(self.log_sigma_bc).detach())
        sigma_ground_val = float(torch.exp(self.log_sigma_ground).detach())
        n_success_buffer = len(self.success_buffer)
        n_collision_buffer = len(self.collision_buffer)
        n_intermediate_buffer = len(self.intermediate_buffer)
        L_eik_alm_val = 0.0  # ALM contribution to critic loss; populated when training

        if extra is not None and "xi_curr" in extra:
            xi_np = extra["xi_curr"]
            xi_t = torch.FloatTensor(xi_np).to(self.device)
            xi_colloc = sample_collocation(xi_t, ratio_real=self.collocation_ratio)

            # Step 7C: append current rollout's terminal/intermediate samples
            # to the replay buffers BEFORE sampling for L_bc / L_ground. New
            # data is therefore visible to this iter's loss; oldest data
            # evicted when buffers reach capacity. See math derivation §9.4.
            self.update_replay_buffers(extra, xi_t)

            # ---- Term 1: L_eik (PDE residual) ----
            rho = eikonal_residual_toa(
                self.aux_critic, xi_colloc, self.dynamics, v_min=self.v_min,
            )
            L_eik = (rho ** 2).mean()
            mean_rho_val = float(rho.detach().abs().mean())

            # ---- Term 2: L_bc — sampled from terminal replay buffers ----
            # Step 7C: replay buffer sustains BC signal even when a single
            # iteration's rollout has no terminals of a given type. See
            # math derivation §9.4.
            succ_states, coll_states = self._sample_bc_data()
            L_bc_succ = torch.tensor(0.0, device=self.device)
            L_bc_coll = torch.tensor(0.0, device=self.device)
            if succ_states is not None:
                T_succ_pred = self.aux_critic(succ_states)
                if T_succ_pred.dim() > 1:
                    T_succ_pred = T_succ_pred.squeeze(-1)
                L_bc_succ = ((T_succ_pred - 0.0) ** 2).mean()
                mean_T_succ = float(T_succ_pred.detach().mean())
            if coll_states is not None:
                T_coll_pred = self.aux_critic(coll_states)
                if T_coll_pred.dim() > 1:
                    T_coll_pred = T_coll_pred.squeeze(-1)
                L_bc_coll = ((T_coll_pred - self.T_max) ** 2).mean()
                mean_T_coll = float(T_coll_pred.detach().mean())
            L_bc = L_bc_succ + L_bc_coll

            # ---- Term 3: L_ground — sampled from replay buffers ----
            ground_states, ground_targets = self._sample_ground_data()
            if ground_states is not None and ground_states.shape[0] > 0:
                T_grounding = self.aux_critic(ground_states)
                if T_grounding.dim() > 1:
                    T_grounding = T_grounding.squeeze(-1)
                L_ground = ((T_grounding - ground_targets) ** 2).mean()
                n_grounding_points = int(ground_states.shape[0])
            else:
                L_ground = torch.tensor(0.0, device=self.device)
                n_grounding_points = 0

            # ---- Term 4: L_distill (soft-KL — T_phi DETACHED) ----
            A_eik = compute_one_step_toa_advantages(
                self.aux_critic, xi_t, self.dynamics, detach=True,
            )
            target_action_dist = F.softmax(A_eik / self.tau, dim=-1)
            actor_logits, _, _, _, _ = self.policy(obs_t, hidden_t)
            n_aligned = min(actor_logits.shape[0], target_action_dist.shape[0])
            log_actor_dist = F.log_softmax(actor_logits[:n_aligned], dim=-1)
            L_distill = self.beta_KL * F.kl_div(
                log_actor_dist, target_action_dist[:n_aligned],
                reduction="batchmean",
            )

            # ---- Step 7D: hybrid critic loss ----
            # ALM (Bertsekas 1996) for the L_eik constraint:
            #     L_eik_alm = λ · L_eik + (μ/2) · L_eik²
            # KGC (Kendall et al. 2018) for the noisy supervised tasks:
            #     L_sup = (1/(2σ_bc²))·L_bc + (1/(2σ_ground²))·L_ground
            #             + log σ_bc + log σ_ground
            # See math derivation §10.3.
            sigma_bc_sq = torch.exp(2 * self.log_sigma_bc)
            sigma_ground_sq = torch.exp(2 * self.log_sigma_ground)
            L_eik_alm = self.alm_lambda * L_eik + 0.5 * self.alm_mu * (L_eik ** 2)
            L_sup = (
                (1.0 / (2.0 * sigma_bc_sq)) * L_bc
                + (1.0 / (2.0 * sigma_ground_sq)) * L_ground
                + self.log_sigma_bc
                + self.log_sigma_ground
            )
            critic_loss = L_eik_alm + L_sup

            # Track L_eik for ALM update rule (Bertsekas §4.2).
            self.alm_l_eik_history.append(float(L_eik.detach()))

            # Critic + log_sigma_{bc,ground} jointly optimised via aux_optimizer.
            # ALM scalars λ, μ are updated by update_alm_parameters() (called
            # externally by the training loop every alm_update_interval steps).
            self.aux_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.aux_critic.parameters())
                + [self.log_sigma_bc, self.log_sigma_ground],
                self.max_grad_norm,
            )
            self.aux_optimizer.step()

            # Actor-side: L_distill (T_phi detached, no φ feedback).
            actor_loss = actor_loss + self.w_distill * L_distill

            L_eik_val = float(L_eik.detach())
            L_bc_raw_val = float(L_bc.detach())
            L_ground_raw_val = float(L_ground.detach()) if isinstance(L_ground, torch.Tensor) else 0.0
            # Step 7C: "normalized" view = L_i / σ_i^2 — the precision-weighted
            # contribution to the gradient. Kept as an alias for backward-
            # compat CSV / metric readers. The Step 7B T_max^2 division is gone.
            L_bc_norm_val = float((L_bc / sigma_bc_sq).detach())
            L_ground_norm_val = (
                float((L_ground / sigma_ground_sq).detach())
                if isinstance(L_ground, torch.Tensor) else 0.0
            )
            L_bc_val = L_bc_raw_val
            L_ground_val = L_ground_raw_val
            L_distill_val = float(L_distill.detach())
            sigma_eik_val = 0.0  # 7D: removed; surrogate value for back-compat readers
            sigma_bc_val = float(torch.exp(self.log_sigma_bc).detach())
            sigma_ground_val = float(torch.exp(self.log_sigma_ground).detach())
            L_eik_alm_val = float(L_eik_alm.detach())
            n_success_buffer = len(self.success_buffer)
            n_collision_buffer = len(self.collision_buffer)
            n_intermediate_buffer = len(self.intermediate_buffer)

        loss = actor_loss + self.ent_coef * entropy_loss + self.vf_coef * vf_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else float(actor_loss),
            "vf_loss": float(vf_loss.item()) if isinstance(vf_loss, torch.Tensor) else float(vf_loss),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
            "L_eik": L_eik_val,
            # Step 7C: L_bc / L_ground refer to RAW (unweighted) squared
            # error. The Kendall-weighted contribution to the gradient is
            # L_i / (2*sigma_i^2); also exposed via L_bc_normalized /
            # L_ground_normalized for transparency. T_max^2 normalisation
            # from Step 7B is removed — superseded by uncertainty weighting.
            "L_bc": L_bc_val,
            "L_bc_raw": L_bc_raw_val,
            "L_bc_normalized": L_bc_norm_val,
            "L_ground": L_ground_val,
            "L_ground_raw": L_ground_raw_val,
            "L_ground_normalized": L_ground_norm_val,
            "L_distill": L_distill_val,
            "mean_rho": mean_rho_val,
            "mean_T_succ": mean_T_succ,
            "mean_T_coll": mean_T_coll,
            "n_grounding_points": n_grounding_points,
            # Step 7C/7D: σ values (sigma_eik retained as 0 for back-compat; 7D
            # uses ALM for L_eik). Step 7D adds alm_lambda, alm_mu, L_eik_alm.
            "sigma_eik": sigma_eik_val,
            "sigma_bc": sigma_bc_val,
            "sigma_ground": sigma_ground_val,
            "alm_lambda": float(self.alm_lambda),
            "alm_mu": float(self.alm_mu),
            "L_eik_alm_contribution": L_eik_alm_val,
            "n_success_buffer": n_success_buffer,
            "n_collision_buffer": n_collision_buffer,
            "n_intermediate_buffer": n_intermediate_buffer,
            # Backwards-compat aliases (used by training scripts that read the
            # legacy column names). These point at the new fields' values.
            "eikonal_residual_mean": mean_rho_val,
            "anchor_loss": L_ground_val,  # closest analogue
            "bc_loss": L_bc_val,
            "distill_loss": L_distill_val,
            "distill_gap": L_distill_val,
        }

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
            policy_optim_state=self.optimizer.state_dict(),
            aux_state=self.aux_critic.state_dict(),
            aux_optim_state=self.aux_optimizer.state_dict(),
            obs_dim=self.obs_dim,
            method="eikonal_aux",
            arch=self._arch_dict(),
            config={
                "w_eik": self.w_eik,
                "w_bc": self.w_bc,
                "w_ground": self.w_ground,
                "w_distill": self.w_distill,
                "beta_KL": self.beta_KL,
                "tau": self.tau,
                "T_max": self.T_max,
                "v_min": self.v_min,
                "gamma": self.gamma,
            },
        )

    def load(self, path: str, strict_arch: bool = True):
        data = load_pde_checkpoint(path, device=self.device)
        verify_arch(data.get("arch"), self._arch_dict(), strict=strict_arch, ckpt_path=path)
        self.policy.load_state_dict(data["policy"])
        if "policy_optimizer" in data:
            self.optimizer.load_state_dict(data["policy_optimizer"])
        self.aux_critic.load_state_dict(data["aux_critic"])
        if "aux_optimizer" in data:
            self.aux_optimizer.load_state_dict(data["aux_optimizer"])

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu", **overrides) -> "EikonalAuxAgent":
        arch = peek_checkpoint_arch(path, device=device)
        ctor_kwargs = {k: arch[k] for k in ("obs_dim", "n_actions", "hidden_dim", "aux_hidden_dim")
                       if k in arch}
        ctor_kwargs.update(overrides)
        if "obs_dim" not in ctor_kwargs:
            raise ValueError(f"Checkpoint at {path} has no 'arch.obs_dim'; pass obs_dim= explicitly.")
        agent = cls(device=device, **ctor_kwargs)
        agent.load(path, strict_arch=True)
        return agent
