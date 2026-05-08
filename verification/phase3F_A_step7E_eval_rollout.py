"""Phase 3F-A Step 7E: post-training eval-rollout diagnostics.

Computes three diagnostics on a trained Eikonal-aux checkpoint that are NOT
captured in the training CSV — they require loading the final policy and
performing fresh deterministic rollouts:

  1. Pearson(T_phi, T_obs) on supervised states (success/collision episodes
     only; T_obs = countdown-to-success for success episodes, T_max for
     collision episodes).
  2. A_eik forward-progress agreement = fraction of states at which
     argmax_a A_eik(s, a) equals the actor's chosen action.
  3. KL(pi_theta || softmax(A_eik / tau)) averaged across rolled-out states.

Entry point ``evaluate_cell()`` loads the checkpoint, runs ``n_episodes``
deterministic eval rollouts of up to ``max_steps`` each, accumulates xi,
actions, logits, and T_obs arrays, and returns the diagnostics dict. On any
failure the function returns ``{"error": "..."}`` rather than raising.

Intended caller: Step 7E driver script that aggregates per-cell diagnostics
into a status JSON.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import traceback
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


# Make repository root importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _select_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_xi(pde_builder, info: dict) -> np.ndarray:
    """Build reduced PDE state xi from env info dict.

    The env populates ``info["built"]`` (StateBuilder output with s_ego,
    s_geom, s_vis, f_agents) and ``info["raw_obs"]``/``info["ttc_min"]``
    in both reset() and step(). ``ReducedPDEState.build`` consumes those
    two structures. See experiments/pde/collect_rollouts.py for the
    canonical reference.
    """
    built = info.get("built", {}) or {}
    return pde_builder.build(built, info)


def _action_logits(policy, obs: np.ndarray, device: str) -> tuple[int, "torch.Tensor"]:
    """Get (action_int, logits_1d) for a deterministic policy step.

    Mirrors ``EikonalAuxAgent.get_action(deterministic=True)`` but ALSO
    returns the actor's logits (which get_action discards). Side effect:
    advances ``policy._hidden`` exactly like get_action would.
    """
    with torch.no_grad():
        o = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
        h_in = policy._hidden
        # RecurrentActorCritic.__call__ returns (logits, value, log_prob, action, new_hidden)
        logits, _value, _lp, _action, new_hidden = policy.policy(o, h_in)
        policy._hidden = new_hidden
        # logits is (1, 1, n_actions) typically — flatten to (n_actions,)
        logits_flat = logits.reshape(-1)
        # Deterministic: argmax of logits.
        action_int = int(logits_flat.argmax().item())
        return action_int, logits_flat.detach().cpu()


def evaluate_cell(
    scenario: str,
    seed: int,
    maneuver: str,
    ckpt_path: str,
    n_episodes: int = 30,
    max_steps: int = 500,
) -> dict:
    """Compute Pearson, A_eik agreement, and KL diagnostics for a checkpoint.

    Parameters
    ----------
    scenario : SUMO scenario name (e.g. "1a").
    seed : eval seed (also drives env reset seeds).
    maneuver : ego maneuver string (e.g. "stem_right").
    ckpt_path : path to a trained EikonalAuxAgent checkpoint .pt file.
    n_episodes : number of deterministic eval episodes per call (default 30).
    max_steps : max env steps per episode (default 500).

    Returns
    -------
    dict with keys
        pearson, a_eik_agreement, kl,
        n_pairs_pearson, n_states_a_eik,
        n_succ_eps, n_coll_eps, n_to_eps
    or, on any failure, ``{"error": "..."}``.
    """
    if torch is None:
        return {"error": "torch is not available"}

    device = _select_device()
    env = None
    try:
        # ---- Lazy imports so that import errors propagate via the error-key path ----
        from env.sumo_env import SumoEnv
        from models.pde.eikonal_aux_agent import EikonalAuxAgent
        from models.pde.state_builder import ReducedPDEState
        from models.pde.dynamics import compute_one_step_toa_advantages
        from models.pde.checkpointing import (
            load_pde_checkpoint, peek_checkpoint_arch, verify_arch,
        )

        if not os.path.isfile(ckpt_path):
            return {"error": f"checkpoint not found: {ckpt_path}"}

        _set_seed(int(seed))

        # ---- Load policy from checkpoint ----
        # We do NOT use EikonalAuxAgent.from_checkpoint() because it also
        # restores the aux_optimizer state, which mismatches across Step 7C/7D
        # changes (log_sigma_eik was removed in 7D; older checkpoints persist
        # the old optimizer param group). For inference-only diagnostics we
        # only need the network weights, so we load policy + aux_critic
        # state dicts directly.
        arch = peek_checkpoint_arch(ckpt_path, device=device)
        ctor_kwargs = {
            k: arch[k]
            for k in ("obs_dim", "n_actions", "hidden_dim", "aux_hidden_dim")
            if k in arch
        }
        if "obs_dim" not in ctor_kwargs:
            return {"error": f"checkpoint at {ckpt_path} has no 'arch.obs_dim'"}
        policy = EikonalAuxAgent(device=device, **ctor_kwargs)
        ckpt_data = load_pde_checkpoint(ckpt_path, device=device)
        verify_arch(ckpt_data.get("arch"), policy._arch_dict(),
                    strict=True, ckpt_path=ckpt_path)
        policy.policy.load_state_dict(ckpt_data["policy"])
        policy.aux_critic.load_state_dict(ckpt_data["aux_critic"])
        # Pull tau / T_max from the checkpoint config if present.
        ck_cfg = ckpt_data.get("config", {}) or {}
        if "tau" in ck_cfg:
            policy.tau = float(ck_cfg["tau"])
        if "T_max" in ck_cfg:
            policy.T_max = float(ck_cfg["T_max"])
        policy.policy.eval()
        policy.aux_critic.eval()
        tau = float(getattr(policy, "tau", 1.0))
        T_max = float(getattr(policy, "T_max", 100.0))

        pde_builder = ReducedPDEState()

        # ---- Build env (must match training kwargs) ----
        env = SumoEnv(
            use_gui=False,
            scenario_name=scenario,
            ego_maneuver=maneuver,
            use_intent=False,
            buildings=True,
            style_filter=None,
            state_ablation=None,
        )

        # ---- Per-episode rollout ----
        # Episode buffers are filtered by termination cause AFTER the episode
        # finishes: success/collision contribute their states to the supervised
        # arrays; timeouts are kept only for A_eik agreement / KL (which are
        # state-level diagnostics independent of T_obs labels).
        all_xi: list[np.ndarray] = []
        all_actions: list[int] = []
        all_logits: list[np.ndarray] = []
        all_T_obs: list[float] = []  # NaN where T_obs undefined (timeouts)

        n_succ_eps = 0
        n_coll_eps = 0
        n_to_eps = 0

        for ep_idx in range(int(n_episodes)):
            ep_seed = int(seed) + 100_000 + ep_idx  # disjoint from training eval seeds
            try:
                obs, info = env.reset(seed=ep_seed)
            except TypeError:
                # Older gym API path (no kwarg)
                obs, info = env.reset()
            policy.reset_hidden()

            ep_xi: list[np.ndarray] = []
            ep_actions: list[int] = []
            ep_logits: list[np.ndarray] = []

            terminated_success = False
            terminated_collision = False
            timed_out = False
            ep_len = 0

            for step_i in range(int(max_steps)):
                # Build xi from the PRE-step info (matches collect_rollouts).
                xi_pre = _build_xi(pde_builder, info)
                # Deterministic action + logits (advances policy._hidden).
                try:
                    action_int, logits_flat = _action_logits(policy, obs, device)
                except Exception:
                    # If logits-extraction path fails, fall back to get_action;
                    # we will not be able to compute KL without logits, so we
                    # skip this state.
                    action_int, _, _, _ = policy.get_action(obs, deterministic=True)
                    logits_flat = None

                ep_xi.append(np.asarray(xi_pre, dtype=np.float32))
                ep_actions.append(int(action_int))
                if logits_flat is not None:
                    ep_logits.append(logits_flat.numpy().astype(np.float32))
                else:
                    ep_logits.append(np.full((policy.n_actions,), np.nan, dtype=np.float32))

                obs, _r, term, trunc, info = env.step(action_int)
                ep_len += 1

                if term:
                    if info.get("collision", False):
                        terminated_collision = True
                    else:
                        terminated_success = True
                    break
                if trunc:
                    timed_out = True
                    break
            else:
                # max_steps exhausted without termination => treat as timeout.
                timed_out = True

            n_states = len(ep_xi)
            if n_states == 0:
                continue

            # Decide T_obs per state in this episode.
            if terminated_success:
                # T_obs(t) = (T_episode - t) where T_episode = (n_states - 1)
                # because the success terminal step is the last appended state's
                # successor. For each state index t, countdown is (last_t - t).
                # We index pre-step states t = 0 .. n_states-1; the success
                # step itself is the post-step transition that ended the loop.
                # Convention used by trainer: success terminal contributes
                # T_obs = 0; intermediate states T_obs = (seg_end - t) in
                # macro-steps. The trainer's "seg_end" is the index of the
                # success state in xi_t; here, n_states - 1 is the last
                # PRE-step state, which is the macro-step BEFORE success was
                # registered. We therefore use (n_states - 1 - t) + 1 = n_states - t
                # as the countdown so that the last pre-success state has
                # T_obs = 1, and earlier states have proportionally larger
                # countdowns. (This matches the natural macro-step unit.)
                T_obs_arr = np.array(
                    [float(n_states - t) for t in range(n_states)], dtype=np.float32
                )
                n_succ_eps += 1
            elif terminated_collision:
                T_obs_arr = np.full((n_states,), float(T_max), dtype=np.float32)
                n_coll_eps += 1
            else:
                # Timeout: T_obs undefined.
                T_obs_arr = np.full((n_states,), np.nan, dtype=np.float32)
                n_to_eps += 1
                _ = timed_out  # silence unused

            all_xi.extend(ep_xi)
            all_actions.extend(ep_actions)
            all_logits.extend(ep_logits)
            all_T_obs.extend(T_obs_arr.tolist())

        if not all_xi:
            return {"error": "no states collected across eval episodes"}

        xi_np = np.stack(all_xi, axis=0).astype(np.float32)            # (N, XI_DIM)
        actions_np = np.asarray(all_actions, dtype=np.int64)            # (N,)
        logits_np = np.stack(all_logits, axis=0).astype(np.float32)     # (N, n_actions)
        T_obs_np = np.asarray(all_T_obs, dtype=np.float32)              # (N,) with NaN

        # ---- Compute T_phi(xi) under no_grad ----
        xi_t = torch.from_numpy(xi_np).to(device)
        with torch.no_grad():
            T_phi_t = policy.aux_critic(xi_t)
            if T_phi_t.dim() > 1:
                T_phi_t = T_phi_t.squeeze(-1)
        T_phi_np = T_phi_t.detach().cpu().numpy().reshape(-1)

        # ---- Pearson(T_phi, T_obs) on valid (non-NaN) states ----
        valid = ~np.isnan(T_obs_np)
        n_pairs_pearson = int(valid.sum())
        if n_pairs_pearson >= 2:
            # Guard against zero-variance arrays.
            x = T_phi_np[valid].astype(np.float64)
            y = T_obs_np[valid].astype(np.float64)
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                pearson = float("nan")
            else:
                with np.errstate(invalid="ignore"):
                    pearson = float(np.corrcoef(x, y)[0, 1])
        else:
            pearson = float("nan")

        # ---- A_eik via compute_one_step_toa_advantages (detach=True is fine; ----
        # ---- the function does not need autograd: it is purely T_net forward) ----
        try:
            A_eik_t = compute_one_step_toa_advantages(
                policy.aux_critic, xi_t, policy.dynamics, detach=True,
            )
            A_eik_np = A_eik_t.detach().cpu().numpy().astype(np.float64)  # (N, 5)
        except Exception as e:
            return {"error": f"A_eik computation failed: {e}"}

        # Agreement = mean(argmax_a A_eik == actor's chosen action).
        a_eik_argmax = A_eik_np.argmax(axis=1)
        n_states_a_eik = int(A_eik_np.shape[0])
        if n_states_a_eik > 0:
            a_eik_agreement = float((a_eik_argmax == actions_np).mean())
        else:
            a_eik_agreement = float("nan")

        # ---- KL(pi_theta || softmax(A_eik / tau)) averaged across states ----
        # Drop any state whose logits row is NaN (fallback path above).
        kl_valid = ~np.isnan(logits_np).any(axis=1)
        if kl_valid.sum() == 0:
            kl_mean = float("nan")
        else:
            log_p = logits_np[kl_valid].astype(np.float64)
            log_p = log_p - log_p.max(axis=1, keepdims=True)  # numerical stability
            log_p_norm = log_p - np.log(np.exp(log_p).sum(axis=1, keepdims=True))
            log_q_pre = (A_eik_np[kl_valid] / max(tau, 1e-8))
            log_q_pre = log_q_pre - log_q_pre.max(axis=1, keepdims=True)
            log_q_norm = log_q_pre - np.log(np.exp(log_q_pre).sum(axis=1, keepdims=True))
            p = np.exp(log_p_norm)
            kl_per_state = (p * (log_p_norm - log_q_norm)).sum(axis=1)
            kl_mean = float(kl_per_state.mean())

        return {
            "pearson": pearson,
            "a_eik_agreement": a_eik_agreement,
            "kl": kl_mean,
            "n_pairs_pearson": n_pairs_pearson,
            "n_states_a_eik": n_states_a_eik,
            "n_succ_eps": int(n_succ_eps),
            "n_coll_eps": int(n_coll_eps),
            "n_to_eps": int(n_to_eps),
        }

    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description="Step 7E eval-rollout diagnostics.")
    p.add_argument("--scenario", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--maneuver", required=True)
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--n_episodes", type=int, default=30)
    p.add_argument("--max_steps", type=int, default=500)
    args = p.parse_args()

    out = evaluate_cell(
        scenario=args.scenario,
        seed=args.seed,
        maneuver=args.maneuver,
        ckpt_path=args.ckpt_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
    )
    print(json.dumps(out, indent=2, default=str))
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
