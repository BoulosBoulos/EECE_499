# Changes Log: Architecture Refactoring

## Summary

Refactoring to prepare the codebase for 4 PDE-based auxiliary critic methods,
remove discarded heuristic methods, fix mathematical errors, and document everything.

---

## Changes

### 1. Fixed HJB residual time-scale error
**File**: `models/pde/residuals.py`

Replaced `dynamics.drift(xi, a)` (which returns `(F_a - xi) / dt`) with
`dynamics.one_step(xi, a) - xi` (raw state change). The old code made the
advection term 10x too large since dt = 0.1.

### 2. Added gamma factor to advection term in PDE Q-values
**File**: `models/pde/residuals.py`

Changed `q_a = r_a + advection` to `q_a = r_a + gamma * advection`.
The gamma parameter was passed to `pde_q_values` but never used in the
Q-value computation.

### 3. Replaced indicator functions with smooth sigmoids in local_reward
**File**: `models/pde/local_reward.py`

Replaced `(ttc_next < ttc_thr).float()` and `(d_pot_next < pothole_thr).float()`
with sigmoid approximations. The step functions had zero gradient everywhere,
breaking the autograd chain in PDE residual computation.

### 4. Refactored pde_q_values to avoid redundant dynamics calls
**Files**: `models/pde/residuals.py`, `models/pde/local_reward.py`

Added `local_reward_from_next` that accepts pre-computed `xi_next`. Refactored
`pde_q_values` to compute `dynamics.one_step` once per action (was called
twice: once in local_reward, once for advection). Halves dynamics evaluations.

### 5. Fixed frame mismatch in intent history CPA computation
**File**: `env/sumo_env.py`

In `_update_agent_history`, the CPA computation mixed world-frame `dp` with
ego-frame `delta_v`. Replaced with `delta_xy` (already in ego frame).

### 6. Removed heuristic physics methods from DRPPO
**File**: `models/drppo.py`

Stripped Design A/B physics losses, safety filter, L_ego, and PhysicsPredictor
dependency. DRPPO is now a clean recurrent PPO baseline. Legacy constructor
parameters are still accepted for backward compatibility but are ignored.

### 7. Removed models/physics.py
Old Cartesian bicycle model with `PhysicsPredictor`, `compute_L_ego`,
`compute_L_stop`, `compute_L_fric`, `compute_risk_from_state`. None used by
the PDE pipeline.

### 8. Implemented EikonalAuxAgent and EikonalAuxCritic
**Files**: `models/pde/eikonal_aux_agent.py`, `models/pde/eikonal_aux_critic.py`

Eikonal residual: `||grad U||^2 - c(xi)^2` where c = 1/v_eff with
safety-modulated effective speed.

### 9. Implemented CBFAuxAgent and CBFAuxCritic
**Files**: `models/pde/cbf_aux_agent.py`, `models/pde/cbf_aux_critic.py`

CBF-PDE residual: `ReLU(-max_a [grad_U*(F_a-xi) + alpha*U])` using the
auxiliary critic as a barrier function for safe set invariance.

### 10. Added comprehensive documentation
**Files**: `docs/ARCHITECTURE.md`, `docs/PDE_METHODS.md`, `docs/PROJECT_STATE.md`, `docs/CHANGES_LOG.md`

### 11. Deprecated configs/residuals/default.yaml
Replaced contents with deprecation comment. Kept file for backward compatibility
with experiment scripts that load it.

### 12. Deleted results/ablation_batch2/
Old heuristic method ablation results showing insignificant differences.
