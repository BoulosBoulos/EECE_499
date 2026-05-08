# Architecture Reference

## Pipeline Overview

```
World (SUMO)
  |
  v
Perception (raw_obs: ego, agents, geom, vis)
  |
  v
Feature Builder (StateBuilder)
  |
  v
Full State (134D)
  |--- [Optional] Intent LSTM --> Augmented State (+30D)
  |
  v
GRU Policy (RecurrentActorCritic)
  |
  v
Discrete Action {STOP=0, CREEP=1, YIELD=2, GO=3, ABORT=4}
  |
  v
SUMO
```

### PDE Auxiliary Critic Branch (parallel to policy)

```
Full State (134D)
  |
  v
ReducedPDEState --> PDE State (79D xi)
  |
  v
Auxiliary Critic U_phi(xi)  [MLP, tanh activations]
  |
  v
PDE Residual (autograd for grad_xi U)
  |
  v
Auxiliary Critic Update (L_anchor + L_pde + L_bc)
  |
  v
Distillation --> PPO Critic (L_distill)
```

---

## State Vectors

### Full State (134D)

| Block | Dims | Description |
|-------|------|-------------|
| s_ego | 6 | v, a, psi_dot, psi, jerk, yaw_accel |
| s_geom | 12 | d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane, g_turn(3), rho(2) |
| s_vis | 6 | alpha_cz, alpha_cross, d_occ, dt_seen, sigma_percep, n_occ |
| f_agents | 5x22=110 | Per-agent: dx, dy, dvx, dvy, delta_psi, v_i, a_i, d_cz_i, d_exit_i, tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i, chi_i, pi_ROW_i, nu_i, sigma_i, type(3), mask |
| **Total** | **134** | |

### PDE State (79D xi)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | v | ego speed (m/s) |
| 1 | a | ego acceleration (m/s^2) |
| 2 | psi_dot | ego yaw rate (rad/s) |
| 3 | d_stop | distance to stop line (m) |
| 4 | d_cz | distance to conflict zone entry (m) |
| 5 | d_exit | distance to conflict zone exit (m) |
| 6 | kappa | path curvature (1/m) |
| 7 | TTC_min | minimum TTC across agents (s) |
| 8 | alpha_cz | visible fraction of conflict zone [0,1] |
| 9 | alpha_cross | visible fraction of cross-traffic [0,1] |
| 10 | d_occ | distance to nearest occlusion boundary (m) |
| 11 | dt_seen | time since cross-traffic last observed (s) |
| 12-33 | Agent 1 | 22D ego-frame relative features |
| 34-55 | Agent 2 | 22D ego-frame relative features |
| 56-77 | Agent 3 | 22D ego-frame relative features |
| 78 | d_pothole | distance to pothole (m) |

**Per-agent 22D layout**: dx, dy, dvx, dvy, delta_psi, v_i, a_i, d_cz_i, d_exit_i, tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i, chi_i, pi_ROW_i, nu_i, sigma_i, type_onehot(3), mask

### Intent Features (30D, optional)

When `use_intent=True`, the Intent LSTM produces 6 features per agent (top 5 agents):
intent class probabilities and style parameters, appended to the full state.

---

## Action Space

| Index | Name | Nominal Control |
|-------|------|-----------------|
| 0 | STOP | a = -a_brake (5.0 m/s^2), delta = 0 |
| 1 | CREEP | Regulate toward v_creep (1.0 m/s) |
| 2 | YIELD | a = -0.5 m/s^2 |
| 3 | GO | a = a_go (2.0 m/s^2) |
| 4 | ABORT | a = -a_abort (8.0 m/s^2) |

---

## Behavioral Dynamics (BehavioralDynamics)

Path-distance propagation model operating on the PDE state xi:

1. Nominal acceleration from action mapping
2. `v_new = clamp(v + a_nom * dt, 0, v_max)`
3. `delta_s = 0.5 * (v + v_new) * dt` (trapezoidal integration)
4. Distances propagated: `d_stop, d_cz, d_exit, d_pothole -= delta_s`
5. Yaw rate updated from path curvature: `psi_dot = (v_new/L) * tan(atan(L*kappa))`
6. Per-agent: constant-velocity propagation of relative positions and recomputation of tau, t_cpa, d_cpa, TTC

**Time step**: dt = 0.1s

---

## Reward Structure

### Environment Reward (env/sumo_env.py)
- Progress: `w_prog * 0.5 * (v + v_new) * dt`
- Time penalty: `w_time * dt`
- Risk penalty: `w_risk * I(TTC < threshold)`
- Pothole penalty: `w_pothole * I(d_pot < threshold)`
- Collision: `w_coll` (terminal, large negative)
- Success: `w_success` (terminal, positive)

### PDE Local Reward Surrogate (models/pde/local_reward.py)
Same structure but uses smooth sigmoid approximations instead of step functions,
and excludes collision/success (handled via boundary conditions in PDE loss).

---

## Agent Variants

| Agent | Method | Auxiliary Critic | Residual |
|-------|--------|-----------------|----------|
| DRPPO | Baseline PPO | None | None |
| HJBAuxAgent | Hard-HJB | HJBAuxCritic | U*ln(gamma) + max_a[r + gamma*grad_U*(F_a-xi)] |
| SoftHJBAuxAgent | Soft-HJB | SoftHJBAuxCritic | U*ln(gamma) + tau*logsumexp([r + gamma*grad_U*(F_a-xi)]/tau) |
| EikonalAuxAgent | Eikonal | EikonalAuxCritic | ||grad_U||^2 - c(xi)^2 |
| CBFAuxAgent | CBF-PDE | CBFAuxCritic | ReLU(-max_a[grad_U*(F_a-xi) + alpha*U]) |

All PDE agents share:
- RecurrentActorCritic as the policy network
- Auxiliary MLP critic (79D input, 256-dim hidden, tanh activations)
- Collocation sampling (mix of rollout states + jittered copies)
- Anchor loss (MSE to GAE returns)
- Boundary conditions (success -> U=0, collision -> U=w_coll)
- Distillation (PPO critic <- auxiliary critic)
