# Hyperparameters Reference

Complete list of all hyperparameters and their current values across config files.

---

## Algorithm (configs/algo/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_pinn` | `true` | Enable physics-informed critic (Design A) |
| `lr` | `3e-4` | Adam learning rate |
| `gamma` | `0.99` | Discount factor |
| `gae_lambda` | `0.95` | GAE λ for advantage estimation |
| `clip_range` | `0.2` | PPO clip range |
| `ent_coef` | `0.01` | Entropy coefficient |
| `vf_coef` | `0.5` | Value function loss coefficient |
| `n_steps` | `4096` | Rollout steps per PPO update |
| `batch_size` | `128` | Minibatch size for PPO |
| `n_epochs` | `8` | PPO epochs per rollout |
| `gru_hidden` | `128` | GRU hidden dimension |
| `gru_layers` | `1` | GRU layers |

---

## Physics-Informed Critic / Residuals (configs/residuals/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lambda_physics_critic` | `0.5` | Overall weight for Design A physics-informed critic loss |
| `lambda_physics_ttc` | `1.0` | Per-residual: TTC violation weight |
| `lambda_physics_stop` | `1.0` | Per-residual: stopping-distance violation weight |
| `lambda_physics_fric` | `1.0` | Per-residual: friction-circle violation weight |
| `lambda_physics_ego` | `0.1` | Per-residual: ego dynamics (L_ego) weight; used when use_l_ego=True |
| `physics_ttc_thr` | `3.0` | TTC threshold (s): penalize V(s) when ttc < this |
| `physics_tau` | `0.5` | Reaction time (s) for d_stop = v·τ + v²/(2a_max) |
| `a_max` | `5.0` | Max deceleration (m/s²) for stopping distance |
| `mu` | `0.8` | Friction coefficient (friction circle) |
| `g` | `9.81` | Gravity (m/s²) |
| `lambda_ego` | `0.05` | (Legacy) Ego dynamics residual weight |
| `lambda_stop` | `0.01` | (Legacy) Stop distance residual weight |
| `lambda_fric` | `0.01` | (Legacy) Friction residual weight |
| `lambda_risk` | `0.01` | (Legacy) Risk residual weight |
| `a_brake` | `5.0` | Brake deceleration (m/s²) |
| `v_creep` | `1.0` | Creep speed (m/s) |
| `a_go` | `2.0` | Go acceleration (m/s²) |
| `eta` | `0.5` | Risk threshold for shield |
| `dt` | `0.1` | Control step (s) |

---

## Reward (configs/reward/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `w_prog` | `1.0` | Progress reward per m |
| `w_time` | `-0.1` | Time penalty per step |
| `w_risk` | `-1.0` | Penalty when TTC < ttc_thr |
| `w_coll` | `-10.0` | Collision penalty per step |
| `w_pothole` | `-5.0` | Pothole penalty |
| `ttc_thr` | `3.0` | TTC threshold (s) for risk penalty |
| `d_coll` | `2.0` | Distance (m) below which counts as collision |

---

## State (configs/state/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_n_agents` | `5` | Max agents in state vector |
| `epsilon` | `1e-6` | Small constant for numerical stability |
| `t_h_cpa` | `3.0` | Horizon (s) for closest-point-of-approach |
| `d_safe` | `2.0` | Safe distance (m) for TTC proxy |

---

## Scenario (configs/scenario/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stem_length` | `100` | Stem lane length (m) |
| `bar_half_length` | `80` | Bar half-length (m) |
| `lane_width` | `3.5` | Lane width (m) |
| `junction_type` | `priority` | Junction type (priority, right_before_left, unregulated) |
| `jm_ignore_probs` | `[0, 0.05, 0.1, 0.15, 0.2]` | Sampled jmIgnoreJunctionFoeProb per episode |

---

## DRPPO Internal (hardcoded / constructor)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_l_ego` | `False` | Enable L_ego (ego dynamics) in physics critic; ablation: pinn_ego uses True |
| `max_grad_norm` | `0.5` | Gradient clipping |
| `hidden_dim` | `128` | GRU hidden (same as gru_hidden) |
| `n_actions` | `5` | STOP, CREEP, YIELD, GO, ABORT |
