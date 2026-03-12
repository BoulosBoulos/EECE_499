# Hyperparameters Reference

Complete list of all hyperparameters and their **current values** across config files.
Synchronized with configs as of the latest update.

---

## Algorithm (configs/algo/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pinn_placement` | `critic` | PINN placement: "critic" / "actor" / "both" / "none" |
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
| `lambda_physics_critic` | `0.5` | Overall weight for physics-informed critic loss (sensitivity sweep: 0.001–1.0) |
| `lambda_physics_ttc` | `1.0` | Per-residual: TTC violation weight |
| `lambda_physics_stop` | `1.0` | Per-residual: stopping-distance violation weight |
| `lambda_physics_fric` | `1.0` | Per-residual: friction-circle violation weight |
| `lambda_physics_ego` | `0.1` | Per-residual: ego dynamics (L_ego) weight |
| `physics_ttc_thr` | `3.0` | TTC threshold (s) |
| `physics_tau` | `0.5` | Reaction time (s) for d_stop |
| `a_max` | `5.0` | Max deceleration (m/s²) |
| `mu` | `0.8` | Friction coefficient |
| `g` | `9.81` | Gravity (m/s²) |

---

## Reward (configs/reward/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `w_prog` | `1.0` | Progress reward per m |
| `w_time` | `-0.1` | Time penalty per step |
| `w_risk` | **`-3.0`** | Penalty when TTC < ttc_thr |
| `w_coll` | **`-20.0`** | Collision penalty (SUMO events + proximity) |
| `w_pothole` | `-5.0` | Pothole penalty |
| `ttc_thr` | `3.0` | TTC threshold (s) |
| `d_coll` | `2.0` | Proximity collision distance (m) |

---

## State (configs/state/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `top_n_agents` | `5` | Max agents in state vector |
| `epsilon` | `1e-6` | Numerical stability constant |
| `t_h_cpa` | `3.0` | CPA horizon (s) |
| `d_safe` | `2.0` | Safe distance for TTC proxy (m) |

---

## Scenario (configs/scenario/default.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stem_length` | **`200`** | Stem arm length (m) |
| `bar_half_length` | **`160`** | Bar arm half-length (m) |
| `lane_width` | `3.5` | Lane width (m) |
| `junction_type` | `priority` | Junction type |

---

## DRPPO Internal

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | `128` | GRU hidden dimension |
| `n_actions` | `5` | STOP, CREEP, YIELD, GO, ABORT |
| `max_grad_norm` | `0.5` | Gradient clipping |

---

## Intent LSTM (models/intent_style.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_dim` | `9` | Per-timestep feature dimension |
| `hidden_dim` | `64` | LSTM hidden dimension |
| `intent_classes` | `3` | yield/proceed/turn |
| `style_classes` | `3` | soft/mid/aggressive |
