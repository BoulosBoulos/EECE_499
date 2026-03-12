# DRPPO Framework: Complete A-to-Z Reference

**Physics-Informed Recurrent RL for Autonomous Driving at T-Intersections**

---

## 1. Project Overview

This project implements a **Deep Recurrent Proximal Policy Optimization (DRPPO)** agent that learns to navigate a T-intersection in SUMO. The agent must handle:
- Other vehicles (cars, motorcycles) with diverse maneuvers and driving styles
- Pedestrians with varied crossing behaviors
- Road hazards (potholes with random placement)
- Right-of-way decisions under uncertainty

The key innovation is a **physics-informed critic** that regularizes the value function using TTC, stopping-distance, and friction-circle constraints, plus optional ego dynamics prediction error (L_ego).

---

## 2. Scenarios

| ID | Actors Present | Purpose |
|----|---------------|---------|
| 1a | Ego + other car | Car-only interaction |
| 1b | Ego + pedestrian | Pedestrian crossing |
| 1c | Ego + motorcycle | Motorcycle interaction |
| 1d | Ego + pothole | Hazard avoidance |
| 2 | Ego + car + pedestrian | Mixed VRU |
| 3 | Ego + car + pedestrian + motorcycle | Full multi-agent |
| 4 | Ego + car + pedestrian + motorcycle + pothole | Everything |

**Network layout**: Bidirectional T-intersection with 2 lanes per direction on all arms.
- Stem: 200m, Bar: 160m each side
- Ego always enters from stem, turns right

---

## 3. Behavior Diversity (BehaviorSampler)

Each episode samples independent behavior configs. File: `scenario/behavior_sampler.py`

### 3.1 Car Maneuvers
| Maneuver | Route | Intent Label |
|----------|-------|-------------|
| straight_left_right | left_in → right_out | proceed (1) |
| straight_right_left | right_in → left_out | proceed (1) |
| turn_left | left_in → stem_out | turn (2) |
| turn_right | right_in → stem_out | turn (2) |

### 3.2 Car Styles
| Style | sigma | accel | tau | jm_ignore | Speed | Label |
|-------|-------|-------|-----|-----------|-------|-------|
| nominal | 0.15 | 2.5 | 0.5 | 0.0 | 13.89 | mid (1) |
| aggressive | 0.3 | 4.0 | 0.3 | 0.15 | 16.67 | aggressive (2) |
| timid | 0.05 | 1.5 | 1.0 | 0.0 | 11.11 | soft (0) |
| distracted | 0.4 | 2.0 | 1.5 | 0.05 | 13.89 | soft (0) |
| erratic | 0.5 | 3.5 | 0.3 | 0.1 | 16.67 | aggressive (2) |
| drunk | 0.6 | 2.0 | 1.2 | 0.2 | 11.11 | aggressive (2) |
| rule_violating | 0.2 | 3.0 | 0.3 | 0.2 | 16.67 | aggressive (2) |

### 3.3 Pedestrian Maneuvers
| Maneuver | Direction | Intent Label |
|----------|-----------|-------------|
| cross_left_right | left_in → right_out | cross (1) |
| cross_right_left | right_out → left_in | cross (1) |

### 3.4 Pedestrian Styles
| Style | Speed (m/s) | Special Behavior | Label |
|-------|-------------|------------------|-------|
| normal_walk | 1.2 | — | mid (1) |
| running | 3.0 | — | aggressive (2) |
| slow_elderly | 0.6 | — | soft (0) |
| stop_midway | 1.2 | Stops in CZ for 2-5s | soft (0) |
| hesitant | 1.0 | Start-stop-start | soft (0) |
| distracted_slow | 0.8 | Late start | soft (0) |
| jaywalking_fast | 2.5 | Early, fast cross | aggressive (2) |

### 3.5 Motorcycle Maneuvers
| Maneuver | Route | Intent Label |
|----------|-------|-------------|
| straight_right_left | right_in → left_out | proceed (1) |
| straight_left_right | left_in → right_out | proceed (1) |
| turn_into_stem | right_in → stem_out | turn (2) |

### 3.6 Motorcycle Styles
| Style | sigma | accel | tau | max_speed | Label |
|-------|-------|-------|-----|-----------|-------|
| nominal | 0.15 | 4.0 | 0.4 | 16.67 | mid (1) |
| aggressive_fast | 0.3 | 6.0 | 0.2 | 22.22 | aggressive (2) |
| cautious | 0.05 | 2.5 | 0.8 | 13.89 | soft (0) |
| late_brake | 0.2 | 5.0 | 0.15 | 19.44 | aggressive (2) |
| swerving | 0.5 | 4.0 | 0.3 | 16.67 | aggressive (2) |
| yield_to_ego | 0.05 | 2.0 | 1.2 | 11.11 | soft (0) |

### 3.7 Pothole
- Position: randomly sampled within junction conflict zone (x ∈ [-8, 8], y ∈ [-5, 5])
- Size: half_w ∈ [1.5, 4.0], half_h ∈ [1.0, 2.5]
- Color: dark brown (0.2, 0.15, 0.1) at layer 1 for visibility

---

## 4. Actions

| ID | Name | Effect |
|----|------|--------|
| 0 | STOP | Hard brake: v -= 5.0 * dt |
| 1 | CREEP | Slow approach: target ~1 m/s |
| 2 | YIELD | Gentle decel: v -= 0.5 * dt |
| 3 | GO | Accelerate: v += 1.0 * dt |
| 4 | ABORT | Emergency stop: v -= 5.0 * dt |

---

## 5. State Vector (134-165 dimensions)

### 5.1 Ego State (6 dims)
v, a, ψ̇, ψ, Δa (jerk), Δψ̇

### 5.2 Geometry (12 dims)
d_stop, d_cz, d_exit, κ, e_y, e_ψ, w_lane, g_turn[3], ρ[2]

### 5.3 Visibility (6 dims)
α_cz, α_cross, d_occ, dt_seen, σ_percep, n_occ

### 5.4 Per-Agent Features (22 × 5 = 110 dims)
Δx, Δy, Δv_x, Δv_y, Δψ, v_i, a_i, d_cz_i, d_exit_i, τ_i, Δτ_i,
t_cpa, d_cpa, TTC_i, χ_i, π_row_i, ν_i, σ_i, type_onehot[3], mask

### 5.5 Intent Features (optional, 30 dims)
5 agents × (3 intent probs + 3 style probs)

### 5.6 Pothole Distance (optional, 1 dim)

---

## 6. Model Architecture (DRPPO)

### 6.1 RecurrentActorCritic
- GRU encoder: obs_dim → hidden_dim (128)
- Actor head: hidden_dim → 5 actions (categorical)
- Critic head: hidden_dim → 1 value
- Hidden state carried across timesteps (reset on episode boundary)

### 6.2 Physics-Informed Critic (Design A)
L_total = L_actor + c_ent * L_entropy + c_vf * (L_vf + λ * L_physics)

L_physics = V(s) × (λ_ttc × violation_ttc + λ_stop × violation_stop + λ_fric × violation_fric)

- violation_ttc = ReLU(ttc_thr - ttc_min)
- violation_stop = ReLU(-(d_cz - d_stop(v)))  where d_stop = v*τ + v²/(2*a_max)
- violation_fric = ReLU(a_lat² + a_lon² - (μg)²)

### 6.3 L_ego (optional)
Penalizes V(s_next) when bicycle-model prediction error is high.

### 6.4 IntentStylePredictor (LSTM)
- Input: per-agent history (B, T, 9)
- LSTM: 9 → 64
- Intent head: 64 → 3 (yield/proceed/turn)
- Style head: 64 → 3 (soft/mid/aggressive)
- Trained separately via `experiments/train_intent.py`
- Features aligned with state-builder agent ordering

---

## 7. Reward Function

r_t = w_prog × progress + w_time × dt + [w_risk if TTC < thr] + [w_coll if collision] + [w_pothole if in_pothole]

| Parameter | Value | Description |
|-----------|-------|-------------|
| w_prog | 1.0 | Per-meter progress |
| w_time | -0.1 | Per-step time penalty |
| w_risk | -3.0 | TTC below threshold |
| w_coll | -20.0 | Actual collision (SUMO event + proximity) |
| w_pothole | -5.0 | Driving over pothole |
| ttc_thr | 3.0 s | Risk threshold |
| d_coll | 2.0 m | Proximity collision distance |

---

## 8. Collision Detection

Two-layer collision detection:
1. **SUMO events**: `traci.simulation.getCollidingVehiclesIDList()` — actual physics collisions
2. **Proximity**: Any agent within d_coll (2m) of ego — catches near-misses SUMO may not flag

Both are logged separately in `info["sumo_collision"]` and `info["proximity_collision"]`.
Combined as `info["collision"]`.

---

## 9. All Hyperparameters (Current Values)

### 9.1 Algorithm (configs/algo/default.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| lr | 3e-4 | Adam learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE λ |
| clip_range | 0.2 | PPO clip |
| ent_coef | 0.01 | Entropy bonus |
| vf_coef | 0.5 | Value loss coefficient |
| n_steps | 4096 | Rollout length |
| batch_size | 128 | Minibatch size |
| n_epochs | 8 | PPO epochs per rollout |
| gru_hidden | 128 | GRU hidden dimension |
| gru_layers | 1 | GRU layers |

### 9.2 Physics Critic (configs/residuals/default.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| lambda_physics_critic | 0.5 | Overall physics loss weight |
| lambda_physics_ttc | 1.0 | TTC violation weight |
| lambda_physics_stop | 1.0 | Stopping distance weight |
| lambda_physics_fric | 1.0 | Friction circle weight |
| lambda_physics_ego | 0.1 | Ego dynamics weight |
| physics_ttc_thr | 3.0 s | TTC threshold |
| physics_tau | 0.5 s | Reaction time |
| a_max | 5.0 m/s² | Max deceleration |
| mu | 0.8 | Friction coefficient |
| g | 9.81 m/s² | Gravity |

### 9.3 Reward (configs/reward/default.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| w_prog | 1.0 | Progress per meter |
| w_time | -0.1 | Time penalty |
| w_risk | -3.0 | TTC risk penalty |
| w_coll | -20.0 | Collision penalty |
| w_pothole | -5.0 | Pothole penalty |
| ttc_thr | 3.0 s | TTC threshold |
| d_coll | 2.0 m | Collision distance |

### 9.4 Scenario (configs/scenario/default.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| stem_length | 200 m | Stem arm length |
| bar_half_length | 160 m | Bar arm half-length |
| junction_type | priority | Yield rules |

### 9.5 State (configs/state/default.yaml)
| Parameter | Value | Description |
|-----------|-------|-------------|
| top_n_agents | 5 | Max agents tracked |
| epsilon | 1e-6 | Numerical stability |
| t_h_cpa | 3.0 s | CPA horizon |
| d_safe | 2.0 m | Safe distance |

---

## 10. Ablation Study Variants

| Variant | pinn_placement | use_l_ego | use_safety_filter | Description |
|---------|---------------|-----------|-------------------|-------------|
| nopinn | none | False | False | Baseline PPO, no physics |
| pinn_critic | critic | False | False | Design A: physics on critic |
| pinn_actor | actor | False | False | Design B: physics on actor |
| pinn_both | both | False | False | Designs A+B combined |
| pinn_ego | critic | True | False | Design A + L_ego dynamics |
| pinn_no_ttc | critic | False | False | Design A without TTC residual |
| pinn_no_stop | critic | False | False | Design A without stopping-distance |
| pinn_no_fric | critic | False | False | Design A without friction-circle |
| safety_filter | none | False | True | No physics loss, safety filter only |
| pinn_critic_sf | critic | False | True | Design A + safety filter |

All ablations run with **5 seeds** (42, 123, 456, 789, 999) and report mean ± std.

### 10.1 Hyperparameter Sensitivity Sweep

Tests `lambda_physics_critic` across: {0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}.
See `docs/ABLATION_HYPERPARAMETERS.md` for full design.

---

## 11. Training Pipeline

```
1. make setup                  # Create venv, install deps
2. make train-intent           # Train intent LSTM (optional)
3. make train                  # Train DRPPO (pinn + nopinn)
4. make eval                   # Evaluate checkpoint
5. make ablation               # Full ablation (6 variants × 5 seeds)
6. make ablation-sensitivity   # Sensitivity sweep (λ_phys: 0.001–1.0)
7. make plot                   # Plot training curves
8. make plot-ablation          # Plot ablation results
9. make dashboard              # Launch Streamlit dashboard (localhost:8501)
10. make visualize-gui         # Watch in SUMO GUI
```

---

## 12. Logged Metrics

### Training CSV columns:
step, episode_return, episode_len, actor_loss, vf_loss, collision_count, collision_rate, mean_ttc, min_ttc, entropy, l_physics, l_actor_physics, l_ego, viol_ttc_rate, viol_stop_rate, viol_fric_rate, viol_ttc_mag, viol_stop_mag, viol_fric_mag

### Eval CSV columns:
seed, eval_mode, episode, return, length, collision_steps, collision_episode, mean_ttc, min_ttc, pothole_hits

### Ablation Eval CSV columns:
scenario, variant, pinn_placement, use_l_ego, use_safety_filter, use_intent, lambda_phys, seed, eval_mode, mean_return, std_return, collision_rate, pothole_hits_mean, mean_ttc, min_ttc

### Ablation Training Log CSV columns:
scenario, variant, lambda_phys, seed, step, actor_loss, vf_loss, entropy, total_loss, l_physics, l_actor_physics, l_ego, viol_ttc_rate, viol_stop_rate, viol_fric_rate, viol_ttc_mag, viol_stop_mag, viol_fric_mag

### Intent Training CSV columns:
epoch, train_loss, val_loss, val_intent_acc, val_style_acc

---

## 13. File Structure

```
configs/
  algo/default.yaml          # PPO hyperparameters
  residuals/default.yaml     # Physics critic weights
  reward/default.yaml        # Reward function weights
  scenario/default.yaml      # T-intersection dimensions
  state/default.yaml         # State builder params
env/
  sumo_env.py                # Main SUMO environment
scenario/
  generator.py               # SUMO network + route generation
  behavior_sampler.py        # Per-episode behavior diversity
state/
  builder.py                 # Raw obs → structured state vector
models/
  drppo.py                   # DRPPO: GRU + PPO + physics critic
  intent_style.py            # IntentStylePredictor LSTM
  physics.py                 # Bicycle model for L_ego
experiments/
  run_train.py               # Train DRPPO
  run_eval.py                # Evaluate with multi-seed + CI
  run_ablation.py            # Ablation study (multi-seed, drop-one, sensitivity)
  train_intent.py            # Train intent LSTM
  dashboard.py               # Streamlit dashboard (localhost:8501)
  plot_comparison.py         # Plot pinn vs nopinn
  plot_ablation.py           # Plot ablation results
  run_visualize_sumo.py      # SUMO visualization
  run_visualize_ablation.py  # Ablation visualization
docs/
  FRAMEWORK.md               # This file (A-to-Z reference)
  ABLATION_HYPERPARAMETERS.md # Ablation design and sensitivity sweep
  HYPERPARAMETERS.md          # Hyperparameter reference
  STATE.md                   # State vector specification
  PHYSICS_INFORMED.md        # Physics critic explanation
  RUNNING.md                 # How to run everything
  PIPELINE.md                # End-to-end pipeline
  SCENARIO.md                # Scenario details
```

---

## 14. Known Limitations and Future Work

### Currently Placeholder:
- κ (curvature): derived from yaw rate and speed in `env.sumo_env` (ψ̇ / v); still 0 in the legacy synthetic env. e_y, e_ψ: fixed at 0 (no path model)
- g_turn: fixed [0,0,1] (always right turn)
- ρ: fixed [0.5, 0.5] (no ROW model)
- All visibility features (α_cz, α_cross, d_occ, dt_seen, σ_percep, n_occ): fixed (no occlusion model)
- Per-agent d_cz_i, d_exit_i: rough proxy (‖p_i − p_e‖ × 0.5), not lane-aware
- χ_i, π_row_i: fixed 0.5 (no per-agent intent in state, only via intent LSTM)

### Implemented:
- ψ̇ (yaw rate): computed from consecutive TraCI headings (was previously always 0)
- Multi-seed ablation with 5 seeds and aggregated results
- Drop-one ablation variants (pinn_no_ttc, pinn_no_stop, pinn_no_fric)
- Hyperparameter sensitivity sweep (λ_phys: 0.001–1.0)
- Per-term physics loss logging (L_physics, L_ego, violation rates/magnitudes)
- Streamlit dashboard for interactive visualization

### Future:
- Implement proper path tracking (curvature from SUMO lane geometry)
- Occlusion model using ray-casting or sensor simulation
- Right-of-way reasoning from traffic light / priority rules
- Lane-aware d_cz_i / d_exit_i from SUMO route geometry
- Action switching penalty (w_switch)
- Comfort penalty (w_comfort, w_jerk)
- Varying ego objectives (turn left, go straight) for generalization
