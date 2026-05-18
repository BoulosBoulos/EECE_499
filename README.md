# EECE 499 — Physics-Informed Auxiliary Critics for Autonomous Behavioral Decision-Making at Unsignalized Intersections

**Author:** Boulos Boulos  
**Supervisor:** Prof. Naseem Daher  
**Institution:** American University of Beirut (AUB), Electrical and Computer Engineering  
**Target Venue:** IEEE Transactions on Intelligent Vehicles (T-IV)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Core Concepts](#3-core-concepts)
4. [Environment: SUMO T-Intersection](#4-environment-sumo-t-intersection)
5. [Observation Space](#5-observation-space)
6. [Action Space](#6-action-space)
7. [Reward Function](#7-reward-function)
8. [Models](#8-models)
   - [DRPPO Baseline](#81-drppo-baseline)
   - [Shared PDE Architecture](#82-shared-pde-architecture)
   - [Hard-HJB Auxiliary Critic](#83-hard-hjb-auxiliary-critic)
   - [Soft-HJB Auxiliary Critic](#84-soft-hjb-auxiliary-critic)
   - [Eikonal Auxiliary Critic](#85-eikonal-auxiliary-critic)
   - [CBF Auxiliary Critic](#86-cbf-auxiliary-critic)
   - [Fusion Auxiliary Critic](#87-fusion-auxiliary-critic)
   - [Rule-Based Baseline](#88-rule-based-baseline)
   - [Intent and Style Encoder](#89-intent-and-style-encoder)
9. [PDE Infrastructure](#9-pde-infrastructure)
   - [Reduced PDE State (xi)](#91-reduced-pde-state-xi)
   - [Behavioral Dynamics](#92-behavioral-dynamics)
   - [Collocation Sampling](#93-collocation-sampling)
   - [Local Reward Surrogate](#94-local-reward-surrogate)
   - [Residual Computations](#95-residual-computations)
   - [Checkpointing](#96-checkpointing)
10. [Scenario Generation](#10-scenario-generation)
11. [Behavior Sampling](#11-behavior-sampling)
12. [Experiment Pipeline](#12-experiment-pipeline)
    - [Training Scripts](#121-training-scripts)
    - [Calibration](#122-calibration)
    - [Ablation Tiers](#123-ablation-tiers)
    - [Evaluation](#124-evaluation)
13. [Analysis Pipeline](#13-analysis-pipeline)
14. [Configuration System](#14-configuration-system)
15. [Verification Suite](#15-verification-suite)
16. [Results Storage](#16-results-storage)
17. [Hardware and Performance](#17-hardware-and-performance)
18. [Installation and Setup](#18-installation-and-setup)
19. [Running Experiments](#19-running-experiments)
20. [Key Design Decisions](#20-key-design-decisions)

---

## 1. Project Overview

This codebase implements and compares five physics-informed reinforcement learning methods for **discrete behavioral decision-making** at unsignalized T-intersections under **partial observability**. The central contribution is an auxiliary critic architecture where a separate neural network is trained to satisfy a PDE residual derived from the system dynamics, then distilled into the main PPO value function.

The problem is framed as a Partially Observable Markov Decision Process (POMDP). The ego autonomous ground vehicle must choose among five discrete behavioral actions — STOP, CREEP, YIELD, GO, ABORT — to navigate a T-intersection shared with mixed traffic (cars, pedestrians, motorcyclists) while buildings create occlusion blind spots.

**Six trainable methods** are compared:

| Key | Method | PDE Family |
|-----|---------|------------|
| `drppo` | Recurrent PPO baseline | None (baseline) |
| `hjb_aux` | Hard-HJB auxiliary critic | Optimality (discrete Bellman) |
| `soft_hjb_aux` | Soft-HJB auxiliary critic | Optimality (entropy-regularized) |
| `eikonal_aux` | Eikonal auxiliary critic | Safety (time-of-arrival) |
| `cbf_aux` | CBF auxiliary critic | Safety (barrier descent) |
| `fusion_aux` | Fusion of optimality + safety critics | Hybrid |

A **rule-based TTC-threshold controller** serves as a non-learning reference.

---

## 2. Repository Structure

```
EECE_499-main/
│
├── env/                        # SUMO Gymnasium environment
│   └── sumo_env.py             # Main environment class (1,478 lines)
│
├── models/                     # All neural network models
│   ├── drppo.py                # DRPPO baseline: RecurrentActorCritic + DRPPO trainer
│   ├── intent_style.py         # Per-agent LSTM intent/style encoder (v9)
│   ├── rule_based_policy.py    # Deterministic TTC-threshold reference policy
│   └── pde/                    # Physics-informed auxiliary critic family
│       ├── state_builder.py    # Reduced PDE state xi (79D), extractor
│       ├── dynamics.py         # BehavioralDynamics: differentiable one-step xi update
│       ├── residuals.py        # HJB, Soft-HJB, Eikonal, CBF residual functions
│       ├── collocation.py      # Collocation point sampler with physics-consistent jitter
│       ├── local_reward.py     # Surrogate reward r(xi, a) for PDE training
│       ├── checkpointing.py    # Save/load for PDE-family checkpoints
│       ├── hjb_aux_agent.py    # Hard-HJB agent (train_step, get_action)
│       ├── hjb_aux_critic.py   # Hard-HJB auxiliary critic MLP
│       ├── soft_hjb_aux_agent.py
│       ├── soft_hjb_aux_critic.py
│       ├── eikonal_aux_agent.py
│       ├── eikonal_aux_critic.py
│       ├── cbf_aux_agent.py
│       ├── cbf_aux_critic.py
│       ├── fusion_aux_agent.py # Fusion agent (506 lines)
│       └── fusion_aux_critic.py
│
├── scenario/                   # SUMO network and behavior generation
│   ├── generator.py            # Generates SUMO XML files for 7 scenarios
│   └── behavior_sampler.py     # Per-episode behavior/style randomizer
│
├── scenarios/                  # Pre-generated SUMO network files
│   └── sumo_1a/               # Network XML for scenario 1a
│       ├── t.net.xml
│       ├── t.nod.xml
│       ├── t.edg.xml
│       └── scenario_dims.yaml
│
├── experiments/
│   └── pde/                    # All experiment entry points
│       ├── train_drppo_baseline.py
│       ├── train_hjb_aux.py
│       ├── train_soft_hjb_aux.py
│       ├── train_eikonal_aux.py
│       ├── train_cbf_aux.py
│       ├── train_fusion_aux.py
│       ├── eval.py             # Unified evaluation script
│       ├── run_calibration.py  # 36-job calibration orchestrator
│       ├── run_ablation.py     # Single-method ablation runner
│       ├── run_full_ablation.py# Full multi-tier ablation orchestrator
│       ├── run_metadata.py     # Column schemas and metadata helpers
│       ├── smoke_test.py       # Single-run smoke test
│       ├── smoke_test_orchestrator.py
│       ├── aggregate_tier_1_results.py
│       ├── preview_tier_1_split.py
│       ├── trajectory_logger.py
│       ├── verify_conflicts.py
│       ├── visualize_sumo.py
│       ├── collect_rollouts.py
│       ├── plot_interaction.py
│       ├── plot_pde.py
│       └── analysis/           # Post-hoc analysis scripts
│           ├── plot_learning_curves.py
│           ├── plot_pde_convergence.py
│           ├── plot_failure_trajectories.py
│           ├── generate_results_tables.py
│           ├── compute_aulc.py
│           └── compute_overhead.py
│
├── analysis/                   # Offline analysis pipeline
│   ├── config.py               # Colors, method ordering, statistical constants
│   ├── loader.py               # Results loader and quality checker
│   ├── plots.py                # Seven plot families (PDF + HTML)
│   ├── stats.py                # Welch t-test, Holm correction, Cohen's d
│   ├── tables.py               # LaTeX table generator
│   ├── metrics.py              # Metric computation from raw CSVs
│   ├── quality.py              # Run quality gate checks
│   ├── run_analysis.py         # Orchestrates full analysis pipeline
│   ├── calibration_analysis.py # Calibration-specific analysis
│   ├── calibration_diagnostic.py
│   └── calibration_action_termination.py
│
├── verification/               # Preflight and validation scripts
│   ├── preflight_5_occlusion.py
│   ├── preflight_6_buildings.py
│   ├── preflight_7_pothole.py
│   ├── preflight_9_dense.py
│   ├── preflight_10_style.py
│   ├── preflight_11_state_ablation.py
│   ├── preflight_12_obsdim.py
│   ├── test_residuals_math.py
│   ├── test_smooth_clamp.py
│   ├── run_determinism.py
│   ├── run_ckpt_round_trip.py
│   ├── validate_smoke_outputs.py
│   └── smoke_hjb/ smoke_rb/    # Smoke test output artifacts
│
├── configs/                    # Per-subsystem YAML overrides
│   ├── algo/default.yaml
│   ├── pde/{hjb,soft_hjb,eikonal,cbf}_aux.yaml
│   ├── reward/default.yaml
│   ├── scenario/default.yaml
│   └── state/default.yaml
│
├── docs/                       # 20+ internal documentation files
│   ├── ARCHITECTURE.md
│   ├── PDE_METHODS.md
│   ├── ENVIRONMENT.md
│   ├── STATE.md
│   ├── EGO_MANEUVERS.md
│   ├── SCENARIO.md
│   ├── HYPERPARAMETERS.md
│   └── ...
│
├── config_frozen_v1.yaml       # Canonical locked configuration (all tiers)
├── config_loader.py            # YAML loader + config-lock enforcer
├── config_lock.json            # Hash of frozen config for integrity checks
├── requirements.txt
├── Makefile
└── CALIBRATION_REPORT.md       # 36-run calibration outcome documentation
```

---

## 3. Core Concepts

### The Dual-Critic Architecture

Every PDE-augmented method (Hard-HJB, Soft-HJB, Eikonal, CBF, Fusion) shares the same structural pattern:

```
Observation (135D or 165D)
        │
        ▼
  GRU Encoder  ─────────────────────────────────────────┐
        │                                               │
        ▼                                               ▼
  PPO Actor (policy π)           PPO Critic V_ψ (value)
        │                                ▲
        │                         L_distill (stop-grad)
        │                                │
        │                         Auxiliary Critic U_φ(xi)
        │                                │
        │                         PDE Residual ρ(xi)
        │                         (HJB / Eikonal / CBF)
        │
        ▼
  Discrete action (STOP/CREEP/YIELD/GO/ABORT)
```

The auxiliary critic `U_φ` operates on a **79-dimensional reduced physics state** `xi` extracted from the full observation. It is trained with a composite loss:

```
L_aux = λ_res * L_residual(U_φ, xi)   # PDE consistency
       + λ_dist * L_distill(V_ψ, U_φ)  # value alignment
       + λ_bc * L_bc(U_φ, xi_terminal) # boundary conditions
       + L_anchor(U_φ, GAE_returns)     # empirical anchoring
```

After each update, `U_φ(xi_t).detach()` is used as a soft supervision target for `V_ψ` through `L_distill = MSE(V_PPO, U_aux.detach())`. This decouples PDE physics from policy optimization — no gradient from the PDE residual ever reaches the actor `π_θ`.

### Why a Reduced State (xi)?

The full observation is 135–165D and includes raw perception features, agent IDs, intent encodings, and geometry that are not physically meaningful for PDE computation. The auxiliary critic needs a state where:

1. Derivatives `∇_xi U` are physically interpretable (speed, distance to collision zone, etc.)
2. The behavioral dynamics `F_a(xi)` can be computed in closed form
3. Autograd can trace through the dynamics to compute the Hamiltonian

The 79D reduced state `xi` contains exactly the physically meaningful subset.

---

## 4. Environment: SUMO T-Intersection

**File:** `env/sumo_env.py` (1,478 lines)  
**Class:** `SumoEnv` — wraps [SUMO](https://sumo.dlr.de) via TraCI as a Gymnasium environment.

### Intersection Geometry

A T-junction with three arms:
- **Stem** (south): where the ego typically enters from
- **Left arm** (west): horizontal road, left direction
- **Right arm** (east): horizontal road, right direction

Four corner buildings (NW, NE, SW, SE) create occlusion blind spots. Buildings can be toggled on/off via the `no_buildings` flag for ablation studies.

### Scenarios

Seven base scenarios, each defining which agent types are present:

| Scenario | Car | Pedestrian | Motorcyclist | Pothole |
|----------|-----|------------|--------------|---------|
| `1a` | ✓ | — | — | — |
| `1b` | — | ✓ | — | — |
| `1c` | — | — | ✓ | — |
| `1d` | — | — | — | ✓ |
| `2` | ✓ | ✓ | — | — |
| `3` | ✓ | ✓ | ✓ | — |
| `4` | ✓ | ✓ | ✓ | ✓ |

**Dense variants** (`2_dense`, `3_dense`, `4_dense`) spawn multiple instances of each agent type for high-conflict stress testing.

### Ego Maneuvers

Six legal T-intersection routes the ego can be assigned each episode:

| Key | Entry | Exit | Description |
|-----|-------|------|-------------|
| `stem_right` | stem_in | right_out | Enter from stem, exit right |
| `stem_left` | stem_in | left_out | Enter from stem, exit left |
| `right_stem` | right_in | stem_out | Enter from right, exit stem |
| `right_left` | right_in | left_out | Through-traffic right-to-left |
| `left_stem` | left_in | stem_out | Enter from left, exit stem |
| `left_right` | left_in | right_out | Through-traffic left-to-right |

### Episode Flow

1. `reset()` — starts a new SUMO process via TraCI, queries actual junction position, offsets occlusion polygon coordinates by the runtime junction center, spawns ego and agents per the sampled `BehaviorConfig`, returns initial observation
2. `step(action)` — translates the discrete action to SUMO speed commands, advances one simulation step (dt=0.1s), computes reward, checks termination conditions
3. Episode terminates on: **collision**, **success** (ego reaches exit edge), **timeout** (500 steps), or **ABORT** (ego retreats out of conflict zone)

### Collision Detection

Collision is detected via two independent mechanisms (both must agree or either can trigger):
- SUMO's built-in collision event system via `traci.simulation.getCollisions()`
- Proximity check: ego-to-agent distance < `d_coll = 2.0m`

### Occlusion Computation

At each step, the environment computes two visibility fractions:
- `alpha_cz` — fraction of the conflict zone (junction center ± radius) visible to the ego, computed by ray-casting against the four building polygons
- `alpha_cross` — fraction of the cross-traffic approach region visible

Building polygon coordinates are computed at runtime by querying `traci.junction.getPosition('center')` inside `reset()` after SUMO starts. This is critical — static origin-centered coordinates would place buildings ~55m from the ego due to SUMO's netconvert offset.

### Pothole

Scenario `1d` and `4` place a random pothole on one of the ego's possible routes. The ego receives a penalty `w_pothole = -5.0` per step spent within 2m of the pothole center. The pothole position is randomized each episode.

---

## 5. Observation Space

The full observation vector has two variants:

**Base observation (135D):**
```
[0:6]    Ego state: x, y, speed, acceleration, yaw, lateral deviation
[6:18]   Geometric: d_stop, d_cz, d_exit, kappa, e_y, e_psi, w_lane, g_turn, rho, sigma_percep, n_occ, d_pothole
[18:24]  Visibility: alpha_cz, alpha_cross, d_occ, dt_seen, n_occ_agents, alpha_hist
[24:134] Agent features (5 agents × 22D each):
           [dx, dy, dvx, dvy, delta_psi, v_i, a_i, d_cz_i, d_exit_i,
            tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i, chi_i, pi_ROW_i,
            nu_i, sigma_i, type_car, type_ped, type_moto, mask]
[134]    d_pothole (scalar)
```

**With intent encoder (165D):**  
Base 135D + 30D intent block (5 agents × 6D: 3 intent probabilities + 3 style probabilities).

Agents are sorted at each step by distance to the conflict zone center, so the closest agent always occupies slots `[24:46]`. This ordering is consistent with the PDE state builder's agent indexing.

---

## 6. Action Space

Five discrete behavioral actions mapping to SUMO speed commands:

| Index | Name | Behavior | Nominal Acceleration |
|-------|------|----------|----------------------|
| 0 | STOP | Emergency brake | -5.0 m/s² |
| 1 | CREEP | Slow approach at 1 m/s | proportional to (1 - v) |
| 2 | YIELD | Gentle deceleration | -0.5 m/s² |
| 3 | GO | Accelerate through | +2.0 m/s² |
| 4 | ABORT | Hard brake and retreat | -8.0 m/s² |

Actions are translated to SUMO via `traci.vehicle.setSpeed()` at each step, with the target speed clamped to physical limits (0 to 13.89 m/s = 50 km/h).

---

## 7. Reward Function

The step reward is a weighted sum of shaped components:

```
r_t = w_prog * Δd_route      # route progress (dense shaping signal)
    + w_time * 1              # time penalty (-0.1 per step)
    + w_risk * r_ttc          # TTC-based risk (-3.0 when TTC < 3s)
    + w_pothole * 1_pothole   # pothole proximity (-5.0)
    + w_abort * 1_abort_action# comfort penalty for ABORT (-0.5)
    + w_rule * 1_yield_viol   # rule violation penalty (-2.0)
    + F(s, a, s')             # potential-based shaping (Ng et al. 1999)

Terminal bonuses:
    w_success = +200.0        # on reaching exit edge
    w_coll    = -20.0         # on collision
    w_switch  = -0.05         # per action change (smoothness)
```

**Potential-based shaping** uses `Φ(s) = -d_route(s)` (route distance to conflict zone exit, queried via `traci.simulation.getDistanceRoad`). The shaping term `F = γ_shaping * Φ(s') - Φ(s)` with `γ_shaping = 1.0` telescopes exactly over any episode length, eliminating drift bias. This forces the agent to drive through the intersection rather than hovering near the exit.

The `w_success = +200` bonus was set empirically during Phase 31 Stage 1B calibration. The original value of +10 was insufficient to provide gradient against accumulated per-step time costs in dense traffic scenarios.

---

## 8. Models

### 8.1 DRPPO Baseline

**Files:** `models/drppo.py`

**RecurrentActorCritic** — single GRU-based network with shared encoder:

```
Input obs (135D or 165D)
    └── GRU (hidden=256, 1 layer)
         ├── Linear → logits (5D) → Categorical distribution → action
         └── Linear → scalar value V(s)
```

**DRPPO** class wraps `RecurrentActorCritic` with standard PPO training:
- Rollout collection with GRU hidden state carryover across steps
- GAE advantage estimation (λ=0.95, γ=0.99)
- Clipped surrogate objective (ε=0.2)
- 8 epochs of minibatch updates per rollout (batch_size=128)
- Gradient clipping at 0.5

When used alone (without any auxiliary critic), this is the DRPPO baseline. All PDE-augmented agents inherit the same backbone.

### 8.2 Shared PDE Architecture

Every PDE-augmented agent (`HJBAuxAgent`, `SoftHJBAuxAgent`, `EikonalAuxAgent`, `CBFAuxAgent`, `FusionAuxAgent`) shares:

- Identical `RecurrentActorCritic` backbone for the policy
- An **auxiliary critic MLP** `U_φ: R^79 → R` that maps the reduced PDE state to a scalar
- The auxiliary critic architecture is fixed: `Linear(79, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1)`
- A `BehavioralDynamics` object for computing `F_a(xi)` across all 5 actions
- Autograd for computing `∇_xi U_φ(xi)` — the gradient is computed via `torch.autograd.grad` at every training step
- A collocation sampler for augmenting real rollout states with jittered copies

The only thing that differs between methods is the **PDE residual function** plugged in at loss computation.

The composite training loss for all PDE methods:

```
L_policy  = PPO clipped surrogate (actor)
L_value   = MSE(V_ψ(s_t), GAE_return_t)  (PPO critic)
L_distill = MSE(V_ψ(s_t), U_φ(xi_t).detach())  (critic←aux)
L_residual = E_xi[ρ(xi)²]   (PDE consistency, method-specific)
L_anchor  = MSE(U_φ(xi_t), GAE_return_t)  (empirical grounding)
L_bc      = boundary condition losses at terminal states

L_total = L_policy + c_vf*L_value + λ_res*L_residual
        + λ_dist*L_distill + L_anchor + L_bc
```

### 8.3 Hard-HJB Auxiliary Critic

**Files:** `models/pde/hjb_aux_agent.py`, `models/pde/hjb_aux_critic.py`

The auxiliary critic is trained to satisfy the discrete-time Hamilton-Jacobi-Bellman equation:

```
ρ_HJB(xi) = U(xi)·ln(γ) + max_a [ r(xi, a) + γ · ∇U(xi)ᵀ · (F_a(xi) - xi) ]
```

This is a discrete-time Taylor expansion of the Bellman optimality equation. For small discount rates, it is equivalent up to O((1-γ)²) error to the continuous-time HJB. The max over discrete actions replaces the continuous supremum, making this formulation valid for the finite action set {STOP, CREEP, YIELD, GO, ABORT}.

**Key parameters:**
- `lambda_residual = 0.2` — weight of `L_residual`
- `lambda_distill = 0.25` — weight of `L_distill`
- `collocation_size = 256` — number of collocation points per update
- `aux_lr = 1e-3` — auxiliary critic learning rate (separate from PPO lr)

**Boundary conditions:** `U(xi_success) = +200`, `U(xi_collision) = -20` (matching env terminal rewards).

### 8.4 Soft-HJB Auxiliary Critic

**Files:** `models/pde/soft_hjb_aux_agent.py`, `models/pde/soft_hjb_aux_critic.py`

Uses logsumexp instead of hard max, implementing the entropy-regularized (exploratory) HJB framework of Wang, Zariphopoulou & Zhou (JMLR 2020):

```
ρ_SoftHJB(xi) = U(xi)·ln(γ) + τ · logsumexp( [r(xi,a) + γ·∇U·(F_a-xi)] / τ )
```

The temperature parameter `τ=1.0` controls the softness. When `τ → 0`, this recovers the Hard-HJB residual.

**Additional term unique to Soft-HJB:** an actor-alignment KL loss that nudges the PPO actor toward the soft policy induced by the PDE Q-values:

```
L_actor_kl = λ_kl · KL( π_θ(·|s) || π_soft(·|xi) )
```

where `π_soft(a|xi) ∝ exp(q_a(xi) / τ)` and `q_a(xi) = r(xi,a) + γ·∇U·(F_a - xi)`.

**Key additional parameters:** `lambda_actor_kl = 0.1`, `tau_soft = 1.0`.

### 8.5 Eikonal Auxiliary Critic

**Files:** `models/pde/eikonal_aux_agent.py`, `models/pde/eikonal_aux_critic.py`

The auxiliary critic is trained to satisfy an Eikonal-like PDE gradient-norm condition:

```
ρ_Eikonal(xi) = ‖∇_xi U(xi)‖² − c(xi)²
```

where `c(xi)` is the **dynamics-derived maximum safe speed** at state `xi`, defined as:

```
c(xi) = max_a [ ‖F_a(xi) - xi‖ / dt ]   (maximum displacement per timestep across all actions)
```

This is inspired by the Eikonal equation from fast marching methods (Sethian 1999) and the Eik-HIQL work of Giammarino & Qureshi (2025), adapted to discrete actions and non-isotropic intersection dynamics. Rather than enforcing a shortest-path structure (which requires goal-conditioning and isotropic dynamics), this variant enforces that the value function gradient magnitude matches the physical reachability of the state — essentially imposing a "no faster than physically possible" constraint on value function gradients.

**Training with Augmented Lagrangian:** the Eikonal residual is minimized subject to the boundary condition `U(xi_terminal) = U_target` using an Augmented Lagrangian Method (ALM), combining exact penalty with Lagrange multiplier updates. This avoids the gradient conflict between residual minimization and boundary enforcement that naive L2 losses suffer from.

**Loss weighting:** `L_eikonal = L_residual_alm + L_distill + L_anchor`. Collision states are **excluded** from the anchor loss to prevent the safety-oriented critic from being pulled toward collision-region values.

**Key parameters:** `w_fail = +50.0` (boundary condition weight for failure states), `lambda_residual = 0.2`.

### 8.6 CBF Auxiliary Critic

**Files:** `models/pde/cbf_aux_agent.py`, `models/pde/cbf_aux_critic.py`

Inspired by Control Barrier Function theory (Ames et al., ECC 2019). The auxiliary critic is trained to satisfy a barrier-descent condition:

```
ρ_CBF(xi) = ReLU( −max_a [ ∇U(xi)ᵀ·(F_a(xi) - xi) + α·U(xi) ] )
```

This penalizes states where no action produces a positive CBF derivative condition `ḣ + α·h ≥ 0`, meaning the system is heading toward unsafe territory with no escape.

**Critical design note — barrier offset:** A standard CBF uses `h(xi) ≥ 0` for safety, but `U(xi)` is a value function that can be positive or negative. This creates a semantic tension: CBF requires `h > 0` in safe regions, but value functions are negative near bad states. This is resolved by defining:

```
h(xi) = U(xi) + barrier_offset     (barrier_offset = 10.0)
```

The offset ensures `h(xi) > 0` in safe regions (where `U > -10`) while preserving the correct CBF gradient semantics.

**Barrier function components:** the CBF h is defined as the smooth minimum of three sub-barriers:
- `h_stop(xi)` — stopping distance barrier (can the ego stop before the conflict zone?)
- `h_friction(xi)` — friction-limited stopping barrier
- `h_ttc(xi)` — time-to-collision barrier (TTC-based safety margin)

**Key parameters:** `alpha_cbf = 1.0` (decay rate), `barrier_offset = 10.0`.

### 8.7 Fusion Auxiliary Critic

**Files:** `models/pde/fusion_aux_agent.py` (506 lines)

Combines one optimality-oriented critic and one safety-oriented critic through a convex combination of their distillation targets:

```
U_distill = w_o · U_optimality(xi) + w_s · U_safety(xi)
```

The fused target `U_distill` is used to supervise `V_PPO` via `L_distill = MSE(V_PPO, U_distill.detach())`. Both critics are trained with **stop-gradient** applied before the combination — each receives its own residual loss independently.

This means the Fusion agent actually trains **two separate auxiliary critics** simultaneously:
- `U_optimality` trained with Hard-HJB residual
- `U_safety` trained with CBF residual

The weight pair `(w_optimality, w_safety)` is a key hyperparameter explored in Tier 2c ablation across 8 combinations from `(1.0, 0.0)` (pure optimality) to `(0.0, 1.0)` (pure safety) to `(3.0, 1.0)` (optimality-dominant).

Fusion also includes the Soft-HJB actor-alignment KL term `L_actor_kl` from its optimality branch.

### 8.8 Rule-Based Baseline

**File:** `models/rule_based_policy.py`  
**Class:** `RuleBasedTTCPolicy`

A deterministic, heuristic policy that requires no training:

```python
if any agent's TTC < ttc_threshold (3.0s):
    action = STOP
elif d_cz > far_zone_dist (5.0m):
    action = GO  # approach junction
else:
    action = GO  # proceed through
```

This policy has no GRU hidden state (it is stateless). It reads TTC directly from the observation vector (index `IDX_TTC_MIN`). The API is intentionally identical to the learned agents (`get_action`, `reset_hidden`) so evaluation code treats all methods uniformly.

This reference performs near-perfectly on simple scenarios (SR~1.0 on 1a) but fails on partially occluded scenarios (SR~0.25 on 1b) because it cannot reason about hidden agents. On dense traffic scenarios it dominates all learned methods under the current training budget due to the hard TTC threshold.

### 8.9 Intent and Style Encoder

**File:** `models/intent_style.py`  
**Class:** `IntentStylePredictor`

A per-agent bidirectional LSTM that takes a sequence of agent kinematic features and outputs:
- **3 intent probabilities:** yield/stop, proceed, turn/merge (for vehicles); cross, wait/slow (for VRUs)
- **3 style probabilities:** cautious, normal, chaotic

**Architecture (v9):**
```
Input: (batch, seq_len, input_dim=12)
  └── Conv1d(12, 32) → GELU → Conv1d(32, 32) → GELU   (1D CNN frontend)
  └── Bidirectional LSTM(hidden=384, layers=3, dropout=0.2)
  └── Concat [forward, backward] → (batch, 768)
       ├── Linear(768, 3) → Softmax → intent probs
       └── Linear(768, 3) → Softmax → style probs
```

An **ensemble of 3 members** is used at inference. The 5-agent intent block produces a 30D feature vector (5 agents × 6D) appended to the base 135D observation, giving the 165D "with intent" observation.

The encoder is pre-trained separately (`experiments/train_intent.py`) and its weights are frozen during PPO training. Checkpoint paths are specified in `config_frozen_v1.yaml :: intent_encoder`.

---

## 9. PDE Infrastructure

### 9.1 Reduced PDE State (xi)

**File:** `models/pde/state_builder.py`  
**Class:** `ReducedPDEState`  
**Dimensionality:** `XI_DIM = 79`

The reduced state is extracted from the full observation and env info dict at every training step:

```
xi[0]       v           ego speed (m/s)
xi[1]       a           ego acceleration (m/s²)
xi[2]       psi_dot     ego yaw rate (rad/s)
xi[3]       d_stop      distance to stop line (m)
xi[4]       d_cz        distance to conflict zone entry (m)
xi[5]       d_exit      distance to conflict zone exit (m)
xi[6]       kappa       local path curvature (1/m)
xi[7]       TTC_min     minimum TTC across all agents (s)
xi[8]       alpha_cz    visible fraction of conflict zone [0,1]
xi[9]       alpha_cross visible fraction of cross-traffic region [0,1]
xi[10]      d_occ       distance to nearest occlusion boundary (m)
xi[11]      dt_seen     time since cross-traffic last fully observed (s)
xi[12:34]   Agent 1 features (22D):
              dx, dy, dvx, dvy, delta_psi, v_i, a_i,
              d_cz_i, d_exit_i, tau_i, delta_tau_i,
              t_cpa, d_cpa, TTC_i, chi_i, pi_ROW_i,
              nu_i, sigma_i, type_car, type_ped, type_moto, mask
xi[34:56]   Agent 2 features (22D): same layout
xi[56:78]   Agent 3 features (22D): same layout
xi[78]      d_pothole   distance to pothole (m)
```

Only the 3 closest agents (by distance to conflict zone) are included in `xi`. Agents 4 and 5 from the full observation are excluded from the PDE state to limit dimensionality.

The `mask` bit (last of each agent's 22D block) is 0 if no agent occupies that slot, allowing the dynamics and residual functions to zero out contributions from absent agents.

### 9.2 Behavioral Dynamics

**File:** `models/pde/dynamics.py`  
**Class:** `BehavioralDynamics`

Implements differentiable one-step dynamics `F_a(xi)` for each of the 5 actions. The dynamics propagate:

- **Ego kinematics** (v, a, d_stop, d_cz, d_exit) using a simple Euler integration of the kinematic bicycle model with action-dependent accelerations
- **Agent relative positions** (dx, dy, dvx, dvy) updated by constant-velocity assumption
- **Conflict metrics** (tau_i, delta_tau_i, t_cpa, d_cpa, TTC_i) recomputed from updated relative positions

**`_smooth_clamp_nonneg`** is used at 6 different sites throughout the dynamics to prevent zero-gradient issues at clamp boundaries. Instead of `max(x, 0)`, it uses:

```python
smooth_clamp(x) = 0.5 * (x + sqrt(x² + ε²)) - ε/2
```

This has a non-zero gradient everywhere, which is essential for autograd-based PDE residual computation.

Nominal accelerations per action:
- STOP: `-a_brake = -5.0 m/s²`
- CREEP: `clamp(v_creep - v, -0.5, 0.5)` where `v_creep = 1.0 m/s`
- YIELD: `-0.5 m/s²`
- GO: `+a_go = +2.0 m/s²`
- ABORT: `-a_abort = -8.0 m/s²`

### 9.3 Collocation Sampling

**File:** `models/pde/collocation.py`

PDE residuals are evaluated on **collocation points** — a mixture of real rollout states and physics-consistent jittered copies. This improves coverage of the state space beyond what was visited during rollout.

**Jitter strategy:** Only **primitive** features (speed, distances, angles) are jittered. **Derived** features (TTC, t_cpa, d_cpa, tau) are recomputed from the jittered primitives to maintain physical consistency. Jittering derived features directly would produce physically impossible states.

Jitter standard deviations are calibrated per feature:
- Ego speed: σ=0.3 m/s
- Distance features (d_stop, d_cz, d_exit): σ=1.0 m
- Agent relative positions: σ=0.5 m
- Visibility fractions: σ=0.05

Physical bounds are enforced after jitter (e.g., speed clamped to [0, 13.89], distances to [0, 100]).

### 9.4 Local Reward Surrogate

**File:** `models/pde/local_reward.py`

The PDE residual involves `r(xi, a)` — the one-step reward as a function of the reduced PDE state and action. Since the real reward comes from SUMO (which the auxiliary critic cannot query at collocation points), a surrogate is implemented.

The surrogate mirrors the env reward structure:
- Progress term: `w_prog * (d_exit_old - d_exit_new)` using the dynamics-computed d_exit change
- Time penalty: `w_time`
- TTC-based risk: `w_risk * max(0, 1 - TTC_min/ttc_thr)`
- Pothole penalty: `w_pothole * (d_pothole < 2.0)`
- Abort comfort: `w_abort * (action == ABORT)`
- Potential-based shaping: `γ_shaping * Φ(xi_next) - Φ(xi)` with `Φ = -d_exit`

**Terminal rewards are excluded** from the surrogate — they are handled separately as boundary conditions in each PDE method's loss.

### 9.5 Residual Computations

**File:** `models/pde/residuals.py`

All four PDE residuals are implemented here. The key shared primitive is `_compute_grad_U`:

```python
def _compute_grad_U(U_net, xi):
    xi_req = xi.detach().requires_grad_(True)
    U_val = U_net(xi_req)
    grad_U = torch.autograd.grad(
        U_val.sum(), xi_req, create_graph=True, retain_graph=True
    )[0]
    return U_val, grad_U
```

`create_graph=True` keeps the autograd graph alive so that higher-order gradients can flow through `grad_U` during backprop on `L_residual`.

**`pde_q_values`** computes the PDE-based Q-value for each action:
```
q_a(xi) = r(xi, a) + γ · ∇U(xi)ᵀ · (F_a(xi) - xi)
```

Note: this uses the raw discrete-time increment `(F_a - xi)` without division by `dt`. This is correct for discrete-time Bellman. The CBF residual, being continuous-time, divides by `dt` internally.

### 9.6 Checkpointing

**File:** `models/pde/checkpointing.py`

Each PDE checkpoint stores:
- `policy_state_dict` — all PPO policy network weights
- `policy_optim_state` — Adam optimizer state for policy
- `aux_state_dict` — auxiliary critic MLP weights
- `aux_optim_state` — auxiliary critic optimizer state
- `obs_dim`, `method`, `config` — reconstructable architecture spec
- `arch` dict — every constructor argument affecting tensor shapes

`Agent.from_checkpoint(path)` can fully reconstruct an agent without the caller knowing the architecture, using the saved `arch` dict. This is critical for evaluation where different runs may have been trained with different hidden dimensions.

---

## 10. Scenario Generation

**File:** `scenario/generator.py`

Generates SUMO network files (`.net.xml`, `.nod.xml`, `.edg.xml`, route files) programmatically for all 7 scenario types. The T-intersection geometry is fixed:
- Stem arm: 60m length, 2 lanes
- Left/right arms: 50m length, 2 lanes each direction
- Junction center at coordinate origin (SUMO shifts this at runtime)

The generator creates:
- All 6 ego route definitions (stem_right, stem_left, right_stem, right_left, left_right, left_stem)
- Other agent route definitions (straight left-right, straight right-left, turns into stem)
- Pedestrian and motorcyclist routes where applicable
- Conflict-guaranteed spawning parameters (agents timed to arrive at the junction simultaneously with the ego)
- Sidewalk definitions for scenarios with pedestrians (1b, 2, 3, 4 and their dense variants)
- Corner building polygons (not in the network file — added as SUMO polygons via TraCI at runtime)

Pre-generated files for scenario `1a` are stored in `scenarios/sumo_1a/`. Other scenarios are generated fresh on first run.

---

## 11. Behavior Sampling

**File:** `scenario/behavior_sampler.py`  
**Class:** `BehaviorSampler` returning `BehaviorConfig`

Every episode, `BehaviorSampler.sample()` returns a configuration specifying:

**For each other agent:**
- Random maneuver selection from the agent-type-specific maneuver list
- Random style selection from the style list
- Departure position (randomized within lane)
- Departure speed (style-dependent, e.g., aggressive cars depart at higher speed)
- Departure time (jittered to ensure conflict with the ego at the junction)
- TraCI-level parameters: `sigma` (driver imperfection), `tau` (reaction time), `accel`, `decel`, `jmDriveAfterYellowTime` (right-of-way aggressiveness)

**Style distributions:**

Car styles (7): `nominal`, `aggressive`, `timid`, `distracted`, `erratic`, `drunk`, `rule_violating`

Pedestrian styles (7): `normal_walk`, `running`, `slow_elderly`, `stop_midway`, `hesitant`, `distracted_slow`, `jaywalking_fast`

Motorcyclist styles (6): `nominal`, `aggressive_fast`, `cautious`, `late_brake`, `swerving`, `yield_to_ego`

The `PED_SYNTHETIC_SIGMA` constant controls per-step position noise for pedestrians, simulating the inherent unpredictability of pedestrian motion.

For ablation studies, a `style_filter` can restrict sampling to `"nominal"` styles only (Tier 3 behavioral robustness baseline) with evaluation on `"adversarial"` styles (Tier 4 held-out).

---

## 12. Experiment Pipeline

### 12.1 Training Scripts

Each method has a dedicated training script in `experiments/pde/`:

```
train_drppo_baseline.py   # DRPPO (no auxiliary critic)
train_hjb_aux.py          # Hard-HJB
train_soft_hjb_aux.py     # Soft-HJB
train_eikonal_aux.py      # Eikonal
train_cbf_aux.py          # CBF
train_fusion_aux.py       # Fusion
```

All training scripts share the same CLI interface:

```bash
python experiments/pde/train_hjb_aux.py \
    --scenario 1b \
    --ego_maneuver stem_right \
    --seed 42 \
    --total_steps 400000 \
    --out_dir results/ablation/my_run \
    --intent_on            # add intent features (optional)
    --no_buildings         # disable occlusion (optional)
```

Each training script:
1. Instantiates the agent with config from `config_frozen_v1.yaml`
2. Creates a `SumoEnv` with the specified scenario and maneuver
3. Runs the PPO rollout-update loop for `total_steps` steps
4. Writes `metrics.csv` (per-iteration training metrics) and `meta.json` (run provenance) to `out_dir`
5. Evaluates every `eval_every_n_iter=10` iterations and writes `eval_*.csv`
6. Saves checkpoints every `save_every_n_iter=50` iterations

**Performance note:** `torch.set_num_threads(2)` is set globally to avoid PyTorch thread contention when running many parallel SUMO workers.

### 12.2 Calibration

**File:** `experiments/pde/run_calibration.py`

The **36-run calibration study** determines the training budget `total_steps` for all subsequent ablation tiers. It runs:

```
6 methods × 2 scenarios (1a, 2_dense) × 3 seeds = 36 jobs
```

All 36 jobs at 500,000 steps each, with 15 running in parallel.

**Convergence criterion (v2):** a run is considered converged when the success rate in the last 50,000-step window achieves `SR_mean ≥ threshold` and `SR_std_rel ≤ 0.05`. The v2 criterion uses success rate as the primary signal (v1 used reward stability, which was unreliable in dense traffic).

**Outcome of the calibration:**
- All 6 methods converge on `(1a, stem_right)` within ~100,000 steps
- No method converges on `(2_dense, stem_right)` within 500,000 steps  
- Hard-HJB reaches SR=1.0 on one seed at step 331,776 on the dense cell
- Calibrated budget: **400,000 steps** (from the Hard-HJB convergence signal, rounded up with 1.1× safety buffer)

This budget is locked in `config_frozen_v1.yaml :: calibration :: total_steps_calibrated = 400000`.

### 12.3 Ablation Tiers

**File:** `experiments/pde/run_full_ablation.py`

The full ablation is organized in four tiers sourced directly from `config_frozen_v1.yaml`:

**Tier 1 — Main Comparison (1,680 jobs)**
```
12 scenario-maneuver combos × 7 methods × 10 seeds × 2 intent variants
```
Combos include all difficulty levels: easy (1a), occluded pedestrian (1b), dense traffic (2_dense), multi-class (3, 4), and various maneuver types.

**Tier 2 — Sensitivity Studies (1,160 jobs)**
- Sub-grid 2a: λ sweep (6 values: 0.01–1.0) × 5 methods × 2 scenarios × 2 maneuvers × 5 seeds = 600 jobs
- Sub-grid 2b: Occlusion on/off × 5 methods × 4 scenarios × 2 maneuvers × 5 seeds = 400 jobs
- Sub-grid 2c: 8 fusion weight pairs × 1 method × 2 scenarios × 2 maneuvers × 5 seeds = 160 jobs

**Tier 3 — Ablation Studies (275 jobs)**
- State ablation: train without visibility features (xi[8:12] zeroed)
- Behavioral robustness: train on nominal styles, evaluate on adversarial styles
- Dense scenario stress: 3 dense scenarios × 5 methods × 5 seeds

**Tier 4 — Held-Out Evaluation (eval only)**  
Applies Tier 1/Tier 3 checkpoints to 5 held-out configurations never seen during training:
- HO1: occlusion-trained → no-occlusion eval
- HO2: no-occlusion-trained → occlusion eval
- HO3: full-trained → adversarial style eval
- HO4: nominal-trained → adversarial eval
- HO5: full-state-trained → no-visibility-features eval

**Running tiers:**
```bash
python experiments/pde/run_full_ablation.py --tier 1 --max_parallel 22
python experiments/pde/run_full_ablation.py --tier 2 --max_parallel 16
python experiments/pde/run_full_ablation.py --tier all --dry_run  # preview job count
```

The orchestrator manages parallel subprocess pool, writing to `results/ablation/<run_id>/`.

### 12.4 Evaluation

**File:** `experiments/pde/eval.py`

Evaluates a trained checkpoint across N episodes and writes per-episode metrics:

```bash
python experiments/pde/eval.py \
    --checkpoint results/ablation/my_run/checkpoint_final.pt \
    --scenario 1b \
    --ego_maneuver stem_right \
    --n_episodes 100 \
    --out_dir results/ablation/my_run/eval
```

Evaluation metrics per episode:
- Total return, episode length, terminal state (success/collision/timeout/abort)
- `min_ttc`, `mean_ttc` — time-to-collision statistics
- `min_distance_to_collision` — closest approach to any agent
- `ego_max_speed` — peak speed during episode
- `n_action_changes` — smoothness indicator

At end of evaluation, an `eval_summary.json` is written with aggregated statistics (mean, std, collision rate, success rate across all episodes).

The evaluation uses `deterministic=True` mode (argmax action selection) for reportable metrics, and `stochastic=True` for behavioral diversity assessment.

---

## 13. Analysis Pipeline

**Directory:** `analysis/`

A full offline analysis suite that operates on `results/ablation/` and produces publication-ready figures and tables.

### Entry Point

```bash
python analysis/run_analysis.py \
    --results_dir results/ablation \
    --output_dir results/analysis \
    --tier 1
```

### Modules

**`analysis/config.py`** — single source of truth for:
- Method colors: DRPPO (gray), HJB (blue), Soft-HJB (orange), Eikonal (green), CBF (red), Fusion (purple), Rule-based (black)
- Method display labels and ordering
- Matplotlib RC settings (serif fonts, 300 DPI, tight bounding boxes)
- Statistical constants pulled from `config_frozen_v1.yaml`

**`analysis/loader.py`** — loads `metrics.csv` and `eval_*.csv` files from all runs. Applies quality gate checks (expected column presence, minimum iteration count, NaN detection). Reports failure rate; halts if > 5% of runs fail quality checks.

**`analysis/metrics.py`** — computes per-run outcome metrics from the final 10% of training iterations (the `FINAL_WINDOW_FRAC` window):
- `final_collision_rate` — collision fraction in last window
- `final_success_rate` — success fraction in last window
- `final_mean_reward` — mean undiscounted return in last window
- `min_ttc_eval`, `mean_return_eval` — from eval episodes

**`analysis/stats.py`** — statistical tests:
- **Welch's t-test** (two-sided, unequal variances) — pairwise comparison of each PDE method vs. DRPPO per metric per combo
- **Holm-Bonferroni step-down correction** — applied within each (family × metric × tier) group to control FWER. Two families: A (PDE-vs-DRPPO), B (pairwise PDE-vs-PDE)
- **Cohen's d** — effect size estimate, thresholded at negligible/small/medium/large (0.2/0.5/0.8)
- **Bootstrap CIs** — 1000 resamples, 95% CI, RNG seed=12345 for determinism

**`analysis/plots.py`** — seven plot families, each rendered as both PDF (paper-ready) and HTML (interactive):
1. Learning curves per scenario-maneuver-method (smoothed reward vs. training steps, seed overlay shading)
2. PDE residual convergence (|ρ| vs. training steps, log scale)
3. Distillation gap (`E[(V_PPO - U_aux)²]` vs. training steps)
4. Success rate bar charts (per-method, per-cell, error bars = ±1 SD)
5. Collision rate comparisons
6. Action distribution pie/bar charts (STOP/CREEP/YIELD/GO/ABORT fractions)
7. TTC distribution plots (min TTC, mean TTC, CDF)

**`analysis/tables.py`** — generates LaTeX tables for the paper:
- `tier1_main_comparison.tex` — full per-cell results table
- `tier2a_lambda_sensitivity.tex` — lambda sweep summary
- `tier2b_occlusion_impact.tex` — occlusion ablation
- `tier2c_fusion_weights.tex` — fusion weight sweep
- `tier4_holdout.tex` — held-out generalization results
- `computational_overhead.tex` — training time per method

**`analysis/calibration_analysis.py`** — specific analysis of the 36-run calibration study, producing the calibration curves figure showing convergence on easy vs. dense cells.

---

## 14. Configuration System

### `config_frozen_v1.yaml`

The canonical configuration file. **All result-affecting hyperparameters are locked here.** Any change requires:
1. Renaming to `config_frozen_v2.yaml`
2. Re-running the config lock generation
3. Re-running calibration

Key sections:
- `ppo:` — all PPO hyperparameters (lr, gamma, gae_lambda, clip_eps, ent_coef, vf_coef, n_epochs_per_update, batch_size, n_steps)
- `architecture:` — network dimensions (GRU hidden=256, policy hidden=256, aux critic hidden=256, xi_dim=79)
- `methods:` — per-method defaults (lambda_residual, lambda_distill, collocation_size, aux_lr, and method-specific params)
- `tier1/tier2/tier3/tier4:` — complete ablation grid definitions
- `analysis:` — statistical test parameters
- `intent_encoder:` — v9 ensemble paths and architecture
- `calibration:` — locked `total_steps_calibrated = 400000` and `criterion_version`
- `env:` — observation dimensionalities, episode length, dt

### `config_loader.py`

Provides:
- `get_config()` — loads and caches `config_frozen_v1.yaml`
- `get_tier_config(tier)` — returns a specific tier's configuration dict
- `check_config_lock()` — verifies the YAML hash against `config_lock.json` to detect unauthorized modifications

### `configs/` directory

Per-subsystem YAML overrides used by some legacy scripts. In the current codebase, CLI defaults and `config_frozen_v1.yaml` take precedence. These files are kept for backward compatibility.

---

## 15. Verification Suite

**Directory:** `verification/`

A comprehensive set of preflight and validation scripts that must all pass before launching any training tier. The pre-Tier-1 micro-launch check (14/14 checks) verified:

| Script | What it checks |
|--------|----------------|
| `preflight_5_occlusion.py` | Occlusion polygons correctly offset to runtime junction position |
| `preflight_6_buildings.py` | Corner buildings appear in SUMO at correct world coordinates |
| `preflight_7_pothole.py` | Pothole spawns within ego route bounds |
| `preflight_9_dense.py` | Dense scenario spawns correct number of agents |
| `preflight_10_style.py` | All 7 car styles / 7 ped styles / 6 moto styles sample without error |
| `preflight_11_state_ablation.py` | State ablation mode zeroes correct xi features |
| `preflight_12_obsdim.py` | Observation dimension invariant across all 7 scenarios |
| `test_residuals_math.py` | PDE residuals produce correct values and non-zero gradients |
| `test_smooth_clamp.py` | `_smooth_clamp_nonneg` has gradient > 0 everywhere |
| `run_determinism.py` | Two runs with same seed produce identical trajectories |
| `run_ckpt_round_trip.py` | Save → load → inference produces identical outputs |
| `validate_smoke_outputs.py` | Smoke test outputs have correct CSV schema |

**Phase verification artifacts** (`phase3F_*`, `phase4_*`, `phase5_*`) are JSON files recording the pass/fail status of each preflight step, used to document exactly which version of the codebase was verified before each training launch.

---

## 16. Results Storage

All training outputs land in `results/`:

```
results/
├── calibration/
│   └── CAL_<method>_<scenario>_s<seed>/
│       ├── metrics.csv         # per-iteration training metrics
│       ├── meta.json           # run provenance and outcome
│       └── checkpoints/
│           └── checkpoint_final.pt
│
├── calibration_analysis/
│   ├── convergence_per_run.csv
│   ├── convergence_per_method.csv
│   └── calibrated_total_steps.json
│
├── ablation/
│   └── <run_id>/               # e.g., 1a_stem_right_hjb_aux_intent_s42
│       ├── metrics.csv
│       ├── meta.json
│       ├── eval_iter_010.csv   # eval at iteration 10
│       ├── eval_iter_020.csv
│       ├── ...
│       ├── eval_final.csv
│       └── checkpoints/
│           ├── checkpoint_050.pt
│           └── checkpoint_final.pt
│
├── analysis/
│   ├── figures/
│   │   ├── learning_curves/    # PDF + HTML
│   │   ├── pde_convergence/
│   │   ├── success_rate/
│   │   ├── action_distribution/
│   │   └── failures/           # failure trajectory plots
│   ├── tables/
│   │   ├── tier1_main_comparison.csv
│   │   ├── tier1_main_comparison.tex
│   │   ├── tier1_statistical_summary.csv
│   │   └── ...
│   └── stats/
│       ├── eval_summary_all.csv    # aggregated per-run eval outcomes
│       ├── aulc_summary.csv        # area-under-learning-curve
│       └── overhead_summary.csv    # training time per method
│
└── intent_model_v9_member{0,1,2}.pt  # pre-trained intent encoder ensemble
```

**`meta.json` schema** (per run):
```json
{
  "run_id": "1a_stem_right_hjb_aux_intent_s42",
  "start_time_iso": "...",
  "end_time_iso": "...",
  "wall_time_seconds": 7230,
  "method": "hjb_aux",
  "scenario": "1a",
  "ego_maneuver": "stem_right",
  "seed": 42,
  "intent_on": true,
  "total_steps_target": 400000,
  "total_steps_actual": 401408,
  "convergence_reason": "steps_exhausted",
  "git_commit": "4c5ead7",
  "device": "cuda:0",
  "config": { ... all hyperparameters ... },
  "result_summary": {
    "final_success_rate": 0.984,
    "final_collision_rate": 0.012,
    "final_mean_reward": 421.8
  }
}
```

**`metrics.csv` key columns** (per training iteration):
`iteration`, `total_steps`, `wall_time_seconds`, `iter_time_seconds`, `L_total`, `L_policy`, `L_value`, `L_entropy`, `L_residual_optimality`, `L_residual_safety`, `L_distill`, `mean_reward`, `n_collisions`, `n_successes`, `action_dist_{stop,creep,yield,go,abort}`

---

## 17. Hardware and Performance

**Target workstation:** Intel i9-14900 (24c/32t), 64GB DDR5, RTX 4000 Ada 20GB VRAM, Ubuntu 24.04

**Parallelism model:** Each training job spawns one SUMO instance via subprocess (TraCI) plus one PyTorch process. Multiple jobs run in parallel as separate OS processes — there is no shared memory between jobs. The GPU is shared implicitly by all concurrent PyTorch processes.

**Recommended settings:**
- `torch.set_num_threads(2)` per worker (set globally in eval.py)
- `--max_parallel 22` for local workstation (leaves headroom for OS)
- GPU enabled with `batch_size=128`

**Throughput characteristics:**  
The workload is **CPU-bound via SUMO**, not GPU-bound. Each SUMO episode requires real-time (or faster) simulation of TraCI calls. The RTX 4000 Ada sits at ~98% GPU utilization when 22 workers are active, but the bottleneck is SUMO's single-threaded simulation.

**Per-job estimated runtimes** (400,000 steps):
- Scenario 1a (easy, few agents): ~2 hours
- Scenario 2_dense (dense, many agents): ~5–6 hours (2–3× more SUMO steps per training step)

**Cloud migration:** For full-scale ablation (1,680+ jobs), cloud CPU instances are recommended (Vast.ai or RunPod). The codebase includes a `cloud_migration_playbook.md` (in project knowledge) with 4-way and 10-way manifest splits verified for byte-identical determinism.

---

## 18. Installation and Setup

### Prerequisites

- Ubuntu 20.04+ (or compatible Linux)
- SUMO ≥ 1.18.0 (installed via `apt-get install sumo sumo-tools`)
- Python 3.10+
- CUDA 11.8+ (for GPU training)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
```
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.5
plotly>=5.18.0
pyyaml>=6.0
scipy          # for statistical tests
stable-baselines3>=2.0.0
```

### Environment Variables

```bash
export SUMO_HOME=/usr/share/sumo        # or wherever SUMO is installed
export PYTHONPATH=$PYTHONPATH:/path/to/EECE_499-main
```

### Generate SUMO Networks

```bash
python scenario/generator.py --all      # generate all 7 scenario networks
# or individual:
python scenario/generator.py --scenario 1b
```

Pre-generated files for scenario `1a` are included in `scenarios/sumo_1a/`.

---

## 19. Running Experiments

### Quick Smoke Test (5,000 steps, ~5 minutes)

```bash
python experiments/pde/smoke_test.py --method hjb_aux --scenario 1a
```

### Single Training Run

```bash
python experiments/pde/train_hjb_aux.py \
    --scenario 1b \
    --ego_maneuver stem_right \
    --seed 42 \
    --total_steps 400000 \
    --out_dir results/ablation/test_run \
    --intent_on
```

### Calibration (36 jobs, ~72 hours on workstation)

```bash
python experiments/pde/run_calibration.py \
    --total_steps 500000 \
    --parallel 15
```

### Tier 1 Ablation (1,680 jobs)

```bash
# Preview without launching:
python experiments/pde/run_full_ablation.py --tier 1 --dry_run

# Launch with 22 parallel workers:
python experiments/pde/run_full_ablation.py --tier 1 --max_parallel 22
```

### Evaluation of a Checkpoint

```bash
python experiments/pde/eval.py \
    --checkpoint results/ablation/my_run/checkpoint_final.pt \
    --scenario 1b \
    --ego_maneuver stem_right \
    --n_episodes 100
```

### Full Analysis Pipeline

```bash
python analysis/run_analysis.py \
    --results_dir results/ablation \
    --output_dir results/analysis \
    --tier 1
```

---

## 20. Key Design Decisions

A summary of important non-obvious decisions embedded in the codebase, each traceable to a specific investigation:

**`w_success = +200` (not +10)**  
Phase 31 Stage 1B. The original +10 success bonus was insufficient against accumulated per-step time costs in dense traffic. Without a large success signal, the policy learned to oscillate in the approach zone. Verified in `verification/phase31_investigation_2_dense.json`.

**`gamma_shaping = 1.0` potential-based shaping**  
Phase 31 Stage 1D. The Wiewiora (2003) finite-horizon form telescopes exactly, eliminating the `(1-γ)·T·mean_d` drift that the γ=0.99 version allowed the agent to exploit (achieving +209k mean reward with 0% success by hovering near the exit). Route distance must go through the junction — Euclidean distance allowed hovering near the exit. Verified by comparing Stage 1C vs. 1D learning curves.

**Occlusion polygons offset at runtime**  
SUMO's `netconvert` shifts the junction center to ~(55, 54) world coordinates. Static origin-centered polygons place buildings 55m from the ego. The fix queries `traci.junction.getPosition('center')` inside `reset()` after SUMO starts and offsets all polygon coordinates. Without this, `alpha_cz = 0.000` at all times. Verified in Phase 3 Step 5 preflight.

**`_smooth_clamp_nonneg` at all 6 dynamics sites**  
Hard `torch.clamp(x, min=0)` has zero gradient at the clamp boundary. Since autograd traces through the dynamics to compute the PDE residual, zero-gradient regions create silent training failures. The smooth approximation `0.5*(x + sqrt(x²+ε²)) - ε/2` maintains non-zero gradient everywhere, verified in `verification/test_smooth_clamp.py`.

**CBF barrier offset = 10.0**  
The value function `U(xi)` can be negative (near bad states), but CBF theory requires `h(xi) ≥ 0` in safe regions. Setting `h = U + 10` ensures the semantic consistency — states where `U > -10` (i.e., not near disaster) are treated as safe by the CBF.

**Eikonal collision excluded from anchor loss**  
Including collision-state returns in the Eikonal anchor loss caused gradient conflicts: the safety-oriented critic was being pushed toward the collision-return value (a large negative), undermining its ability to learn smooth gradient fields. Excluding collision anchors resolves this. Other PDE methods do not need this exclusion.

**`criterion_version = "v2_sr_primary_post_stage3"`**  
The calibration convergence criterion was updated twice. v1 used reward stability (unreliable in dense traffic). v2 uses success rate as the primary signal, with collision rate as secondary. The version string is locked in `config_frozen_v1.yaml` to ensure all subsequent calibration-dependent decisions use the same criterion.

**`torch.set_num_threads(2)` globally**  
Without this, each PyTorch process grabs all available CPU threads, causing severe contention when 22+ workers run simultaneously. With 2 threads per worker and 22 workers, PyTorch consumes 44 threads of the available 32 (with hyperthreading) — manageable without starvation.

---

*This README was generated from the codebase at commit `4c5ead7` (tag: `v2_phase_3F_step_12-tier_1_ready`). For questions about the mathematical formulations, see `docs/PDE_METHODS.md`. For environment details, see `docs/ENVIRONMENT.md` and `docs/STATE_SCHEMA.md`.*
