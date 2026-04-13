# Project State

## Implemented and Verified

- **RecurrentActorCritic**: GRU(obs_dim, 128) + Linear actor + Linear critic
- **DRPPO**: Clean recurrent PPO baseline (heuristic physics code removed)
- **StateBuilder**: Full 134D state from SUMO perception
- **ReducedPDEState**: 79D PDE state extraction from full state
- **BehavioralDynamics**: Differentiable one-step dynamics on 79D PDE state
- **HJBAuxAgent / HJBAuxCritic**: Hard-HJB auxiliary critic method
- **SoftHJBAuxAgent / SoftHJBAuxCritic**: Soft-HJB auxiliary critic method with KL alignment
- **Collocation sampler**: Mixed real/jittered PDE state sampling
- **Local reward surrogate**: Smooth sigmoid approximations for PDE-compatible rewards
- **PDE checkpointing**: Save/load for all PDE agent families
- **All 7 SUMO scenarios**: 1a, 1b, 1c, 1d, 2, 3, 4 (T-intersection variants)
- **BehaviorSampler**: Stochastic cross-traffic behavior generation
- **IntentStylePredictor**: LSTM-based intent/style prediction (optional 30D augmentation)

## Newly Implemented (This Refactoring)

- **EikonalAuxAgent / EikonalAuxCritic**: Eikonal travel-time residual method
- **CBFAuxAgent / CBFAuxCritic**: CBF-PDE safety regularization method
- **eikonal_residual**: `||grad U||^2 - c(xi)^2` with safety-modulated effective speed
- **cbf_residual**: `ReLU(-max_a[grad_U*(F_a-xi) + alpha*U])` barrier function regularizer
- **local_reward_from_next**: Optimized reward computation (avoids redundant dynamics calls)
- **Corrected HJB/Soft-HJB residuals**: Fixed time-scale error (delta_xi instead of drift), added gamma factor
- **Smooth reward indicators**: Sigmoid approximations in local_reward (non-zero gradients)
- **Frame mismatch fix**: CPA computation in env/sumo_env.py now uses ego-frame consistently

## Removed (This Refactoring)

- **models/physics.py**: Old Cartesian bicycle model (PhysicsPredictor)
- **Heuristic DRPPO physics**: Design A/B, safety filter, L_ego, physics violations
- **configs/residuals/default.yaml**: Replaced with deprecation stub
- **results/ablation_batch2/**: Old heuristic ablation results (insignificant differences)

## NOT Yet Done

- End-to-end training validation with new PDE methods
- Ablation study across all 4 PDE methods + baseline
- Hyperparameter tuning for Eikonal and CBF-PDE methods
- Evaluation on all 7 scenarios
- Paper writing and results analysis
- Training script updates for new methods (experiments/pde/)

## Known Mathematical Approximations

1. **Constant-velocity agents**: BehavioralDynamics propagates other agents with constant velocity (no acceleration prediction)
2. **Isotropic Eikonal**: The Eikonal residual assumes isotropic propagation; actual vehicle dynamics are anisotropic
3. **CBF-inspired (not strict CBF)**: Discrete actions and learned dynamics mean strict forward invariance is not guaranteed
4. **Markov assumption on xi**: The 79D PDE state is treated as Markov, though the full system has partial observability (mitigated by GRU in the policy)
5. **Taylor expansion**: HJB residuals use first-order Taylor expansion of V(F_a(xi)), which is approximate for large state changes
6. **Sigmoid reward approximation**: PDE local reward uses sigmoid instead of indicator, which slightly changes the reward semantics near thresholds
