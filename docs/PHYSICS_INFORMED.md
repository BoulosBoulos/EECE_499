# Physics-Informed Neural Network (PINN) Augmentation: Design A

This document explains how the PIRL (Physics-Informed Reinforcement Learning) framework augments DRPPO with a **physics-informed critic** (Design A). The augmentation penalizes the value function \( V(s) \) when physics constraints are violated, so that gradients flow through the critic and improve both value estimates and policy learning.

---

## 1. Overview

### 1.1 What is Design A?

**Design A** is a physics-informed critic loss that:

1. **Uses physics constraints** from the intersection domain (TTC, stopping distance, friction circle).
2. **Penalizes the critic output** \( V(s) \) when these constraints are violated.
3. **Ensures gradient flow** — because the loss depends on \( V(s) \), backpropagation updates the critic (and indirectly the policy via advantages).

### 1.2 Why the Critic (Not the Actor)?

| Aspect | Critic | Actor |
|--------|--------|-------|
| Output | Continuous scalar \( V(s) \) | Discrete action (0–4) |
| Gradient flow | Direct | Blocked by discrete sampling |
| Physics link | "Value should be low when physics violated" | "Action should be physically plausible" |

The critic is the natural place for physics-informed losses because its output is differentiable. The policy benefits indirectly: better value estimates → better advantages → better policy updates.

---

## 2. Physics Constraints Used

### 2.1 Time-to-Collision (TTC)

**Constraint:** When TTC is below a threshold (e.g. 3 s), the situation is dangerous.

**Violation:**
\[
\text{violation}_{\text{ttc}} = \text{ReLU}(\text{ttc}_{\text{thr}} - \text{ttc}_{\min})
\]

- When \( \text{ttc}_{\min} < \text{ttc}_{\text{thr}} \): violation > 0 (dangerous).
- When \( \text{ttc}_{\min} \geq \text{ttc}_{\text{thr}} \): violation = 0 (safe).

**Config:** `physics_ttc_thr` (default 3.0 s).

### 2.2 Stopping Distance

**Constraint:** Ego should not enter the conflict zone unless it can stop in time.

**Stopping distance** (kinematic model):
\[
d_{\text{stop}}(v) = v \cdot \tau + \frac{v^2}{2 a_{\max}}
\]

- \( \tau \): reaction time (s).
- \( a_{\max} \): max deceleration (m/s²).

**Violation:** Ego is too close when \( d_{\text{cz}} < d_{\text{stop}}(v) \):
\[
\text{violation}_{\text{stop}} = \text{ReLU}(-(d_{\text{cz}} - d_{\text{stop}}(v)))
\]

**Config:** `physics_tau` (0.5 s), `a_max` (5.0 m/s²).

### 2.3 Friction Circle

**Constraint:** Combined lateral and longitudinal acceleration must stay within the friction circle:
\[
a_{\text{lat}}^2 + a_{\text{lon}}^2 \leq (\mu g)^2
\]

- \( a_{\text{lat}} = v^2 |\kappa| \) (lateral accel from curvature).
- \( a_{\text{lon}} \): longitudinal acceleration from env.

**Violation:**
\[
\text{violation}_{\text{fric}} = \text{ReLU}(a_{\text{lat}}^2 + a_{\text{lon}}^2 - (\mu g)^2)
\]

**Config:** `mu` (0.8), `g` (9.81).

### 2.4 Ego Dynamics (L_ego, optional)

**Constraint:** The actual next state should match the bicycle model prediction given the current state and action.

**Violation (prediction error):**
\[
\text{err}_{\text{ego}} = \|(x_{t+1}, y_{t+1}, \psi_{t+1}, v_{t+1}) - \hat{x}_{t+1}(s_t, a_t)\|^2
\]

- \( \hat{x}_{t+1} \): one-step Euler prediction from bicycle kinematics.
- When SUMO dynamics differ from the simple model, this error is non-zero.

**Usage:** Set `use_l_ego=True` in DRPPO to enable. The critic is penalized when \( V(s_{t+1}) \cdot \text{err}_{\text{ego}} \) is high — i.e., when we land in a state that was poorly predicted, assign lower value.

**Config:** `lambda_physics_ego` (0.1). **Ablation:** Compare `pinn` (L_ego off) vs `pinn_ego` (L_ego on).

---

## 3. Physics-Informed Critic Loss

### 3.1 Formula

The total physics violation for a state is:
\[
\text{violation} = \text{violation}_{\text{ttc}} + \text{violation}_{\text{stop}} + \text{violation}_{\text{fric}}
\]

The physics-informed critic loss penalizes the critic for assigning high value when violation is high:
\[
\mathcal{L}_{\text{physics\_critic}} = \frac{1}{B} \sum_{i=1}^{B} V(s_i) \cdot \text{violation}_i
\]

- \( B \): batch size.
- \( V(s_i) \): critic output for state \( i \).
- \( \text{violation}_i \): physics violation for state \( i \).

**Gradient:** \( \frac{\partial \mathcal{L}}{\partial V} = \text{violation} \). When violation is high, we push \( V \) down. The critic learns to assign lower value to dangerous states.

### 3.2 L_ego: Ego Dynamics Prediction Error (Ablation)

**Optional:** When `use_l_ego=True`, an additional term penalizes \( V(s) \) when the ego dynamics prediction error is high. The bicycle model predicts \( (x', y', \psi', v') \) from \( (x, y, \psi, v) \) and action \( a \); the actual next state comes from SUMO. When the prediction error is large, the critic is penalized for assigning high value — encouraging conservative estimates when the world deviates from the simple model.

\[
\mathcal{L}_{\text{ego}} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} V(s_i) \cdot \|s_{i,\text{actual}} - s_{i,\text{pred}}\|^2
\]

- \( \mathcal{V} \): indices with valid consecutive ego states (transition \( t \to t+1 \)).
- Config: `lambda_physics_ego` (default 0.1), `use_l_ego` (default False for ablation).

### 3.3 Total Critic Loss

\[
\mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{physics}} \cdot \mathcal{L}_{\text{physics\_critic}} + \lambda_{\text{ego}} \cdot \mathcal{L}_{\text{ego}} \quad \text{(if use\_l\_ego)}
\]

- \( \mathcal{L}_{\text{MSE}} = \|V(s) - \text{returns}\|^2 \): standard TD/GAE target.
- \( \lambda_{\text{physics}} \): weight for physics term (config: `lambda_physics_critic`).

---

## 4. Implementation Details

### 4.1 Data Flow

1. **Rollout collection:** For each step, the env returns `info` with `ttc_min`, and the state builder provides `d_cz`, `v`, `kappa`, `a_lon` from `s_geom` and `s_ego`.
2. **Extra dict:** `collect_rollouts` builds `extra` with arrays: `ttc_min`, `d_cz`, `v`, `kappa`, `a_lon` (one per step).
3. **Minibatch:** During PPO updates, we slice `extra` to match the minibatch indices.
4. **Train step:** `DRPPO.train_step` computes violations from the batch, then \( \mathcal{L}_{\text{physics\_critic}} \), and adds it to the critic loss.

### 4.2 Code Location

- **DRPPO:** `models/drppo.py` — `train_step()` method, `_compute_physics_critic_loss()` helper.
- **Config:** `configs/residuals/default.yaml` — `lambda_physics_critic`, `physics_ttc_thr`, `physics_tau`, `a_max`, `mu`, `g`.
- **Rollout:** `experiments/run_train.py` — `collect_rollouts()` builds `extra`.

### 4.3 Enabling/Disabling

Set `use_pinn=True` (default) to enable Design A. Set `use_pinn=False` for standard PPO without physics augmentation.

---

## 5. Ablation

The ablation study compares **three variants** across all scenarios:

| Variant | use_pinn | use_l_ego | Description |
|---------|----------|-----------|-------------|
| `nopinn` | False | False | Standard PPO, no physics |
| `pinn` | True | False | Design A: TTC + stop + friction only |
| `pinn_ego` | True | True | Design A + L_ego (ego dynamics) |

```bash
make ablation       # trains and evals all 3 variants × 7 scenarios
```

Or manually:
```bash
python experiments/run_ablation.py --total_steps 50000 --eval_episodes 50
python experiments/run_ablation.py --skip_train   # eval only (existing checkpoints)
```

Results: `results/ablation/ablation_results.csv` and `.json`.

---

## 6. Design B (Future)

**Design B** (learnable physics model with ODE coefficients) is not currently implemented. It would:

- Learn ODE coefficients to better match SUMO dynamics.
- Use prediction error as an auxiliary signal.

Design A alone provides a working, gradient-flowing physics-informed augmentation. Design B can be added later if needed.

---

## 7. References

- Physics formulas: `models/physics.py`
- State schema: `docs/STATE_SCHEMA.md`, `docs/STATE.md`
- Hyperparameters: `docs/HYPERPARAMETERS.md`
