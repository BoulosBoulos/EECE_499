# PDE Methods: Mathematical Specification

## Shared Framework

All four PDE methods share the following components:

### Auxiliary Critic U_phi(xi)
- MLP: Linear(79, 256) -> Tanh -> Linear(256, 256) -> Tanh -> Linear(256, 1)
- Trained on the reduced 79D PDE state xi
- Separate from the PPO recurrent critic

### Autograd Gradient
```
U_val = U_phi(xi)
grad_U = autograd.grad(U_val.sum(), xi, create_graph=True)
```

### Behavioral Dynamics F_a(xi)
- `BehavioralDynamics.one_step(xi, a)` propagates xi by one dt=0.1s step
- Delta: `delta_xi_a = F_a(xi) - xi` (raw state change, NOT divided by dt)
- 5 discrete actions: {STOP, CREEP, YIELD, GO, ABORT}

### Local Reward Surrogate r(xi, a)
```
r = w_prog * progress + w_time * dt
    + w_risk * sigmoid((ttc_thr - ttc_next) / 0.3)
    + w_pothole * sigmoid((pothole_thr - d_pot_next) / 0.2)
```
Uses smooth sigmoid approximations (not step functions) to maintain non-zero gradients.

### Collocation Sampling
Mix of real rollout PDE states (70%) and jittered copies (30%).
Jitter applied to primitive features only; derived quantities (tau, t_cpa, d_cpa, TTC) are recomputed.

### Anchor Loss
```
L_anchor = MSE(U_phi(xi_rollout), GAE_returns)
```

### Boundary Conditions
```
L_bc = MSE(U_phi(xi_success), 0) + MSE(U_phi(xi_collision), w_coll)
```
where w_coll = -20.0 (large negative penalty for collision states).

### Distillation
```
L_distill = MSE(V_ppo(s), U_phi(xi).detach())
```
Added to the PPO value loss to transfer physics-informed structure.

### Total Auxiliary Loss
```
L_aux = lambda_anchor * L_anchor + lambda_pde * L_pde + lambda_bc * L_bc
```
where L_pde is the method-specific residual loss (see below).

---

## Method 1: Hard-HJB

### Derivation

From the discrete Bellman equation:
```
V(xi) = max_a [r_step(xi,a) + gamma * V(F_a(xi))]
```

Taylor expand V(F_a(xi)) ~ V(xi) + grad_V^T * (F_a(xi) - xi):
```
V(xi)(1 - gamma) ~ max_a [r_step + gamma * grad_V^T * (F_a(xi) - xi)]
```

Since (1 - gamma) ~ -ln(gamma) for gamma close to 1:
```
rho = U(xi)*ln(gamma) + max_a [r(xi,a) + gamma * grad_U^T * (F_a(xi) - xi)] = 0
```

### PDE Q-values
```
q_a = r(xi, a) + gamma * grad_U^T * (F_a(xi) - xi)
```

### Residual
```
rho = U(xi) * ln(gamma) + max_a q_a
L_hjb = E[rho^2]
```

### Config: `configs/pde/hjb_aux.yaml`

---

## Method 2: Soft-HJB

### Derivation

Entropy-regularized version using logsumexp instead of hard max:
```
rho = U(xi)*ln(gamma) + tau * logsumexp([q_a / tau])
```

### Actor Alignment

The Soft-HJB method also includes a KL alignment term:
```
pi_soft(a|xi) = softmax(q_a / tau)
L_align = KL(pi_soft || pi_theta)
```
This guides the PPO actor toward the PDE-derived soft policy.

### Config: `configs/pde/soft_hjb_aux.yaml`

---

## Method 3: Eikonal

### Formulation

The Eikonal equation governs the "travel cost" field:
```
||grad_U(xi)||^2 = c(xi)^2
```
where c(xi) = 1 / v_eff(xi) is the local cost (inverse of effective safe speed).

### Effective Safe Speed

```
v_eff = v * sigma_safe, clamped to >= v_min

sigma_safe = sigma_stop * sigma_ttc * sigma_vis
```

where:
- `sigma_stop = sigmoid((m_stop - 1.0) / 1.0)` with `m_stop = d_cz - d_stop(v)`
- `sigma_ttc = sigmoid((TTC_min - ttc_thr) / 0.5)`
- `sigma_vis = clamp(alpha_cz, min=0.1)`

### Residual
```
rho = ||grad_U||^2 - c(xi)^2
L_eik = E[rho^2]
```

### Theoretical Note

The Eikonal equation is a structural regularizer on the auxiliary critic: it
encourages the learned value landscape to reflect travel-time structure under
safety constraints. With discrete actions, this is an approximation -- the
true Hamilton-Jacobi equation would require continuous control. The Eikonal
acts as an isotropic simplification.

### Config: `configs/pde/eikonal_aux.yaml`

---

## Method 4: CBF-PDE

### Formulation

Uses a shifted barrier function h(xi) = U(xi) + c_offset, where c_offset = |w_coll|/2 = 10.0.
This shifts the zero-level set so the safe/unsafe boundary occurs at U = -10 (midpoint
between typical safe returns and the collision penalty -20).

| State | U value | h = U + 10 | Interpretation |
|-------|---------|------------|----------------|
| Success | ~0 | +10 | Safe (goal reached) |
| Typical safe | +10 | +20 | Deeply safe |
| Safety boundary | -10 | 0 | Transition point |
| Collision | -20 | -10 | Deeply unsafe |

Since c_offset is constant, grad_h = grad_U. The CBF condition requires
forward invariance of {xi : h(xi) >= 0}:

```
h_dot(xi, a) + alpha * h(xi) >= 0
```
where h_dot = grad_U^T * (F_a(xi) - xi).

### Residual
```
h_val = U(xi) + cbf_safe_offset
max_val = max_a [grad_U^T * (F_a(xi) - xi) + alpha_cbf * h_val]
rho = ReLU(-max_val)
L_cbf = E[rho^2]
```

### Theoretical Note

This is a CBF-inspired regularizer, not a strict CBF guarantee. With learned
dynamics and a discrete action set, strict forward invariance is not guaranteed.
The regularizer encourages the auxiliary critic to learn a value landscape
consistent with the existence of safe control actions.

### Config: `configs/pde/cbf_aux.yaml`

---

## Mathematical Notes

### Approximations Used

| Method | Approximation | Justification |
|--------|--------------|---------------|
| Hard-HJB | First-order Taylor: U(F_a) ~ U(xi) + grad_U * (F_a - xi) | Valid for small dt, one-step transitions |
| Soft-HJB | Same Taylor + log-sum-exp soft-max | Entropy regularization prevents policy collapse |
| Eikonal | Dynamics-derived v_eff via post-action TTC check | Grounds speed constraint in actual action space |
| CBF-PDE | Shifted barrier h = U + c, c = |w_coll|/2 | Places zero-level at meaningful safety boundary |
| All methods | Top-3 agent reduction in xi (79D) | Captures most relevant interactions within fixed budget |

---

## Summary Table

| Method | Residual | Key Idea |
|--------|----------|----------|
| Hard-HJB | `U*ln(gamma) + max_a[r + gamma*grad_U*(F_a-xi)]` | Discrete Bellman consistency |
| Soft-HJB | `U*ln(gamma) + tau*logsumexp([r + gamma*grad_U*(F_a-xi)]/tau)` | Entropy-regularized Bellman |
| Eikonal | `||grad_U||^2 - (1/v_eff)^2`, v_eff from dynamics | Travel-time with dynamics-derived speed |
| CBF-PDE | `ReLU(-max_a[grad_U*(F_a-xi) + alpha*(U+offset)])` | Shifted barrier safe set invariance |
