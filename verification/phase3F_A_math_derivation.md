# Phase 3F-A — Mathematical Derivation: Eikonal Reformulation as Time-of-Arrival Critic

**Status:** Paper-grade methodology document. Gate before any code change.
**Companion spec:** `SPEC_PHASE_3F_A_EIKONAL_REFORMULATION.md` (Phase 31 follow-up).
**Branch:** `phase2-verification-gate`.
**Authors:** EECE 499 PINN-RL working notes.

This document derives, from first principles, the four-term auxiliary loss

$$
\mathcal L_{\mathrm{aux}}(\phi,\theta) = w_{\mathrm{eik}}\,\mathcal L_{\mathrm{eik}}(\phi)
+ w_{\mathrm{bc}}\,\mathcal L_{\mathrm{bc}}(\phi)
+ w_{\mathrm{ground}}\,\mathcal L_{\mathrm{ground}}(\phi)
+ w_{\mathrm{distill}}\,\mathcal L_{\mathrm{distill}}(\theta;\,\mathrm{stop\_grad}(T_\phi))
$$

used to train an auxiliary critic $T_\phi:\mathcal S\to\mathbb R_{\ge 0}$ that represents the **expected time-of-arrival** to a goal set, alongside a PPO actor $\pi_\theta$ that is biased toward time-minimising actions via a soft-KL penalty. Each term is justified from a separate mathematical foundation: $\mathcal L_{\mathrm{eik}}$ from the canonical Eikonal Hamilton–Jacobi PDE of time-optimal control; $\mathcal L_{\mathrm{bc}}$ from the canonical Eikonal boundary condition adapted to a finite horizon; $\mathcal L_{\mathrm{ground}}$ from the observed-data fitting paradigm of Physics-Informed Neural Networks (PINNs) and Monte-Carlo estimation; $\mathcal L_{\mathrm{distill}}$ from KL-regularised reinforcement learning. Mutual consistency is established: $\mathcal L_{\mathrm{eik}},\mathcal L_{\mathrm{bc}},\mathcal L_{\mathrm{ground}}$ all describe the *same physical scalar field* $T(s)$, and the stop-gradient on $T_\phi$ inside $\mathcal L_{\mathrm{distill}}$ guarantees the actor’s loss does not pollute the critic’s training.

The reformulation resolves the structural ill-posedness identified in Phase 31 Stage 2 (`verification/phase31_investigation_eikonal.json`), in which the original Eikonal critic was simultaneously asked to track GAE returns (range $[-1500, +200]$), anchor at $+200$ at terminal success states, and satisfy the kinematic constraint $\|\nabla U\|^2 = c(\xi)^2 \approx 4$. Numerical analysis on production checkpoints showed $\|\nabla U\|^2 / c(\xi)^2 \in [8.4,\,15.9]$ at step $5\times10^5$, with $89\%$ of states violating the residual in the same direction; the residual grew during training rather than decreasing.

---

## 1. The canonical Eikonal Hamilton–Jacobi PDE

Consider an autonomous time-optimal control problem on a closed state set $\mathcal S\subset\mathbb R^n$ with a target set $\mathcal G\subset\mathcal S$. A controlled trajectory $s(t)$ obeys
$$
\dot s(t) = f(s(t), a(t)), \qquad a(t)\in\mathcal A,\quad s(0)=s_0,
$$
and the *time-to-arrival* function is defined by
$$
T(s_0) := \inf_{a(\cdot)\in\mathcal A^{[0,\infty)}} \big\{\,\tau\ge 0 : s(\tau)\in\mathcal G \mid s(0)=s_0\,\big\}.
$$
$T:\mathcal S\to\mathbb R_{\ge 0}\cup\{+\infty\}$ is a non-negative scalar field with the obvious boundary condition $T(s)=0$ for $s\in\mathcal G$ and $T(s)=+\infty$ for any $s$ from which $\mathcal G$ is unreachable.

### 1.1 The Hamilton–Jacobi–Bellman equation

By Bellman's principle of optimality, for any state $s\notin\mathcal G$ and any small $\delta>0$,
$$
T(s) = \delta + \min_{a\in\mathcal A} T\big(\Phi_a^{\delta}(s)\big),
$$
where $\Phi_a^{\delta}(s)$ is the time-$\delta$ flow of the control $a$ applied to $s$. Subtracting $T(s)$ from both sides, dividing by $\delta$, and taking $\delta\to 0^+$ yields the Hamilton–Jacobi–Bellman (HJB) equation
$$
0 \;=\; 1 \;+\; \min_{a\in\mathcal A}\; \nabla T(s)^{\top} f(s,a).
$$
Equivalently,
$$
\max_{a\in\mathcal A}\;\big[-\nabla T(s)^{\top} f(s,a)\big]\;=\;1,
\qquad s\in\mathcal S\setminus\mathcal G.
\tag{HJB}
$$
This is the static, time-stationary HJB equation for time-optimal control. The boundary condition is $T|_{\partial\mathcal G}=0$.

### 1.2 Reduction to the Eikonal form

When the dynamics are *isotropic* — meaning the available velocity at $s$ has the same magnitude in every direction the control can produce, so that $\{f(s,a):a\in\mathcal A\}$ is the closed ball $\overline B_{v(s)}$ of some radius $v(s)>0$ — the HJB equation simplifies. The maximisation over $a$ is then equivalent to maximisation over the unit sphere, and
$$
\max_{\|f\|\le v(s)} \big[-\nabla T(s)^{\top} f\big] = v(s)\,\|\nabla T(s)\|.
$$
Substituting into (HJB),
$$
v(s)\,\|\nabla T(s)\| \;=\; 1
\;\Longleftrightarrow\;
\|\nabla T(s)\|\;=\;\frac{1}{v(s)}\;=:\;c(s),
\qquad s\in\mathcal S\setminus\mathcal G,\ \ T|_{\partial\mathcal G}=0.
\tag{Eik}
$$
This is the *canonical Eikonal equation* (Sethian, 1996; Mitchell, 2007). The function $c(s)>0$ is called the *slowness*; it is the reciprocal of the local speed at which time-of-arrival information propagates through the state space.

### 1.3 Squared form used in this work

Squaring (Eik) gives the form used in the auxiliary loss:
$$
\|\nabla T(s)\|^2 \;=\; c(s)^2,
\qquad
\rho(s,T) := \|\nabla T(s)\|^2 - c(s)^2 \;\stackrel{!}{=}\; 0.
\tag{Eik²}
$$
Two reasons to prefer the squared form for neural-network optimisation: (a) $\|\nabla T\|^2$ is smooth in the network parameters $\phi$ even when $\|\nabla T\|=0$ (the unsquared form has a non-differentiable kink at zero); (b) when implemented as $(\rho)^2$ the loss enjoys quadratic local geometry, which is well-conditioned for first-order optimisers.

### 1.4 The slowness $c(\xi)$ in this project

The reduced PDE state is $\xi\in\mathbb R^{79}$ (`models/pde/state_builder.py`). The slowness is operationalised via the existing $v_{\mathrm{eff}}$ computation in `models/pde/residuals.py`:
$$
v_{\mathrm{eff}}(\xi)\;=\;\Big(\max_{a\in\mathcal A}\big[v_{\mathrm{next}}(\xi,a)\cdot\sigma\big((\mathrm{TTC}_{\mathrm{next}}-\mathrm{TTC}_{\mathrm{thr}})/0.5\big)\big]\Big)\cdot\max(\alpha_{\mathrm{cz}},0.1),
$$
$$
c(\xi)\;=\;\frac{1}{\max\big(v_{\mathrm{eff}}(\xi),\,v_{\min}\big)},\qquad v_{\min}=0.5,
$$
where $\sigma$ is the logistic, $\alpha_{\mathrm{cz}}\in[0,1]$ is conflict-zone visibility, and the $v_{\min}$ floor is a numerical regulariser preventing division-by-zero. The reformulation reuses this $c(\xi)$ unchanged: the Stage 2 diagnosis showed $c(\xi)$ itself is well-formed; the issue was the auxiliary critic's *semantics*, not the slowness.

---

## 2. Discretisation to the discrete-time MDP

Our environment is a discrete-time Markov decision process $(\mathcal S,\mathcal A,P,r,\gamma)$ with timestep $\Delta t=0.1\,\mathrm s$ (one SUMO macro-step). We need to show that the canonical continuous-time time-of-arrival function $T$ has a natural discrete analogue and that the Eikonal PDE is a valid asymptotic limit of the discrete Bellman equation.

### 2.1 Discrete time-of-arrival

For each state $s\in\mathcal S$, define the random variable
$$
\tau(s) := \inf\{\,k\ge 0 : s_k\in\mathcal G \mid s_0=s,\ a_t\sim\pi^\star_t\,\},
$$
where $\pi^\star$ is a time-minimal policy and $\inf\emptyset:=+\infty$. Define
$$
T(s) := \mathbb E_{\pi^\star,P}\big[\tau(s)\big].
$$
With deterministic dynamics (which our reduced-state surrogate $f_a:\xi\to\xi'$ is), the expectation collapses to the deterministic $\tau(s)$. We will work in the deterministic regime; the stochastic case differs only in that $T$ becomes an expectation over $P$ but the structural results are unchanged.

### 2.2 Discrete Bellman equation

By Bellman's principle, for $s\notin\mathcal G$,
$$
T(s) \;=\; 1 \;+\; \min_{a\in\mathcal A}\; T\big(f_a(s)\big),
$$
with $T|_{\mathcal G}=0$. The "1" is the unit step cost (one timestep). This is a discrete shortest-path equation (Bertsekas, 2017, §1.2).

### 2.3 Continuum limit recovers the Eikonal PDE

Write the one-step transition as $f_a(s) = s + \Delta t\cdot\dot f(s,a) + O(\Delta t^2)$ where $\dot f$ is the underlying continuous-time vector field. Taylor-expand $T(f_a(s))$ around $s$:
$$
T\big(f_a(s)\big) = T(s) + \Delta t\,\nabla T(s)^{\top}\dot f(s,a) + O(\Delta t^2).
$$
Substituting into the discrete Bellman equation, dividing by $\Delta t$, and rearranging:
$$
\frac{1}{\Delta t} \;+\; \min_{a\in\mathcal A}\,\nabla T(s)^{\top}\dot f(s,a) \;=\; 0 \;+\; O(\Delta t).
$$
Multiplying by $\Delta t$ and letting $\Delta t\to 0$ gives back the continuous HJB equation (HJB) of §1.1, hence (Eik) under the isotropy reduction. In the discrete setting we enforce (Eik²) at the *macroscopic* timestep $\Delta t = 0.1$, which is a valid approximation provided $\|\nabla T\|\,\|\dot f\|\,\Delta t \ll 1$, i.e. the time-of-arrival changes by $\ll 1$ step per timestep — true away from the boundary $\partial\mathcal G$.

**Practical consequence.** We train $T_\phi$ such that $T_\phi(s)$ is the time *in units of timesteps* (not seconds): $T_\phi(s)\approx k$ means "approximately $k$ macro-steps to the goal." This convention matches the supervised target $T_{\mathrm{obs}}(s_t)=t_{\mathrm{succ}}-t$ in §3 below.

### 2.4 The auxiliary critic $T_\phi$

The auxiliary critic is a multi-layer perceptron $T_\phi:\mathbb R^{79}\to\mathbb R_{\ge 0}$ with architecture matching the existing `EikonalAuxCritic` (`models/pde/eikonal_aux_critic.py`): two hidden layers of width 256 with `tanh` activations and a final scalar head. Non-negativity is encouraged but not strictly enforced (a softplus head was considered and rejected in §6.2 below).

---

## 3. Finite-horizon boundary-condition adaptation

The canonical (Eik) prescribes $T(s_{\mathrm{succ}})=0$ on the goal set and $T(s)=+\infty$ for unreachable states. Naïvely transferring this to neural-network training fails: a $+\infty$ target is uniformly the worst possible target for a regression loss. Numerical Eikonal solvers (Sethian, 1996, §3.2; Mitchell, 2007) handle the same issue by truncating the unreachable region or by initialising it at a *large but finite* value chosen to be well above any feasible reachable $T(s)$. We import the same truncation here.

### 3.1 Choice of $T_{\max}$

Define
$$
T_{\max} := 100,
$$
in units of timesteps. The justification chain is:

1. **Empirical bounds.** Inspection of the 36-run Phase 3 calibration metrics shows that successful trajectories in the $\texttt{1a}$ scenario complete in 50–250 macro-steps (mean $\approx 110$ in late training, after Stage 1A+1B+1D). Setting $T_{\max}$ above this band ensures collision states are anchored *strictly larger* than any value that any successful state legitimately produces.

2. **Avoids confounding with timeout.** The maximum episode length is $K=500$ steps. We deliberately set $T_{\max}=100\ll K$ so collision-anchored values $T_\phi(s_{\mathrm{coll}})\approx 100$ are not identified with timeout-truncated trajectories (whose true $T$ is undefined). $T_{\max}$ is a *boundary-condition target*, not an upper bound on $T_\phi$.

3. **Numerical conditioning.** With $T_\phi$ outputs roughly in $[0,T_{\max}]$, $\|\nabla T_\phi\|$ on the order of unity is consistent with the kinematic $c(\xi)\approx 1/\max(v_{\mathrm{eff}},v_{\min}) \in [1/14, 2]$ in our state space. Specifically, integrating $\|\nabla T\|=c(\xi)\approx 1$ along a path of length $\approx 100$ macro-step-equivalents gives $T\approx 100$, which matches $T_{\max}$ self-consistently. (Stage 1’s $w_{\mathrm{success}}=200$ and $\|\nabla U\|^2\approx 30\text{–}50$ were *inconsistent* in this exact sense — the squared-gradient and reward-anchor magnitudes were on different scales.)

4. **Literature analogue.** Sethian's Fast Marching Methods initialise unaccepted nodes at a finite "large value" (often $10^9$ in floating-point); we use the same idea but pick the value to be *physically meaningful* (≈ longest plausible feasible trajectory) rather than numerically infinite, because this is a *training target* not a bookkeeping placeholder.

### 3.2 Boundary-condition loss

$$
\mathcal L_{\mathrm{bc}}(\phi) \;=\;
\frac{1}{|\mathcal B_{\mathrm{succ}}|}\sum_{s\in\mathcal B_{\mathrm{succ}}}\big(T_\phi(s)-0\big)^2
\;+\;
\frac{1}{|\mathcal B_{\mathrm{coll}}|}\sum_{s\in\mathcal B_{\mathrm{coll}}}\big(T_\phi(s)-T_{\max}\big)^2
\tag{$\mathcal L_{\mathrm{bc}}$}
$$
where $\mathcal B_{\mathrm{succ}}$ and $\mathcal B_{\mathrm{coll}}$ are mini-batches of terminal states sampled from the rollout buffer. Timeout-terminated trajectories are excluded from $\mathcal L_{\mathrm{bc}}$ because the true $T$ at a timeout state is unknown (the trajectory was truncated, not run to completion).

### 3.3 Why this BC is consistent with $\mathcal L_{\mathrm{eik}}$

In canonical Eikonal solvers, the BC is an *external constraint* imposed on the PDE solution; the PDE governs the interior and the BCs anchor it. Our $\mathcal L_{\mathrm{bc}}$ plays the same role: it pins $T_\phi$ at $0$ on $\partial\mathcal G$ and at $T_{\max}$ on a "failure boundary" that approximates the unreachable region. The compatibility holds because both $\mathcal L_{\mathrm{eik}}$ and $\mathcal L_{\mathrm{bc}}$ pertain to the *same scalar field* $T$ defined in §2.1.

By contrast, the Stage-2 *original* Eikonal critic anchored $U_\phi$ at $+200$ at success and at $+50$ at collision, treating $U$ as an env-reward-shaped value function. That role conflicted with the kinematic constraint $\|\nabla U\|^2=c^2$, and indeed Stage 2 verified the conflict empirically: $\|\nabla U\|^2$ grew during training to fit the reward-anchored $U$. Phase 3F-A removes that conflict by re-defining the field's semantics.

---

## 4. The discrete-time Eikonal advantage $A_{\mathrm{eik}}(s,a)$

A central goal of the reformulation is to transmit the time-of-arrival information learned by $T_\phi$ to the PPO actor $\pi_\theta$, biasing it toward time-minimising actions. The vehicle for this is a per-action *Eikonal advantage*
$$
A_{\mathrm{eik}}(s,a) \;:=\; T_\phi(s)\;-\;T_\phi\big(f_a(s)\big),\qquad a\in\mathcal A.
\tag{$A_{\mathrm{eik}}$}
$$
This section derives $A_{\mathrm{eik}}$ as the discrete-time analogue of the time-minimal Hamiltonian and shows that $\arg\max_a A_{\mathrm{eik}}(s,a)$ is the time-optimal action under $T_\phi$.

### 4.1 The continuous-time time-minimal Hamiltonian

For the continuous-time time-minimal control problem with cost $\int 1\,dt$ until reaching $\mathcal G$, Pontryagin's minimum principle (Pontryagin et al., 1962; Bertsekas, 2017, §3.3) gives the Hamiltonian
$$
H(s,p,a) \;=\; 1 + p^{\top} f(s,a),\qquad p:=\nabla T(s).
$$
The optimal action satisfies $a^\star(s)\in\arg\min_a H(s,\nabla T(s),a)$, i.e.
$$
a^\star(s) \;\in\; \arg\min_{a\in\mathcal A}\;\nabla T(s)^{\top} f(s,a) \;=\; \arg\max_{a\in\mathcal A}\;\big[-\nabla T(s)^{\top} f(s,a)\big].
$$
Recognising $-\nabla T(s)^{\top} f(s,a)$ as the directional rate of decrease of $T$ along action $a$, the optimal action is the one that *decreases time-of-arrival fastest*.

### 4.2 Discrete analogue

In the discrete-time MDP, the analogous quantity is
$$
\mathrm{rate}_a(s) \;:=\; \frac{T(s) - T(f_a(s))}{1\ \text{step}} \;=\; T(s) - T(f_a(s)) \;=:\; A_{\mathrm{eik}}(s,a),
$$
where the denominator $\Delta t = 1$ step is absorbed into the units (recall §2.3: $T$ is in units of timesteps). By the discrete Bellman equation (§2.2),
$$
T(s) \;=\; 1 + \min_a T(f_a(s))
\ \;\Longleftrightarrow\;\
\max_a \big[T(s) - T(f_a(s))\big] \;=\; 1.
$$
Thus $\max_a A_{\mathrm{eik}}(s,a)=1$ everywhere on $\mathcal S\setminus\mathcal G$ for the *exact* solution, with the maximiser being the time-optimal action $a^\star(s)$. For a learned approximation $T_\phi$, $A_{\mathrm{eik}}(s,a)$ is a continuous proxy that retains the directional information: actions with high $A_{\mathrm{eik}}$ "point downhill" in $T_\phi$, hence toward $\mathcal G$.

### 4.3 Why $T(s) - T(s_{\mathrm{next}}|a)$ rather than $\nabla T^{\top}\Delta s$

A natural alternative would be to compute the linearised advantage
$$
\tilde A_{\mathrm{eik}}(s,a) \;=\; -\nabla T_\phi(s)^{\top}\big(f_a(s)-s\big),
$$
mirroring the continuous-time form. We choose $A_{\mathrm{eik}}=T_\phi(s)-T_\phi(s_{\mathrm{next}}|a)$ instead for three reasons:

1. **Higher-order accurate.** $\tilde A_{\mathrm{eik}}$ is the first-order Taylor approximation of $A_{\mathrm{eik}}$. For $\Delta t=0.1$ macro-steps and our network smoothness, the second-order term is non-negligible. The finite-difference form is exact at the chosen discretisation.

2. **Cheaper.** Computing $A_{\mathrm{eik}}$ requires $|\mathcal A|+1=6$ network forward passes (one for $T_\phi(s)$ and one for each $T_\phi(f_a(s))$). Computing $\tilde A_{\mathrm{eik}}$ requires one forward pass plus one backward pass to obtain $\nabla T_\phi(s)$, plus $|\mathcal A|=5$ dynamics evaluations. The dominant cost in our setup is dynamics, so the comparison is roughly equal — but $A_{\mathrm{eik}}$ avoids the autograd backward, which is desirable when the actor's loss must not modify $\phi$ (cf. §6).

3. **Sign-invariant under network rescaling.** Both forms depend linearly on $T_\phi$, so their *ranking* of actions is the same under multiplicative rescaling of $T_\phi$. This robustness matters during early training when $T_\phi$ has not yet been correctly scaled.

### 4.4 Connection to $\mathcal L_{\mathrm{eik}}$

The exact Eikonal solution satisfies $\max_a A_{\mathrm{eik}}(s,a)=1$ pointwise and $\|\nabla T(s)\|^2 = c(s)^2$. Both encode the same time-optimality condition, but in different forms:
- $\mathcal L_{\mathrm{eik}}$ enforces the *magnitude* of the gradient $\|\nabla T\|=c$ via collocation-state regression.
- $A_{\mathrm{eik}}$ exposes the *direction* of the gradient through finite differences over the action space.

This complementarity is essential for transmitting Eikonal information to the actor: $\mathcal L_{\mathrm{eik}}$ trains $T_\phi$ as a scalar field with the right gradient *magnitude*; $A_{\mathrm{eik}}$ tells the actor *which way to go* in that field.

---

## 5. Soft-KL distillation: actor guidance with exploration preserved

Once $T_\phi$ is trained, we wish to inform $\pi_\theta(\cdot|s)$ that high-$A_{\mathrm{eik}}$ actions are time-saving. The naïve choice would be to take $a^\star(s)=\arg\max_a A_{\mathrm{eik}}(s,a)$ and constrain $\pi_\theta$ to be deterministic at $a^\star$. We reject this as too aggressive and instead use a *soft* (entropic) target.

### 5.1 The soft target distribution

Define the temperature-$\tau$ Boltzmann policy under the Eikonal advantage:
$$
\pi_{\mathrm{eik}}(a\mid s)\;:=\;\frac{\exp\big(A_{\mathrm{eik}}(s,a)/\tau\big)}{\sum_{a'}\exp\big(A_{\mathrm{eik}}(s,a')/\tau\big)},\qquad\tau>0.
\tag{$\pi_{\mathrm{eik}}$}
$$
$\pi_{\mathrm{eik}}$ is a smooth approximation of the greedy maximiser: as $\tau\to 0$ it concentrates on $a^\star$; as $\tau\to\infty$ it tends to uniform. With $\tau=1$ (default) it is the maximum-entropy policy *conditioned on a unit average advantage*.

### 5.2 KL-distillation loss

We pull $\pi_\theta$ toward $\pi_{\mathrm{eik}}$ via a forward KL penalty:
$$
\mathcal L_{\mathrm{distill}}(\theta;\,\mathrm{stop\_grad}(T_\phi))\;:=\;\beta_{\mathrm{KL}}\cdot \mathbb E_{s\sim\rho_\theta}\big[\,\mathrm{KL}\big(\pi_\theta(\cdot\mid s)\,\big\|\,\pi_{\mathrm{eik}}(\cdot\mid s)\big)\,\big].
\tag{$\mathcal L_{\mathrm{distill}}$}
$$
$\rho_\theta$ is the on-policy state distribution of the rollout buffer; $\beta_{\mathrm{KL}}>0$ is a hyperparameter; $\mathrm{stop\_grad}$ is detailed in §6.

### 5.3 Why soft-KL preserves PPO exploration

PPO's standard exploration mechanism is the entropy bonus $\beta_{\mathrm{ent}}\,\mathcal H[\pi_\theta]$ (Schulman et al., 2017). A hard greedy target $\delta_{a^\star}$ would *fight* the entropy bonus directly: minimising $\mathrm{KL}(\pi_\theta\|\delta_{a^\star})$ is unbounded below only when $\pi_\theta=\delta_{a^\star}$, which has zero entropy. The optimisation balances entropy bonus against distillation, and the equilibrium is highly sensitive to $\beta_{\mathrm{KL}}$.

The soft target $\pi_{\mathrm{eik}}$ has entropy $\mathcal H[\pi_{\mathrm{eik}}] = \log|\mathcal A| - \tau^{-1}\sum_a\pi_{\mathrm{eik}}(a)A_{\mathrm{eik}}(s,a)$, bounded below by zero and above by $\log|\mathcal A|=\log 5$. Pulling $\pi_\theta$ toward a positive-entropy target *aligns* with the entropy bonus rather than fighting it: $\pi_\theta$ is encouraged to retain at least the entropy of $\pi_{\mathrm{eik}}$. Empirically (Haarnoja et al., 2018, Soft Actor–Critic), KL-regularisation toward a soft target is the standard mechanism for entropy-regularised exploration in policy-gradient methods.

A second motivation: $\pi_{\mathrm{eik}}$ is a *hypothesis* about good actions, computed from a learned $T_\phi$ that may be imperfect. A hard target propagates errors in $T_\phi$ directly into a deterministic policy. A soft target spreads the influence across multiple actions, weighted by their advantage; this is robust to low-confidence regions where multiple actions have similar $A_{\mathrm{eik}}$ (e.g. STOP and CREEP near a stationary obstacle). This robustness argument mirrors that of TRPO/PPO's *soft* trust region (Schulman et al., 2015, 2017).

### 5.4 Forward KL versus reverse KL

We use the *forward* KL $\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{eik}})$ rather than the reverse $\mathrm{KL}(\pi_{\mathrm{eik}}\|\pi_\theta)$. Forward KL is *mean-seeking*: it penalises $\pi_\theta$ whenever it places mass where $\pi_{\mathrm{eik}}$ has none, but is more permissive where $\pi_{\mathrm{eik}}$ is broad. This is appropriate for distilling from a *coarse* hypothesis: we want $\pi_\theta$ to cover the support of $\pi_{\mathrm{eik}}$ but allow further structure that PPO discovers. Reverse KL would be mode-seeking and could cause $\pi_\theta$ to collapse onto a single mode of $\pi_{\mathrm{eik}}$, defeating the soft-distillation rationale.

---

## 6. The four-term loss decomposition: mutual consistency

The auxiliary loss is
$$
\mathcal L_{\mathrm{aux}}(\phi,\theta)
\;=\; w_{\mathrm{eik}}\,\mathcal L_{\mathrm{eik}}(\phi)
\;+\; w_{\mathrm{bc}}\,\mathcal L_{\mathrm{bc}}(\phi)
\;+\; w_{\mathrm{ground}}\,\mathcal L_{\mathrm{ground}}(\phi)
\;+\; w_{\mathrm{distill}}\,\mathcal L_{\mathrm{distill}}\big(\theta;\,\mathrm{stop\_grad}(T_\phi)\big).
\tag{4-term}
$$
This section justifies why the four terms are mutually consistent — each pertains to a coherent aspect of the *same* time-of-arrival field — and why the stop-gradient on $T_\phi$ in $\mathcal L_{\mathrm{distill}}$ is essential for that consistency.

### 6.1 The supervised grounding term $\mathcal L_{\mathrm{ground}}$

For each completed trajectory $\tau=(s_0,a_0,\ldots,s_K)$ in the rollout buffer with $K<\mathrm{max\_steps}$, define the observed time-of-arrival
$$
T_{\mathrm{obs}}(s_t) \;:=\;
\begin{cases}
K-t, & \text{if }\tau\text{ ended in success at step }K,\\
T_{\max}, & \text{if }\tau\text{ ended in collision at any step (all }t\text{)},\\
\text{undefined,} & \text{if }\tau\text{ was truncated by timeout.}
\end{cases}
$$
Successful trajectories give *exact* time-to-arrival from each visited state. Collision-terminated trajectories give a globally consistent BC of $T_{\max}$ (matching the BC value of §3.1) — the trajectory failed to reach the goal, so all its states inherit the failure-boundary value. Timeout-terminated trajectories are *omitted* from $\mathcal L_{\mathrm{ground}}$ because the true $T(s_t)$ is unknown (the trajectory did not complete and we cannot retrospectively assign a value).

The supervised loss is
$$
\mathcal L_{\mathrm{ground}}(\phi) \;=\;
\frac{1}{|\mathcal D_{\mathrm{ground}}|}\sum_{(s_t,T_{\mathrm{obs}})\in\mathcal D_{\mathrm{ground}}}\big(T_\phi(s_t) - T_{\mathrm{obs}}(s_t)\big)^2,
\tag{$\mathcal L_{\mathrm{ground}}$}
$$
where $\mathcal D_{\mathrm{ground}}$ is the set of $(s_t,T_{\mathrm{obs}})$ pairs from non-timeout trajectories.

### 6.2 Why all three of $\mathcal L_{\mathrm{eik}},\mathcal L_{\mathrm{bc}},\mathcal L_{\mathrm{ground}}$ describe the same field

**Claim.** Let $T:\mathcal S\to\mathbb R_{\ge 0}$ be the time-of-arrival function defined in §2.1 with finite-horizon truncation $T(s_{\mathrm{coll}})=T_{\max}$. Then $T$ simultaneously satisfies:
1. $\|\nabla T(s)\|^2 = c(s)^2$ on the interior $\mathcal S\setminus(\mathcal G\cup\partial\mathcal G_{\mathrm{coll}})$,
2. $T(s)=0$ on $\partial\mathcal G$ (success boundary) and $T(s)=T_{\max}$ on $\partial\mathcal G_{\mathrm{coll}}$ (failure boundary),
3. $T(s_t)=K-t$ for every state $s_t$ on a successfully-completed trajectory of length $K$.

The first two are by construction (canonical Eikonal + finite-horizon adaptation). The third is the *definition* of $T$: along a time-minimal trajectory from $s_t$ to $\mathcal G$ taking exactly $K-t$ steps, the time-of-arrival from $s_t$ is $K-t$. (For a *non-optimal* trajectory the relation holds in expectation only; we use only the on-policy observed values, which is consistent with the standard PINN paradigm of fitting both PDE residual and observed data.)

**Implication.** The three loss terms $\mathcal L_{\mathrm{eik}},\mathcal L_{\mathrm{bc}},\mathcal L_{\mathrm{ground}}$ each penalise deviations of $T_\phi$ from $T$ along orthogonal axes:
- $\mathcal L_{\mathrm{eik}}$ constrains the *gradient magnitude* of $T_\phi$ throughout the interior;
- $\mathcal L_{\mathrm{bc}}$ pins $T_\phi$ at boundary states;
- $\mathcal L_{\mathrm{ground}}$ supplies *interior* point-wise values from observed trajectories.

There is no contradiction in the joint minimum, because the *true* $T$ is a fixed point of all three losses simultaneously. Indeed, this is precisely the structural fix relative to the Stage-2 original: there, $\mathcal L_{\mathrm{anchor}}$ pulled $U_\phi$ toward GAE returns (range $[-1500,+200]$) while $\mathcal L_{\mathrm{eik}}$ demanded $\|\nabla U_\phi\|^2\approx c^2\approx 4$, and these were not compatible because GAE returns are not a function with that gradient norm. Here, $T_{\mathrm{obs}}$ *is* the correct value of $T$ on its support, so $\mathcal L_{\mathrm{ground}}$ and $\mathcal L_{\mathrm{eik}}$ have the same fixed point.

### 6.3 The stop-gradient on $T_\phi$ in $\mathcal L_{\mathrm{distill}}$

$\mathcal L_{\mathrm{distill}}$ is constructed from $A_{\mathrm{eik}}(s,a)$, which depends on $T_\phi$. If we backpropagate through $T_\phi$ when computing $\partial \mathcal L_{\mathrm{distill}}/\partial\phi$, the actor's loss exerts a force on $T_\phi$'s parameters, and that force is *inconsistent* with the time-of-arrival semantics: the actor "wants" $T_\phi$ to make the actions it likes look better, which is reward-coupled rather than physics-coupled. To preserve the property that $T_\phi$ is governed only by Eikonal + BC + observed-data losses, we apply $\mathrm{stop\_grad}$ to $T_\phi$'s output inside $A_{\mathrm{eik}}$:
$$
A_{\mathrm{eik}}^{\mathrm{(detach)}}(s,a) \;=\; \mathrm{stop\_grad}\big[T_\phi(s)\big] \;-\; \mathrm{stop\_grad}\big[T_\phi(f_a(s))\big].
$$
Equivalently, $A_{\mathrm{eik}}$ is computed under `torch.no_grad()` for the purpose of $\mathcal L_{\mathrm{distill}}$. The gradient $\partial \mathcal L_{\mathrm{distill}}/\partial\phi=0$ identically; only $\partial \mathcal L_{\mathrm{distill}}/\partial\theta$ is non-zero.

This separation gives a clean two-network decomposition:
$$
\frac{\partial \mathcal L_{\mathrm{aux}}}{\partial \phi} \;=\; w_{\mathrm{eik}}\frac{\partial\mathcal L_{\mathrm{eik}}}{\partial\phi}+w_{\mathrm{bc}}\frac{\partial\mathcal L_{\mathrm{bc}}}{\partial\phi}+w_{\mathrm{ground}}\frac{\partial\mathcal L_{\mathrm{ground}}}{\partial\phi},\qquad
\frac{\partial \mathcal L_{\mathrm{aux}}}{\partial \theta} \;=\; w_{\mathrm{distill}}\frac{\partial\mathcal L_{\mathrm{distill}}}{\partial\theta}.
$$
$T_\phi$ is trained to be a correct time-of-arrival function; $\pi_\theta$ is independently trained (jointly with the standard PPO losses) to follow $T_\phi$'s recommendations. The two are decoupled via stop-gradient; this is the *cleanest possible* implementation of "train a physics critic, then bias the policy with it" within a single optimiser pass.

### 6.4 Hyperparameter weights

Default weights, justified *a priori*:

- $w_{\mathrm{eik}}=1.0$ — base scale.
- $w_{\mathrm{ground}}=1.0$ — matched to $w_{\mathrm{eik}}$. Both terms describe $T$ at *interior* points but from different angles (gradient constraint vs observed value). Empirically their squared errors should be on the same order ($T_\phi$ typical magnitude $\sim T_{\max}/2 \sim 50$, expected MSE on order of $10^2\!-\!10^3$).
- $w_{\mathrm{bc}}=0.5$ — reduced because BCs are anchored at *only two values* (0 and $T_{\max}$) and are sampled at fewer states per minibatch. The smaller weight prevents BC-overfitting of $T_\phi$'s output range.
- $w_{\mathrm{distill}}=0.5$ — reduced because $\mathcal L_{\mathrm{distill}}$ acts on $\theta$, not $\phi$, and its gradient combines additively with the much larger PPO surrogate loss; an over-large $w_{\mathrm{distill}}$ would dominate PPO's own policy gradient.
- $\beta_{\mathrm{KL}}=0.1,\ \tau=1.0$ — soft-KL distillation strength and temperature; default values consistent with SAC-family entropy regularisation.

Final tuning is empirical (Phase 3F-A Step 5 verification).

---

## 7. Compatibility with PINN literature

The four-term loss is not an invented hybrid; it is the standard *Physics-Informed Neural Network* (PINN) training paradigm (Karniadakis et al., 2021; Raissi et al., 2019), instantiated for our specific PDE.

### 7.1 The PINN paradigm

A PINN trains a neural network $u_\phi:\Omega\to\mathbb R$ to approximate the solution of a PDE $\mathcal N[u]=0$ on $\Omega$ with boundary condition $u|_{\partial\Omega}=g$, using a loss of the form
$$
\mathcal L_{\mathrm{PINN}}(\phi)\;=\; w_r\,\underbrace{\mathbb E_{\Omega}[(\mathcal N[u_\phi])^2]}_{\text{PDE residual}}
\;+\;w_b\,\underbrace{\mathbb E_{\partial\Omega}[(u_\phi-g)^2]}_{\text{boundary loss}}
\;+\;w_d\,\underbrace{\mathbb E_{\mathcal D}[(u_\phi-u_{\mathrm{obs}})^2]}_{\text{observed-data loss}}.
$$
Each term enforces the same target field $u$ from a different angle: PDE constraint + boundary anchoring + observed values. The well-known successes of PINNs for inverse problems and noisy-data PDE recovery (Raissi et al., 2019, §3.2) come from precisely this combination: residual alone is under-constrained (any function with the right gradient norm satisfies it locally); BC alone over-constrains terminal regions; observed data alone is sparse but exact at the points where it exists.

### 7.2 Mapping to Phase 3F-A

| PINN term | Phase 3F-A term | What it constrains |
|---|---|---|
| $\mathcal N[u]=\|\nabla u\|^2-c^2$ | $\mathcal L_{\mathrm{eik}}$ | gradient magnitude on collocation states |
| $u\|_{\partial\Omega}=g$ | $\mathcal L_{\mathrm{bc}}$ | boundary anchoring at $\partial\mathcal G$ and $\partial\mathcal G_{\mathrm{coll}}$ |
| $u(x_i)=u_i$ on observed points | $\mathcal L_{\mathrm{ground}}$ | observed time-of-arrival from completed trajectories |

The $\mathcal L_{\mathrm{distill}}$ term is *not* part of the PINN paradigm — it is an RL-specific addition that bridges $T_\phi$ (the PINN output) to the PPO actor. It does not modify $T_\phi$; the PINN training of $T_\phi$ is unaffected. From the perspective of the PINN literature, $T_\phi$ is trained by a standard three-term PINN loss; the fourth term is purely policy-side regularisation.

### 7.3 Why this configuration is well-posed

PINNs for first-order Hamilton–Jacobi PDEs (Yang & Mei, 2023; Bansal & Tomlin, 2021) consistently report that *all three* of residual + BC + observed-data are needed for stable training:
- BC + residual alone fails to converge in regions far from the boundary because the residual is under-constrained (any gradient *direction* satisfies it pointwise).
- Residual + observed-data without BC fails near boundaries because the observed data is concentrated on interior trajectories and offers no anchor at terminal sets.
- BC + observed-data without residual reduces to plain supervised regression and ignores the PDE structure; the network may interpolate the observed data without satisfying the PDE constraint at unobserved states.

All three are present in Phase 3F-A. This is what Stage 2 was missing: only $\mathcal L_{\mathrm{eik}}+\mathcal L_{\mathrm{bc}}+\mathcal L_{\mathrm{anchor}}$ was in place, with $\mathcal L_{\mathrm{anchor}}$ playing the role of "observed data" but pointing at the wrong target (GAE returns instead of observed time-of-arrival). $\mathcal L_{\mathrm{ground}}$ replaces $\mathcal L_{\mathrm{anchor}}$ with a target that is consistent with $T$.

---

## 8. References

**Eikonal HJ-PDE theory & numerical solvers**
- Sethian, J. A. (1996). *Level Set Methods and Fast Marching Methods: Evolving Interfaces in Computational Geometry, Fluid Mechanics, Computer Vision, and Materials Science.* Cambridge University Press. — canonical Eikonal solvers, finite-value initialisation of unaccepted nodes.
- Mitchell, I. M. (2007). *A toolbox of level set methods.* UBC Tech. Rep. TR-2007-11. — practical numerical Eikonal/HJ solvers; truncation conventions.
- Bansal, S., & Tomlin, C. J. (2021). *DeepReach: A deep learning approach to high-dimensional reachability.* In *ICRA*. — neural-network Hamilton–Jacobi solvers; PINN-style training for HJ-PDEs.

**Optimal control & dynamic programming**
- Pontryagin, L. S., Boltyanskii, V. G., Gamkrelidze, R. V., & Mishchenko, E. F. (1962). *The Mathematical Theory of Optimal Processes.* Wiley. — minimum principle for time-optimal control.
- Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control* (4th ed., Vol. 1, §1.2 & §3.3). Athena Scientific. — discrete-time time-minimal control; Bellman shortest-path equations.
- Lygeros, J. (2004). *On reachability and minimum cost optimal control.* *Automatica*, 40(6), 917–927. — HJ-PDE in continuous-time reachability and safe control.

**Physics-Informed Neural Networks**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* *Journal of Computational Physics*, 378, 686–707. — seminal PINN paper; combination of PDE residual, boundary conditions, and observed data.
- Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). *Physics-informed machine learning.* *Nature Reviews Physics*, 3(6), 422–440. — comprehensive PINN survey.
- Yang, L., & Mei, S. (2023). *Solving Hamilton-Jacobi-Bellman equations using physics-informed neural networks.* — PINNs for HJ-PDEs in optimal-control settings.

**KL-regularised reinforcement learning**
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms.* arXiv:1707.06347. — PPO; entropy-regularised policy gradient.
- Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *Trust region policy optimization.* In *ICML*. — soft trust-region argument for KL-regularised policy updates.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft actor–critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor.* In *ICML*. — entropy-regularised RL; soft-max policy as Boltzmann distribution over advantages.

**Eikonal in RL — most directly relevant prior**
- "Eik-HIQL" (arXiv:2509.06782; preprint citation as referenced by parent spec). — Eikonal-style auxiliary in goal-conditioned RL; closest published precedent for the present method, although Eik-HIQL formulates the auxiliary as a value function rather than a time-of-arrival critic and therefore does not encounter the structural conflict that Stage 2 diagnosed.

---

## Summary

The Phase 3F-A reformulation re-defines the Eikonal auxiliary critic from a *value function* to a *time-of-arrival function* $T_\phi:\mathcal S\to\mathbb R_{\ge 0}$. Four loss terms train this critic and bias the actor without conflicting:

$$
\mathcal L_{\mathrm{aux}}(\phi,\theta)
\;=\; w_{\mathrm{eik}}\,\mathbb E_{\mathrm{coll}}\big[(\|\nabla T_\phi\|^2 - c^2)^2\big]
\;+\; w_{\mathrm{bc}}\,\mathbb E_{\partial\mathcal G\cup\partial\mathcal G_{\mathrm{coll}}}\big[(T_\phi - g)^2\big]
\;+\; w_{\mathrm{ground}}\,\mathbb E_{\mathcal D_{\mathrm{ground}}}\big[(T_\phi - T_{\mathrm{obs}})^2\big]
\;+\; w_{\mathrm{distill}}\,\beta_{\mathrm{KL}}\,\mathbb E_{\rho_\theta}\big[\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{eik}})\big],
$$
where
$g(s)=0$ on $\partial\mathcal G$, $g(s)=T_{\max}$ on $\partial\mathcal G_{\mathrm{coll}}$;
$T_{\mathrm{obs}}(s_t)=K-t$ on completed-success trajectories and $T_{\max}$ on collision trajectories;
$\pi_{\mathrm{eik}}=\mathrm{softmax}(\mathrm{stop\_grad}(A_{\mathrm{eik}})/\tau)$;
and $A_{\mathrm{eik}}(s,a)=T_\phi(s)-T_\phi(f_a(s))$.

All terms describe the same time-of-arrival field $T$: the first three constrain $T_\phi$'s gradient, boundary, and interior values; the fourth uses a frozen $T_\phi$ to bias the actor without modifying $\phi$. The mathematical formulation is self-consistent and closes the ill-posedness identified in Phase 31 Stage 2.

**Status:** *Step 1 complete.* Ready for human review. **Implementation (Steps 2–12) is gated on approval of this document.**

---

## 9. Adaptive loss balancing and sparse-data handling (Step 7C)

Steps 2–7B implemented and verified the four-term loss with fixed weights. Step 7 ran the loss without weight balancing and observed $\mathcal L_{\mathrm{ground}}$ dominating $\mathcal L_{\mathrm{eik}}$ by a factor of ~300 (because $T_{\max}^2/c^2 \approx 10^4/4 = 2500$ is the natural magnitude ratio). Step 7B normalised by $T_{\max}^2$ and over-corrected, with $\mathcal L_{\mathrm{eik}}$ now dominating $\mathcal L_{\mathrm{ground}}$ by ~9×. Both manual interventions are brittle. Step 7C replaces the fixed weights with learnable uncertainty parameters following Kendall, Gal, Cipolla (CVPR 2018). It also adds a terminal replay buffer to address sparse-event boundary-condition training in scenarios with rare successes or collisions.

### 9.1 Kendall et al. 2018 uncertainty-weighted multi-task loss

For a multi-task regression problem with $K$ tasks, suppose the residual of task $i$ is approximately Gaussian with task-specific homoscedastic noise variance $\sigma_i^2$:
$$
y_i \;=\; f_i(s; \phi) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0,\sigma_i^2).
$$
The negative log-likelihood of observed data given the network parameters $\phi$ and per-task variances $\boldsymbol\sigma$ is
$$
-\log p(\mathbf y \mid \phi, \boldsymbol\sigma)
\;=\; \sum_i \Big[\frac{(y_i - f_i(s;\phi))^2}{2\sigma_i^2} + \log\sigma_i + \tfrac12\log 2\pi\Big].
$$
Dropping the constant and recognising $(y_i - f_i)^2 \to \mathcal L_i(\phi)$ as the per-task squared error, the joint loss minimised over $(\phi,\boldsymbol\sigma)$ is
$$
\mathcal L_{\mathrm{KGC}}(\phi,\boldsymbol\sigma)
\;=\; \sum_{i=1}^K \frac{1}{2\sigma_i^2}\,\mathcal L_i(\phi) \;+\; \sum_{i=1}^K \log\sigma_i.
\tag{KGC}
$$
The first term is the precision-weighted task loss; the second is the regulariser that prevents the trivial solution $\sigma_i \to \infty$ (which would zero out the contribution of task $i$). Optimising over $\sigma_i$ at fixed $\phi$ gives $\sigma_i^{\star 2} = \mathcal L_i(\phi)$, the maximum-likelihood homoscedastic variance estimate; substituting back recovers $\sum_i\big(\tfrac12 + \tfrac12\log\mathcal L_i\big)$, showing the asymptotic balance is "all task losses contribute the same precision-weighted amount."

For numerical stability we parameterise $\sigma_i = e^{\log\sigma_i}$ via a learnable scalar $\log\sigma_i$; then $\sigma_i^2 = e^{2\log\sigma_i}$ is always positive and the gradient is well-conditioned everywhere. Initialise $\log\sigma_i = 0 \Rightarrow \sigma_i = 1$ for all three critic-side terms; the optimiser adjusts each as training progresses.

### 9.2 Application to the Phase 3F-A critic loss

Substitute (KGC) for the fixed-weight critic loss of Step 1:
$$
\mathcal L_{\mathrm{critic}}(\phi, \log\sigma_{\mathrm{eik}}, \log\sigma_{\mathrm{bc}}, \log\sigma_{\mathrm{ground}})
\;=\; \frac{\mathcal L_{\mathrm{eik}}(\phi)}{2 e^{2\log\sigma_{\mathrm{eik}}}}
\;+\; \frac{\mathcal L_{\mathrm{bc}}(\phi)}{2 e^{2\log\sigma_{\mathrm{bc}}}}
\;+\; \frac{\mathcal L_{\mathrm{ground}}(\phi)}{2 e^{2\log\sigma_{\mathrm{ground}}}}
\;+\; \log\sigma_{\mathrm{eik}} + \log\sigma_{\mathrm{bc}} + \log\sigma_{\mathrm{ground}}.
\tag{4-term-KGC}
$$
The actor-side $\mathcal L_{\mathrm{distill}}$ keeps its fixed weight $w_{\mathrm{distill}}=0.5$ and is added to the *actor's* loss outside (KGC); applying uncertainty weighting to a distillation regulariser combined with the PPO surrogate is not appropriate (their dynamics are governed by different optimisers and clip schedules).

The Step 7B $T_{\max}^2$ normalisation is **removed** in this formulation. Uncertainty weighting subsumes manual scale balancing: $\sigma_{\mathrm{ground}}$ will grow in proportion to the variance of $\mathcal L_{\mathrm{ground}}$, automatically downweighting it when its squared error is naturally large (which is the case at training start when $T_\phi$ is far from $T_{\mathrm{obs}}$).

### 9.3 Stop-gradient property is preserved

The actor-side $\mathcal L_{\mathrm{distill}}$ continues to use $\mathrm{stop\_grad}(T_\phi)$ inside $A_{\mathrm{eik}}$ as derived in §4 and §6.3. Adding $\log\sigma_i$ parameters to the critic-side loss does not couple the actor's gradient to $T_\phi$. Specifically:
$$
\frac{\partial \mathcal L_{\mathrm{aux}}}{\partial \phi}
\;=\; \frac{1}{2 e^{2\log\sigma_{\mathrm{eik}}}}\frac{\partial \mathcal L_{\mathrm{eik}}}{\partial\phi}
+ \frac{1}{2 e^{2\log\sigma_{\mathrm{bc}}}}\frac{\partial \mathcal L_{\mathrm{bc}}}{\partial\phi}
+ \frac{1}{2 e^{2\log\sigma_{\mathrm{ground}}}}\frac{\partial \mathcal L_{\mathrm{ground}}}{\partial\phi}
$$
$$
\frac{\partial \mathcal L_{\mathrm{aux}}}{\partial \log\sigma_i}
\;=\; -\frac{\mathcal L_i(\phi)}{e^{2\log\sigma_i}} + 1
\quad\Longrightarrow\quad
\log\sigma_i^\star \;=\; \tfrac12 \log\mathcal L_i(\phi)
$$
$$
\frac{\partial \mathcal L_{\mathrm{aux}}}{\partial \theta}
\;=\; w_{\mathrm{distill}}\,\frac{\partial \mathcal L_{\mathrm{distill}}}{\partial\theta}.
$$
The actor's gradient depends only on $\theta$ (and indirectly on the *current* values of $T_\phi$ via the detached $A_{\mathrm{eik}}$ — but those values are constants from $\theta$'s perspective). The critic's gradient depends only on $\phi$ and $\log\sigma_i$. The two-network decomposition of §6.3 is intact.

### 9.4 Terminal replay buffer for sparse-data PINN training

The supervised-grounding loss $\mathcal L_{\mathrm{ground}}$ (cf. §6.1) and the boundary-condition loss $\mathcal L_{\mathrm{bc}}$ (cf. §3.2) both estimate expectations over the terminal/observed-state distribution:
$$
\mathcal L_{\mathrm{bc}}(\phi)
\;=\; \mathbb E_{s\sim\partial\mathcal G\cup\partial\mathcal G_{\mathrm{coll}}}\big[(T_\phi(s)-g(s))^2\big],
\qquad
\mathcal L_{\mathrm{ground}}(\phi)
\;=\; \mathbb E_{(s,T_{\mathrm{obs}})\sim\mathcal D_{\mathrm{ground}}}\big[(T_\phi(s)-T_{\mathrm{obs}})^2\big].
$$
An empirical Monte-Carlo estimator using only the *current* iteration's rollout produces a **biased estimate** when the current rollout has zero terminal events of the relevant type — the BC contribution to the gradient is exactly zero on those iterations, leaving the high-T region of the state space *unobserved* during the update.

The cure is a moving-window Monte-Carlo estimator that augments the current iteration's data with samples from past iterations. Specifically, maintain three FIFO ring buffers (capacity $N=1000$ each):
$$
\mathcal B_{\mathrm{succ}}, \mathcal B_{\mathrm{coll}}, \mathcal B_{\mathrm{intermediate}}
$$
that store $(s_t, T_{\mathrm{obs}})$ pairs from the last $N$ terminal events of each type. At each iteration we (a) append new terminal events from the current rollout, evicting oldest when capacity is reached; (b) sample mini-batches of size $B$ uniformly from each buffer to estimate the BC and ground losses.

This is the standard technique in DeepXDE (Lu, Meng, Mao, Karniadakis, *SIAM Review* 2021) for sparse-data PDE problems where boundary conditions are observed at a small number of specific points. The estimator is consistent in the limit $N\to\infty$ and biased only by $O(N^{-1})$ for finite buffer; for our use the bias is negligible because the underlying terminal distribution evolves slowly across iterations (the policy changes only modestly between consecutive PPO updates).

The replay buffer changes **only the sample distribution** from which $\mathcal L_{\mathrm{bc}}$ and $\mathcal L_{\mathrm{ground}}$ are estimated; the loss formulations themselves are unchanged. The four-term decomposition, mutual-consistency claim of §6.2, and stop-gradient property of §6.3 all remain intact.

### 9.5 Why both interventions preserve the math derivation

The math of §1–§8 derives the loss *forms* — the PDE residual $\rho^2$, the boundary anchors, the observed-time MSE, the soft-KL distillation. Step 7C changes (a) the *relative weights* applied to those forms (uncertainty weighting replaces fixed weights) and (b) the *empirical estimator* used to compute the expectations (replay buffer replaces single-iteration Monte-Carlo). Neither is a change to the loss form itself. Both are standard PINN/multi-task-learning engineering practices with peer-reviewed citations. The fixed point $T^\star$ that simultaneously minimises all four losses is unchanged: $T^\star$ has $\mathcal L_{\mathrm{eik}}(T^\star)=\mathcal L_{\mathrm{bc}}(T^\star)=\mathcal L_{\mathrm{ground}}(T^\star)=0$, and the optimal $\sigma_i^\star=0$ in that limit (with the $\log\sigma_i$ regulariser becoming $-\infty$, so any clipped/regularised lower bound suffices).

### 9.6 References (Section 9 additions)

- Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018. — homoscedastic uncertainty weighting for multi-task learning.
- Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). *DeepXDE: A deep learning library for solving differential equations.* *SIAM Review* 63(1), 208–228. — boundary-data resampling and replay-style estimators for sparse-data PINN training.

---

**Status:** *Step 1 (Sections 1–8) and Step 7C update (Section 9) complete.* Ready for human review.

---

## 10. Augmented Lagrangian for the Eikonal constraint (Step 7D)

Step 7C empirically confirmed a structural mismatch between the Kendall–Gal–Cipolla (KGC) homoscedastic uncertainty weighting (§9.1–9.3) and the Eikonal residual loss $\mathcal L_{\mathrm{eik}}$. The mechanism downweighted $\mathcal L_{\mathrm{eik}}$ during training ($\sigma_{\mathrm{eik}}: 1.08\to 2.21$ in 1a; $1.07\to 2.71$ in 2_dense), causing the residual to *grow* rather than shrink. This section diagnoses the failure mode and derives the canonical fix: treat $\mathcal L_{\mathrm{eik}}$ as a *constraint*, not a noisy supervised task, and enforce it via the Augmented Lagrangian Method (ALM; Hestenes 1969; Powell 1969; Bertsekas 1996; applied to PINNs in Lu et al. 2021).

### 10.1 Why KGC fails for $\mathcal L_{\mathrm{eik}}$

The KGC derivation (§9.1) starts from the *probabilistic* assumption that each task's loss arises from a noisy observation model
$$
y_i \;=\; f_i(s;\phi) + \varepsilon_i,\qquad \varepsilon_i \sim \mathcal N(0,\sigma_i^2),
$$
so that $\mathcal L_i$ is the negative log-likelihood of *observed* data under task-specific homoscedastic Gaussian noise. The optimal weight $\sigma_i^{\star 2} = \mathcal L_i(\phi^\star)$ is the *empirical noise variance* at the optimum: large $\sigma_i$ means task $i$ has high inherent noise, so its precision-weighted contribution should be small.

This story is correct for $\mathcal L_{\mathrm{bc}}$ and $\mathcal L_{\mathrm{ground}}$:
- $\mathcal L_{\mathrm{bc}}$ regresses $T_\phi(s_{\mathrm{succ}})\to 0$ and $T_\phi(s_{\mathrm{coll}})\to T_{\max}$. The targets are *fixed* and the residual at any $\phi$ has interpretable variance-of-prediction across the batch of terminal states.
- $\mathcal L_{\mathrm{ground}}$ regresses $T_\phi(s_t)\to T_{\mathrm{obs}}(s_t)=K-t$. The observed time-of-arrival from on-policy trajectories is *itself* a noisy estimate of the true expected time-to-arrival (depends on stochastic actor / env outcomes), so per-state residuals have legitimate Gaussian-like noise.

But $\mathcal L_{\mathrm{eik}} = \mathbb E[(\|\nabla T_\phi\|^2 - c(\xi)^2)^2]$ is *not* a noisy observation. The PDE constraint $\|\nabla T\|^2 = c^2$ is a *deterministic equation* that the true solution $T^\star$ satisfies pointwise; the residual at $T^\star$ is *exactly zero*, not "Gaussian noise around zero." The KGC update rule treats large $\mathcal L_{\mathrm{eik}}$ as evidence of high task noise and downweights it, which is exactly the opposite of what is needed: the residual should be *enforced* toward zero, not *modelled* as noise.

This is a known failure mode of multi-task uncertainty weighting when applied to mixed soft-target / hard-constraint problems (Sener & Koltun 2018, §4 discussion; Wang et al. 2022 specifically for PINNs). The remedy is to enforce $\mathcal L_{\mathrm{eik}}$ as a constraint via Lagrangian methods.

### 10.2 The Augmented Lagrangian formulation

Reformulate the critic-side loss as a constrained problem:
$$
\min_\phi \big[\mathcal L_{\mathrm{ground}}(\phi) + \mathcal L_{\mathrm{bc}}(\phi)\big]
\qquad \text{subject to}\qquad
\mathcal L_{\mathrm{eik}}(\phi) \;=\; 0.
\tag{C}
$$
The classical penalty method approximates (C) by $\min_\phi [f(\phi) + \mu g(\phi)^2]$ with $\mu\to\infty$, but pure penalty methods are ill-conditioned for large $\mu$. The Augmented Lagrangian, due to Hestenes (1969) and Powell (1969), instead introduces a Lagrange multiplier $\lambda$:
$$
\mathcal L_{\mathrm{aug}}(\phi;\lambda,\mu)
\;:=\; f(\phi) + \lambda\,g(\phi) + \frac{\mu}{2}\,g(\phi)^2,
\tag{ALM}
$$
where $f := \mathcal L_{\mathrm{bc}}+\mathcal L_{\mathrm{ground}}$ and $g := \mathcal L_{\mathrm{eik}}$. The dual ascent / penalty schedule alternates:
1. **Primal:** $\phi_{k+1} = \arg\min_\phi \mathcal L_{\mathrm{aug}}(\phi; \lambda_k, \mu_k)$ — performed by SGD over multiple inner steps.
2. **Multiplier update (dual ascent):** $\lambda_{k+1} = \lambda_k + \mu_k\,g(\phi_{k+1})$.
3. **Penalty update (Bertsekas 1996, §4.2):**
$$
\mu_{k+1} \;=\; \begin{cases}
\beta\,\mu_k & \text{if } g(\phi_{k+1}) > \alpha\,g(\phi_k)\\
\mu_k & \text{otherwise}
\end{cases}
\quad
\alpha\in(0,1),\ \beta>1.
$$

The penalty $\mu$ grows by factor $\beta=5$ when the constraint residual fails to decrease by factor $\alpha=0.25$ (i.e. $g_{k+1} > 0.25\,g_k$). This is the canonical update from Bertsekas 1996 Chapter 4.

**Why the ALM works where pure penalty fails.** With $\lambda$ updated by dual ascent, the augmented Lagrangian's stationarity condition $\nabla_\phi f + (\lambda + \mu g)\nabla_\phi g = 0$ converges to KKT $\nabla_\phi f + \lambda^\star \nabla_\phi g = 0$ at the constraint boundary $g(\phi^\star)=0$, *without* requiring $\mu\to\infty$. The numerical conditioning of the inner problem stays bounded; the constraint is satisfied exactly in the limit (Bertsekas 1996, Theorem 4.5).

### 10.3 The Step 7D hybrid critic loss

Combining ALM for the constraint with KGC for the noisy supervised tasks gives the Phase 3F-A Step 7D auxiliary loss:
$$
\mathcal L_{\mathrm{aux}}(\phi, \log\sigma_{\mathrm{bc}}, \log\sigma_{\mathrm{ground}}; \theta)
\;=\;
\underbrace{\lambda\,\mathcal L_{\mathrm{eik}}(\phi) + \frac{\mu}{2}\,\mathcal L_{\mathrm{eik}}^2(\phi)}_{\text{ALM (constraint)}}
$$
$$
\;+\;\underbrace{\frac{\mathcal L_{\mathrm{bc}}(\phi)}{2 e^{2\log\sigma_{\mathrm{bc}}}}
+ \frac{\mathcal L_{\mathrm{ground}}(\phi)}{2 e^{2\log\sigma_{\mathrm{ground}}}}
+ \log\sigma_{\mathrm{bc}} + \log\sigma_{\mathrm{ground}}}_{\text{KGC (noisy supervised)}}
\;+\;\underbrace{w_{\mathrm{distill}}\,\mathcal L_{\mathrm{distill}}\big(\theta;\,\mathrm{stop\_grad}(T_\phi)\big)}_{\text{actor-side (unchanged)}}.
\tag{4-term-7D}
$$
Each term is justified separately:
- **ALM** for $\mathcal L_{\mathrm{eik}}$: the residual is a hard constraint, not a noisy observation; ALM is the canonical method.
- **KGC** for $\mathcal L_{\mathrm{bc}}, \mathcal L_{\mathrm{ground}}$: these *are* noisy supervised regressions; KGC's assumptions hold; retain Step 7C's adaptive weighting.
- **Stop-grad** $\mathcal L_{\mathrm{distill}}$: actor-side soft-KL distillation; preserves the two-network decomposition of §6.3.

### 10.4 Application to PINNs and prior work

ALM-based PINN training is established practice when the PDE residual must be driven to zero rather than just made small. Lu et al. 2021 ("Physics-informed neural networks with hard constraints for inverse design", *SIAM J. Sci. Comput.*) apply ALM to PINNs for inverse-design problems with hard PDE constraints; their Algorithm 1 is essentially the formulation above. McClenny & Braga-Neto 2020 ("Self-Adaptive Physics-Informed Neural Networks via a Soft Attention Mechanism") use a related self-adaptive λ for boundary residuals. Wang, Yu, Perdikaris 2022 ("When and why PINNs fail to train: A neural tangent kernel perspective", *J. Comp. Phys.*) document the same KGC-failure mode and recommend constraint-aware reweighting.

The convergence theory (Bertsekas 1996, Theorem 4.5) guarantees that under mild assumptions (continuous gradients, bounded constraint set), the sequence $(\phi_k, \lambda_k)$ converges to a KKT point of (C) provided $\mu_k$ does not need to grow unbounded. If $\mu_k\to\mu_{\max}$ without convergence, the constraint is *infeasible at the current network capacity* — a documented phenomenon (Wang et al. 2022) handled empirically by capacity increase (E.1 in Step 7D's Decision E).

### 10.5 Why this preserves all earlier derivations

The PDE residual $\rho = \|\nabla T\|^2 - c^2$ is unchanged (§1, §2). The boundary conditions $T(s_{\mathrm{succ}})=0$, $T(s_{\mathrm{coll}})=T_{\max}$ are unchanged (§3). The supervised grounding $T_{\mathrm{obs}}(s_t)=K-t$ is unchanged (§6.1). The advantage $A_{\mathrm{eik}}=T_\phi(s)-T_\phi(f_a(s))$ and the soft-KL distillation are unchanged (§4, §5). The mutual-consistency property (§6.2) — that $\mathcal L_{\mathrm{eik}}, \mathcal L_{\mathrm{bc}}, \mathcal L_{\mathrm{ground}}$ all share the fixed point $T^\star$ — is preserved. The stop-gradient on $T_\phi$ in $\mathcal L_{\mathrm{distill}}$ (§6.3) is preserved. The PINN paradigm (§7) is unchanged.

The change is *only* how the three critic-side loss components are aggregated into a single scalar for SGD. Step 7C used KGC for all three; Step 7D uses ALM for the constraint and KGC for the two supervised tasks. The fixed point of the joint loss is the same.

### 10.6 References (Section 10 additions)

- Hestenes, M. R. (1969). *Multiplier and gradient methods.* Journal of Optimization Theory and Applications 4(5):303–320. — original ALM paper.
- Powell, M. J. D. (1969). *A method for nonlinear constraints in minimization problems.* In *Optimization* (R. Fletcher, ed.), Academic Press. — concurrent ALM development.
- Bertsekas, D. P. (1996). *Constrained Optimization and Lagrange Multiplier Methods.* Athena Scientific (reprint of 1982 Academic Press edition). — canonical ALM reference; Chapter 4 covers the augmented Lagrangian and its update rule.
- Lu, L., Pestourie, R., Yao, W., Wang, Z., Verdugo, F., & Johnson, S. G. (2021). *Physics-informed neural networks with hard constraints for inverse design.* *SIAM Journal on Scientific Computing* 43(6):B1105–B1132. — ALM applied to PINNs.
- McClenny, L. D., & Braga-Neto, U. M. (2020). *Self-Adaptive Physics-Informed Neural Networks via a Soft Attention Mechanism.* arXiv:2009.04544.
- Sener, O., & Koltun, V. (2018). *Multi-Task Learning as Multi-Objective Optimization.* NeurIPS 2018. — KGC-style weighting limitations and alternatives.
- Wang, S., Yu, X., & Perdikaris, P. (2022). *When and why PINNs fail to train: A neural tangent kernel perspective.* *Journal of Computational Physics* 449:110768. — empirical evidence for capacity-bounded constraint feasibility in PINNs.

---

**Status:** *Sections 1–8 (Step 1), §9 (Step 7C update), §10 (Step 7D update) complete.* The Step 7D ALM hybrid is the final principled engineering iteration on the loss formulation.
