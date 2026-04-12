# PDE auxiliary critics (HJB / Soft-HJB) and SUMO environment updates

This document explains the **research idea**, **mathematics**, **what was added or changed** in the codebase, and how **artifacts** are laid out. It complements `docs/RUNNING.md`, which remains the general command reference for legacy DRPPO / PINN ablations.

---

## 1. Design goals

- **Legacy path preserved:** Training, evaluation, and ablations under `experiments/run_train.py`, `experiments/run_eval.py`, `experiments/run_ablation.py`, and `results/ablation/` are unchanged in purpose. Legacy checkpoints and CSVs are not overwritten by the new pipelines unless you point outputs at the same paths.
- **New path (PDE family):** HJB-residual and Soft-HJB **auxiliary critics** live under `models/pde/` and `experiments/pde/`, with configs in `configs/pde/`. Checkpoints and CSVs default to `results/pde/` and `results/pde_ablation/`.
- **Ego remains the RL decision-maker:** The recurrent actor (`RecurrentActorCritic` in `models/drppo.py`) still chooses actions. The auxiliary network \(U(\xi)\) shapes learning via PDE residuals, value anchoring, terminal boundary conditions, PPO critic distillation, and (for Soft-HJB) an actor–soft-policy KL term.

---

## 2. Conceptual idea

### 2.1 From scalar “physics penalties” to a PDE viewpoint

The original PINN-style loss in this repo applied **scalar constraint terms** on the PPO critic (e.g., coupling value with TTC or stopping-distance violations). The new methods treat a **reduced state** \(\xi\) as the spatial variable for a **Bellman / HJB-style** consistency condition:

- A separate network \(U_\theta(\xi)\) (the **auxiliary critic**) is trained so that a **residual** \(\rho(\xi)\), built from \(U\), local dynamics, and local rewards, is driven toward zero on collocation points.
- The **PPO value head** is still trained with TD/GAE targets but receives an extra **distillation** term toward \(U_\theta(\xi)\), coupling the simulator-grounded representation to the PDE-side value.

This is **not** a full continuous-time boundary-value solver; it is a **practical PINN-style surrogate** on a low-dimensional \(\xi\) with discrete-time local dynamics \(f_a(\xi)\).

### 2.2 Reduced state \(\xi\) (dimension 79)

`ReducedPDEState` in `models/pde/state_builder.py` maps the usual `StateBuilder` output plus `info` into `XI_DIM = 79` floats:

- Ego / path: speed, accel, yaw rate, stopping distance, distances to conflict zone and exit, curvature proxy, `ttc_min`.
- Visibility block: `alpha_cz`, `alpha_cross`, `d_occ`, `dt_seen`.
- Up to **three** agents, **22 features** each (positions, relative velocities, type, lane distances, conflict timing, TTC-related scalars, mask).
- Pothole distance from `info["raw_obs"]["d_pothole"]`.

Constants that are less informative for a local PDE (e.g. fixed lane width, fixed turn indicator) are intentionally omitted from \(\xi\) to keep the state focused on negotiated geometry and traffic.

### 2.3 Local dynamics \(f_a(\xi)\)

`BehavioralDynamics` in `models/pde/dynamics.py` implements a **differentiable one-step map** \(\xi' = \xi + f_a(\xi)\,\Delta t\) (implemented as `one_step`), with nominal longitudinal accelerations per discrete action:

| Action | Intent (nominal) |
|--------|-------------------|
| STOP (0) | Strong deceleration (`a_brake`) |
| CREEP (1) | Toward creep speed |
| YIELD (2) | Mild deceleration |
| GO (3) | Moderate acceleration |
| ABORT (4) | Stronger deceleration (`a_abort`) |

The map updates ego kinematic scalars, distance-to-go fields, agent-relative features, and recomputes derived quantities (e.g. TTC) consistently inside the module. This is a **model-based scaffold** for the PDE losses, aligned with the SUMO action semantics at a high level, not a micro-simulator clone.

### 2.4 Local reward \(r(\xi, a)\)

`local_reward` in `models/pde/local_reward.py` defines a **surrogate** one-step reward:

- Progress term from average speed over \(\Delta t\).
- Small time penalty per step.
- Penalty when **post-step** `ttc_min` falls below a threshold.
- Penalty when **post-step** pothole distance is below a threshold.

**Collision costs** are **not** folded into \(r\); they appear in **terminal boundary conditions** on \(U\) (see below), matching the idea that collisions are catastrophic terminal events.

---

## 3. Mathematics

### 3.1 Auxiliary value and Q-type quantities

Let \(U_\theta: \mathbb{R}^{79} \to \mathbb{R}\) be the auxiliary critic. For batch inputs \(\xi\), autograd gives \(\nabla_\xi U_\theta(\xi)\).

For each discrete action \(a \in \{0,\dots,4\}\), define:

\[
Q_a(\xi) = r(\xi, a) + \nabla_\xi U_\theta(\xi)^\top f_a(\xi)
\]

where \(f_a(\xi)\) is the drift in `BehavioralDynamics.drift` (the continuous-time spirit is “advection” of the value along the mode-dependent vector field). Implementations: `pde_q_values` in `models/pde/residuals.py`.

### 3.2 Discounted hard-HJB residual

With discount \(\gamma \in (0,1)\), the **hard** residual is:

\[
\rho_{\mathrm{HJB}}(\xi) = U_\theta(\xi)\,\log\gamma + \max_{a} Q_a(\xi)
\]

At an optimum, one expects \(\rho_{\mathrm{HJB}}(\xi) \approx 0\) on the support of the state distribution (under idealized assumptions). Training minimizes \(\mathbb{E}[\rho_{\mathrm{HJB}}^2]\) on collocation samples. Implemented in `hjb_residual`.

### 3.3 Soft-HJB (entropy-smoothed) residual

With temperature \(\tau > 0\):

\[
\rho_{\mathrm{soft}}(\xi) = U_\theta(\xi)\,\log\gamma + \tau \log \sum_a \exp\left(\frac{Q_a(\xi)}{\tau}\right)
\]

This replaces the hard max by a **log-sum-exp**, smoothing the effective “Bellman backup” and yielding an implied **soft policy** \(\pi_{\mathrm{soft}}(a|\xi) \propto \exp(Q_a/\tau)\). Implemented in `soft_hjb_residual` and `soft_policy_from_q`.

### 3.4 Auxiliary loss (shared structure)

On each PPO minibatch with valid PDE extras, the auxiliary parameters \(\theta\) are updated with:

\[
\mathcal{L}_{\mathrm{aux}} =
\lambda_{\mathrm{anchor}}\,\mathcal{L}_{\mathrm{anchor}}
+ \lambda_{\mathrm{pde}}\,\mathcal{L}_{\mathrm{pde}}
+ \lambda_{\mathrm{bc}}\,\mathcal{L}_{\mathrm{bc}}
\]

- **Anchor:** \(\mathcal{L}_{\mathrm{anchor}} = \|U_\theta(\xi) - G\|^2\) where \(G\) are **MC returns** from the rollout (same targets scale as the PPO critic sees).
- **PDE:** \(\mathcal{L}_{\mathrm{pde}} = \mathbb{E}[\rho^2]\) with \(\rho = \rho_{\mathrm{HJB}}\) or \(\rho_{\mathrm{soft}}\).
- **Boundary / terminal (BC):**
  - Successful terminal states: push \(U_\theta \to 0\).
  - Collision terminal states: push \(U_\theta \to w_{\mathrm{coll}}\) (large negative scalar, default \(-20\)).

Coefficients \(\lambda_*\) come from `configs/pde/hjb_aux.yaml` or `configs/pde/soft_hjb_aux.yaml`.

### 3.5 Distillation onto the PPO critic

Let \(V_\phi\) be the scalar value head. An extra term is added to the PPO value loss:

\[
\mathcal{L}_{\mathrm{vf}} \;\leftarrow\; \mathcal{L}_{\mathrm{vf}} + \lambda_{\mathrm{distill}}\,\|V_\phi - \mathrm{sg}(U_\theta)\|^2
\]

(`sg` = stop-gradient on \(U_\theta\).)

This keeps **policy gradients** flowing through the standard PPO objective while **transferring** PDE-regularized value structure into the main critic.

### 3.6 Soft-HJB only: actor–soft-policy alignment

For Soft-HJB, after updating the auxiliary critic, form \(\pi_{\mathrm{soft}}\) from \(Q_a\) and compare to the policy logits \(\pi_\theta\). A **KL**-style term (implemented as \(\sum_a \pi_{\mathrm{soft}} (\log \pi_{\mathrm{soft}} - \log \pi_\theta)\)) is added to the **actor** loss with weight `lambda_align` in `configs/pde/soft_hjb_aux.yaml`. This nudges the discrete actor toward the entropy-regularized backup implied by \(U\).

### 3.7 Collocation

`sample_collocation` in `models/pde/collocation.py` builds a batch mixing:

- A fraction `collocation_ratio` of **real** \(\xi\) from rollouts, and  
- The rest as **jittered** copies with noise on **primitive** channels only; derived agent scalars (`tau`, TTC-related terms) are **recomputed** for consistency.

---

## 4. Implementation map (new files)

| Path | Role |
|------|------|
| `models/pde/__init__.py` | Package |
| `models/pde/state_builder.py` | `ReducedPDEState`, `XI_DIM` |
| `models/pde/dynamics.py` | `BehavioralDynamics` |
| `models/pde/local_reward.py` | `local_reward` |
| `models/pde/collocation.py` | `sample_collocation` |
| `models/pde/residuals.py` | `pde_q_values`, `hjb_residual`, `soft_hjb_residual`, `soft_policy_from_q` |
| `models/pde/checkpointing.py` | `save_pde_checkpoint` / `load_pde_checkpoint` |
| `models/pde/hjb_aux_critic.py` | MLP for \(U\) (HJB path) |
| `models/pde/soft_hjb_aux_critic.py` | MLP for \(U\) (Soft-HJB path; separate class for clarity) |
| `models/pde/hjb_aux_agent.py` | `HJBAuxAgent`: PPO + HJB auxiliary + distill |
| `models/pde/soft_hjb_aux_agent.py` | `SoftHJBAuxAgent`: + soft residual + actor KL |
| `experiments/pde/collect_rollouts.py` | Rollouts with `xi_curr`, terminal flags for PDE |
| `experiments/pde/train_hjb_aux.py` | Single-run HJB training CLI |
| `experiments/pde/train_soft_hjb_aux.py` | Single-run Soft-HJB training CLI |
| `experiments/pde/eval.py` | Multi-seed eval for PDE checkpoints |
| `experiments/pde/run_ablation.py` | Grid over scenarios × variants × \(\lambda\) × seeds |
| `experiments/pde/visualize_sumo.py` | SUMO GUI / CSV logs for PDE checkpoints |
| `experiments/pde/plot_pde.py` | Training and ablation figures + legacy comparison hooks |
| `experiments/pde/plot_interaction.py` | Trajectory / interaction analysis from CSV |
| `configs/pde/hjb_aux.yaml` | HJB hyperparameters |
| `configs/pde/soft_hjb_aux.yaml` | Soft-HJB hyperparameters |

---

## 5. Implementation map (modified files — environment & credibility)

These changes **apply to all** trainers (legacy and PDE) that use `SumoEnv`:

| Area | File(s) | Summary |
|------|---------|---------|
| Scenario layout | `configs/scenario/default.yaml`, `scenario/generator.py` | Smaller stem/bar lengths; optional tighter conflict geometry; ground polygon scaling |
| Spawning / behavior | `scenario/behavior_sampler.py` | Tighter `depart_time` windows; `depart_pos` for vehicles; pothole bias toward ego path |
| SUMO control & negotiation | `env/sumo_env.py` | Ego `setSpeedMode` enabling safety/negotiation; `slowDown`-based actions with distinct decels for STOP vs ABORT; shorter warmup; **real** `d_cz_i`, `d_exit_i`, pedestrian heading, dynamic visibility and per-agent uncertainty, ROW-aware `rho`, dynamic `e_y` / `e_psi` |
| Legacy visualizer | `experiments/run_visualize_sumo.py` | Default `--policy` is `checkpoint`; **requires** `--checkpoint` for that mode; clear error message |
| Dashboard | `experiments/dashboard.py` | **PDE Methods** tab: curves, ablation summary, PDE vs legacy comparison, interaction viewer |
| Automation | `Makefile` | Targets: `train-hjb-aux`, `train-soft-hjb-aux`, `eval-pde`, `ablation-pde`, `plot-pde`, `visualize-pde-gui`, `plot-interaction`, `regen-scenarios` |

---

## 6. Outputs and naming

### 6.1 Single training runs (default `results/pde/`)

| Method | Training log | Checkpoint |
|--------|----------------|------------|
| HJB | `train_hjb_aux_{scenario}.csv` | `model_hjb_aux_{scenario}.pt` |
| Soft-HJB | `train_soft_hjb_aux_{scenario}.csv` | `model_soft_hjb_aux_{scenario}.pt` |

### 6.2 Ablation (`results/pde_ablation/` by default)

- `ablation_results.csv` — eval metrics per scenario, variant, `lambda_hjb`, seed, eval mode  
- `ablation_train_log.csv` — training diagnostics (PDE residuals, distill, KL for soft)  
- Checkpoints: `ablation_{scenario}_{variant}_lh{lambda}_s{seed}.pt`

Legacy ablation remains under `results/ablation/` when you use `experiments/run_ablation.py` or `make ablation`.

---

## 7. How this relates to legacy “PINN”

| Aspect | Legacy PINN (`models/drppo.py` / residuals config) | New PDE path |
|--------|-----------------------------------------------------|--------------|
| **Where “physics” lives** | Extra terms on **PPO critic** / actor depending on `pinn_placement` | Separate **\(U_\theta(\xi)\)** + residual on \(\xi\) |
| **Equation** | Hand-designed violation scalars | HJB / soft-HJB **residual** with **dynamics** \(f_a\) and **local reward** |
| **Training scripts** | `experiments/run_train.py`, `run_ablation.py` | `experiments/pde/train_*.py`, `run_ablation.py` |
| **Checkpoints** | e.g. `results/model_pinn_*.pt`, `results/ablation/*.pt` | `results/pde/*.pt`, `results/pde_ablation/*.pt` |

You can compare empirically via the dashboard’s PDE tab and `experiments/pde/plot_pde.py` (legacy dir defaults to `results/ablation`).

---

## 8. References in code

- PPO + auxiliary step (HJB): ```164:176:models/pde/hjb_aux_agent.py
            aux_loss = (self.lambda_anchor * L_anchor +
                        self.lambda_hjb * L_hjb +
                        self.lambda_bc * L_bc)
...
            L_distill = F.mse_loss(value[:len(U_distill)], U_distill)
            vf_loss = vf_loss + self.lambda_distill * L_distill
```
- Residual definitions: ```63:95:models/pde/residuals.py
def hjb_residual(
...
    rho = U_val * math.log(gamma) + max_q
    return rho
...
def soft_hjb_residual(
...
    rho = U_val * math.log(gamma) + soft_max
    return rho
```

---

## 9. Quick start: shell commands

Run everything from the repository root (the folder that contains `Makefile`), with your virtual environment activated.

```bash
cd /path/to/EECE_499    # repository root (folder containing Makefile)
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export SUMO_HOME=/usr/share/sumo   # typical Linux install; adjust if needed
```

**First-time setup (if you have not already):**

```bash
make setup
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export SUMO_HOME=/usr/share/sumo
```

**Optional — regenerate SUMO scenario folders** (after changing `configs/scenario/default.yaml` or generator code):

```bash
make regen-scenarios
```

### 9.1 Train PDE methods (single scenario)

HJB auxiliary critic (default scenario `1a`, 50k steps):

```bash
make train-hjb-aux
```

Soft-HJB auxiliary critic:

```bash
make train-soft-hjb-aux
```

Equivalent explicit invocations:

```bash
python3 experiments/pde/train_hjb_aux.py --scenario 1a --total_steps 50000 --out_dir results/pde
python3 experiments/pde/train_soft_hjb_aux.py --scenario 1a --total_steps 50000 --out_dir results/pde
```

Useful flags: `--config`, `--algo_config`, `--seed`, `--use_intent`, `--sumo_gui` (slow; debugging only).

**Outputs:** `results/pde/model_hjb_aux_1a.pt`, `results/pde/model_soft_hjb_aux_1a.pt`, and CSV training logs in the same folder.

### 9.2 Evaluate a PDE checkpoint

```bash
make eval-pde
```

This runs the Makefile example (HJB, scenario `1a`). Customize:

```bash
python3 experiments/pde/eval.py \
  --checkpoint results/pde/model_hjb_aux_1a.pt \
  --method hjb_aux \
  --scenario 1a \
  --episodes 50 \
  --seeds 42 123 \
  --out_dir results/pde
```

For Soft-HJB, set `--method soft_hjb_aux` and point `--checkpoint` to `model_soft_hjb_aux_*.pt`.

### 9.3 PDE ablation grid (multi-scenario, multi-seed)

Writes to **`results/pde_ablation/`** (does not touch `results/ablation/`):

```bash
make ablation-pde
```

Customize:

```bash
python3 experiments/pde/run_ablation.py \
  --out_dir results/pde_ablation \
  --total_steps 50000 \
  --eval_episodes 50 \
  --scenarios 1a 1b \
  --variants hjb_aux soft_hjb_aux \
  --lambda_hjb 0.1 0.2 \
  --seeds 42 123
```

### 9.4 Plots (PDE + optional legacy comparison)

```bash
make plot-pde
```

Requires `results/pde_ablation/ablation_train_log.csv` (and optionally `results/ablation/` for legacy comparison). Override:

```bash
python3 experiments/pde/plot_pde.py \
  --pde_dir results/pde_ablation \
  --legacy_dir results/ablation \
  --out_dir results/pde_ablation
```

### 9.5 SUMO GUI: watch a PDE-trained policy

```bash
make visualize-pde-gui
```

Customize:

```bash
python3 experiments/pde/visualize_sumo.py \
  --gui --episodes 1 --scenario 1a \
  --method hjb_aux \
  --checkpoint results/pde/model_hjb_aux_1a.pt \
  --out_dir results/pde
```

### 9.6 Interaction plots from a trajectory CSV

After producing a trajectory (e.g. from `visualize_sumo.py`):

```bash
make plot-interaction
```

Or pass a specific CSV:

```bash
python3 experiments/pde/plot_interaction.py \
  --csv results/pde/pde_trajectory_hjb_aux_ep0.csv \
  --out_dir results/pde
```

### 9.7 Dashboard (legacy + PDE tab)

```bash
make dashboard
```

Open `http://localhost:8501` and use the **PDE Methods** tab for training curves, ablation summaries, and comparisons.

### 9.8 Legacy DRPPO / PINN (unchanged entry points)

```bash
make train
make eval
make ablation
make visualize-gui
```

**Legacy visualization with a checkpoint** (default policy is `checkpoint`; you must pass `--checkpoint`):

```bash
python3 experiments/run_visualize_sumo.py --gui \
  --policy checkpoint \
  --checkpoint results/model_pinn_1a.pt \
  --episodes 3 --scenario 1a --out_dir results
```

---

## 10. Version note

This document describes the repository state that includes the PDE package and SUMO interaction/credibility updates listed in Section 5. Legacy workflows are documented in `docs/RUNNING.md`.
