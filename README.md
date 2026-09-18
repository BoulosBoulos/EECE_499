# Physics-Informed Auxiliary Critics for Behavioral Decision-Making at Unstructured Intersections

**Author:** Boulos Boulos · **Supervisor:** Prof. Naseem Daher
**Institution:** American University of Beirut, Department of Electrical and Computer Engineering
**Paper:** submitted to IEEE Transactions on Intelligent Vehicles (T-IV)
**Provenance commit:** `68aaf0627329e186e7300d17b20406d64749acc5`

This repository contains the code, configuration, experiment orchestration and
analysis outputs behind the paper. Every number reported in the paper is
computed from `results/tables/` at the commit above, and section 4 maps each
claim to the file it comes from.

---

## Contents

1. [What this project asks, and what it found](#1-what-this-project-asks-and-what-it-found)
2. [Quick start: reproduce a number from the paper in two minutes](#2-quick-start-reproduce-a-number-from-the-paper-in-two-minutes)
3. [Repository layout](#3-repository-layout)
4. [Results provenance: paper claim to data file](#4-results-provenance-paper-claim-to-data-file)
5. [Method](#5-method)
6. [Environment](#6-environment)
7. [Observation, action and reward](#7-observation-action-and-reward)
8. [The five physics-informed methods](#8-the-five-physics-informed-methods)
9. [PDE infrastructure](#9-pde-infrastructure)
10. [Experiment design](#10-experiment-design)
11. [Statistical methodology](#11-statistical-methodology)
12. [Configuration and the config lock](#12-configuration-and-the-config-lock)
13. [Installation](#13-installation)
14. [Running experiments](#14-running-experiments)
15. [Analysis pipeline](#15-analysis-pipeline)
16. [Known gaps and coverage limits](#16-known-gaps-and-coverage-limits)
17. [Citation and licence](#17-citation-and-licence)

---

## 1. What this project asks, and what it found

The question is narrow and falsifiable: **does the choice of partial
differential equation used to regularize a value function change how well a
reinforcement-learning agent negotiates an unstructured intersection, and if
so, how?**

A recurrent PPO agent (single-layer GRU, hidden width 256) selects among five
discrete behavioral actions (STOP, CREEP, YIELD, GO, ABORT) at an unstructured
T-intersection in SUMO, under partial observability from corner-building
occlusion. Alongside the policy critic $V_\psi$, a second **auxiliary critic**
$U_\phi$ is trained on a PDE residual over a reduced physics state
$\xi \in \mathbb{R}^{79}$ (the policy observation is
$o_t \in \mathbb{R}^{135}$). The auxiliary critic reaches the policy through
**one or both of two coupling channels**, and the whole empirical contribution
turns on which channel is used.

| Channel | What it does | Where it enters |
|---|---|---|
| **1: critic** | $U_\phi(\xi_t)$ detached, appended to the value loss as a squared consistency term at $\lambda_{\mathrm{distill}} = 0.25$ | physics reaches the actor only through advantage estimates, where realized returns can correct it |
| **2: actor** | the residual's action values are converted to a policy and matched to $\pi_\theta$ by a forward-KL term on the actor loss | overrides what the policy gradient learned from outcomes |

Five instantiations:

| Method | Paradigm | Channel 1 (critic) | Channel 2 (actor) |
|---|---|---|---|
| Hard-HJB | optimality | yes | no |
| Soft-HJB | optimality | yes | yes |
| Eikonal | safety / geometry | **no** | yes |
| CBF | safety / geometry | yes | no |
| Fusion (Soft-HJB + CBF) | both | yes | yes |

Compared against **DRPPO**, an identical-backbone recurrent PPO baseline with
no auxiliary critic, and a deterministic time-to-collision rule-based
reference.

### The three findings

**1. Residual choice redistributes competence rather than adding it.** Of 60
method-cell comparisons against the baseline on success rate, exactly **one**
is a Holm-significant improvement (Eikonal on (1a, right_stem), 0.983,
$d = +2.05$, corrected $p = 0.018$) and **twelve** are significant
degradations, split 4/4/4 across Soft-HJB, Eikonal and Fusion. The two
formulations that never degrade (Hard-HJB, CBF) never improve either. Pooled
success rates: DRPPO 0.575, Hard-HJB 0.595, CBF 0.609, Soft-HJB 0.406,
Eikonal 0.379, Fusion 0.402.

**2. The damage comes from the actor coupling, not from the residual.**
Sweeping $\lambda_{\mathrm{res}}$ over two orders of magnitude leaves the
deficits of the coupled methods flat. Disabling the alignment KL recovers
Eikonal on (1b, stem_right) from 0.015 to 0.218 at corrected $p = 0.003$, and
moves both coupled methods the same way on each dense arm tested. **This is the
paper's contribution.** The design rule: a physics prior may regularize the
critic at little risk; coupling it to the actor is safe only where the
surrogate dynamics are accurate.

**3. Occlusion is not the binding difficulty; behavioral heterogeneity is.**
Four independent manipulations of occlusion and visibility return nulls (224
testable comparisons, smallest corrected $p = 0.238$). Training on the nominal
driving style alone collapses success to 0.000 on the pedestrian cell for every
method run in that suite including the baseline, while the same policies stay
at 0.980 to 0.999 on the car-only cell.

The negative results are the portable ones. This is a characterization study,
not a "our method wins" paper.

---

## 2. Quick start: reproduce a number from the paper in two minutes

You do not need SUMO, a GPU or any training to check the paper's numbers. The
analysis tables are in the repository.

```bash
git clone --depth 1 https://github.com/BoulosBoulos/physics-informed-auxiliary-critics
cd physics-informed-auxiliary-critics
pip install pandas numpy
```

```python
import csv, numpy as np

# Claim: "Of the sixty method-cell comparisons against DRPPO, exactly one is a
#         Holm-significant improvement ... twelve are significant degradations,
#         four each for Soft-HJB, Eikonal and Fusion."
rows = [r for r in csv.DictReader(open("results/tables/all_comparisons.csv"))
        if r["metric"] == "success_rate" and r["method"] != "rule_based"]
sig  = [r for r in rows if r["significant_welch"].lower() == "true"]
up   = [r for r in sig if float(r["cohens_d"]) > 0]
down = [r for r in sig if float(r["cohens_d"]) < 0]
print(len(rows), "comparisons;", len(up), "improvement,", len(down), "degradations")
# -> 60 comparisons; 1 improvement, 12 degradations

# Claim: pooled DRPPO success rate 0.57 [0.51, 0.65]
x = np.array([float(r["success_rate"]) for r in
              csv.DictReader(open("results/tables/baseline_per_run_tier1.csv"))])
rng = np.random.default_rng(12345)                      # analysis/config.py seed
s = rng.choice(x, size=(1000, x.size), replace=True).mean(axis=1)
print(round(x.mean(), 3), np.percentile(s, [2.5, 97.5]).round(3))
# -> 0.575 [0.507 0.646]
```

The bootstrap parameters (1000 resamples, 95 per cent, seed 12345) come from
`config_frozen_v1.yaml :: analysis`, and the procedure is
`analysis/stats.py :: bootstrap_ci`.

---

## 3. Repository layout

```
physics-informed-auxiliary-critics/
├── env/
│   └── sumo_env.py              SUMO Gymnasium environment (TraCI wrapper)
├── models/
│   ├── drppo.py                 RecurrentActorCritic + DRPPO trainer (baseline)
│   ├── intent_style.py          Per-agent LSTM intent/style encoder
│   ├── rule_based_policy.py     Deterministic TTC-threshold reference
│   └── pde/
│       ├── state_builder.py     Reduced physics state xi (XI_DIM = 79)
│       ├── dynamics.py          BehavioralDynamics: differentiable one-step map
│       ├── residuals.py         The four residuals + shared autograd helpers
│       ├── collocation.py       Collocation sampler with consistent jitter
│       ├── local_reward.py      Surrogate reward r(xi, a)
│       ├── checkpointing.py     Save/load with architecture spec
│       └── {hjb,soft_hjb,eikonal,cbf,fusion}_aux_{agent,critic}.py
├── scenario/
│   ├── generator.py             SUMO network generation (nodes, edges, buildings)
│   └── behavior_sampler.py      Per-episode style and route randomizer
├── scenarios/                   Pre-generated SUMO network XML
├── experiments/
│   ├── pde/train_*.py           One training entry point per method
│   ├── pde/eval.py              Unified evaluation
│   ├── pde/run_calibration.py   36-job calibration orchestrator
│   └── train_intent.py          Intent encoder pre-training
├── analysis/                    Offline analysis pipeline (see section 15)
├── verification/                Preflight checks, audits, determinism tests
├── configs/                     Per-subsystem YAML overrides
├── docs/                        25 internal design documents
├── scripts/                     Cluster submission and aggregation shell scripts
├── results/
│   ├── tables/                  *** the analysis tables behind the paper ***
│   ├── analysis/                figures, statistical tests, PAPER_REPORT.md
│   ├── tier_1_full/             1,636 Tier-1 run directories
│   ├── tier_1_machine_*/        per-machine run trees (cmu1..cmu8, local)
│   └── ablation/                Tier-2 and Tier-3 run outputs
├── config_frozen_v1.yaml        Canonical locked configuration
├── config_lock.json             SHA-256 of the above, checked at import
├── config_loader.py             Loader and lock enforcer
├── requirements.txt
├── Makefile
└── CALIBRATION_REPORT.md        36-run calibration outcome
```

`results/tables/` is the directory that matters for checking the paper. It
holds 18 CSVs and 16 LaTeX tables, about 1.2 MB in total.

---

## 4. Results provenance: paper claim to data file

Every claim in the paper resolves to one of these files. The unit of analysis
is the **training run**, not the episode.

| Paper claim | File | How to check |
|---|---|---|
| Table III, all 12 cells × 6 methods | `all_comparisons.csv`, `baseline_per_cell_tier1.csv` | filter `metric == success_rate` |
| 1 improvement, 12 degradations 4/4/4 | `all_comparisons.csv` | count `significant_welch` by sign of `cohens_d` |
| Pooled DRPPO 0.57 [0.51, 0.65] | `baseline_per_run_tier1.csv` | 120 rows, bootstrap per section 2 |
| Six pedestrian cells, 36 pairwise, median $d = 1.71$ | `pde_vs_pde_comparisons.csv` | **contains each pair twice (A-vs-B and B-vs-A); deduplicate on `frozenset({method_a, method_b})` first** |
| $\lambda_{\mathrm{res}}$ sweep flat for the coupled methods | `tier2a_lambda_sensitivity.csv` | 6 values × 5 methods × 4 cells |
| Occlusion nulls, 224 testable, min corrected $p = 0.238$ | `tier2b_occlusion_impact.csv` | 240 rows, 16 untestable (both arms at zero collisions) |
| Fusion weight sweep, 8 pairs | `tier2c_fusion_weights.csv` | pure optimality 0.394 vs unweighted 0.160 on (2_dense, right_left) |
| **Actor-KL ablation (the key result)** | `tier2d_actor_kl_ablation.csv` | Eikonal (1b, stem_right) on 0.015 → off 0.218, `welch_p_adj = 0.0031` |
| Nominal-style collapse to 0.000 | `tier3_behavioral_robustness.csv` | |
| Dense stress, learned 0.053 to 0.168 | `tier3_dense_stress.csv` | |
| Visibility-feature ablation | `tier3_state_ablation.csv` | |
| Held-out transfer, 23 of 25 rise | `heldout_comparisons.csv` | 5 conditions HO1..HO5 |
| Intent encoder, 74 cells, median 0.018 | `tier1_intent_effect.csv` | |
| Eikonal residual floor 1.36 | `pde_diagnostics_summary.csv` | `median_residual_late` |
| Fusion distillation gap 6.4 → 34.3 in 225 of 240 runs | `pde_diagnostics.csv` | `distill_early`, `distill_late` |
| Training cost 8.6 to 18.3 per cent | `computational_overhead.csv` | `residual_frac_of_iter`; the `walltime_ratio_vs_drppo_HW_CONFOUNDED` column is named for a reason and should not be quoted |

**Two traps worth repeating.** `pde_vs_pde_comparisons.csv` contains each pair
twice, so counting without deduplication doubles every Family-B statistic.
Aggregating at the episode level instead of the run level inflates the sample
size by an integer factor and deflates $p$-values.

---

## 5. Method

### The dual-critic architecture

```
Observation o_t (135D, or 165D with the intent encoder)
        │
        ▼
    GRU encoder (hidden 256) ──────────────┐
        │                                  │
        ▼                                  ▼
   PPO actor π_θ                    PPO critic V_ψ
        ▲                                  ▲
        │  channel 2: forward-KL           │  channel 1: L_distill (stop-grad)
        │  (Soft-HJB, Eikonal, Fusion)     │  (all except Eikonal)
        │                                  │
        └────── auxiliary critic U_φ(ξ) ───┘
                        ▲
                 PDE residual ρ(ξ)
```

**Important, and the point of the paper:** it is *not* true that physics never
reaches the actor. Hard-HJB and CBF reach the policy through the value function
alone. Soft-HJB, Eikonal and Fusion additionally add a Kullback-Leibler term
directly to the actor loss, and those three are exactly the three that degrade.
Eikonal uses the actor channel only, with no critic distillation at all.

### Why a reduced state

The auxiliary critic operates on $\xi \in \mathbb{R}^{79}$ rather than the full
observation because the residual needs a state where derivatives
$\nabla_\xi U$ are physically meaningful, where the behavioral dynamics
$f_a(\xi)$ can be written in closed form, and where autograd can trace through
that map. Reducing the agent slots from $K=5$ to 3 accounts for 44 of the 56
discarded dimensions; the rest are perceptual features the surrogate cannot
propagate.

---

## 6. Environment

`env/sumo_env.py` wraps SUMO 1.18 via TraCI as a Gymnasium environment at
$\Delta t = 0.1$ s.

**Geometry.** A T-junction with a 100 m main road (50 m per branch) and a 60 m
stem. Lanes are 3.2 m (SUMO default; the generator writes `numLanes="2"` with no
explicit width). Posted limit 13.89 m/s.

**Buildings and occlusion.** Four corner buildings create the blind spots.
Their inner edges sit **8 m from the junction centre in the car-only scenario
and 11 m in the pedestrian-bearing scenarios**, the difference being exactly the
3 m sidewalk width (`scenario/generator.py`, `_ped_scenarios`). Building polygon
coordinates are computed at runtime from `traci.junction.getPosition('center')`
inside `reset()`; static origin-centred coordinates would place them roughly
55 m off because of the netconvert offset.

**Pedestrian crossings.** `--crossings.guess` produces exactly three crossings,
each 8.90 m from the junction centre: east arm at (+8.90, 0), stem at
(0, −8.90), west arm at (−8.90, 0). Both pedestrian routes
(`cross_left_right`, `cross_right_left`) run along the southern sidewalk
corridor and use **only the stem crossing**. Neither enters the north half of
the junction. This is why the right-turn deficit is a property of the sidewalk
topology rather than of right turns in general.

**Scenarios.** Ten are specified; eight were trained and evaluated.

| Scenario | Car | Pedestrian | Motorcyclist | Pothole | Trained |
|---|---|---|---|---|---|
| `1a` | yes | | | | yes |
| `1b` | | yes | | | yes |
| `1c` | | | yes | | no |
| `1d` | | | | yes | no |
| `2` | yes | yes | | | yes |
| `3` | yes | yes | yes | | yes |
| `4` | yes | yes | yes | yes | yes |
| `2_dense`, `3_dense`, `4_dense` | + a third conflicting vehicle | | | | yes |

**Maneuvers.** Six are defined; five were evaluated. `left_stem` is the
reflection of `right_stem` about the main-road axis and was skipped.

**Episode termination.** Exit edge reached (success), collision, or timeout at
500 steps. Collision is detected both by `traci.simulation.getCollisions()` and
by a proximity check at $d_{\mathrm{coll}} = 2.0$ m.

---

## 7. Observation, action and reward

**Observation (135D).** Ego state, junction geometry, visibility indicators,
and 5 agent slots of 22 features each sorted by distance to the conflict zone,
with occluded or out-of-range agents zero-padded and their mask bit cleared.
With the intent encoder enabled the vector is 165D (5 agents × 6D of intent and
style probabilities appended).

**Actions.**

| Index | Name | Nominal acceleration |
|---|---|---|
| 0 | STOP | −5.0 m/s² |
| 1 | CREEP | `clip(1.0 − v, −0.5, 0.5)` toward 1 m/s |
| 2 | YIELD | −0.5 m/s² |
| 3 | GO | +2.0 m/s² |
| 4 | ABORT | −8.0 m/s² |

The STOP rate is firm rather than comfortable: the primitive has to arrest
motion within the available stopping sight distance, and a comfort rate would
make it behaviorally indistinguishable from YIELD. Ride comfort and jerk are
therefore not evaluation objectives.

**Reward coefficients** (`config_frozen_v1.yaml :: reward`):

| Term | Symbol | Value |
|---|---|---|
| Route progress | $w_{\mathrm{prog}}$ | +1.0 |
| Time | $w_{\mathrm{time}}$ | −0.1 |
| Proximity risk | $w_{\mathrm{risk}}$ | −3.0 |
| Collision | $w_{\mathrm{coll}}$ | −20.0 |
| Success | $w_{\mathrm{succ}}$ | +200.0 |
| Potential shaping | $w_{\mathrm{shape}}$ | +3.0 |
| Action switch | $w_{\mathrm{switch}}$ | −0.05 |
| Right-of-way violation | $w_{\mathrm{rule}}$ | −2.0 |
| Pothole | $w_{\mathrm{pothole}}$ | −5.0 |
| TTC alarm threshold | | 3.0 s |
| Collision distance | $d_{\mathrm{coll}}$ | 2.0 m |
| Shaping discount | $\gamma_{\mathrm{shape}}$ | 1.0 |

Two choices matter. Unit shaping discount makes the cumulative shaping
telescope exactly to $d_{\mathrm{exit}}(s_0) - d_{\mathrm{exit}}(s_T)$ whatever
the termination cause, removing the drift bias a discounted term introduces.
And the success bonus exceeds the collision penalty tenfold, which is needed to
supply positive gradient against accumulated per-step costs in dense episodes;
the original +10 was insufficient.

---

## 8. The five physics-informed methods

All five share the identical `RecurrentActorCritic` backbone, a two-hidden-layer
auxiliary critic MLP ($79 \to 256 \to 256 \to 1$, $\tanh$), a
`BehavioralDynamics` object, autograd for $\nabla_\xi U_\phi$, and the
collocation sampler. Only the residual and the coupling channels differ.

### 8.1 Hard-HJB

$$\rho^{\mathrm{HJB}}(\xi) = U_\phi(\xi)\ln\gamma + \max_a q_a(\xi), \qquad
q_a(\xi) = r(\xi,a) + \gamma\,\nabla_\xi U_\phi(\xi)\cdot\Delta\xi_a$$

Derived by replacing the successor value in the Bellman equation with its
first-order Taylor expansion, collecting $(1-\gamma)U$ on the left and applying
$1-\gamma \approx -\ln\gamma$. The increment $\Delta\xi_a$ enters **undivided by
$\Delta t$**, which is correct for a discrete-time backup. Critic channel only.

### 8.2 Soft-HJB

$$\rho^{\mathrm{Soft}}(\xi) = U_\phi(\xi)\ln\gamma
+ \tau\,\mathrm{logsumexp}_a\!\left(q_a(\xi)/\tau\right), \qquad \tau = 1.0$$

As $\tau \to 0$ this collapses onto the hard residual. The soft operator induces
a Boltzmann policy $\pi_{\mathrm{soft}} \propto \exp(q_a/\tau)$, and the actor
is drawn toward it by a **forward** KL term appended to the actor loss:

$$\mathcal{L}^{\mathrm{align}}(\theta) = \lambda_{\mathrm{align}}\,
D_{\mathrm{KL}}\!\left(\pi_{\mathrm{soft}} \,\|\, \pi_\theta\right),
\qquad \lambda_{\mathrm{align}} = 0.1$$

The direction matters and the code computes it as written
(`soft_hjb_aux_agent.py`: `kl = (pi_s * (pi_s.log() - pi_t.log())).sum()`).
Both channels.

### 8.3 Eikonal time-of-arrival

The auxiliary critic is reinterpreted as an arrival-time field
$T_\phi : \mathbb{R}^{79} \to \mathbb{R}_{\geq 0}$, constrained by

$$\rho^{\mathrm{Eik}}(\xi) = \|\nabla_\xi T_\phi(\xi)\|^2 - c(\xi)^2,
\qquad c(\xi) = 1/v_{\mathrm{eff}}(\xi)$$

The slowness field is **derived from the behavioral dynamics and the state of
the scene**, which is the substance of the adaptation. With
$\omega_a(\xi) = \mathrm{sig}\!\left((\mathrm{TTC}'_{\min,a} -
\mathrm{TTC}_{\mathrm{thr}})/0.5\right)$ weighting each action by the margin it
produces,

$$v_{\mathrm{eff}}(\xi) = \Bigl[\max_a v'_a(\xi)\,\omega_a(\xi)\Bigr]
\cdot \mathrm{clip}(\alpha_{\mathrm{cz}}, 0.1, 1)$$

clamped below at $v_{\min} = 0.5$ m/s. An occluded or conflicted state is
therefore slow to traverse by construction.
(`residuals.py :: _compute_v_eff_and_c_sq`.)

Enforced as a **constraint** by an augmented Lagrangian, with the boundary and
grounding terms balanced by learned homoscedastic uncertainties
(Kendall et al.). The penalty may grow by four orders of magnitude, so a
residual that still fails to vanish indicates the constraint and the anchors are
not jointly satisfiable rather than that the pressure was insufficient. It
settles at a median late magnitude of **1.36**.

**Eikonal uses the actor channel only.** It has no critic distillation term.

### 8.4 CBF barrier-descent

$$\rho^{\mathrm{CBF}}(\xi) = \mathrm{ReLU}\!\left(-\max_a\bigl[
\nabla_\xi U_\phi(\xi)\cdot f_a(\xi) + \alpha_h h(\xi)\bigr]\right),
\qquad h(\xi) = U_\phi(\xi) + c_{\mathrm{offset}}$$

with $\alpha_h = 1.0$ and $c_{\mathrm{offset}} = 10.0$. The offset places the
safe-unsafe boundary $h = 0$ midway between a typical successful state and the
collision terminal at $U^- = -20$, so states of positive learned value lie
inside the safe set. This uses the **rate** form $f_a = \Delta\xi_a/\Delta t$,
because the barrier condition is intrinsically continuous-time, unlike the HJB
backup.

The existential quantifier (max over actions, not min) is deliberate: under
partial observability the policy needs one safe action to exist, not safety
under every action simultaneously.

This is a critic regularizer that encourages barrier-like gradient structure,
**not** a certified safety filter: $h$ is learned rather than designed, so
there is no invariance certificate, and the first-order discrete-time drift
approximates a continuous-time inequality. Critic channel only.

### 8.5 Fusion

Two independent auxiliary critics, $U_{\mathrm{opt}}$ under the **Soft-HJB**
residual and $U_{\mathrm{safe}}$ under the CBF residual, sharing no parameters
and no gradient path, each with its own optimizer and its own residual, anchor
and boundary losses. Their detached outputs are combined into one distillation
target and their action values in the same proportion into a fused policy:

$$\hat U^{\mathrm{fuse}} = \frac{w_o U_{\mathrm{opt}} + w_s U_{\mathrm{safe}}}
{w_o + w_s}$$

Reported configuration $w_o = w_s = 1$. Both channels. The training logs show
the failure mechanism directly: the gap between the policy critic and the fused
target grows in **225 of 240** runs, from a median of 6.4 early to 34.3 late,
the signature of two critics supplying competing supervision to one value head.

### 8.6 Rule-based reference

`models/rule_based_policy.py`. Stateless. Commands STOP whenever the minimum
observed TTC falls below 3.0 s and a 5 m far-zone gate is cleared, GO
otherwise. It uses no CREEP, YIELD or ABORT. The threshold is held at the same
value in every cell rather than tuned per scenario, so it characterizes a
standard TTC gate rather than the best achievable heuristic, and it is excluded
from all statistical tests.

Read its results carefully. It is near-perfect where the conflict involves only
vehicles (0.996 to 1.000 success at 0.004 collision or below) and it either
crosses or strikes on the pedestrian cells: on (1b, stem_right) it succeeds in
0.254 of episodes and collides in 0.740. Pooled across the grid it reaches 0.68
success at 0.22 collision, against 0.13 to 0.18 collision for the learned
family.

---

## 9. PDE infrastructure

### Reduced physics state, `XI_DIM = 79`

| Index | Block | Contents |
|---|---|---|
| 0–7 | Ego (8) | $v$, $a$, $\dot\psi$, $d_{\mathrm{stop}}$, $d_{\mathrm{cz}}$, $d_{\mathrm{exit}}$, $\kappa$, $\mathrm{TTC}_{\min}$ |
| 8–11 | Visibility (4) | $\alpha_{\mathrm{cz}}$, $\alpha_{\mathrm{cross}}$, $d_{\mathrm{occ}}$, $\Delta t_{\mathrm{seen}}$ |
| 12–33 | Agent 1 (22) | relative kinematics, conflict metrics, class one-hot, mask |
| 34–55 | Agent 2 (22) | same layout |
| 56–77 | Agent 3 (22) | same layout |
| 78 | Pothole (1) | $d_{\mathrm{pothole}}$ |

Slots are filled by the three participants with the smallest estimated time to
conflict. The mask bit lets the dynamics and residuals zero out absent agents.

### Behavioral dynamics

Speed advances as
$v'_a = \mathrm{clip}(v + a^{\mathrm{nom}}_a(v)\Delta t, 0, v_{\max})$ and the
path advance by trapezoidal integration
$\delta s_a = \tfrac12 (v + v'_a)\Delta t$. Route distances decrease by
$\delta s_a$ and must stay non-negative, but a hard clamp would zero the
gradient wherever a distance has already reached zero, so a smooth surrogate is
used at every clamping site:

$$\sigma_+(x;\varepsilon) = \tfrac12\left(x + \sqrt{x^2 + \varepsilon^2}\right)
- \tfrac{\varepsilon}{2}, \qquad \varepsilon = 0.1$$

which approximates $\max(x,0)$ and is strictly increasing everywhere. Agents
advance by constant-velocity extrapolation, and TTC and closest-point-of-approach
quantities are recomputed from the updated geometry. Both the drift
$f_a = \Delta\xi_a/\Delta t$ and the undivided increment $\Delta\xi_a$ are
retained, because the HJB backup needs one and the barrier condition the other.

### Collocation

256 points per auxiliary update, 0.7 of them real rollout states and the rest
jitter-augmented copies. **Only primitive features are jittered** (speed,
acceleration, distances, curvature, visibility fractions); derived quantities
(TTC, CPA, $\tau_i$) are recomputed from the perturbed primitives, because
jittering them directly would produce physically impossible states. Per-feature
standard deviations are in `config_frozen_v1.yaml`.

### Local reward surrogate

The residual needs $r(\xi, a)$ at collocation points the simulator cannot be
queried at, so the surrogate mirrors the environment reward from quantities
available in $\xi$. It **excludes the collision penalty and the success bonus**,
because those terminal signals would conflate the residual, which governs smooth
regions of the value landscape, with boundary conditions; they are imposed
instead by anchoring $U_\phi$ at terminal states to $U^+ = +200$ and
$U^- = -20$. It inherits the constant-velocity extrapolation, so it degrades
exactly where agent motion departs from it. That inheritance is the mechanism
behind finding 2.

---

## 10. Experiment design

**Training budget: 400,000 steps**, established by calibration rather than
assumed. A 36-job study covered the six methods across two cells at the
difficulty extremes at three seeds each, run to 500,000 steps. A run was
declared converged at the first step whose trailing 50,000-step window held a
mean rolling success rate of at least 0.70, a standard deviation of at most
0.10, and a mean collision rate of at most 0.05. Every method converged on the
low-conflict cell and none on the dense cell; the latest convergence observed,
one Hard-HJB seed at 331,776 steps, fixed the budget. See
`CALIBRATION_REPORT.md`.

**Tier 1, the main grid.** Twelve scenario-maneuver cells × six methods × ten
seeds = **719 runs** (71 cells at ten seeds, Soft-HJB on (3, left_right) at
nine). Cells: (1a, right_stem), (1a, stem_left), (1a, stem_right),
(1b, stem_left), (1b, stem_right), (2, stem_left), (2, stem_right),
(2_dense, right_left), (3, left_right), (3, right_left), (4, stem_left),
(4, stem_right).

**Tier 2, sensitivity and mechanism.** Five runs per arm on four cells.
2a residual-weight sweep over $\{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$; 2b occlusion
on/off; 2c eight fusion weight pairs; **2d the actor-KL ablation**, which is the
result the paper turns on.

**Tier 3, robustness of the recipe.** Five runs per arm. Visibility features
removed; nominal-style-only training; dense variants.

**Tier 4, held-out transfer.** Five zero-shot conditions (HO1 occlusion removed,
HO2 occlusion imposed, HO3 and HO4 adversarial styles, HO5 visibility features
withheld).

Seeds control network initialization, rollout sampling and SUMO episode
randomization, so reported spreads reflect genuine run-to-run variability.

---

## 11. Statistical methodology

- **Unit of analysis is the training run.** Each run contributes one value per
  metric over its own evaluation window (the last ten per cent of iterations, at
  ten evaluation episodes per logged iteration). Per-cell sample sizes are seed
  counts, not episode counts.
- **Two-sided Welch $t$-tests**, which do not assume equal variance, because the
  augmented methods and the baseline have genuinely different across-seed
  variance.
- **Two families per metric**, corrected independently: Family A is the five
  tests of each augmented method against the baseline; Family B the ten pairwise
  comparisons among the augmented methods.
- **Holm-Bonferroni** step-down correction at $\alpha = 0.05$.
- **Cohen's $d$** on the pooled standard deviation with every significant
  difference, classified negligible below 0.2, then small, medium, and large at
  or above 0.8.
- **Percentile bootstrap** confidence intervals, $B = 1000$, seed 12345 for
  determinism.
- No claim of superiority is made when a significant difference is negative.

Implementation: `analysis/stats.py`. Constants: `config_frozen_v1.yaml ::
analysis`.

---

## 12. Configuration and the config lock

`config_frozen_v1.yaml` is the single source of truth for every
result-affecting constant: PPO hyperparameters, method-specific weights,
reward coefficients, scenario dimensions, tier definitions and analysis
constants. `config_loader.py` reads it, and `check_config_lock()` compares its
SHA-256 against `config_lock.json`, so an edit to the configuration is detected
rather than silently changing results.

Method defaults live under `methods:` and are read as the CLI defaults of each
training script, so the value in the YAML is the value the runs used unless a
job script overrode it on the command line.

**One provenance caveat.** The Eikonal method's actor-KL weight
$\beta_{\mathrm{KL}}$ and its $w_{\mathrm{eik}}$ are **not** in
`config_frozen_v1.yaml`. They come from `configs/pde/eikonal_aux.yaml`
($\beta_{\mathrm{KL}} = 0.1$, matching the other methods' alignment weight) via
`_pick()` in `train_eikonal_aux.py`. Anyone reconstructing the configuration
needs both files.

---

## 13. Installation

**Prerequisites:** Python 3.10+, SUMO 1.18 installed system-wide with
`SUMO_HOME` set, and a CUDA-capable GPU for training (evaluation and analysis
run on CPU).

```bash
git clone https://github.com/BoulosBoulos/physics-informed-auxiliary-critics
cd physics-informed-auxiliary-critics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
export SUMO_HOME=/usr/share/sumo        # adjust to your install
python -c "from config_loader import check_config_lock; print(check_config_lock())"
```

To check the analysis tables only, you need nothing but `numpy` and `pandas`;
see section 2.

Generate the SUMO networks before training:

```bash
python -c "from scenario.generator import ScenarioGenerator as G; \
           [G().generate(f'scenarios/sumo_{s}', s) for s in \
            ('1a','1b','2','3','4','2_dense','3_dense','4_dense')]"
```

---

## 14. Running experiments

```bash
# Smoke test, about five minutes
python experiments/pde/smoke_test.py --total_steps 5000

# One training run
python experiments/pde/train_hjb_aux.py \
    --scenario 1b --maneuver stem_right --seed 42 --total_steps 400000

# Calibration study (36 jobs)
python experiments/pde/run_calibration.py

# Tier 1 on a cluster node
bash scripts/run_tier1_node.sh

# Evaluate a checkpoint
python experiments/pde/eval.py --checkpoint path/to/ckpt.pt --episodes 100
```

Method-specific flags follow the `config_frozen_v1.yaml` names, for example
`--lambda_residual`, `--lambda_distill`, `--lambda_actor_kl` (Soft-HJB,
Fusion), `--beta_KL` (Eikonal). The actor-KL ablation is
`scripts/run_job2_actor_kl_ablation.sh`, which passes `--lambda_actor_kl 0` and
`--beta_KL 0`.

---

## 15. Analysis pipeline

`analysis/run_analysis.py` orchestrates the whole thing.

| Module | Responsibility |
|---|---|
| `config.py` | colors, method ordering, statistical constants from the frozen YAML |
| `loader.py` | reads run directories, applies quality gates |
| `metrics.py` | computes per-run metrics from raw CSVs |
| `quality.py` | run quality checks |
| `stats.py` | Welch $t$-test, Holm correction, Cohen's $d$, percentile bootstrap |
| `tables.py` | LaTeX table generation |
| `plots.py` | seven plot families |
| `calibration_*.py` | calibration-specific analysis |

Outputs land in `results/tables/` (the tables the paper cites) and
`results/analysis/` (figures, statistical tests, `PAPER_REPORT.md`).

**`results/analysis/PAPER_REPORT.md` contains pre-audit values that were later
falsified and should not be cited.** The file opens with a banner saying so and
mapping each superseded section to its replacement. Use `results/tables/` for
anything that goes into a document.

---

## 16. Known gaps and coverage limits

Stated plainly so that nobody has to discover them.

- **Tier 1 is complete.** All 12 cells × 6 methods present; 59 of the 60
  augmented method-cells at ten seeds, Soft-HJB on (3, left_right) at nine, the
  baseline at ten everywhere.
- **Every ablation arm is at five seeds**, by design. That supports detection of
  large effects and leaves moderate ones inconclusive. One exception: the
  Hard-HJB $\lambda = 0.1$ arm on (2_dense, right_left) was replicated to ten
  seeds because the interval was wide on that bimodal cell.
- **Fusion was not run** in the three Tier-3 suites (behavioral robustness,
  dense stress, state ablation).
- **The rule-based reference appears only in the dense-stress arm** among Tier-3
  suites.
- **CBF training diagnostics cover 224 runs of 240**, because the full
  per-iteration loss traces were retained for that subset.
- **Held-out method coverage is asymmetric.** HO2 includes Fusion and omits
  DRPPO; the other four include DRPPO and omit Fusion.
- **Intent-enabled runs are excluded from the primary tables** because
  evaluation attrition left per-cell sizes as low as two seeds. Analyzed
  separately across 74 measurable cells the encoder changes nothing (median
  absolute change 0.018, no cell survives correction).
- **Locally executed baseline runs are excluded** on grounds of instrumentation
  uniformity rather than any demonstrated defect. The two populations differ on
  one pedestrian cell, 0.112 against 0.216, not significant under an
  unequal-variance test.
- **The dense through cell is bimodal.** Success rates for an identical
  configuration range from 0.008 to 0.970 across seeds, so per-cell means there
  must be read with their intervals rather than as point estimates.
- **Hard-HJB is $\lambda$-sensitive on that cell**, which leaves open that the
  shared $\lambda_{\mathrm{res}} = 0.2$ was not its best setting.
- **Scenarios 1c and 1d and the `left_stem` maneuver were specified but not
  trained.**
- **End-to-end wall-clock ratios are not comparable** across methods, because
  runs were distributed over heterogeneous machines. Use
  `residual_frac_of_iter` from `computational_overhead.csv` instead.

---

## 17. Citation and licence

If you use this code or these results, please cite the paper (see
`CITATION.cff`). Until it appears, cite the repository at tag
[`paper-tiv-v1`](https://github.com/BoulosBoulos/physics-informed-auxiliary-critics/releases/tag/paper-tiv-v1),
which pins commit `68aaf062`, the state every reported number was computed from.

Licence: see `LICENSE`. Code and analysis outputs are released for research use.

**A note on the commit history.** Training runs were executed on a shared
compute machine, and commits pushed from it carry that machine's configured git
identity rather than the author's. All work in this repository is the author's.

The repository was renamed from `EECE_499` in September 2026. GitHub redirects
the old address, but the current one is preferred.

---

*Questions about the data or the analysis are best answered by opening the
relevant CSV in `results/tables/` and recomputing. Section 4 says which file
corresponds to which claim.*
