# MASTER_PLAN_RECONCILIATION — Authoritative tier-size record

**Generated:** Phase 2 / Verification gate (`SPEC_PHASE_2_VERIFICATION_GATE.md` Step 4).
**Source of truth:** `experiments/pde/run_full_ablation.generate_jobs()` against the frozen
`config_frozen_v1.yaml` (config hash `337cc19b82009bf99e7b497f9ee33c58a63de22bad6f286ad62e78d53bdcd8e5`).
This file supersedes any tier-count number in `MASTER_PLAN_MAY11.md` or in earlier
phase reports.

## Tier 1 — Main comparison

| Quantity | Value |
|---|---:|
| Combos (scenario × maneuver) | 12 |
| Methods | 7 (`hjb_aux`, `soft_hjb_aux`, `eikonal_aux`, `cbf_aux`, `fusion_aux`, `drppo`, `rule_based`) |
| Seeds | 10 (42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555) |
| Intent settings | 2 (False, True) |
| **Total jobs (full grid)** | **1680** |
| ↳ trainable jobs | 1440 (12 × 6 × 10 × 2) |
| ↳ rule_based eval-only | 240 (12 × 1 × 10 × 2) |

`run_full_ablation.py --tier 1 --dry_run` (default, rule_based excluded by `_apply_filters`)
prints `Generated 1440 jobs for tier '1'`. Pass `--include_rule_based` to see the full 1680.

## Tier 2 — Hyperparameter / occlusion / fusion sweeps

| Sub-grid | Description | Jobs |
|---|---|---:|
| 2a | λ residual sweep (per Phase 1D) | 600 |
| 2b | occlusion sweep | 400 |
| 2c | fusion weight sweep | 160 |
| **Total Tier 2** | | **1160** |

(Phase 1F final value; spec text at line 22 of `SPEC_PHASE_2_VERIFICATION_GATE.md` cites
"Tier 2 = 1,160 jobs", confirmed by `--tier 2 --dry_run` final summary.)

## Tier 3 — State / behavioral / dense (audit target of this step)

Live `python3 experiments/pde/run_full_ablation.py --tier 3 --dry_run` reports
**`Generated 275 jobs for tier '3'`**.

Per-sub-grid breakdown from `generate_jobs("3", 50000)`:

| Sub-grid | Tag prefix | Multipliers | Jobs |
|---|---|---|---:|
| State ablation (`no_visibility`) | `T3S_` | 4 combos × 5 methods × 5 seeds | **100** |
| Behavioral robustness (`style_filter=nominal`) | `T3B_` | 4 combos × 5 methods × 5 seeds | **100** |
| Dense scenarios | `T3D_` | 3 scenarios × 5 methods × 5 seeds | **75** |
| **Total Tier 3** | | | **275** |

### Source lists (from `config_frozen_v1.yaml` via run_full_ablation.py)

```
TIER3_STATE_COMBOS    = [('1a','stem_right'), ('1b','stem_right'), ('2','stem_right'), ('4','stem_right')]
TIER3_STATE_METHODS   = ['hjb_aux','soft_hjb_aux','eikonal_aux','cbf_aux','drppo']
TIER3_STATE_SEEDS     = [42, 123, 456, 789, 999]
TIER3_BEHAV_COMBOS    = [('1a','stem_right'), ('1b','stem_right'), ('2','stem_right'), ('4','stem_right')]
TIER3_BEHAV_METHODS   = ['hjb_aux','soft_hjb_aux','eikonal_aux','cbf_aux','drppo']
TIER3_BEHAV_SEEDS     = [42, 123, 456, 789, 999]
TIER3_DENSE_SCENARIOS = ['2_dense', '3_dense', '4_dense']
TIER3_DENSE_METHODS   = ['hjb_aux','soft_hjb_aux','eikonal_aux','cbf_aux','drppo']
TIER3_DENSE_SEEDS     = [42, 123, 456, 789, 999]
```

### Reconciliation against earlier estimates

| Source | Claim | Status |
|---|---:|---|
| `SPEC_PHASE_1F_FOLLOWUP_TIER1_COMBO_SWAP.md` (Phase 1F report) | 275 | ✅ matches actual |
| `MASTER_PLAN_MAY11.md` (older estimate) | ~165 | ⚠ stale — superseded by Phase 1F's expanded methods set (5 instead of 3) and 5 seeds (instead of 3) per sub-grid |
| `SPEC_PHASE_2_VERIFICATION_GATE.md` line 22 | 1,160 (Tier 2 only; doesn't claim Tier 3) | ✅ no Tier 3 number to compare |
| `MASTER_PLAN_MAY11.md` "What success looks like" line 37 | "~175 runs" | ⚠ stale by Phase 1F |

The 275 figure is the binding number; the older 165/175 estimates predate the Phase 1F
expansion of methods (3 → 5) and seeds (3 → 5) within each Tier 3 sub-grid.

## Tier 4 — Held-out evaluation

Eval-only on Tier 1 checkpoints. Job count depends on which Tier 1 checkpoints exist
on disk at orchestrator launch time (`generate_tier4_jobs` walks the filesystem).
Pre-Tier-1 launch, this returns **0 jobs** (verified by `--tier 4 --dry_run`); after
Tier 1 completes, the count is implied by the cross-product of held-out combos × the
existing checkpoint set.

## Supplementary

`--tier supp --dry_run` reports **126 jobs** (per Phase 0 fingerprint marker matching
SPEC_FINAL_VERIFICATION's earlier number).

## Reproducibility

Regenerate this file by running:
```bash
python3 experiments/pde/run_full_ablation.py --tier 3 --dry_run
```
and confirming the printed `Generated N jobs for tier '3':` matches the **275** above.

---

## Determinism Notes (Phase 2 / Prompt11 decision)

### Problem

`SPEC_PHASE_2_VERIFICATION_GATE.md` (Decision D) originally required **two** determinism guarantees:

1. *Strict CPU determinism* — same seed + `torch.set_num_threads(1)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` produces byte-identical `metrics.csv` across two consecutive runs.
2. *Relaxed GPU determinism* — same seed produces results within **1 × 10⁻⁵** absolute float tolerance.

Both checks failed empirically: two consecutive 5,000-step `train_hjb_aux` runs at `seed=42` produced `mean_reward` deltas of **703.47** (GPU) and **917.82** (CPU strict). Per-iteration action distributions diverged from iteration 1, ruling out CUDA atomic-add as the primary cause. CPU strict mode was as non-deterministic as GPU, ruling out CUDA reduction non-determinism. The non-determinism is upstream of all torch compute and traces to **TraCI / SUMO subprocess timing**: TraCI port allocation, OS scheduling between the Python policy thread and the SUMO subprocess, and local-socket buffer ordering all introduce per-launch variability that downstream PPO updates amplify into the policy weights.

### Decision

Following Prompt11 (Option A), strict bit-reproducibility is acknowledged as **unachievable** for SUMO-coupled RL training. Pursuing fixes (deterministic SUMO port allocation, single-threaded TraCI, deterministic socket I/O) would constitute a separate research project and is not budgeted for the paper. **Statistical claims hold across seed-aggregated experiments**, not within a single seed.

### Tolerance — measured empirically

5 consecutive runs at `seed=42`, `scenario=1a`, `ego_maneuver=stem_right`, `total_steps=5000` produced the following sample statistics (across-runs std at ddof=1):

| Metric | Mean | Std | 3 σ tolerance |
|---|---:|---:|---:|
| `final_collision_rate` | 0.0000 | 0.0000 | **0.0000** |
| `final_success_rate`   | 0.0000 | 0.0000 | **0.0000** |
| `final_mean_reward`    | −7526.80 | 570.57 | **1711.72** |
| `n_collisions_sum`     | 0 | 0 | 0 |
| `n_successes_sum`      | 0 | 0 | 0 |

(Collision and success rates have zero variance because at 5,000 training steps the policy is undertrained and times out on every episode in the final-window; n_collisions and n_successes are integer 0 across all 5 runs. Reward variance is the dominant non-determinism signal at this stage.)

Source data: `verification/phase29_determinism_variance.json`.

### Relaxed determinism test (Phase 29.8 / 29.9)

Tests 29.8 (GPU) and 29.9 (CPU) collapse into a single relaxed test with the same tolerance — there's no longer a meaningful distinction between "strict CPU" and "relaxed GPU" because both share TraCI/SUMO non-determinism. Acceptance criteria for two consecutive runs at the same seed:

- **Integer columns exactly equal** (`n_episodes`, `n_collisions`, `n_successes`, `n_timeouts`, `n_aborts`, `iteration`, `total_steps`) — if these differ, there is a worse bug than SUMO timing.
- `|Δ final_collision_rate| ≤ 0.0` (must equal at this training length).
- `|Δ final_success_rate|   ≤ 0.0` (must equal at this training length).
- `|Δ final_mean_reward|    ≤ 1711.72`.

The driver is `verification/run_determinism_relaxed.py`; outputs are written to `verification/phase29_determinism_{gpu,cpu}.json`.

### Implications for paper claims

- Single-seed metrics on this stack are not bit-reproducible. **All paper-grade claims must aggregate ≥ 5 seeds** (matching `TIER1_SEEDS = [42, 123, 456, 789, 999, 1111, 2222, 3333, 4444, 5555]` — n=10).
- Holm-corrected significance, bootstrap CIs on Cohen's d, and AULC (the primary Tier 1 metric) are robust to this non-determinism by construction — the across-seed std subsumes it.
- Calibration runs (Phase 3) must use ≥ 3 seeds per (method, scenario, maneuver); the current spec already requires this.
- For exact-replication needs (e.g. debugging a specific run), pin the SUMO version, TraCI port, and OS scheduler — that is a deeper-than-paper-budget remediation and not part of the published methodology.
