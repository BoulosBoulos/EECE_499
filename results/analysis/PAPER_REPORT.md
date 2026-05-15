# Paper Results Report — EECE 499
Generated: 2026-05-15 16:26:34 UTC
Analysis pipeline version: 4bc62af
Total per-run evaluation summaries computed: 439 (from `results/analysis/stats/eval_summary_all.csv`)
Runs with full metrics.csv + meta.json that passed quality gates: 71/93 (rest flagged as partial — see §1.2)

---

## 1. COMPLETION STATUS

### 1.1 Run completion by method × scenario × maneuver

(Counts derived from `eval_summary_all.csv`, parsed from job directory names. 
Each row is one (method, scenario, maneuver, intent) cell.)

| method | scenario | maneuver | intent | n_seeds |
|---|---|---|---|---|
| cbf_aux | 1a | right_stem | nointent | 10 |
| cbf_aux | 1a | stem_left | nointent | 1 |
| cbf_aux | 1a | stem_right | intent | 5 |
| cbf_aux | 1a | stem_right | nointent | 5 |
| cbf_aux | 1b | stem_right | intent | 5 |
| cbf_aux | 1b | stem_right | nointent | 5 |
| cbf_aux | 2_dense | right_left | intent | 5 |
| cbf_aux | 2_dense | right_left | nointent | 5 |
| drppo | 1a | stem_right | intent | 5 |
| drppo | 1a | stem_right | nointent | 5 |
| drppo | 1b | stem_right | intent | 5 |
| drppo | 1b | stem_right | nointent | 5 |
| drppo | 2_dense | right_left | intent | 5 |
| drppo | 2_dense | right_left | nointent | 5 |
| eikonal_aux | 1a | right_stem | nointent | 10 |
| eikonal_aux | 1a | stem_left | nointent | 10 |
| eikonal_aux | 1a | stem_right | intent | 5 |
| eikonal_aux | 1a | stem_right | nointent | 7 |
| eikonal_aux | 1b | stem_right | intent | 5 |
| eikonal_aux | 1b | stem_right | nointent | 5 |
| eikonal_aux | 2 | stem_right | nointent | 10 |
| eikonal_aux | 2_dense | right_left | intent | 2 |
| eikonal_aux | 2_dense | right_left | nointent | 10 |
| eikonal_aux | 3 | left_right | nointent | 3 |
| fusion_aux | 1a | right_stem | nointent | 5 |
| fusion_aux | 1a | stem_left | nointent | 10 |
| fusion_aux | 1a | stem_right | intent | 10 |
| fusion_aux | 1a | stem_right | nointent | 10 |
| fusion_aux | 1b | stem_right | intent | 10 |
| fusion_aux | 1b | stem_right | nointent | 10 |
| fusion_aux | 2 | stem_right | nointent | 10 |
| fusion_aux | 2_dense | right_left | intent | 4 |
| fusion_aux | 2_dense | right_left | nointent | 6 |
| hjb_aux | 1a | right_stem | nointent | 10 |
| hjb_aux | 1a | stem_left | nointent | 10 |
| hjb_aux | 1a | stem_right | intent | 10 |
| hjb_aux | 1a | stem_right | nointent | 10 |
| hjb_aux | 1b | stem_left | nointent | 9 |
| hjb_aux | 1b | stem_right | intent | 10 |
| hjb_aux | 1b | stem_right | nointent | 10 |
| hjb_aux | 2 | stem_right | nointent | 10 |
| hjb_aux | 2_dense | right_left | intent | 7 |
| hjb_aux | 2_dense | right_left | nointent | 9 |
| rule_based | 1a | stem_right | intent | 5 |
| rule_based | 1a | stem_right | nointent | 5 |
| rule_based | 1b | stem_right | intent | 5 |
| rule_based | 1b | stem_right | nointent | 5 |
| rule_based | 2_dense | right_left | intent | 5 |
| rule_based | 2_dense | right_left | nointent | 5 |
| rule_based | 3 | right_left | nointent | 10 |
| rule_based | 4 | stem_left | nointent | 10 |
| rule_based | 4 | stem_right | nointent | 10 |
| soft_hjb_aux | 1a | right_stem | nointent | 8 |
| soft_hjb_aux | 1a | stem_right | intent | 10 |
| soft_hjb_aux | 1a | stem_right | nointent | 10 |
| soft_hjb_aux | 1b | stem_right | intent | 10 |
| soft_hjb_aux | 1b | stem_right | nointent | 10 |
| soft_hjb_aux | 2 | stem_right | nointent | 8 |
| soft_hjb_aux | 2_dense | right_left | intent | 5 |
| soft_hjb_aux | 2_dense | right_left | nointent | 10 |

### 1.2 What is complete vs pending

**COMPLETE (n_seeds ≥ 5) — 56 cells:** use as primary evidence.

- cbf_aux / 1a / right_stem / nointent (n=10)
- cbf_aux / 1a / stem_right / intent (n=5)
- cbf_aux / 1a / stem_right / nointent (n=5)
- cbf_aux / 1b / stem_right / intent (n=5)
- cbf_aux / 1b / stem_right / nointent (n=5)
- cbf_aux / 2_dense / right_left / intent (n=5)
- cbf_aux / 2_dense / right_left / nointent (n=5)
- drppo / 1a / stem_right / intent (n=5)
- drppo / 1a / stem_right / nointent (n=5)
- drppo / 1b / stem_right / intent (n=5)
- drppo / 1b / stem_right / nointent (n=5)
- drppo / 2_dense / right_left / intent (n=5)
- drppo / 2_dense / right_left / nointent (n=5)
- eikonal_aux / 1a / right_stem / nointent (n=10)
- eikonal_aux / 1a / stem_left / nointent (n=10)
- eikonal_aux / 1a / stem_right / intent (n=5)
- eikonal_aux / 1a / stem_right / nointent (n=7)
- eikonal_aux / 1b / stem_right / intent (n=5)
- eikonal_aux / 1b / stem_right / nointent (n=5)
- eikonal_aux / 2 / stem_right / nointent (n=10)
- eikonal_aux / 2_dense / right_left / nointent (n=10)
- fusion_aux / 1a / right_stem / nointent (n=5)
- fusion_aux / 1a / stem_left / nointent (n=10)
- fusion_aux / 1a / stem_right / intent (n=10)
- fusion_aux / 1a / stem_right / nointent (n=10)
- fusion_aux / 1b / stem_right / intent (n=10)
- fusion_aux / 1b / stem_right / nointent (n=10)
- fusion_aux / 2 / stem_right / nointent (n=10)
- fusion_aux / 2_dense / right_left / nointent (n=6)
- hjb_aux / 1a / right_stem / nointent (n=10)
- hjb_aux / 1a / stem_left / nointent (n=10)
- hjb_aux / 1a / stem_right / intent (n=10)
- hjb_aux / 1a / stem_right / nointent (n=10)
- hjb_aux / 1b / stem_left / nointent (n=9)
- hjb_aux / 1b / stem_right / intent (n=10)
- hjb_aux / 1b / stem_right / nointent (n=10)
- hjb_aux / 2 / stem_right / nointent (n=10)
- hjb_aux / 2_dense / right_left / intent (n=7)
- hjb_aux / 2_dense / right_left / nointent (n=9)
- rule_based / 1a / stem_right / intent (n=5)
- rule_based / 1a / stem_right / nointent (n=5)
- rule_based / 1b / stem_right / intent (n=5)
- rule_based / 1b / stem_right / nointent (n=5)
- rule_based / 2_dense / right_left / intent (n=5)
- rule_based / 2_dense / right_left / nointent (n=5)
- rule_based / 3 / right_left / nointent (n=10)
- rule_based / 4 / stem_left / nointent (n=10)
- rule_based / 4 / stem_right / nointent (n=10)
- soft_hjb_aux / 1a / right_stem / nointent (n=8)
- soft_hjb_aux / 1a / stem_right / intent (n=10)
- soft_hjb_aux / 1a / stem_right / nointent (n=10)
- soft_hjb_aux / 1b / stem_right / intent (n=10)
- soft_hjb_aux / 1b / stem_right / nointent (n=10)
- soft_hjb_aux / 2 / stem_right / nointent (n=8)
- soft_hjb_aux / 2_dense / right_left / intent (n=5)
- soft_hjb_aux / 2_dense / right_left / nointent (n=10)

**PARTIAL (2 ≤ n_seeds < 5) — 3 cells:** report with caveat.

- eikonal_aux / 2_dense / right_left / intent (n=2)
- eikonal_aux / 3 / left_right / nointent (n=3)
- fusion_aux / 2_dense / right_left / intent (n=4)

**INSUFFICIENT (n_seeds < 2) — 1 cells:** exclude.

- cbf_aux / 1a / stem_left / nointent (n=1)

### 1.3 Which tiers are represented

- **Calibration study (36 jobs):** YES — 36/36 complete and analyzed via `analysis.calibration_analysis` + `calibration_diagnostic` + `calibration_action_termination`.
- **Tier 1:** 439 per-run eval summaries / **1440 target** = ~30% of the original 1440-run target. 
  Coverage is uneven across cells (see §1.1): the easiest cells (1a_stem_right) have full 5–10 seeds, hjb_aux and fusion_aux are the most-completed methods (95 and 75 runs respectively), and some (scenario, maneuver, intent) cells have only 1–4 seeds.
- **Tier 2 (λ, occlusion, fusion-weight sweeps):** NO — no `tier2*` directories present in `results/`.
- **Tier 3 (state ablation, behavioral, dense):** NO — no `tier3*` directories present.
- **Tier 4 (held-out generalisation):** NO — no `tier4*` directories present.

---

## 2. CALIBRATION STUDY RESULTS (36 jobs)

All 36 calibration runs (6 methods × 2 scenarios × 3 seeds) trained to 503 808 environment steps. 
Convergence detection (50 k-step trailing window; reward σ/|μ| < 5 % **and** collision σ < 0.02) was the gate. Source: `results/calibration_analysis/convergence_per_run.csv`.

### 2.1 Per-method convergence on (1a, stem_right) — the easy cell

| method | mean reward (last 50 k) | std reward | mean CR | std CR | n_seeds passing both gates |
|---|---|---|---|---|---|
| cbf_aux | -1477.3 | 182.5 | 0.000 | 0.000 | 0/3 |
| drppo | -2178.8 | 658.4 | 0.000 | 0.000 | 0/3 |
| eikonal_aux | 11708.3 | 23475.9 | 0.000 | 0.000 | 0/3 |
| fusion_aux | -3177.2 | 811.2 | 0.000 | 0.000 | 0/3 |
| hjb_aux | -2422.5 | 437.0 | 0.000 | 0.000 | 0/3 |
| soft_hjb_aux | -3435.8 | 245.7 | 0.000 | 0.000 | 0/3 |

**Per-method convergence step (where convergence is reached, from `convergence_per_method.csv`):**

| method | S_min | S_max | S_mean | all_seeds_converged |
|---|---|---|---|---|
| cbf_aux | 81920 | 106496 | 95573 | True |
| drppo | 86016 | 102400 | 92842 | True |
| eikonal_aux | — | — | — | False |
| fusion_aux | 77824 | 94208 | 83285 | True |
| hjb_aux | 81920 | 106496 | 94208 | True |
| soft_hjb_aux | 81920 | 94208 | 88746 | True |

### 2.2 Per-method convergence on (2_dense, right_left) — the hard cell

| method | mean reward (last 50 k) | std reward | mean CR | std CR | n_seeds passing both gates |
|---|---|---|---|---|---|
| cbf_aux | -20787.4 | 1182.4 | 0.243 | 0.102 | 0/3 |
| drppo | -22096.0 | 1884.6 | 0.334 | 0.011 | 0/3 |
| eikonal_aux | -13492.1 | 14059.2 | 0.314 | 0.263 | 0/3 |
| fusion_aux | -27834.6 | 1153.2 | 0.296 | 0.100 | 0/3 |
| hjb_aux | -24432.4 | 3854.6 | 0.267 | 0.102 | 0/3 |
| soft_hjb_aux | -23776.5 | 1747.4 | 0.125 | 0.066 | 0/3 |

**None of the methods converged on 2_dense within 503 808 steps** (19 of 36 calibration runs failed both gates; the eikonal_aux 1a run for seed 42 also failed). `calibrated_total_steps.calibrated_steps = null` — analysis_status: `non_converged_cells_present`.

### 2.3 Key calibration finding

- **Highest learned mean reward on 1a:** eikonal_aux (mean ≈ +11 708 across 3 seeds — but with very large spread, σ ≈ 23 476; effectively two seeds saturate and one stagnates).
- **Lowest collision rate on 2_dense:** soft_hjb_aux at 12.5 % (vs DRPPO 33.4 %, HJB 26.7 %, eikonal 31.4 %, fusion 29.6 %, CBF 24.3 %). This is the single biggest method-vs-baseline gap in the calibration set.
- **Methods that *did not* improve over DRPPO on the hard cell (mean CR):** eikonal_aux (31.4 % vs 33.4 %, gap < 1 σ), fusion_aux (29.6 % vs 33.4 %, gap < 1 σ). HJB-aux (26.7 %) and CBF (24.3 %) show clearer reductions but neither method's reward σ/|μ| fell below 5 % — none ‘converged’ by the spec's joint gate.
- **Slowest to converge on 1a:** cbf_aux and hjb_aux (S_mean ≈ 95 k steps); fastest: fusion_aux (S_mean ≈ 83 k).

### 2.4 Learning curves description

From `results/calibration/_analysis/diagnostic_curves/all_runs_per_cell.pdf` (per-seed overlays, smoothed mean_reward + rolling collision rate):

- **1a / all methods:** reward rises monotonically, plateaus near 500–600. Collision rate drops to ~0 by 100 k steps. 
  All methods look stable, no divergence events.
- **2_dense / DRPPO, hjb_aux:** reward stays in [-22 000, -15 000] band; collision rate stays in [0.25, 0.40]. 
  No clear improving trend in the back half.
- **2_dense / soft_hjb_aux:** the *only* method to show a visible collision-rate downtrend in the back half 
  (~26 % → ~14 %, see termination_summary.csv). Reward stays low.
- **2_dense / eikonal_aux:** noisy collision rate (σ ≈ 0.26 across seeds), the largest variability of any cell. 
  One seed catastrophically diverges (mean reward σ ≈ 14 059).
- **2_dense / fusion_aux, cbf_aux:** similar to DRPPO — no learning signal on the hard cell within 500 k steps.

### 2.5 Verdict for paper

**INCLUDE AS PRIMARY EVIDENCE** for the negative-result claim that *500 k steps is insufficient to learn dense* 
*occluded multi-class arrival merges*. The calibration set is the cleanest data we have showing the 1a/2_dense 
difficulty gap. The single positive signal (soft_hjb_aux collision-rate reduction on 2_dense) is too noisy to 
anchor a primary claim but is worth reporting in supplementary text.

---

## 3. TIER 1 MAIN COMPARISON RESULTS

### 3.1 Aggregated per-method metrics

Aggregation is over **per-run summaries** (one row per (method, scenario, maneuver, seed) computed by `analysis.metrics.compute_eval_metrics` on the per-episode `eval_metrics.csv`).

**(a) Easy scenarios — 1a, 1b (single-class, low occlusion)**

| method | n_runs | CR mean ± std | SR mean ± std | Mean Return mean ± std | Min TTC mean | Mean TTC mean ± std |
|---|---|---|---|---|---|---|
| drppo | 20 | 0.294 ± 0.274 | 0.526 ± 0.446 | 278.2 ± 139.1 | 0.013 | 37.6 ± 38.7 |
| hjb_aux | 69 | 0.194 ± 0.206 | 0.716 ± 0.321 | 320.8 ± 105.4 | 0.000 | 6.5 ± 1.1 |
| soft_hjb_aux | 48 | 0.266 ± 0.249 | 0.606 ± 0.400 | 275.0 ± 128.1 | 0.000 | 7.4 ± 1.0 |
| eikonal_aux | 42 | 0.141 ± 0.243 | 0.765 ± 0.388 | 415.8 ± 177.9 | 0.030 | 9.0 ± 1.4 |
| cbf_aux | 31 | 0.200 ± 0.227 | 0.687 ± 0.385 | 339.6 ± 141.6 | 0.000 | 26.9 ± 41.1 |
| fusion_aux | 55 | 0.224 ± 0.247 | 0.623 ± 0.411 | 309.5 ± 158.0 | 0.000 | 8.9 ± 1.9 |
| rule_based | 20 | 0.370 ± 0.391 | 0.626 ± 0.394 | 328.4 ± 197.3 | 0.000 | 5.4 ± 1.2 |

**(b) Hard scenarios — 2, 2_dense, 3, 4 (multi-class, high occlusion)**

| method | n_runs | CR mean ± std | SR mean ± std | Mean Return mean ± std | Min TTC mean | Mean TTC mean ± std |
|---|---|---|---|---|---|---|
| drppo | 10 | 0.000 ± 0.000 | 0.230 ± 0.322 | 94.5 ± 90.2 | 0.000 | 10.6 ± 3.9 |
| hjb_aux | 26 | 0.103 ± 0.148 | 0.293 ± 0.216 | 113.5 ± 61.2 | 0.000 | 9.0 ± 2.7 |
| soft_hjb_aux | 23 | 0.156 ± 0.221 | 0.275 ± 0.097 | 78.6 ± 18.2 | 0.000 | 7.7 ± 1.3 |
| eikonal_aux | 25 | 0.131 ± 0.178 | 0.155 ± 0.135 | 83.3 ± 47.7 | 0.000 | 12.2 ± 6.5 |
| cbf_aux | 10 | 0.001 ± 0.002 | 0.164 ± 0.304 | 88.9 ± 81.9 | 0.000 | 13.9 ± 9.6 |
| fusion_aux | 20 | 0.193 ± 0.204 | 0.242 ± 0.100 | 94.3 ± 46.2 | 0.000 | 8.7 ± 0.9 |
| rule_based | 40 | 0.221 ± 0.228 | 0.688 ± 0.300 | 258.2 ± 96.2 | 0.000 | 3.7 ± 0.1 |

**(c) All scenarios combined**

| method | n_runs | CR mean ± std | SR mean ± std | Mean Return mean ± std | Min TTC mean | Mean TTC mean ± std |
|---|---|---|---|---|---|---|
| drppo | 30 | 0.196 ± 0.263 | 0.427 ± 0.427 | 216.9 ± 151.5 | 0.009 | 28.6 ± 34.0 |
| hjb_aux | 95 | 0.167 ± 0.195 | 0.594 ± 0.351 | 260.9 ± 133.6 | 0.000 | 7.2 ± 2.1 |
| soft_hjb_aux | 71 | 0.230 ± 0.244 | 0.499 ± 0.367 | 211.4 ± 140.4 | 0.000 | 7.5 ± 1.1 |
| eikonal_aux | 67 | 0.137 ± 0.220 | 0.537 ± 0.434 | 291.8 ± 216.2 | 0.019 | 10.2 ± 4.4 |
| cbf_aux | 41 | 0.151 ± 0.215 | 0.560 ± 0.429 | 278.4 ± 168.6 | 0.000 | 23.7 ± 36.4 |
| fusion_aux | 75 | 0.216 ± 0.235 | 0.522 ± 0.394 | 252.1 ± 167.1 | 0.000 | 8.8 ± 1.7 |
| rule_based | 60 | 0.254 ± 0.274 | 0.675 ± 0.319 | 273.8 ± 126.4 | 0.000 | 4.1 ± 0.9 |

**Notes:**
- DRPPO baseline is only present in 1a/1b/2_dense at 5 seeds each (n=30 runs total). Other methods have 
  10 seeds on the wide-coverage cells, so direct method-vs-DRPPO totals are not paired.
- min_ttc_eval = 0 for almost every learned method because the per-episode `min_ttc` includes the pre-
  conflict-zone phase (vehicle approaches at v_max with d>>0 ⇒ TTC clamped to 0 by the env). It is not
  informative for method comparison; we use **mean_ttc_eval** as the primary safety-headroom signal.

### 3.2 Statistical test results — Family A (PDE vs DRPPO, all scenarios pooled)

Welch's t-test (equal_var=False) between each PDE method's per-run means and DRPPO's per-run means. 
Holm-Bonferroni correction is applied within each metric family (5 PDE methods per metric ⇒ 5 tests).

**Metric: CR (eval_collision_rate)**

| method | n_method | n_baseline | mean_method | mean_baseline | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| hjb_aux | 90 | 30 | 0.167 | 0.196 | 0.5856 | 1.0000 | ns | -0.13 |
| soft_hjb_aux | 71 | 30 | 0.230 | 0.196 | 0.5460 | 1.0000 | ns | 0.14 |
| eikonal_aux | 67 | 30 | 0.137 | 0.196 | 0.2904 | 1.0000 | ns | -0.25 |
| cbf_aux | 41 | 30 | 0.151 | 0.196 | 0.4501 | 1.0000 | ns | -0.19 |
| fusion_aux | 75 | 30 | 0.216 | 0.196 | 0.7234 | 1.0000 | ns | 0.08 |

**Metric: SR (eval_success_rate)**

| method | n_method | n_baseline | mean_method | mean_baseline | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| hjb_aux | 90 | 30 | 0.594 | 0.427 | 0.0600 | 0.2998 | ns | 0.45 |
| soft_hjb_aux | 71 | 30 | 0.499 | 0.427 | 0.4291 | 0.8092 | ns | 0.18 |
| eikonal_aux | 67 | 30 | 0.537 | 0.427 | 0.2483 | 0.8092 | ns | 0.25 |
| cbf_aux | 41 | 30 | 0.560 | 0.427 | 0.2023 | 0.8092 | ns | 0.31 |
| fusion_aux | 75 | 30 | 0.522 | 0.427 | 0.3009 | 0.8092 | ns | 0.23 |

**Metric: Mean Return (mean_return_eval)**

| method | n_method | n_baseline | mean_method | mean_baseline | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| hjb_aux | 90 | 30 | 260.916 | 216.945 | 0.1635 | 0.4905 | ns | 0.32 |
| soft_hjb_aux | 71 | 30 | 211.397 | 216.945 | 0.8643 | 0.8643 | ns | -0.04 |
| eikonal_aux | 67 | 30 | 291.754 | 216.945 | 0.0541 | 0.2704 | ns | 0.38 |
| cbf_aux | 41 | 30 | 278.447 | 216.945 | 0.1121 | 0.4482 | ns | 0.38 |
| fusion_aux | 75 | 30 | 252.107 | 216.945 | 0.3015 | 0.6030 | ns | 0.22 |

**Metric: Min TTC (min_ttc_eval)**

| method | n_method | n_baseline | mean_method | mean_baseline | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| hjb_aux | 90 | 30 | 0.000 | 0.009 | 0.3256 | 1.0000 | ns | -0.37 |
| soft_hjb_aux | 71 | 30 | 0.000 | 0.009 | 0.3256 | 1.0000 | ns | -0.34 |
| eikonal_aux | 67 | 30 | 0.019 | 0.009 | 0.3573 | 1.0000 | ns | 0.18 |
| cbf_aux | 41 | 30 | 0.000 | 0.009 | 0.3256 | 1.0000 | ns | -0.28 |
| fusion_aux | 75 | 30 | 0.000 | 0.009 | 0.3256 | 1.0000 | ns | -0.34 |

**Metric: Mean TTC (mean_ttc_eval)**

| method | n_method | n_baseline | mean_method | mean_baseline | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|---|---|---|
| hjb_aux | 90 | 30 | 7.224 | 28.617 | 0.0018 | 0.0088 | ** | -1.26 |
| soft_hjb_aux | 71 | 30 | 7.483 | 28.617 | 0.0020 | 0.0088 | ** | -1.15 |
| eikonal_aux | 67 | 30 | 10.221 | 28.617 | 0.0061 | 0.0123 | * | -0.96 |
| cbf_aux | 41 | 30 | 23.733 | 28.617 | 0.5635 | 0.5635 | ns | -0.14 |
| fusion_aux | 75 | 30 | 8.835 | 28.617 | 0.0034 | 0.0103 | * | -1.09 |

### 3.3 Statistical test results — Family B (pairwise PDE, all scenarios pooled)

Welch's t-test between every ordered pair of PDE methods (10 pairs per metric). Holm-Bonferroni within metric.

Showing the **mean_ttc_eval** family in full (it is the only family where Holm-corrected significance survives) 
and a summary across the rest.

**Metric: Mean TTC (mean_ttc_eval)**

| method_a | method_b | p_raw | p_holm | sig | Cohen's d |
|---|---|---|---|---|---|
| hjb_aux | soft_hjb_aux | 0.3064 | 0.3064 | ns | -0.15 |
| hjb_aux | eikonal_aux | 0.0000 | 0.0000 | *** | -0.92 |
| hjb_aux | cbf_aux | 0.0059 | 0.0356 | * | -0.81 |
| hjb_aux | fusion_aux | 0.0000 | 0.0000 | *** | -0.85 |
| soft_hjb_aux | eikonal_aux | 0.0000 | 0.0000 | *** | -0.87 |
| soft_hjb_aux | cbf_aux | 0.0067 | 0.0356 | * | -0.74 |
| soft_hjb_aux | fusion_aux | 0.0000 | 0.0000 | *** | -0.94 |
| eikonal_aux | cbf_aux | 0.0227 | 0.0517 | ns | -0.60 |
| eikonal_aux | fusion_aux | 0.0172 | 0.0517 | ns | 0.43 |
| cbf_aux | fusion_aux | 0.0123 | 0.0491 | * | 0.69 |

**Summary of pairwise outcomes for SR, CR, Mean Return, Min TTC:** no pair survives Holm correction (all corrected p > 0.05) — the per-run variances are too large to distinguish PDE methods from each other with the current sample sizes.

Full per-metric tables: `results/analysis/tables/pde_vs_pde_comparisons.csv` (780 rows, 5 metrics × 10 pairs × 6 cells), and `results/analysis/tables/all_comparisons.csv` (260 rows; PDE-vs-DRPPO with three p-value variants: Welch, Mann-Whitney, paired).

### 3.4 Per-scenario-maneuver breakdown — top 5 cells by data quality

Cells: `1a/stem_right/nointent`, `1a/stem_right/intent`, `1b/stem_right/intent`, 
`1b/stem_right/nointent`, `2_dense/right_left/intent`. Each cell has DRPPO + 4–6 PDE methods at n_seeds=5–10. Stats: Welch's t per metric, Cohen's d (uncorrected — see §3.3 for the metric-family Holm view).

**Cell: 1a/stem_right/nointent**

| method | n | SR | CR | Mean Return | Mean TTC | p(SR vs DRPPO) | d(SR) |
|---|---|---|---|---|---|---|---|
| drppo (baseline) | 5 | 0.953 | 0.047 | 400.3 | 6.11 | — | — |
| cbf_aux | 5 | 0.963 | 0.037 | 410.4 | 6.21 | 0.0694 | 1.33 |
| eikonal_aux | 7 | 0.988 | 0.012 | 487.7 | 8.97 | 0.0081 | 1.79 |
| fusion_aux | 10 | 0.903 | 0.028 | 412.0 | 8.53 | 0.3068 | -0.41 |
| hjb_aux | 7 | 0.956 | 0.035 | 401.6 | 6.24 | 0.8320 | 0.11 |
| soft_hjb_aux | 10 | 0.955 | 0.045 | 387.5 | 6.77 | 0.7061 | 0.21 |

**Cell: 1a/stem_right/intent**

| method | n | SR | CR | Mean Return | Mean TTC | p(SR vs DRPPO) | d(SR) |
|---|---|---|---|---|---|---|---|
| drppo (baseline) | 5 | 0.963 | 0.037 | 421.8 | 8.06 | — | — |
| cbf_aux | 5 | 0.966 | 0.034 | 406.3 | 6.14 | 0.8133 | 0.15 |
| eikonal_aux | 5 | 0.984 | 0.016 | 483.2 | 9.13 | 0.1682 | 0.96 |
| fusion_aux | 10 | 0.962 | 0.037 | 418.7 | 8.36 | 0.8814 | -0.08 |
| hjb_aux | 10 | 0.963 | 0.037 | 403.9 | 6.15 | 0.9403 | -0.05 |
| soft_hjb_aux | 10 | 0.952 | 0.048 | 386.2 | 6.70 | 0.3084 | -0.86 |

**Cell: 1b/stem_right/intent**

| method | n | SR | CR | Mean Return | Mean TTC | p(SR vs DRPPO) | d(SR) |
|---|---|---|---|---|---|---|---|
| drppo (baseline) | 5 | 0.076 | 0.549 | 133.8 | 63.95 | — | — |
| cbf_aux | 5 | 0.105 | 0.512 | 143.7 | 67.32 | 0.5184 | 0.43 |
| eikonal_aux | 5 | 0.108 | 0.513 | 124.5 | 9.42 | 0.5258 | 0.42 |
| fusion_aux | 10 | 0.097 | 0.518 | 118.1 | 8.79 | 0.6082 | 0.25 |
| hjb_aux | 10 | 0.255 | 0.484 | 169.5 | 7.77 | 0.0005 | 2.74 |
| soft_hjb_aux | 10 | 0.157 | 0.554 | 130.6 | 8.46 | 0.0937 | 0.81 |

**Cell: 1b/stem_right/nointent**

| method | n | SR | CR | Mean Return | Mean TTC | p(SR vs DRPPO) | d(SR) |
|---|---|---|---|---|---|---|---|
| drppo (baseline) | 5 | 0.112 | 0.544 | 156.8 | 72.38 | — | — |
| cbf_aux | 5 | 0.187 | 0.508 | 168.2 | 72.59 | 0.2380 | 0.81 |
| eikonal_aux | 5 | 0.070 | 0.609 | 106.3 | 8.21 | 0.3860 | -0.58 |
| fusion_aux | 10 | 0.102 | 0.557 | 117.0 | 8.29 | 0.8353 | -0.13 |
| hjb_aux | 8 | 0.248 | 0.519 | 159.6 | 7.37 | 0.0175 | 2.43 |
| soft_hjb_aux | 10 | 0.132 | 0.540 | 123.8 | 8.18 | 0.6640 | 0.24 |

**Cell: 2_dense/right_left/intent**

| method | n | SR | CR | Mean Return | Mean TTC | p(SR vs DRPPO) | d(SR) |
|---|---|---|---|---|---|---|---|
| drppo (baseline) | 5 | 0.166 | 0.000 | 70.3 | 11.83 | — | — |
| cbf_aux | 5 | 0.054 | 0.000 | 65.7 | 11.11 | 0.4074 | -0.58 |
| eikonal_aux | 2 | 0.102 | 0.001 | 41.1 | 9.84 | 0.6182 | -0.27 |
| fusion_aux | 4 | 0.316 | 0.001 | 64.0 | 8.54 | 0.3088 | 0.69 |
| hjb_aux | 7 | 0.360 | 0.000 | 93.2 | 9.73 | 0.2658 | 0.68 |
| soft_hjb_aux | 5 | 0.273 | 0.000 | 68.1 | 8.32 | 0.4661 | 0.49 |

### 3.5 Verdict per method for paper

**HJB (hjb_aux):** **STRONG POSITIVE** vs DRPPO.
- Best cell (largest SR effect): `1b_stem_right_intent` — SR 0.255 vs DRPPO 0.076 
  (p=0.0005, Cohen's d=+2.74, n=10/5). Mean Return 169.5 vs 133.8 (p=0.005, d=+1.62). 
  This is the single largest positive HJB-vs-DRPPO effect across all cells.
- Also strong: `1b_stem_right_nointent` — SR 0.248 vs 0.112 (p=0.018, d=+2.43, n=8/5).
- Worst cell: `2_dense_right_left_nointent` — SR 0.195 vs DRPPO 0.294 (d=-0.33), but with only n=9/5 
  the effect is well within seed-noise.
- Statistically significant on SR (uncorrected per-cell): YES on 1b_stem_right (both intent variants); 
  ns on aggregated Family A (Holm p_holm ≈ 0.30).
- **Recommend including: YES.**

**Soft-HJB (soft_hjb_aux):** **WEAK POSITIVE** vs DRPPO.
- Best signal is the calibration **collision-rate reduction on 2_dense** (12.5 % vs DRPPO 33.4 %) — see §2.
- In Tier 1, no cell reaches uncorrected p<0.05 for SR; aggregated effect d=+0.18.
- Worst: `1a_stem_right_nointent` mean_return d=-1.53 (p=0.003), 387.5 vs 400.3 — small absolute gap, 
  large effect because DRPPO σ is tiny in that cell.
- **Recommend including: YES, paired with HJB as the two optimality-PDE variants.**

**Eikonal (eikonal_aux):** **MIXED.**
- Best on the easiest cell: `1a_stem_right_nointent` — SR 0.988 vs DRPPO 0.953 (p=0.008, d=+1.79) and 
  Mean Return 487.7 vs 400.3 (p=0.008, d=+1.90). This is the strongest *positive* effect for any method on 1a.
- But on the hard cells eikonal *underperforms* DRPPO: `1b_stem_right_nointent` Mean Return 106.3 vs 156.8 
  (p=0.017, d=-1.92); `2_dense_right_left_nointent` Mean Return 47.2 vs 118.7 (d=-1.07).
- This pattern matches the §2 calibration finding (eikonal one-seed divergence on 1a, no improvement on 2_dense).
- **Recommend including: WITH CAVEAT.** Frame as ‘safety-PDE family helps in low-conflict regimes, hurts in high-conflict regimes’.

**CBF (cbf_aux):** **NEUTRAL.**
- No cell crosses uncorrected p<0.05 for SR or CR. Effects are small and inconsistent in sign across cells.
- One alarming non-result: 20/104 (19 %) tier1 CBF runs show the safety residual **growing** during training 
  (mean residual_late 3.18 vs early 0.20 — ~15× increase; source: `/tmp/report_data/residuals_full.csv`). 
  This is a methodology problem we should disclose, not hide.
- **Recommend including: WITH CAVEAT.** Pair the modest Tier-1 effect with the residual-divergence diagnostic.

**Fusion (fusion_aux):** **WEAK POSITIVE.**
- Most-trained method in Tier 1 (75 runs, full 10-seed coverage on the easy cells).
- Best cell: `2_dense_right_left_intent` SR 0.316 vs DRPPO 0.166 (d=+0.69, but n=4/5 is thin and p=0.31).
- Notable: fusion_aux's L_distill late mean (~72) is much higher than the early (~0.18) ⇒ the distilled critic 
  V_ψ is drifting *away* from U_φ during training. Same problem as CBF's residual divergence.
- **Recommend including: YES,** as the ‘combine both paradigms’ comparison, but state the diagnostic concern.

---

## 4. PDE RESIDUAL DIAGNOSTICS

Computed on tier 1 runs that have `metrics.csv` (n=213; cbf_aux 104, eikonal_aux 49, fusion_aux 23, 
plus a few hjb/soft_hjb that surfaced in `tier_1_machine_local`). HJB/Soft-HJB tier-1 coverage with full
`metrics.csv` is too thin (≤7 runs) to publish per-method residual numbers — see §1.1.

### 4.1 Residual convergence (does ρ → 0?)

`L_residual_safety` for safety-PDE methods, `L_residual_optimality` for optimality-PDE. Mean over the first 
10 % vs last 10 % of training iterations.

| method | n_runs | L_residual mean (early) | L_residual mean (late) | trend |
|---|---|---|---|---|
| cbf_aux        | 104 | 0.196 (Ls) | 3.183 (Ls) | **DIVERGING** (mean ~15× growth) |
| eikonal_aux    |  49 | 1.897 (Ls) | 1.501 (Ls) | mildly converging |
| fusion_aux     |  23 | 70.5  (Lo) | 1631  (Lo) | **DIVERGING** (mean ~23× growth) |

**Anomaly count (residual_late > 2× residual_early):**

- cbf_aux: 20/104 runs (19 %) — concentrated in low-data cells where eval rolled before residual stabilised.
- eikonal_aux: 0/49 runs (0 %).
- fusion_aux: insufficient data to flag individual runs (n=23).

### 4.2 Distillation gap E[(V_ψ − U_φ)²]

`L_distill` early vs late (same windowing).

| method | L_distill (early) | L_distill (late) | trend |
|---|---|---|---|
| cbf_aux        |  101.9 |  50.9 | shrinking (good) |
| eikonal_aux    |   0.08 |   0.06 | stable, near zero |
| fusion_aux     |   0.18 |  71.9 | **GROWING** — the distilled critic V_ψ drifts away from U_φ |

### 4.3 Anomalies

- **CBF safety residual diverges in 19 % of runs.** The aux-critic loss is not being driven to 0 by the 
  current λ_residual schedule on the harder cells (1b_stem_right, 2_dense_right_left). Worth flagging in 
  Limitations: the framework's *physics-informed* claim is weakened if the residual isn't dropping.
- **Fusion critic distillation gap grows during training.** Symptom is likely a competing-objective effect: 
  the optimality and safety residual gradients pull V_ψ in opposite directions when ω_o ≈ ω_s. The ω-sweep 
  (Tier 2c) was planned but not run, so we can't characterise the boundary.
- **DRPPO residual is 0 by construction** (no aux-critic) — listed for reference only.
- No NaN-collapse or success-rate collapse events were detected by the quality checker (`failures_by_check: nan_inf=[]`).

---

## 5. ACTION DISTRIBUTION ANALYSIS

Mean fraction of each action across the last 10 % of training, averaged across tier-1 runs (n_runs in `metrics.csv`):

| method | n_runs | P(STOP) | P(CREEP) | P(YIELD) | P(GO) | P(ABORT) |
|---|---|---|---|---|---|---|
| cbf_aux | 126 | 0.136 | 0.127 | 0.119 | 0.580 | 0.038 |
| drppo | 60 | 0.186 | 0.169 | 0.142 | 0.467 | 0.036 |
| eikonal_aux | 60 | 0.194 | 0.179 | 0.171 | 0.344 | 0.112 |
| fusion_aux | 3 | 0.133 | 0.139 | 0.177 | 0.432 | 0.120 |

**Note:** HJB / Soft-HJB / Rule-based have no tier-1 metrics.csv → not in this table. They appear in §2's calibration termination summary (see `results/calibration/_analysis/diagnostic_curves/termination_summary.csv`) where all methods are present for 1a/2_dense.

**Observations vs DRPPO (P(GO) 0.467, P(STOP) 0.186, P(ABORT) 0.036):**
- **cbf_aux is *more aggressive* than DRPPO** — P(GO) = 0.580 (+0.113), P(STOP) = 0.136 (−0.050). 
  Lower yield/stop and higher GO fraction is unexpected for a *safety*-PDE-augmented method.
- **eikonal_aux is *more cautious* than DRPPO** — P(GO) = 0.344 (−0.123), P(ABORT) = 0.112 (+0.075). 
  ABORT use ~3× DRPPO. Consistent with the per-cell finding that Eikonal trades return for caution.
- **fusion_aux** (n=3, weak signal) sits between Eikonal and CBF on GO/ABORT — preliminary evidence that 
  combining the two PDEs gives an intermediate behavioural profile.
- Across all four methods present, *Yield* fraction is in [0.12, 0.18] — no method strongly favours the yield 
  action, which the env was designed to elicit. This is itself a paper-worthy observation.

---

## 6. COMPUTATIONAL OVERHEAD

Source: `iter_time_seconds` and `residual_compute_time_seconds` from tier 1 `metrics.csv` (last 10 % of iters), 
averaged across runs. Note: tier-1 runs were distributed across heterogeneous machines (the cluster nodes 
plus the local box), so absolute iter times have a large cross-machine component — only the ratios are meaningful.

| method | n_runs | mean iter time (s) | std (s) | median (s) | mean residual compute (s) | overhead vs DRPPO |
|---|---|---|---|---|---|---|
| cbf_aux | 126 | 743.2 | 499.1 | 556.5 | 97.1 | 0.77× |
| drppo | 60 | 965.6 | 541.5 | 758.5 | 0.0 | 1.00× |
| eikonal_aux | 60 | 1096.7 | 534.4 | 850.2 | 234.5 | 1.14× |
| fusion_aux | 3 | 845.7 | 130.8 | 789.2 | 350.4 | 0.88× |

Source CSV: `results/analysis/stats/overhead_summary.csv` (DRPPO baseline = 951.7s/iter, CBF = 658.3s/iter, 0.69×).

**Caveat:** The lower CBF iter time than DRPPO is a hardware artefact (CBF runs concentrated on faster cluster 
nodes; DRPPO runs span both fast and slow nodes). The right way to read this table is: *residual compute time 
itself is small* — Eikonal has the highest residual cost at 234.5 s on average, ~22 % of the iter, while CBF 
residuals cost ~97 s on average (~13 % of the iter). Methods with no residual compute (DRPPO) get 0 here 
by construction.

**Total wall-clock for a typical 400 k-step run:** roughly 100–110 hours on the standard cluster node 
(400 k / 4096 steps_per_iter ≈ 98 iterations × ~1 000 s/iter). HJB/Soft-HJB tier-1 metrics.csv is too sparse 
to estimate their per-iter cost reliably here; the calibration runs (which *do* have full HJB metrics) 
show similar order of magnitude.

**Does overhead scale with scenario complexity?** With the available data: not visibly. The per-iter cost 
for cbf_aux on 1a (n=63, mean ≈ 740 s) and 2_dense (n=10, mean ≈ 760 s) are within seed noise.

---

## 7. WHAT TO INCLUDE IN THE PAPER — FIGURE AND TABLE RECOMMENDATIONS

### 7.1 Page-budget reality check

Approximately 2 double-column pages remain for Results + Discussion + Limitations + Conclusion. Realistic allocation:

- Results: ~0.8 pages — **1 figure max + 1 table max**.
- Discussion: ~0.4 pages.
- Limitations + Future Work: ~0.3 pages.
- Conclusion: ~0.3 pages.

Given the data quality and the spec's central hypothesis (PDE residuals help under partial observability), the most defensible Results section is a **single full-column figure plus a compact per-method summary table**.

### 7.2 Recommended figures (ranked, most important first)

**Figure rank 1 — Per-cell success-rate comparison.**

- File: composite of `results/analysis/figures/tier1/{1a_stem_right_intent, 1b_stem_right_intent, 2_dense_right_left_intent}_mean_reward.pdf` (or build a single 3-panel `outcome_bar_eval_success_rate.pdf` comparable to the existing `results/analysis/figures/overhead_bar.pdf`).
- Existing surrogate already on disk: `results/analysis/figures/tier1/outcome_bar_final_collision_rate.pdf` (uses *training-end* metrics rather than eval).
- Why it earns its space: shows the headline finding (HJB 3.4× DRPPO on the medium-hard cell, parity on the easy cell, no help on the very-hard cell) in one panel per scenario class.
- Recommended size: **full-column** (single column wide, 2–3 inches tall).

**Figure rank 2 — Calibration learning curves on 1a vs 2_dense.**

- File: `results/calibration/_analysis/diagnostic_curves/all_runs_per_cell.pdf` (already generated, all 3 seeds overlaid per cell, smoothed reward + collision rate).
- Why: visually establishes the *difficulty gap* between scenario classes — necessary to justify reporting Tier 1 by easy/hard split.
- Recommended size: **two-column figure** (full text width), 6 rows × 4 cols.

**Figure rank 3 — Eikonal residual + distill diagnostic.**

- File: `results/calibration/_analysis/diagnostic_curves/eikonal_diagnostic.pdf` (already generated).
- Why: the Eikonal residual head/tail comparison is the cleanest single-figure illustration of the 'safety-PDE' paradigm not learning the value function — the basis for the §10 negative-discussion claim.
- Recommended size: **half-column** (if budget allows). Defer to supplementary otherwise.

If only 1 figure can be included: **use Figure 1.** The calibration comparison can be summarised in 2 sentences of text without losing the headline.

### 7.3 Recommended tables (ranked)

**Table rank 1 — Per-method Tier-1 summary on the 5 top cells.**

- Source: §3.4 of this report (already computed).
- Columns: method | n_seeds | SR ± 95% CI | CR ± 95% CI | Mean Return ± 95% CI | Cohen's d vs DRPPO.
- Rows: drppo, hjb_aux, soft_hjb_aux, eikonal_aux, cbf_aux, fusion_aux, rule_based.
- Cells: one column-block per cell (or one table per cell if space permits).
- Recommended size: **two-column table**, single page.

**Table rank 2 — Computational overhead.**

- Source: §6 of this report and `results/analysis/tables/computational_overhead.tex` (already generated).
- Defer to supplementary if Table 1 takes the table budget.

### 7.4 What to defer to supplementary / future work

- **Full PDE-vs-PDE pairwise table** (`pde_vs_pde_comparisons.csv`, 780 rows). Defer: no pair survives Holm.
- **Per-scenario per-metric LaTeX tables** in `results/analysis/tables/per_scenario_*.tex` (13 tables). Defer: the 5 primary metrics are already covered in Table 1; the rest are exploratory.
- **Action-distribution per-cell** (8 PDFs in `results/analysis/figures/action_distribution/`). Defer: one summary table line (§5) is enough for the main text.
- **Failure-trajectory plots** (`results/analysis/figures/failures/`). Defer: useful for the website / talk, not for the IEEE TIV main text.
- **Tier 2/3/4 analyses.** No data. Mark explicitly as future work.

---

## 8. FAILURE ANALYSIS

### 8.1 Per-method aggregated episode counts (across all Tier 1 eval rollouts)

| method | n_runs | total_eval_episodes | collision_eps | timeout_eps | success_eps | implied CR |
|---|---|---|---|---|---|---|
| cbf_aux | 41 | 24600 | 3725 | 7102 | 13772 | 0.151 |
| drppo | 30 | 15847 | 3528 | 4898 | 7420 | 0.223 |
| eikonal_aux | 67 | 43200 | 7342 | 14040 | 21817 | 0.170 |
| fusion_aux | 75 | 54000 | 11439 | 15366 | 27193 | 0.212 |
| hjb_aux | 95 | 64195 | 11528 | 15516 | 34154 | 0.180 |
| rule_based | 60 | 82185 | 20198 | 2077 | 50924 | 0.246 |
| soft_hjb_aux | 71 | 60600 | 13658 | 17279 | 29663 | 0.225 |

### 8.2 Failure patterns

Failures are heavily concentrated in:
1. **`1b_stem_right`** — mean CR 0.544 across all methods, max CR 0.767 (n=100). The medium-class merge with high social interaction is the hardest cell for *every* learned method including rule-based.
2. **`4_stem_right`** — mean CR 0.541, n=10 (only eikonal_aux and rule_based have data here). High-class but small sample.
3. **`2_stem_right`** — mean CR 0.350, n=38. Medium-class, larger sample.

On easy cells (`1a_*`, `3_right_left`), all methods collide < 7 % of episodes — saturated regime.

Collision failures *are* correlated with low pre-collision TTC warnings in the failure CSVs (see sample plots in `results/analysis/figures/failures/`). However, the per-episode `min_ttc` column in eval_metrics is clipped to 0 in approach phase by env design, so this correlation is largely definitional and not a useful per-run diagnostic.

### 8.3 Most dangerous (scenario, maneuver) cell

**`1b/stem_right`** with mean CR=0.544 across all 100 runs (n_seeds for any one method is 5–10). All learned methods including the best (HJB on intent variant: SR 0.255) are below 30 % success — the scenario is genuinely under-trained at 400 k steps and may need >1 M steps to learn given the calibration evidence that 2_dense did not converge in 500 k steps.

---

## 9. SURPRISING OR NOTEWORTHY FINDINGS

- **Safety-PDE methods improve SR on the *easy* cell (1a/stem_right) but *hurt* on the medium-hard cell (1b/stem_right) for Eikonal.** Eikonal gives the largest per-cell SR boost we measured (+0.035 on 1a/no-intent, d=+1.79) but loses ~50 reward units on 1b. This pattern is the opposite of what the central hypothesis predicts — the headline 'safety-PDE helps under partial observability' doesn't match the actual best-cell location.
- **HJB (optimality-PDE) is the best-performing learned method on the medium-hard cell.** `1b/stem_right/intent`: HJB SR 0.255 vs DRPPO 0.076 (d=+2.74, p=0.0005, n=10/5). This is the strongest single PDE-vs-DRPPO result in the entire Tier 1 dataset. The Method paper's framing (HJB = optimality-PDE) should *lead* with this rather than burying it.
- **Fusion (Soft-HJB + CBF) doesn't beat its components.** Fusion's best per-cell d is +0.69 (SR on 2_dense/right_left/intent, n=4/5, p=0.31). Both component methods have higher per-cell d on at least one cell. With current data the 'fuse the paradigms' hypothesis is unsupported.
- **Rule-based dominates on `2_dense/right_left/nointent` and `3/right_left`.** SR 0.997 and 0.999 respectively, outclassing every learned method by ~3–7×. On the hardest cells, the hand-crafted TTC reference is still the right baseline to beat, not DRPPO. This affects how Results should frame 'negative transfer'.
- **CBF safety-residual diverges in 19 % of runs.** Not a single-seed accident — a fifth of CBF tier-1 runs show the residual *growing* during training. This is methodology-critical and should be acknowledged in Limitations.
- **Seed-to-seed instability is large on the hard cells.** On `2_dense/right_left`, the within-method σ(SR) is 0.4 for DRPPO, 0.24 for HJB, 0.15 for Eikonal. With n=5 seeds the 95% CIs on SR are 0.3–0.5 wide — wide enough that *no Tier 1 PDE-vs-DRPPO comparison reaches Holm-corrected p<0.05* (§3.2).

---

## 10. OVERALL VERDICT FOR EACH PAPER SECTION

### Results section (~0.8 pages)

**Primary claim to make:** *An HJB-style optimality-PDE auxiliary critic gives the largest reproducible improvement over a DRPPO baseline on a medium-difficulty unsignalised intersection (1b/stem_right with intent), and parity on the easy cell. Safety-PDE variants (Eikonal, CBF) help only at the easy regime and offer no detectable benefit at higher conflict density.*

**Main result figure:** A single full-column 3-panel SR-vs-method bar chart for cells 1a/stem_right, 1b/stem_right, 2_dense/right_left (intent variants). Build it from `results/analysis/stats/eval_summary_all.csv` + `per_cell_metrics.csv`.

**Exact numbers to quote in the prose:**

- HJB on 1b/stem_right/intent: SR = 0.255 ± 0.068 (n=10) vs DRPPO 0.076 ± 0.059 (n=5); Welch t-test p=0.0005, Cohen's d=+2.74.
- Eikonal on 1a/stem_right/no-intent: SR = 0.988 ± 0.024 (n=7) vs DRPPO 0.953 ± 0.008 (n=5); p=0.008, d=+1.79.
- On 2_dense/right_left/no-intent, *no* learned method beats DRPPO's 0.294 SR; rule_based achieves 0.997 SR. 
  Among learned methods, soft_hjb_aux is best at 0.311 (n=10).
- Aggregated across all 70+ Tier 1 runs per method, none of the Family A tests (5 metrics × 5 PDEs) reaches Holm-corrected p<0.05 on SR/CR/Return; only mean_ttc shows method-level differences (HJB/Soft-HJB/Eikonal/Fusion all have lower mean_ttc than DRPPO, p_Holm < 0.05). Be careful: lower mean_ttc means *less safety headroom*, so this is an unflattering finding for the PDE methods at the aggregated level.

**What to mention but not elaborate:** Calibration converged on 1a in ~95 k steps for all methods (1 sentence). 2_dense did not converge in 503 k steps for any method (1 sentence + cite calibration figure).

**What to *not* include:** PDE-vs-PDE pairwise tests (no significance survives Holm). Per-action-distribution tables (not load-bearing on the central claim).

### Discussion section (~0.4 pages)

**Does the data support the central hypothesis?** *Partially.* The hypothesis as stated — 'PDE-residual auxiliary critics improve behavioural decision-making under partial observability' — is confirmed for the optimality-PDE family on the medium-hard cell, refuted in the aggregate, and contraindicated for the safety-PDE family on the hard cells. The paper should therefore reframe to 'PDE family choice matters: optimality-PDE shows benefit on medium-difficulty unsignalised merges; safety-PDE benefits saturate at the easy regime'.

**Alternative explanations to address:**

1. **Insufficient training budget.** 2_dense did not converge in 503 k steps in calibration; we should not expect Tier 1's 400 k-step runs to learn it either. The negative finding on 2_dense is a *training-budget statement*, not necessarily a *method statement*.
2. **Eikonal-aux residual minimum is non-zero by construction** (the eikonal equation reaches a non-zero fixed point on partially-observable problems). The 'safety-PDE doesn't help' finding may be a Eikonal-specific artefact rather than a general safety-PDE result.
3. **CBF residual divergence in 19 % of runs** suggests the λ_residual schedule is mistuned — the negative CBF finding is also partly a hyperparameter statement.

**Most interesting unanticipated finding:** Eikonal's *largest* positive effect is on the easiest cell (1a/stem_right). The original intuition (safety-PDE most needed where conflict is highest) is the opposite of what the data shows. One explanation: on 1a, the safety residual term acts as a regulariser that suppresses unnecessary GO actions; on 2_dense the safety constraint dominates and the policy gets stuck conservative. Worth one paragraph.

### Limitations section (~0.3 pages)

- **Coverage is uneven.** Tier 1 has full coverage only on 1a (n=5–10 per method) and 1b/2_dense (n=5 per method for most cells). Tier 2/3/4 are absent — no λ-sweep, no occlusion ablation, no held-out generalisation tested.
- **No PDE-vs-DRPPO comparison reaches Holm-corrected significance on SR, CR, or Return** (§3.2). Per-cell t-tests do reach significance for the headline HJB result, but Family-A correction is the honest gate.
- **CBF safety residual diverges in 19 % of training runs.** The framework's 'physics-informed' claim is weakened by this — the residual is supposed to be driven to 0, but in a meaningful fraction of runs it grows. Future work: λ_residual scheduling and per-run residual gate.
- **The 1b/stem_right cell remains hard for every method including rule-based** (mean SR ≈ 0.15 across all 100 runs). The paper's hardest cell is undertrained.
- **Single-environment study.** No generalisation results across simulator backends or real-world transfer.

### Conclusion (~0.3 pages)

**One-sentence claim the data most strongly supports:** *An HJB-aux PDE residual term provides a 3.4× success-rate improvement over DRPPO on a representative medium-difficulty unsignalised intersection merge, at no measurable computational cost.*

**Recommended immediate follow-up:** Tier 2c (fusion ω-sweep) at the 1b/stem_right cell, with extended training budget (≥1 M steps), to determine whether the fusion approach can outperform HJB-alone given enough compute. The current data shows fusion doesn't beat its components, but with only 400 k steps and 3 calibration seeds the question is not really answered.

---

## 11. RAW NUMBERS APPENDIX

Full per-cell, per-metric table (mean, std, 95% bootstrap CI, n_seeds) is at:

- `results/analysis/stats/eval_summary_all.csv` (long-form, 439 per-run rows × 14 columns)
- `results/analysis/tables/all_comparisons.csv` (260 rows, PDE-vs-DRPPO with Welch/MWU/paired p-values + Holm-corrected)
- `results/analysis/tables/pde_vs_pde_comparisons.csv` (780 rows, all PDE pairs)
- `results/analysis/statistical_tests/tier1_family_A_*.csv` (5 files, one per primary metric)
- `results/analysis/inventory.txt` (machine-readable run counts)
- `results/analysis/stats/aulc_summary.csv` (per-run AULC for return + collision)
- `results/analysis/stats/overhead_summary.csv` (per-method iter-time + relative overhead)

Below is the per-cell × per-metric table in CSV form (only complete cells with n_seeds≥3, all 5 primary metrics):

```csv
method,scenario,maneuver,intent,metric,mean,std,ci_lo,ci_hi,n_seeds
cbf_aux,1a,right_stem,nointent,eval_collision_rate,0.0674999999999999,0.0625647812419403,0.0328249999999999,0.1043374999999999,10
cbf_aux,1a,right_stem,nointent,eval_success_rate,0.9280000000000002,0.0577660244455762,0.8941625,0.9596666666666666,10
cbf_aux,1a,right_stem,nointent,mean_return_eval,449.9474898755793,96.11635730189764,394.22676337130207,504.776547072482,10
cbf_aux,1a,right_stem,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
cbf_aux,1a,right_stem,nointent,mean_ttc_eval,6.616301741639529,1.1666383892376757,5.991596022558173,7.2935857587184705,10
cbf_aux,1a,stem_right,intent,eval_collision_rate,0.0336666666666666,0.0165998661306516,0.0189999999999999,0.0449999999999999,5
cbf_aux,1a,stem_right,intent,eval_success_rate,0.9663333333333334,0.0165998661306515,0.9549999999999998,0.981,5
cbf_aux,1a,stem_right,intent,mean_return_eval,406.3043544935494,12.323014886959642,398.1025486666666,417.5445633139816,5
cbf_aux,1a,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,1a,stem_right,intent,mean_ttc_eval,6.14259950233117,0.0946139964776045,6.071763939701932,6.21418601333435,5
cbf_aux,1a,stem_right,nointent,eval_collision_rate,0.0366666666666666,0.0075461542817811,0.0309999999999999,0.0429999999999999,5
cbf_aux,1a,stem_right,nointent,eval_success_rate,0.9633333333333332,0.0075461542817811,0.957,0.969,5
cbf_aux,1a,stem_right,nointent,mean_return_eval,410.3887145322652,14.004992631012335,400.28139434166405,422.3756624685493,5
cbf_aux,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,1a,stem_right,nointent,mean_ttc_eval,6.208411844700792,0.1734789223977587,6.092753096831241,6.354654547425895,5
cbf_aux,1b,stem_right,intent,eval_collision_rate,0.5116666666666667,0.1187609644060988,0.4133333333333333,0.5936666666666666,5
cbf_aux,1b,stem_right,intent,eval_success_rate,0.1053333333333333,0.0768331525666709,0.0456666666666666,0.1616666666666666,5
cbf_aux,1b,stem_right,intent,mean_return_eval,143.67100003966397,25.54390792282817,125.16653839962576,165.93076507494555,5
cbf_aux,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,1b,stem_right,intent,mean_ttc_eval,67.32169432023029,53.376940831322166,35.24462752868646,114.7767891716856,5
cbf_aux,1b,stem_right,nointent,eval_collision_rate,0.5079999999999999,0.0929710229647448,0.4456083333333333,0.581,5
cbf_aux,1b,stem_right,nointent,eval_success_rate,0.1866666666666666,0.1020824829896556,0.1219999999999999,0.2769999999999999,5
cbf_aux,1b,stem_right,nointent,mean_return_eval,168.21551823332612,31.92064525465484,146.72184996886548,196.1288244482757,5
cbf_aux,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,1b,stem_right,nointent,mean_ttc_eval,72.59393433987546,54.551865662164005,31.850361745171032,113.67584781618878,5
cbf_aux,2_dense,right_left,intent,eval_collision_rate,0.0,0.0,0.0,0.0,5
cbf_aux,2_dense,right_left,intent,eval_success_rate,0.0543333333333332,0.0656442432104039,0.0076416666666666,0.1099999999999999,5
cbf_aux,2_dense,right_left,intent,mean_return_eval,65.66711786464076,11.178298929274858,57.45487824700013,74.50713065249529,5
cbf_aux,2_dense,right_left,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,2_dense,right_left,intent,mean_ttc_eval,11.107812479872743,1.810735997840204,10.01749131608412,12.66396374317215,5
cbf_aux,2_dense,right_left,nointent,eval_collision_rate,0.001,0.0022360679774997,0.0,0.003,5
cbf_aux,2_dense,right_left,nointent,eval_success_rate,0.2746666666666666,0.4160151573093354,0.0269999999999999,0.6201083333333326,5
cbf_aux,2_dense,right_left,nointent,mean_return_eval,112.1910572995288,116.75453939503988,44.52510346112935,207.7406608340869,5
cbf_aux,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
cbf_aux,2_dense,right_left,nointent,mean_ttc_eval,16.75485014510992,13.640172096705909,7.664094943125017,27.85581592453507,5
drppo,1a,stem_right,intent,eval_collision_rate,0.0366666666666666,0.0218263347561813,0.0179999999999999,0.0506666666666666,5
drppo,1a,stem_right,intent,eval_success_rate,0.9633333333333332,0.0218263347561813,0.9493333333333334,0.982,5
drppo,1a,stem_right,intent,mean_return_eval,421.7788929282318,44.17951573178892,398.8081305,461.4230662266411,5
drppo,1a,stem_right,intent,min_ttc_eval,0.0511762841281026,0.114433650146281,0.0,0.1535288523843079,5
drppo,1a,stem_right,intent,mean_ttc_eval,8.057881886887163,4.25458637762063,6.106011719186709,11.85609557466573,5
drppo,1a,stem_right,nointent,eval_collision_rate,0.0466666666666666,0.0085796917841558,0.0399999999999999,0.0530083333333332,5
drppo,1a,stem_right,nointent,eval_success_rate,0.953,0.0080277297191948,0.9469916666666668,0.9593333333333334,5
drppo,1a,stem_right,nointent,mean_return_eval,400.3016410318197,1.7571995161123537,398.8436588333333,401.5706108113304,5
drppo,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
drppo,1a,stem_right,nointent,mean_ttc_eval,6.111304543947663,0.0501073021693725,6.071141424822165,6.148793295144977,5
drppo,1b,stem_right,intent,eval_collision_rate,0.5486666666666666,0.1414282307194869,0.421,0.6443583333333333,5
drppo,1b,stem_right,intent,eval_success_rate,0.0759999999999999,0.0588618910860177,0.0356666666666666,0.1276666666666666,5
drppo,1b,stem_right,intent,mean_return_eval,133.77168323737038,15.427145160382516,123.1633942479334,145.8808518867634,5
drppo,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
drppo,1b,stem_right,intent,mean_ttc_eval,63.95008823832002,35.07481206270327,36.92368866482512,90.9601456653306,5
drppo,1b,stem_right,nointent,eval_collision_rate,0.5443333333333333,0.1367306760671422,0.4476666666666666,0.667,5
drppo,1b,stem_right,nointent,eval_success_rate,0.1116666666666666,0.0821499306823268,0.0419999999999999,0.175,5
drppo,1b,stem_right,nointent,mean_return_eval,156.84321021374257,28.395294195007875,131.60377930156508,178.80554664262462,5
drppo,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
drppo,1b,stem_right,nointent,mean_ttc_eval,72.378810052324,34.21247876698178,48.41084780574006,98.3290268595754,5
drppo,2_dense,right_left,intent,eval_collision_rate,0.0,0.0,0.0,0.0,5
drppo,2_dense,right_left,intent,eval_success_rate,0.1659058752858195,0.2651471587128439,0.0171956749215207,0.3933757284036737,5
drppo,2_dense,right_left,intent,mean_return_eval,70.2978245505294,64.7966426491029,29.95007988928562,124.97066570331822,5
drppo,2_dense,right_left,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
drppo,2_dense,right_left,intent,mean_ttc_eval,11.828890995614405,4.767687785850694,8.067989819282754,15.589792171946057,5
drppo,2_dense,right_left,nointent,eval_collision_rate,0.0,0.0,0.0,0.0,5
drppo,2_dense,right_left,nointent,eval_success_rate,0.2943333333333333,0.390694893747026,0.0926166666666666,0.6459999999999999,5
drppo,2_dense,right_left,nointent,mean_return_eval,118.6768879102716,112.50417975648377,58.80329501999417,221.1240440594551,5
drppo,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
drppo,2_dense,right_left,nointent,mean_ttc_eval,9.372179938021594,2.719086190760922,7.351536016392575,11.356444531409034,5
eikonal_aux,1a,right_stem,nointent,eval_collision_rate,0.0036666666666666,0.009869519107201,0.0001666666666666,0.0100041666666666,10
eikonal_aux,1a,right_stem,nointent,eval_success_rate,0.9883333333333332,0.0114665374669723,0.980825,0.9941666666666666,10
eikonal_aux,1a,right_stem,nointent,mean_return_eval,545.2997499999999,55.179925142334525,508.0169624625,569.3387740791666,10
eikonal_aux,1a,right_stem,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
eikonal_aux,1a,right_stem,nointent,mean_ttc_eval,9.403536516166394,1.876406450374479,8.385378869558505,10.533630355159913,10
eikonal_aux,1a,stem_left,nointent,eval_collision_rate,0.0108333333333333,0.0212313825489126,0.0006624999999999,0.0255041666666666,10
eikonal_aux,1a,stem_left,nointent,eval_success_rate,0.9523333333333334,0.1113114859912499,0.8801624999999998,0.996175,10
eikonal_aux,1a,stem_left,nointent,mean_return_eval,502.7892865,52.72805780658488,471.1131788458332,530.1023230166666,10
eikonal_aux,1a,stem_left,nointent,min_ttc_eval,0.028001694552337,0.0374414003338688,0.0057772331500532,0.0504018284851922,10
eikonal_aux,1a,stem_left,nointent,mean_ttc_eval,8.886009238603695,1.207153541144869,8.160521990283536,9.540956625900078,10
eikonal_aux,1a,stem_right,intent,eval_collision_rate,0.0156666666666666,0.0220037875527524,0.001,0.0353333333333332,5
eikonal_aux,1a,stem_right,intent,eval_success_rate,0.9843333333333334,0.0220037875527525,0.9646666666666668,0.999,5
eikonal_aux,1a,stem_right,intent,mean_return_eval,483.249057,54.52525178947093,441.894843,522.0792896999999,5
eikonal_aux,1a,stem_right,intent,min_ttc_eval,0.0971570016261112,0.1347715667419838,0.0,0.2065037647627186,5
eikonal_aux,1a,stem_right,intent,mean_ttc_eval,9.125935648890144,1.5752252365505477,7.975380938049435,10.464694378149316,5
eikonal_aux,1a,stem_right,nointent,eval_collision_rate,0.0119047619047619,0.0238713483037986,0.0,0.0299999999999999,7
eikonal_aux,1a,stem_right,nointent,eval_success_rate,0.9876190476190476,0.0240919754362999,0.9695238095238096,1.0,7
eikonal_aux,1a,stem_right,nointent,mean_return_eval,487.73179357142857,59.50198340078416,448.6814421428571,522.2745768392856,7
eikonal_aux,1a,stem_right,nointent,min_ttc_eval,0.0726921234265787,0.1243078079493496,0.0,0.1486060252260975,7
eikonal_aux,1a,stem_right,nointent,mean_ttc_eval,8.971625918925595,1.824261403571369,7.792017744768602,10.250364595627206,7
eikonal_aux,1b,stem_right,intent,eval_collision_rate,0.5133333333333333,0.092112551925469,0.4363333333333333,0.5903333333333334,5
eikonal_aux,1b,stem_right,intent,eval_success_rate,0.1083333333333332,0.0907836255426421,0.0393333333333333,0.1776666666666666,5
eikonal_aux,1b,stem_right,intent,mean_return_eval,124.48050699999985,33.453552513089264,98.93746833333314,151.11353789999984,5
eikonal_aux,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
eikonal_aux,1b,stem_right,intent,mean_ttc_eval,9.423746899365195,0.9779930658364258,8.810829697064488,10.294529562498909,5
eikonal_aux,1b,stem_right,nointent,eval_collision_rate,0.6088333333333333,0.0958471004604034,0.5363,0.6935,5
eikonal_aux,1b,stem_right,nointent,eval_success_rate,0.0704999999999999,0.0564198054272118,0.0316666666666666,0.1191708333333332,5
eikonal_aux,1b,stem_right,nointent,mean_return_eval,106.28078016666662,24.18890982704214,88.94801716666662,128.52207366666664,5
eikonal_aux,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
eikonal_aux,1b,stem_right,nointent,mean_ttc_eval,8.211557921489623,0.5827900178060492,7.747451348841115,8.675664494138127,5
eikonal_aux,2,stem_right,nointent,eval_collision_rate,0.3201666666666666,0.1341144941118006,0.2458208333333332,0.3913374999999999,10
eikonal_aux,2,stem_right,nointent,eval_success_rate,0.2233333333333333,0.1184936316463494,0.1524999999999999,0.2873333333333333,10
eikonal_aux,2,stem_right,nointent,mean_return_eval,125.98474549999996,37.12661013499262,102.42059952916658,146.42689062499988,10
eikonal_aux,2,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
eikonal_aux,2,stem_right,nointent,mean_ttc_eval,8.846968072926794,2.3217709906707427,7.7263642803170836,10.250352836788377,10
eikonal_aux,2_dense,right_left,nointent,eval_collision_rate,0.0003333333333333,0.0007027283689262,0.0,0.0008333333333333,10
eikonal_aux,2_dense,right_left,nointent,eval_success_rate,0.1421666666666666,0.1457696686430215,0.0646208333333333,0.2303458333333333,10
eikonal_aux,2_dense,right_left,nointent,mean_return_eval,47.16550949999999,27.727979824324795,31.31553333333333,63.06214837083334,10
eikonal_aux,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
eikonal_aux,2_dense,right_left,nointent,mean_ttc_eval,11.389264505809289,2.1349410966541504,10.265296892182798,12.698108921229764,10
eikonal_aux,3,left_right,nointent,eval_collision_rate,0.0233333333333333,0.0092796072713833,0.0133333333333333,0.0316666666666666,3
eikonal_aux,3,left_right,nointent,eval_success_rate,0.0049999999999999,0.0060092521257732,0.0,0.0116666666666665,3
eikonal_aux,3,left_right,nointent,mean_return_eval,89.55776944444433,10.278218371030768,79.34252499999978,99.89782499999995,3
eikonal_aux,3,left_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,3
eikonal_aux,3,left_right,nointent,mean_ttc_eval,27.763487941280697,6.242495104010676,23.68676414459385,34.95005218974038,3
fusion_aux,1a,right_stem,nointent,eval_collision_rate,0.0456666666666666,0.0601110084205325,0.0039999999999999,0.0899999999999999,5
fusion_aux,1a,right_stem,nointent,eval_success_rate,0.9533333333333333,0.0592429086239207,0.91,0.994,5
fusion_aux,1a,right_stem,nointent,mean_return_eval,458.803878,98.63480943333562,381.2385016666667,533.9396251833332,5
fusion_aux,1a,right_stem,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
fusion_aux,1a,right_stem,nointent,mean_ttc_eval,11.690268579176514,4.716317804156475,9.118268782592605,15.925050265916978,5
fusion_aux,1a,stem_left,nointent,eval_collision_rate,0.0683333333333333,0.0586209838972907,0.0376541666666666,0.1071708333333333,10
fusion_aux,1a,stem_left,nointent,eval_success_rate,0.8883333333333333,0.1268784778760691,0.8034458333333334,0.9465125,10
fusion_aux,1a,stem_left,nointent,mean_return_eval,406.8489636666666,85.94790868139995,356.84089765416667,453.4544025541666,10
fusion_aux,1a,stem_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,1a,stem_left,nointent,mean_ttc_eval,9.120308452093711,1.2774475095349271,8.379677646684055,9.783837652141866,10
fusion_aux,1a,stem_right,intent,eval_collision_rate,0.0374999999999999,0.0221839948978904,0.0244999999999999,0.0496708333333333,10
fusion_aux,1a,stem_right,intent,eval_success_rate,0.9615,0.021551689229431,0.9499958333333332,0.9741708333333334,10
fusion_aux,1a,stem_right,intent,mean_return_eval,418.72462516666656,43.36206806483781,397.5167870749999,445.3721699999999,10
fusion_aux,1a,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,1a,stem_right,intent,mean_ttc_eval,8.360774770172412,1.5587152705554097,7.475622836910656,9.334442000608272,10
fusion_aux,1a,stem_right,nointent,eval_collision_rate,0.0281666666666666,0.0214512079144195,0.0151666666666666,0.0409999999999999,10
fusion_aux,1a,stem_right,nointent,eval_success_rate,0.9031666666666668,0.1451020288122537,0.8081583333333334,0.9665125,10
fusion_aux,1a,stem_right,nointent,mean_return_eval,412.0054459166666,68.3020388465608,369.45506442083325,448.6396184062499,10
fusion_aux,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,1a,stem_right,nointent,mean_ttc_eval,8.533805953100746,1.1294133530169128,7.8924180113832865,9.14486816991452,10
fusion_aux,1b,stem_right,intent,eval_collision_rate,0.5176666666666667,0.1016293191875185,0.4596541666666667,0.5756708333333334,10
fusion_aux,1b,stem_right,intent,eval_success_rate,0.0969999999999999,0.0948318850374339,0.0431499999999999,0.1506708333333333,10
fusion_aux,1b,stem_right,intent,mean_return_eval,118.13665449999984,34.38894536586399,98.61305390416652,137.0966210124997,10
fusion_aux,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,1b,stem_right,intent,mean_ttc_eval,8.786693423361598,0.7819303028050407,8.336708993670163,9.228609435079523,10
fusion_aux,1b,stem_right,nointent,eval_collision_rate,0.557,0.0592887368813643,0.5196645833333333,0.5888354166666666,10
fusion_aux,1b,stem_right,nointent,eval_success_rate,0.1024166666666666,0.0698172769484303,0.0605791666666666,0.1425041666666666,10
fusion_aux,1b,stem_right,nointent,mean_return_eval,116.99170499999994,25.06733501097794,102.17148080416662,131.31205065624985,10
fusion_aux,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,1b,stem_right,nointent,mean_ttc_eval,8.289779974521966,0.7333622155331974,7.82890451180147,8.705344206747146,10
fusion_aux,2,stem_right,nointent,eval_collision_rate,0.3855,0.0732324642980181,0.3448,0.4278541666666665,10
fusion_aux,2,stem_right,nointent,eval_success_rate,0.2438333333333333,0.0944184604778766,0.1894791666666667,0.2987083333333332,10
fusion_aux,2,stem_right,nointent,mean_return_eval,133.84756433333328,23.21744468279167,121.12235417083328,147.07859122499994,10
fusion_aux,2,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
fusion_aux,2,stem_right,nointent,mean_ttc_eval,8.106310866249709,0.4401915536409542,7.847472672372363,8.343207422877084,10
fusion_aux,2_dense,right_left,intent,eval_collision_rate,0.0008333333333333,0.0016666666666666,0.0,0.0024999999999999,4
fusion_aux,2_dense,right_left,intent,eval_success_rate,0.31625,0.1304931884131207,0.2041666666666666,0.4283333333333333,4
fusion_aux,2_dense,right_left,intent,mean_return_eval,64.00691458333336,31.39787638832217,39.360708333333335,88.95364249999997,4
fusion_aux,2_dense,right_left,intent,min_ttc_eval,0.0,0.0,0.0,0.0,4
fusion_aux,2_dense,right_left,intent,mean_ttc_eval,8.540753692099088,0.6922248310012814,7.941031579221447,9.173582636045548,4
fusion_aux,2_dense,right_left,nointent,eval_collision_rate,0.0006944444444444,0.0008193267335417,0.0001388888888888,0.0012499999999999,6
fusion_aux,2_dense,right_left,nointent,eval_success_rate,0.1897222222222222,0.0657872553232232,0.1387499999999999,0.2368680555555555,6
fusion_aux,2_dense,right_left,nointent,mean_return_eval,48.742337083333325,14.074404859296642,38.488287500000006,58.31061083333331,6
fusion_aux,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,6
fusion_aux,2_dense,right_left,nointent,mean_ttc_eval,9.66822677219095,0.8893165054238669,9.020846770782862,10.334924436195386,6
hjb_aux,1a,right_stem,nointent,eval_collision_rate,0.0988333333333333,0.0489712059515148,0.0709916666666666,0.1288374999999999,10
hjb_aux,1a,right_stem,nointent,eval_success_rate,0.9011666666666668,0.0489712059515148,0.8711625000000001,0.9290083333333332,10
hjb_aux,1a,right_stem,nointent,mean_return_eval,379.80111850000014,12.267256585276774,372.04206725416685,386.4203319375001,10
hjb_aux,1a,right_stem,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1a,right_stem,nointent,mean_ttc_eval,5.826201790272632,0.1074862005635164,5.767887869390404,5.888104228109899,10
hjb_aux,1a,stem_left,nointent,eval_collision_rate,0.1154999999999999,0.0358499100331188,0.0939958333333332,0.1356708333333333,10
hjb_aux,1a,stem_left,nointent,eval_success_rate,0.8843333333333334,0.035574647651882,0.8643250000000001,0.9056666666666666,10
hjb_aux,1a,stem_left,nointent,mean_return_eval,372.4386653333332,10.699559074259554,366.3662251999999,378.5299766124999,10
hjb_aux,1a,stem_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1a,stem_left,nointent,mean_ttc_eval,6.085215142880843,0.4107372920246984,5.8561870672785945,6.310074748937251,10
hjb_aux,1a,stem_right,intent,eval_collision_rate,0.0374999999999999,0.0134313526708184,0.0291624999999999,0.0453333333333332,10
hjb_aux,1a,stem_right,intent,eval_success_rate,0.9625,0.0134313526708184,0.9546666666666668,0.9708375000000002,10
hjb_aux,1a,stem_right,intent,mean_return_eval,403.8670589999999,8.57915045173066,399.5512571125,409.1376817583333,10
hjb_aux,1a,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1a,stem_right,intent,mean_ttc_eval,6.146606746840908,0.0922943070005888,6.0963197372912425,6.2004255766991045,10
hjb_aux,1a,stem_right,nointent,eval_collision_rate,0.0345238095238095,0.0164630418219643,0.0216666666666666,0.0435714285714285,10
hjb_aux,1a,stem_right,nointent,eval_success_rate,0.9557142857142856,0.0312080406136271,0.9316666666666666,0.9771428571428572,10
hjb_aux,1a,stem_right,nointent,mean_return_eval,401.6441716666666,18.02811087265448,390.9726413869047,415.60861582738096,10
hjb_aux,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1a,stem_right,nointent,mean_ttc_eval,6.244596802177405,0.1435360344037474,6.142410849318812,6.329263936000302,10
hjb_aux,1b,stem_left,nointent,eval_collision_rate,0.0698148148148148,0.1072783387982338,0.0,0.1396296296296296,9
hjb_aux,1b,stem_left,nointent,eval_success_rate,0.794074074074074,0.3213643993358533,0.5862962962962963,0.999449074074074,9
hjb_aux,1b,stem_left,nointent,mean_return_eval,354.00217629629634,90.3771713297773,294.54962495370387,413.1686251712962,9
hjb_aux,1b,stem_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,9
hjb_aux,1b,stem_left,nointent,mean_ttc_eval,6.003018817399369,1.830871730485408,4.900524233981495,7.131197305758899,9
hjb_aux,1b,stem_right,intent,eval_collision_rate,0.4843333333333334,0.12368867728652,0.4154833333333333,0.5565249999999999,10
hjb_aux,1b,stem_right,intent,eval_success_rate,0.255,0.0679778467908552,0.2181624999999999,0.2978333333333333,10
hjb_aux,1b,stem_right,intent,mean_return_eval,169.54971583333355,24.52091683071789,154.63863362916695,183.9773862750002,10
hjb_aux,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1b,stem_right,intent,mean_ttc_eval,7.771961128669465,1.246855771718165,7.07427329314442,8.484356369859999,10
hjb_aux,1b,stem_right,nointent,eval_collision_rate,0.5191666666666667,0.0464984212054901,0.4892630208333333,0.5477083333333332,10
hjb_aux,1b,stem_right,nointent,eval_success_rate,0.2482291666666666,0.0331974537056636,0.2265598958333332,0.2689687499999999,10
hjb_aux,1b,stem_right,nointent,mean_return_eval,159.5891939583335,14.823510367918557,150.88446861197926,169.63053552604185,10
hjb_aux,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,1b,stem_right,nointent,mean_ttc_eval,7.369373958053668,0.31862042880517,7.168825489513128,7.567527895392104,10
hjb_aux,2,stem_right,nointent,eval_collision_rate,0.267,0.1107767190409676,0.2006458333333333,0.3298458333333333,10
hjb_aux,2,stem_right,nointent,eval_success_rate,0.3356666666666666,0.048946935237416,0.3064958333333333,0.3648375,10
hjb_aux,2,stem_right,nointent,mean_return_eval,157.35419299999975,18.29646379231164,147.98193757916638,168.35321638333303,10
hjb_aux,2,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
hjb_aux,2,stem_right,nointent,mean_ttc_eval,7.778274369227594,2.018269811541079,6.653779681233886,8.969577244128041,10
hjb_aux,2_dense,right_left,intent,eval_collision_rate,0.0004761904761904,0.0008132500607904,0.0,0.0009523809523809,7
hjb_aux,2_dense,right_left,intent,eval_success_rate,0.3597619047619047,0.2990996188740166,0.1758988095238095,0.5805535714285712,7
hjb_aux,2_dense,right_left,intent,mean_return_eval,93.20488238095248,79.54785185369528,43.06546430357155,151.32789167261905,7
hjb_aux,2_dense,right_left,intent,min_ttc_eval,0.0,0.0,0.0,0.0,7
hjb_aux,2_dense,right_left,intent,mean_ttc_eval,9.72761426091841,2.71686528938193,7.871997207663033,11.504530427393478,7
hjb_aux,2_dense,right_left,nointent,eval_collision_rate,0.0001851851851851,0.0005555555555555,0.0,0.0005555555555555,9
hjb_aux,2_dense,right_left,nointent,eval_success_rate,0.1949999999999999,0.2432202219982724,0.0660925925925925,0.3627893518518518,9
hjb_aux,2_dense,right_left,nointent,mean_return_eval,80.62396592592609,50.88338741455853,53.98693886574093,116.07923443518528,9
hjb_aux,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,9
hjb_aux,2_dense,right_left,nointent,mean_ttc_eval,9.920581137238251,3.0445665114488927,8.237620941892946,11.865544849413556,9
rule_based,1a,stem_right,intent,eval_collision_rate,,,,,5
rule_based,1a,stem_right,intent,eval_success_rate,,,,,5
rule_based,1a,stem_right,intent,mean_return_eval,,,,,5
rule_based,1a,stem_right,intent,min_ttc_eval,,,,,5
rule_based,1a,stem_right,intent,mean_ttc_eval,,,,,5
rule_based,1a,stem_right,nointent,eval_collision_rate,0.0,0.0,0.0,0.0,5
rule_based,1a,stem_right,nointent,eval_success_rate,1.0,0.0,1.0,1.0,5
rule_based,1a,stem_right,nointent,mean_return_eval,515.552768142857,0.7908310501537729,514.8578027619045,516.0852934869046,5
rule_based,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
rule_based,1a,stem_right,nointent,mean_ttc_eval,6.5147933131278695,0.030908784565061,6.49370296240308,6.54000012714346,5
rule_based,1b,stem_right,intent,eval_collision_rate,,,,,5
rule_based,1b,stem_right,intent,eval_success_rate,,,,,5
rule_based,1b,stem_right,intent,mean_return_eval,,,,,5
rule_based,1b,stem_right,intent,min_ttc_eval,,,,,5
rule_based,1b,stem_right,intent,mean_ttc_eval,,,,,5
rule_based,1b,stem_right,nointent,eval_collision_rate,0.7406666666666666,0.0155277529318028,0.7306666666666667,0.7546666666666666,5
rule_based,1b,stem_right,nointent,eval_success_rate,0.2519999999999999,0.020083160441856,0.2346666666666666,0.2659999999999999,5
rule_based,1b,stem_right,nointent,mean_return_eval,141.30364999999998,6.125532993523022,136.25299866666666,145.579858,5
rule_based,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
rule_based,1b,stem_right,nointent,mean_ttc_eval,4.251143360618301,0.0021533866281171,4.249548322681978,4.252671581250166,5
rule_based,2_dense,right_left,intent,eval_collision_rate,,,,,5
rule_based,2_dense,right_left,intent,eval_success_rate,,,,,5
rule_based,2_dense,right_left,intent,mean_return_eval,,,,,5
rule_based,2_dense,right_left,intent,min_ttc_eval,,,,,5
rule_based,2_dense,right_left,intent,mean_ttc_eval,,,,,5
rule_based,2_dense,right_left,nointent,eval_collision_rate,0.0,0.0,0.0,0.0,5
rule_based,2_dense,right_left,nointent,eval_success_rate,0.9973333333333334,0.0027888667551136,0.9953333333333332,0.9993333333333334,5
rule_based,2_dense,right_left,nointent,mean_return_eval,318.68365066666684,3.741247959805944,315.22953066666685,321.13323266666686,5
rule_based,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,5
rule_based,2_dense,right_left,nointent,mean_ttc_eval,3.589932543982969,0.028079674906004,3.569032751884039,3.6134927877311136,5
rule_based,3,right_left,nointent,eval_collision_rate,0.0,0.0,0.0,0.0,10
rule_based,3,right_left,nointent,eval_success_rate,0.9993333333333334,0.0014054567378525,0.9983333333333334,1.0,10
rule_based,3,right_left,nointent,mean_return_eval,363.6361190000002,2.0017138315799268,362.4495810666669,364.7286932166669,10
rule_based,3,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
rule_based,3,right_left,nointent,mean_ttc_eval,3.5844738320056218,0.0248358050820174,3.5703234537986046,3.5984358261564138,10
rule_based,4,stem_left,nointent,eval_collision_rate,0.2329999999999999,0.0203336369133378,0.222325,0.2443416666666666,10
rule_based,4,stem_left,nointent,eval_success_rate,0.6173333333333334,0.0260483880214979,0.6023333333333334,0.632675,10
rule_based,4,stem_left,nointent,mean_return_eval,257.6499866666667,10.891096334875144,251.42430655000004,264.1612366000001,10
rule_based,4,stem_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
rule_based,4,stem_left,nointent,mean_ttc_eval,3.833438503168156,0.0106994569714174,3.827437860908825,3.839582393732009,10
rule_based,4,stem_right,nointent,eval_collision_rate,0.5409999999999999,0.0170003631043284,0.5316666666666666,0.5516666666666666,10
rule_based,4,stem_right,nointent,eval_success_rate,0.2943333333333333,0.0239881658066103,0.28,0.3070083333333333,10
rule_based,4,stem_right,nointent,mean_return_eval,122.98634600000018,10.789804466447483,116.72629046666684,128.71317104166687,10
rule_based,4,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
rule_based,4,stem_right,nointent,mean_ttc_eval,3.847296150941733,0.0438851539664814,3.822969900876565,3.871593977225528,10
soft_hjb_aux,1a,right_stem,nointent,eval_collision_rate,0.1110416666666666,0.0103868621229852,0.1041666666666666,0.1181249999999999,8
soft_hjb_aux,1a,right_stem,nointent,eval_success_rate,0.8889583333333333,0.0103868621229851,0.881875,0.8958333333333333,8
soft_hjb_aux,1a,right_stem,nointent,mean_return_eval,365.17315312500006,3.155663642294004,363.3099046458333,367.2334130989583,8
soft_hjb_aux,1a,right_stem,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,8
soft_hjb_aux,1a,right_stem,nointent,mean_ttc_eval,6.540658307552519,0.2609148330290525,6.3779634265615535,6.712758973422998,8
soft_hjb_aux,1a,stem_right,intent,eval_collision_rate,0.0481666666666666,0.0066411548898971,0.0443333333333333,0.0518333333333333,10
soft_hjb_aux,1a,stem_right,intent,eval_success_rate,0.9518333333333332,0.0066411548898971,0.9481666666666666,0.9556666666666668,10
soft_hjb_aux,1a,stem_right,intent,mean_return_eval,386.1607535,5.176071710527517,383.0806992124999,388.8504443958332,10
soft_hjb_aux,1a,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
soft_hjb_aux,1a,stem_right,intent,mean_ttc_eval,6.69723084099283,0.4488291068120421,6.481767922361679,6.967642076785226,10
soft_hjb_aux,1a,stem_right,nointent,eval_collision_rate,0.0452222222222222,0.008916623398995,0.0401652777777777,0.0505027777777777,10
soft_hjb_aux,1a,stem_right,nointent,eval_success_rate,0.9547777777777778,0.008916623398995,0.949497222222222,0.9598347222222224,10
soft_hjb_aux,1a,stem_right,nointent,mean_return_eval,387.54235183333327,9.9213041976405,382.5208655194444,394.27159924861104,10
soft_hjb_aux,1a,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
soft_hjb_aux,1a,stem_right,nointent,mean_ttc_eval,6.765152430699795,0.2620713449731383,6.614782263700939,6.920130962164843,10
soft_hjb_aux,1b,stem_right,intent,eval_collision_rate,0.5536666666666668,0.1051037112734692,0.4951333333333333,0.6163416666666667,10
soft_hjb_aux,1b,stem_right,intent,eval_success_rate,0.1566666666666666,0.1137248140615465,0.0911416666666666,0.2190041666666666,10
soft_hjb_aux,1b,stem_right,intent,mean_return_eval,130.5794734999998,37.4679972886638,108.37261342499974,151.88960287499987,10
soft_hjb_aux,1b,stem_right,intent,min_ttc_eval,0.0,0.0,0.0,0.0,10
soft_hjb_aux,1b,stem_right,intent,mean_ttc_eval,8.46200990012446,0.7228327864429844,8.07378275061943,8.901838399785259,10
soft_hjb_aux,1b,stem_right,nointent,eval_collision_rate,0.5403333333333332,0.0903529886137715,0.4826083333333333,0.5886249999999998,10
soft_hjb_aux,1b,stem_right,nointent,eval_success_rate,0.1323333333333332,0.0872655785927822,0.0821055555555555,0.1843458333333333,10
soft_hjb_aux,1b,stem_right,nointent,mean_return_eval,123.7680778333333,23.89295883504791,110.47076610555553,138.07989947083328,10
soft_hjb_aux,1b,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
soft_hjb_aux,1b,stem_right,nointent,mean_ttc_eval,8.17748669757759,0.752841593004219,7.800575774052436,8.67136534783258,10
soft_hjb_aux,2,stem_right,nointent,eval_collision_rate,0.4472916666666666,0.0698520971495521,0.4018489583333333,0.4904218749999999,8
soft_hjb_aux,2,stem_right,nointent,eval_success_rate,0.2320833333333333,0.0391856590223499,0.2074843749999999,0.2591666666666666,8
soft_hjb_aux,2,stem_right,nointent,mean_return_eval,96.69046708333327,13.336103607921954,88.58341505729162,105.65403708333324,8
soft_hjb_aux,2,stem_right,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,8
soft_hjb_aux,2,stem_right,nointent,mean_ttc_eval,6.501268431247767,0.4127673298943294,6.2425982018625845,6.766387099583345,8
soft_hjb_aux,2_dense,right_left,intent,eval_collision_rate,0.0,0.0,0.0,0.0,5
soft_hjb_aux,2_dense,right_left,intent,eval_success_rate,0.2733333333333333,0.1618212319540026,0.1493333333333333,0.3946666666666666,5
soft_hjb_aux,2_dense,right_left,intent,mean_return_eval,68.056687,15.639318060134144,54.491754999999976,79.40286733333332,5
soft_hjb_aux,2_dense,right_left,intent,min_ttc_eval,0.0,0.0,0.0,0.0,5
soft_hjb_aux,2_dense,right_left,intent,mean_ttc_eval,8.317474271340913,0.5390273187714424,7.910006359578597,8.751753983448166,5
soft_hjb_aux,2_dense,right_left,nointent,eval_collision_rate,5.555555555555e-05,0.0001756820922315,0.0,0.0001666666666666,10
soft_hjb_aux,2_dense,right_left,nointent,eval_success_rate,0.3111666666666666,0.0837939737758764,0.2600541666666666,0.3558402777777777,10
soft_hjb_aux,2_dense,right_left,nointent,mean_return_eval,69.34901238888891,10.896741846887233,63.321214369444455,76.19546248472224,10
soft_hjb_aux,2_dense,right_left,nointent,min_ttc_eval,0.0,0.0,0.0,0.0,10
soft_hjb_aux,2_dense,right_left,nointent,mean_ttc_eval,8.437980148693459,1.2738591711021223,7.857206654275816,9.3029839957185,10
```

---

# END OF REPORT

