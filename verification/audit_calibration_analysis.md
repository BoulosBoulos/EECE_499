# Audit — `analysis/calibration_analysis.py`

**Type**: read-only audit (PromptF Phase 3C).
**Date**: 2026-05-09.
**File**: `analysis/calibration_analysis.py` @ HEAD `b648cda` (513 lines).
**Scope**: v2 SR-primary criterion correctness; module-wide self-consistency.

The active convergence criterion is v2 SR-primary (post-Phase-3F-Stage-3). The module also retains v1 (reward stability) for retroactive comparison. Audit covers both functions, the wrapper `detect_convergence_in_run`, and the analysis pipeline that consumes them.

---

## Findings

### F1 (W) — `write_outputs` summary JSON includes v1 thresholds even when active criterion is v2
- **Where**: lines 460-466 (`summary` dict in `write_outputs`).
- **Issue**: the summary JSON written by `write_outputs` includes `reward_std_rel_threshold` and `collision_std_abs_threshold` — the v1 thresholds — but the active analysis (via `detect_convergence_in_run` → `evaluate_convergence_v2`) uses v2 thresholds (`V2_TAU_SR=0.70`, `V2_SIGMA_SR=0.10`, `V2_TAU_COLL=0.05`). The summary does NOT record `criterion_version` or the v2 thresholds.
- **Impact**: someone reading `calibrated_total_steps.json` would conclude (incorrectly) that the calibration used the v1 reward-stability criterion. The actual `t_first` values came from v2.
- **Suggested fix**: add to summary dict:
  ```python
  "criterion_version": CRITERION_VERSION,            # = v2_sr_primary_post_stage3
  "v2_tau_sr": V2_TAU_SR,
  "v2_sigma_sr": V2_SIGMA_SR,
  "v2_tau_coll": V2_TAU_COLL,
  ```
  and remove or re-label the v1 fields (`reward_std_rel_threshold`, `collision_std_abs_threshold`) so they're only present when v1 is active.

### F2 (W) — Plot threshold annotation is v1-only and misleading on v2 results
- **Where**: line 396 `ax_c.axhline(COLLISION_STD_ABS_THRESHOLD, ...)`.
- **Issue**: the per-scenario calibration plot draws a dashed horizontal line at `COLLISION_STD_ABS_THRESHOLD = 0.02` and labels it "collision std threshold". Under v2, the relevant threshold is `V2_TAU_COLL = 0.05` and it's a *mean* (not std) bound on the rolling-5 collision rate. The plot annotation references the wrong criterion.
- **Impact**: anyone reading the calibration plots will see a 0.02 reference line and believe the collision criterion is "std < 0.02" when v2 actually uses "mean ≤ 0.05".
- **Suggested fix**: replace with `ax_c.axhline(V2_TAU_COLL, ..., label=f"v2 mean coll threshold ({V2_TAU_COLL:.2f})")` and update the comment in the plotting code.

### F3 (I) — `evaluate_convergence_v2` excludes early steps where the trailing window doesn't fit
- **Where**: lines 181-182 `if t < window_steps: continue`.
- **Issue**: the trailing window [t - window_steps, t] requires t ≥ window_steps to fit fully. With `window_steps = 50_000`, the earliest possible `t_first` is 50,000. A run that converges in (say) 30,000 steps would have `t_first = None` even though it satisfied all three v2 thresholds at every iteration after step 30k.
- **Impact**: the calibrated total-steps recommendation is inflated by up to 50,000 steps for very fast-converging runs. Tier 1 calibration cells took 78,000–331,776 steps — well above 50k — so v1 vs v2 discrepancy is small in practice, but on smaller scenarios (e.g., a future Tier 2 lambda sweep at 100k) the floor would matter.
- **Note**: this matches the spec ("trailing window must be fully within the trajectory") so it's intentional. Documented here for awareness — NOT a bug.

### F4 (I) — `success_rates`/`collision_rates` rolling smoothing is asymmetric
- **Where**: line 174-175 `pd.Series(...).rolling(window=5, min_periods=1).mean()`.
- **Issue**: `pd.rolling` with default `center=False` produces a TRAILING (causal, right-aligned) window. So the smoothed value at iteration i is the mean of iterations [i-4, i]. This is correct for online convergence detection but means `t_first` is shifted ~2-3 iterations later than a centered rolling would produce.
- **Note**: this matches the spec docstring's description ("rolling-5") and is a sensible default. NOT a bug.

### F5 (W) — `CALIBRATION_METHODS` and `CALIBRATION_SCENARIOS` are hardcoded, not YAML-sourced
- **Where**: lines 77-78.
- **Issue**: `CALIBRATION_METHODS = ["drppo", "hjb_aux", ...]` and `CALIBRATION_SCENARIOS = ["1a", "2_dense"]` are module-level constants. They don't read from `config_frozen_v1.yaml`. If the YAML's tier1 methods or calibration scope changes (e.g., adding a 7th method, or extending calibration to a third scenario), this file becomes silently stale.
- **Impact**: low for current codebase — the actual calibration scope hasn't changed. But the hardcoded list creates drift risk when next someone adds a method to the YAML and forgets to update calibration_analysis.py.
- **Suggested fix**: read from `config_loader.get_config()`:
  ```python
  from config_loader import get_config
  _FROZEN = get_config()
  CALIBRATION_METHODS = [m for m in _FROZEN["tier1"]["methods"] if m != "rule_based"]
  ```
  (Calibration analyses trainable methods only; rule_based has no metrics.csv.)

### F6 (W) — `evaluate_convergence_v2` doesn't validate column presence
- **Where**: line 168-172.
- **Issue**: assumes `total_steps`, `n_episodes`, `n_successes`, `n_collisions` are all present. If a metrics.csv has a different schema (e.g., from an older training run), the call raises `KeyError` mid-iteration.
- **Impact**: confusing crash. Better: graceful skip with warning.
- **Suggested fix**: at function entry, verify required columns; if missing, return the `base` (non-converged) struct with a `criterion_version` annotation noting "missing_columns" or similar.

### F7 (I) — `analyze_calibration` tolerates non-converged seeds via group filter
- **Where**: lines 270-292.
- **Issue**: If any seed in a (method, scenario) group did NOT converge, the group is recorded with `S_to_use=None`, `all_seeds_converged=False`. The downstream check at line 296 uses this to set `analysis_status = "non_converged_cells_present"` and refuses to compute a calibrated steps value. Correct behavior — just noting it.
- **Note**: NOT a bug. The conservative behavior matches the spec's "STOP — calibration spec requires explicit human approval before 1M extension."

### F8 (W) — `_load_run` swallows all exceptions
- **Where**: lines 226-230 `except Exception as e: print(...); return None`.
- **Issue**: `Exception` catches everything including `KeyboardInterrupt` (no, that's `BaseException`), `MemoryError`, `OSError`. A genuine "disk full" during read becomes a silent skip.
- **Suggested fix**: narrow to `(json.JSONDecodeError, pd.errors.ParserError, OSError, ValueError)`.

### F9 (I) — `mean_SR_post_first` calculation
- **Where**: lines 137 (v1), 205 (v2).
- **Issue**: `mean_SR_post_first = float(success_rates[first_idx:].mean())`. This is the mean of all SR samples FROM `t_first` to the end, including the trailing window itself. So it double-counts the window into both the convergence test and the post-convergence average. Mild but acknowledged in the docstring's intent (it's a robust quality metric).
- **Note**: NOT a bug. The intended semantic is "average post-convergence quality", which includes the convergence window.

### F10 (I) — Plotting `_compute_per_method_seedmean` uses minimum-length truncation
- **Where**: lines 326-329.
- **Issue**: when seeds have different recorded iteration counts (e.g., one crashed early, one ran longer), the plotter truncates to the SHORTEST. A 90% complete seed and a 100% complete seed both get truncated to 90%, hiding the longer-seed tail.
- **Note**: this is the conservative choice for cross-seed averaging; the spec doesn't promise tail visibility. Documented for awareness.

---

## v2 mathematical correctness — no findings

The implementation matches the spec's v2 definition exactly (lines 142-207):
- Three thresholds (mean SR ≥ τ_SR, std SR ≤ σ_SR, mean coll ≤ τ_coll).
- Rolling-5 smoothing on both SR and collision rates.
- Trailing 50k window with t ≥ window_steps gating.
- `t_first` = first satisfying step, `t_last` = last satisfying step.
- `n_satisfied_evals` = total satisfying iterations.
- `mean_SR_post_first` = mean SR from `t_first` to end.
- Returns `criterion_version = "v2_sr_primary_post_stage3"`.

The implementation is correct under the spec.

---

## Severity summary

| ID | Severity | Issue |
|---|:---:|---|
| F1 | W | Summary JSON labels v1 thresholds while active criterion is v2 |
| F2 | W | Plot annotation references v1 collision-std threshold instead of v2 mean |
| F5 | W | CALIBRATION_METHODS / CALIBRATION_SCENARIOS hardcoded, not YAML-sourced |
| F6 | W | No graceful fallback for metrics.csv missing required columns |
| F8 | W | _load_run swallows all exceptions (incl. unrelated ones) |
| F3 | I | Trailing-window floor excludes very-fast convergence (intentional) |
| F4 | I | Asymmetric trailing rolling (intentional) |
| F7 | I | Non-converged seeds correctly halt analysis |
| F9 | I | mean_SR_post_first includes window (intentional) |
| F10 | I | Plot truncates to min-length seed |

No B (blocking) findings.

---

## Recommendation

The v2 implementation is mathematically correct. F1, F2, and F5 are **paper-relevant cleanups**: when the calibration JSON or plots are referenced in the paper or supplementary, they should use v2 labels and v2 thresholds. F1 in particular is misleading — `calibrated_total_steps.json` claims v1 thresholds for v2 results.

None of the findings block Tier 1 launch. They affect downstream **post-hoc** consumption of calibration outputs. Patch can wait until paper preparation phase.
