# Phase 3F Stage 3 — Bug branch artifact

**Branch:** Bug (algorithmic-mechanism hypothesis)
**Status:** Produced as one of two competing artifacts under an **inconclusive** Stage 3 verdict; the other artifact is `verification/phase3F_stage3_phenomenon_artifact.md`.
**See also:** `verification/phase3F_stage3_status.json` for full numerical results and indicator firings.
**No fix has been applied. No re-verification has been run.** This artifact is a candidate fix proposal awaiting user authorization.

---

## 1. Bug-hypothesis localisation

### 1.1 Honest caveat on candidate localisation

I could not localise a clearly-identifiable bug in `models/pde/soft_hjb_aux_agent.py` from the diagnostic data alone. The Soft-HJB auxiliary critic produces **stable, near-optimal policies** on 1a (mean post-`t_first` success rate 0.984–0.998 across 3 seeds, collision rate 0.0 throughout) — empirically the algorithm "works" on the simpler scenario. The bug indicators that fired on 1a (drift fraction > 0.5; weak correlation with mechanism signals on 2/3 seeds) are most plausibly a methodological artefact: the convergence criterion is too tight for the natural reward variance, and once the policy is at SR ≈ 1.0 there are essentially no success-rate changes to correlate with mechanism signals (denominator-zero correlation problem).

### 1.2 Strongest available bug hypothesis

The single bug-like signature in the data that is *not* fully explained by the criterion-tightness hypothesis is the **2_dense non-convergence**: across 3 seeds, peak SR ≤ 0.125 and final SR = 0.0. The Soft-HJB critic does not produce a useful policy on 2_dense within 500 k training steps. This is a hypothesis-class bug in the auxiliary loss-balance: under the harder 2_dense scenario, the L_anchor signal (MSE between aux U and full GAE returns) dominates the L_soft signal (HJB residual squared), preventing the critic from learning a value function with the soft-HJB structure that the actor-side soft-KL alignment needs.

**Suspected source:** the relative weight `lambda_soft = 0.2` may be too small relative to `lambda_anchor = 1.0` for scenarios with sparse/extreme reward signals.

- **File / line range:** `models/pde/soft_hjb_aux_agent.py:174-176` and `configs/pde/soft_hjb_aux.yaml:2-3`
- **Mechanism:** Total auxiliary loss is `lambda_anchor * L_anchor + lambda_soft * L_soft + lambda_bc * L_bc` with `lambda_anchor = 1.0` and `lambda_soft = 0.2`. On 2_dense the GAE returns (returns range −2700 to +200 in this calibration) produce L_anchor with much larger gradient magnitude than the HJB residual L_soft. The critic learns to track returns rather than the soft-HJB structure, so the soft-policy distillation signal `softmax(q_a/τ)` doesn't carry the reachability information the actor needs to escape collision-prone behavior on 2_dense.
- **Empirical signal:** in `metrics.csv` for 2_dense, `L_residual_optimality` (= L_soft pre-weight) starts in the 5–50 range and grows to 100+ over training — i.e. the residual is *increasing*, not decreasing. On 1a, the residual is also high but the simpler scenario tolerates it because the reachability structure is implicit in the geometry.

### 1.3 Confidence assessment

- **Confidence the proposed fix would address the 2_dense non-convergence:** moderate-low. Higher `lambda_soft` is a standard PINN/HJB-PDE knob to prioritise residual minimisation, and the 2_dense data is consistent with under-weighted residual. But the underlying issue could also be (i) reward-shaping insufficiency, (ii) collocation-batch undercoverage on rare collision states, or (iii) τ_soft = 0.1 producing excessively concentrated soft policies that collapse exploration in dense traffic. None of these can be ruled out from the available data.
- **Confidence the proposed fix would address the 1a 16k/115k/233k convergence-time spread:** very low. The 1a seeds *all converge* at near-perfect SR; the 14× t_first spread reflects criterion sensitivity (the criterion latches at different points along the same noisy plateau), not algorithmic instability. If the 2_dense fix succeeds, the 1a spread would still need to be addressed by the criterion-adjustment proposed in the phenomenon-branch artifact.
- **Therefore: the proposed bug fix and the phenomenon-branch criterion adjustment are not mutually exclusive — both may be needed.**

## 2. Proposed fix (unified diff)

```diff
--- a/configs/pde/soft_hjb_aux.yaml
+++ b/configs/pde/soft_hjb_aux.yaml
@@ -1,8 +1,9 @@
 # Soft-HJB auxiliary critic configuration
-lambda_anchor: 1.0
-lambda_soft: 0.2
+# Phase 3F Stage 3 bug-branch hypothesis: lambda_anchor was dominating L_soft on
+# 2_dense (peak SR ≤ 0.125 across 3 seeds, residual L_residual_optimality
+# growing rather than shrinking through training). Rebalance toward the residual.
+lambda_anchor: 0.5
+lambda_soft: 1.0
 lambda_bc: 0.5
 lambda_distill: 0.25
 lambda_align: 0.05
 tau_soft: 0.1
 aux_hidden_dim: 256
 aux_lr: 1e-3
 collocation_ratio: 0.7
```

The change rebalances total auxiliary-loss gradient weight away from L_anchor (the env-return-fitting term) and toward L_soft (the HJB residual). The 5× ratio shift (1:0.2 → 0.5:1.0) targets the order-of-magnitude scale gap visible in metrics.csv between L_anchor (50–200 typical post-convergence) and L_residual_optimality (5–100). After rebalancing, the gradient contributions become comparable.

**No code-file changes** to `models/pde/soft_hjb_aux_agent.py`. The fix is hyperparameter-only.

## 3. Re-verification spec snippet (NOT EXECUTED — produced for user authorization only)

```yaml
# SPEC SNIPPET — Phase 3F Stage 3 bug-branch re-verification
# Single Soft-HJB job at 100k on (1a, stem_right, seed=42).
# Compute budget: ~30 min on RTX 4000 Ada.
# DO NOT EXECUTE without explicit user authorization.

cell:
  scenario: 1a
  ego_maneuver: stem_right
  seed: 42
  total_steps: 100000

config_change:
  file: configs/pde/soft_hjb_aux.yaml
  changes:
    lambda_anchor: 1.0 -> 0.5
    lambda_soft:   0.2 -> 1.0
  all_other_params_unchanged: true

success_criteria:
  - drift_fraction < 0.3 on the re-verified seed (per Phase 3 multi-metric plateau criterion)
  - no_catastrophic_regression: post_t_first_success_rate_min >= 0.5 * post_t_first_success_rate_peak
  - L_residual_optimality_decreases_overall: tail_value < head_value within the 100k window

failure_action_if_criteria_fail:
  - Bug-branch hypothesis is rejected.
  - Status JSON's step_11_implication remains "requires_user_direction".
  - User must choose between phenomenon-branch criterion adjustment, an alternative bug
    hypothesis (τ_soft tuning, collocation strategy, reward-shaping reformulation), or
    accepting Soft-HJB's 2_dense limitation as a documented finding for Tier 1.

failure_action_if_criteria_pass:
  - Bug-branch hypothesis is supported on the verified seed.
  - User may authorize Step 11 to launch with the new λ weights, but the 1a t_first spread
    issue (criterion sensitivity) remains unresolved by this fix and would still require
    the phenomenon-branch criterion adjustment.
```

The re-verification is bounded — single cell, 100k steps, ~30 min — exactly per spec §4.2.

## 4. What this artifact does NOT do

- **Does NOT apply the diff.** The user must explicitly authorize the change.
- **Does NOT run any training.** Re-verification spec is a snippet for the user to authorize as a separate spec.
- **Does NOT propose a Step 11 spec.** Step 11 spec is a separate authored deliverable.
- **Does NOT speculate on additional fixes** (τ_soft tuning, collocation rebalancing, etc.) beyond the single-knob hyperparameter change.

## 5. Recommendation against this branch standalone

If the user picks this branch:
- The fix targets the *2_dense non-convergence*, not the 1a criterion-oscillation. The latter still needs the phenomenon-branch criterion adjustment in Step 11 even after this fix.
- The fix is unverified; one 100k re-verification on 1a (the easier scenario) does not by itself prove the fix would resolve 2_dense — the spec's re-verification cell choice is locked at (1a, stem_right, 42) per spec §4.2.
- A more thorough bug-branch resolution would require a second 100k re-verification on a 2_dense seed; the spec does not authorize this.

If the user picks the phenomenon branch instead:
- The criterion adjustment resolves the 1a "drift" issue cleanly.
- The 2_dense non-convergence remains as a documented Soft-HJB limitation for Tier 1, framed as "Soft-HJB does not produce a useful policy on the dense-traffic scenario at 500k training steps with the current λ-weight configuration".

The status JSON's `step_11_implication = requires_user_direction` reflects this trade-off explicitly.
