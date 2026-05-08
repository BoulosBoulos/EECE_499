# Fusion-aux pre-Tier-1 verification — BUG branch artifact

**Verdict:** Bug — Fusion's safety critic `U_safety` is trained but **never queried by the policy**, contradicting the docstring's claim of dual-critic policy influence. Bug indicator §2.3 #2 fires (effective `w_distill_cbf = 0`).
**No fix has been applied. No smoke test has been run.** This artifact is a candidate fix proposal awaiting user authorization.

---

## 1. Bug localisation

### 1.1 Source

**File:** `models/pde/fusion_aux_agent.py`
**Lines:** 280–360 of `train_step` (the U_safety auxiliary update + the policy distillation steps)

### 1.2 Mechanism

The Fusion-aux agent constructs **two independent critics** — `U_optimality` (Soft-HJB-shaped) and `U_safety` (CBF-shaped) — with three independent optimizers (`policy_optimizer`, `aux_opt_optimality`, `aux_opt_safety`). Both critics are trained correctly on their respective residual losses + anchor + BC. **However, only `U_optimality` is queried by the policy:**

- **Value-head distillation** (line 322–326): `U_distill = self.U_optimality(xi_t).detach()`. `U_safety` is not consulted.
- **Actor-KL alignment** (line 331–351): `q_all, _ = pde_q_values(self.U_optimality, ...)`. The soft policy `pi_soft = softmax(q_all/τ)` is built from `U_optimality` only.
- **No other path exists in `policy_optimizer`'s loss** (line 363–367). `vf_loss` only contains `L_distill` from `U_optimality`; `actor_loss` only contains the actor-KL term from `U_optimality`. `L_cbf` is in `aux_loss_saf` (line 309–313) which is back-propagated through `aux_opt_safety` and updates only `U_safety`'s parameters.

The class docstring (lines 7–15) claims:
> "U_safety constrains the policy only indirectly through the gradient flow from λ_residual · w_safety · L_cbf in the total loss."

This claim is **factually incorrect from the code**: `L_cbf` does not enter the policy_optimizer's loss. The "Three optimizers" enumeration later in the docstring is consistent with the implementation (`aux_opt_safety` owns only `U_safety`'s params), but the earlier "U_safety constrains the policy" sentence describes intended behaviour that was not implemented (or was removed).

The cited Phase 1C "Decision F" comment at line 322 (`# Decision F: distill target is U_optimality only.`) is consistent with the implementation but conflicts with the dual-critic design intent stated in the docstring.

### 1.3 Why this matches the Step 11 finding

If `U_safety` is never used by the policy, Fusion is effectively `Soft-HJB-aux + a parallel-trained but unused U_safety`. Step 11 cell-mean SR comparison supports this:

| Cell | Soft-HJB-aux | Fusion-aux | Δ |
|---|---:|---:|---:|
| 1a × stem_right | 0.953 | 0.948 | −0.005 (within seed-noise) |
| 2_dense × right_left | 0.013 | 0.032 | +0.019 (within seed-noise) |

Fusion tracks Soft-HJB-aux's signature (low SR on 2_dense; near-baseline on 1a). HJB-aux's strong 2_dense signal (SR 0.511) is not inherited because the HJB structural component does not exist in Fusion — `U_optimality` is Soft-HJB-shaped, and CBF is silenced.

## 2. Verdict criteria — what fired

**Bug indicators fired (per spec §2.3):**
- ✅ **#2 — Distillation weights silently zero or mis-scaled:** effective `w_distill_hjb = lambda_distill = 0.25` (applied to `U_optimality`); effective `w_distill_cbf = 0` (no path from `U_safety` to the policy). Sum = 0.25 < 0.5. Fires.

**Bug indicators NOT fired:**
- ❌ #1 HJB-component residual >2× standalone consistently: median F_opt/S_opt ratio = 0.498 (range 0.0004–43.78 across 6 seeds). Highly seed-variant; not consistent.
- ❌ #3 Stop-gradient missing: `U_optimality.detach()` is correctly applied at line 324 and `q_all = q_all.detach()` at line 335. Stop-grad on `U_safety` is moot since `U_safety` never enters the policy's gradient path.
- ❌ #4 Sign error in critic combination: not applicable — no combination exists.
- ❌ #5 Per-action combined advantage vanishes: not applicable — no combination exists.

**Finding indicators NOT all met** — indicator #2 (distillation weights balanced) fails. Per spec verdict logic ("any bug indicator → bug"), verdict is **bug**.

## 3. Proposed fix (unified diff — NOT APPLIED)

The minimal fix routes `U_safety` into the policy's distillation target via a convex combination, honouring the existing `w_optimality` and `w_safety` weights (default 1.0 / 1.0). This is the simplest revision of Decision F that makes the dual-critic architecture functional.

```diff
--- a/models/pde/fusion_aux_agent.py
+++ b/models/pde/fusion_aux_agent.py
@@ -319,12 +319,28 @@ class FusionAuxAgent:
             nn.utils.clip_grad_norm_(
                 self.U_safety.parameters(), self.max_grad_norm,
             )
             self.aux_opt_safety.step()
 
-            # ── Distillation: V_PPO ← U_optimality.detach() ──────────────
-            # Decision F: distill target is U_optimality only.
+            # ── Distillation: V_PPO ← convex(U_optimality, U_safety).detach() ──
+            # Decision F revised (pre-Tier-1 verification, fusion_bug_artifact):
+            # the original Decision F locked the distill target to U_optimality
+            # only, which Step 11 confirmed makes Fusion track Soft-HJB-aux's
+            # signature instead of inheriting HJB-aux's 2_dense advantage. We
+            # restore the dual-critic dependency by distilling V_PPO from a
+            # convex combination weighted by w_optimality / w_safety (the
+            # parameters were already exposed for this purpose).
             with torch.no_grad():
-                U_distill = self.U_optimality(xi_t).detach()
+                U_distill_opt = self.U_optimality(xi_t).detach()
+                U_distill_saf = self.U_safety(xi_t).detach()
+                w_total = max(self.w_optimality + self.w_safety, 1e-9)
+                U_distill = (
+                    self.w_optimality * U_distill_opt
+                    + self.w_safety * U_distill_saf
+                ) / w_total
             L_distill = F.mse_loss(value[:len(U_distill)], U_distill)
             vf_loss = vf_loss + self.lambda_distill * L_distill
 
             # ── Actor-KL alignment (Decision D) ──────────────────────────
-            # pde_q_values uses autograd to compute grad_U; we detach the
-            # outputs so the actor-KL doesn't backprop into U_optimality.
-            q_all, _ = pde_q_values(
-                self.U_optimality, xi_t, self.dynamics,
-                gamma=self.gamma, reward_kwargs=self.reward_kwargs,
-            )
-            q_all = q_all.detach()
+            # pde_q_values uses autograd to compute grad_U; we detach the
+            # outputs so the actor-KL doesn't backprop into either critic.
+            # Same convex combination as the distillation target above.
+            q_opt, _ = pde_q_values(
+                self.U_optimality, xi_t, self.dynamics,
+                gamma=self.gamma, reward_kwargs=self.reward_kwargs,
+            )
+            q_saf, _ = pde_q_values(
+                self.U_safety, xi_t, self.dynamics,
+                gamma=self.gamma, reward_kwargs=self.reward_kwargs,
+            )
+            q_all = (
+                self.w_optimality * q_opt.detach()
+                + self.w_safety * q_saf.detach()
+            ) / w_total
             pi_soft = soft_policy_from_q(q_all, tau=self.tau_soft).detach()
```

The diff also fixes the actor-KL term (line 331 onward) for consistency: the soft policy used by the actor-KL term is built from the same convex combination of q-values, so the actor and the critic are pulled toward consistent dual-critic distillation rather than U_optimality only.

**Note on the docstring:** the class docstring lines 7–15 should also be revised post-fix to reflect the new behaviour. Not in the diff because it's clarifying text, not code-correctness — but worth doing in a follow-up patch.

## 4. Smoke-test specification (NOT EXECUTED)

```yaml
# SPEC SNIPPET — Fusion bug-branch fix smoke-test
# Single Fusion-aux job at 50k on (2_dense, right_left, seed=42)
# Compute budget: ~25-30 min on RTX 4000 Ada
# DO NOT EXECUTE without explicit user authorization.

cell:
  scenario: 2_dense
  ego_maneuver: right_left
  seed: 42
  total_steps: 50000

config_change:
  file: models/pde/fusion_aux_agent.py
  patch: see "Proposed fix (unified diff)" above
  expectation: "U_safety now contributes to V_PPO distillation target and to actor-KL soft-policy q-values via convex combination weighted by w_optimality/w_safety (defaults 1.0/1.0 → 0.5/0.5 split)"
  no_other_files_modified: true

success_criteria:
  - mean_SR_post_first >= 0.3 (substantially above 0.131 baseline,
    partially recovering HJB-aux's 0.511 advantage)
  - L_residual_safety should track magnitude similar to standalone CBF-aux
    on 2_dense (within 5x; was 2.7x median in Step 11)
  - L_residual_optimality should not blow up (>10x its standalone Soft-HJB
    counterpart) — sanity check that the combined distillation target
    doesn't destabilise the optimality critic

failure_action_if_criteria_fail:
  - Conclude that Decision F (distill from U_optimality only) was a sound
    architectural choice and the dual-critic design fundamentally cannot
    benefit from CBF distillation in this domain. Document as finding
    rather than bug; revert the diff. Tier 1 launches with current Fusion
    implementation and the documented limitation.

failure_action_if_criteria_pass:
  - Apply the diff to models/pde/fusion_aux_agent.py.
  - Update docstring to reflect new dual-critic distillation behaviour.
  - Update Tier 1 spec to include the patched fusion_aux_agent.py.
  - Optionally run a 100k confirmation on 1a × stem_right × seed=42 to
    verify the fix doesn't regress on the easy cell.
```

## 5. What this artifact does NOT do

- **Does NOT apply the diff.** User must authorize.
- **Does NOT run the smoke test.** Requires separate authorization.
- **Does NOT propose a Tier 1 spec.** Independent decision.
- **Does NOT revise the Phase 1C Decision F lock document** (`SPEC_PHASE_1C_FUSION_ARCHITECTURE.md`) — that's a separate change if the smoke test passes.

## 6. Confidence assessment

The bug is concretely localised to the absence of a U_safety → policy gradient path. The diff is small, principled (convex combination using existing exposed weights), and isolated to the `train_step` distillation block. **Confidence the fix is correct: high.** **Confidence the fix recovers HJB-aux's 2_dense advantage: moderate.** Reasons for moderate-not-high:

1. The CBF residual measures a different geometric structure (control barrier function) than HJB optimality. Adding it to V_PPO's distillation target may not by itself convey the HJB-style reachability signal that drives HJB-aux's 2_dense success.
2. The bug-fix produces a "Soft-HJB + CBF" dual-critic architecture, *not* a "HJB + CBF" dual-critic. To recover HJB-aux's specific 2_dense signal in Fusion, the optimality critic would need to be HJB (not Soft-HJB). That is a deeper architectural change than this diff.
3. The smoke-test threshold (`mean_SR_post_first ≥ 0.3`) is set to detect partial recovery, not full recovery. Full HJB-aux replication on 2_dense would require swapping `U_optimality` from `SoftHJBAuxCritic` to `HJBAuxCritic`, which is a separate (larger) decision.

**Recommendation:** Run the smoke test to confirm the diff is non-destructive and at minimum recovers a portion of the dual-critic design intent. If the partial recovery is meaningful (≥ 0.3 SR), authorize the fix for Tier 1. If not, the finding stands and Fusion proceeds to Tier 1 with the documented limitation that "Decision F as implemented makes Fusion architecturally equivalent to Soft-HJB-aux."

## 7. Tier 1 implication

`apply_fix_then_smoke_test_then_tier_1` (per spec §4 `tier_1_implication` enumeration). The smoke test is a 25–30 minute compute investment that resolves the bug-vs-finding ambiguity before committing to 240 Fusion jobs at Tier 1.

If the smoke test fails (Fusion still ≈ baseline on 2_dense), the verdict converts to "finding" and Fusion proceeds to Tier 1 with the implementation locked at Phase 1C Decision F + documented limitation.
