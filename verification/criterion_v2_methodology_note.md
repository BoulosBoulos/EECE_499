# Convergence-criterion methodology note (v1 → v2)

**Phase:** 3F Stage 3 follow-up — convergence-criterion redesign post-diagnostic
**Status:** Active criterion is v2 (`v2_sr_primary_post_stage3`); v1 retained for retroactive analysis
**Source:** `analysis/calibration_analysis.py` — see `evaluate_convergence_v1`, `evaluate_convergence_v2`, and the `CRITERION_VERSION` constant

---

## v1 criterion (Decision B; pre-Stage-3)

A run was declared converged at step S if, over the forward window [S, S+50,000]:

1. `std(rolling-5 mean_reward) / |mean(window)| < 0.05`
2. `std(rolling-5 collision_rate) < 0.02` (absolute)

This formulation tracked *reward-signal stability* as a proxy for policy convergence. It assumed that once the policy stops changing, the reward distribution would also stabilise.

## Stage 3 finding that prompted the change

Phase 3F Stage 3 diagnostic (`verification/phase3F_stage3_status.json`) showed the proxy assumption fails on near-perfect policies in this domain. On the three 1a Soft-HJB Phase 3 calibration seeds, post-`t_first` mean success rate was 0.984 / 0.998 / 0.998 (essentially perfect), yet the v1 criterion was violated on 88.5 % / 94.2 % / 91.0 % of post-`t_first` evaluation iterations, producing a 14× spread in `t_first` across seeds (16,384 / 114,688 / 233,472). The proximate cause is `mean_reward` exhibiting 5–30 % relative noise from per-episode randomness — variable episode lengths, stochastic SUMO traffic, and soft-policy entropy structure — even when the action distribution is essentially constant. The mechanism hypothesis (auxiliary-critic distillation pulling the policy past optimum as the critic continues refining) was falsified: post-`t_first` |corr(Δsuccess, Δ{HJB residual, KL alignment, entropy})| < 0.31 on every 1a seed. The "drift" is criterion-induced, not policy-induced.

## v2 criterion (current)

A run is converged at step t if, over the trailing window [t − 50,000, t]:

1. `mean(rolling-5 success_rate) ≥ τ_SR = 0.70`
2. `std(rolling-5 success_rate) ≤ σ_SR = 0.10`
3. `mean(rolling-5 collision_rate) ≤ τ_coll = 0.05`

`t_first` is the smallest t at which the trailing window satisfies; `t_last` is the largest. The reward-stability check is dropped entirely. Reward magnitude depends on shaping, scenario, and ego maneuver in non-comparable ways across cells; success rate is the actual outcome metric and is directly comparable.

## Threshold calibration

τ_SR = 0.70, σ_SR = 0.10, τ_coll = 0.05 satisfy the spec's hard limits (τ_SR ≥ 0.65, σ_SR ≤ 0.15) and produce convergence on all three 1a Soft-HJB seeds at the spec-proposed values without loosening. v2 `t_first` values for the three seeds: 90,112 / 81,920 / 94,208 — a 1.15× spread (versus v1's 14×), aligning closely with the step at which smoothed success rate first crossed τ_SR. Post-`t_first` mean success rate: 0.998 / 0.998 / 0.999.

## Retroactive re-analysis on Phase 3 (36 jobs)

v2 produced 8 disagreements with v1, all in the direction v1=False → v2=True, all on 1a (none on 2_dense). The recovered runs are: `drppo` s123, `hjb_aux` s42 / s456, `eikonal_aux` s123 / s456, `cbf_aux` s42 / s123, `fusion_aux` s456 — each with mean post-`t_first` SR of 1.000 (or 0.997 for fusion_aux s456). All 18 2_dense Phase 3 jobs return v2_converged = False, matching v1's verdict and the empirical reality that no method produced a useful 2_dense policy at 500 k steps. There are zero v2_True ↔ v1_True transitions where v2 disagrees on `t_first` direction beyond the 1a Soft-HJB seeds (which are the calibration target). No qualitative change to Phase 3 method comparisons: the cross-method ranking on 1a is unchanged when restricted to runs where both criteria agree.

## Paper-ready statement

> We define convergence as a success-rate plateau over the trailing 50,000 training steps, requiring mean SR ≥ τ_SR (= 0.70), std SR ≤ σ_SR (= 0.10), and mean collision rate ≤ τ_coll (= 0.05). This formulation directly tracks the policy outcome of interest rather than reward-signal stability, which we found to be below the irreducible per-episode noise floor on near-optimal policies and therefore prone to false-negative convergence reports under reasonable noise levels in stochastic-traffic discrete-action driving.
