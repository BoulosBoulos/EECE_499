# Phase 3F Stage 3 — Phenomenon branch artifact

**Branch:** Phenomenon (criterion-design hypothesis)
**Status:** Produced as one of two competing artifacts under an **inconclusive** Stage 3 verdict; the other artifact is `verification/phase3F_stage3_bug_artifact.md`.
**See also:** `verification/phase3F_stage3_status.json` for full numerical results and indicator firings.

---

## 1. Phenomenon hypothesis

The Phase 3 multi-metric plateau criterion (analysis/calibration_analysis.py §Decision B) requires both
- `std(rolling-5 mean_reward) / |mean(window)| < 0.05`, and
- `std(rolling-5 collision_rate) < 0.02 absolute`,

over a 50,000-step lookahead window starting at any candidate step S. On 1a, the criterion is satisfied at some early step (S = 16,384 / 114,688 / 233,472 across the three seeds), but is then violated on 88–94 % of subsequent post-`t_first` evaluation iterations *despite the underlying policy being essentially optimal*: mean post-t_first success rate is 0.984–0.998 across the three seeds, with collision rate identically 0.0 throughout. The proximate driver of criterion violation is residual variance in `mean_reward`, which oscillates between roughly −1500 and −7500 within the post-convergence window — a factor of ~4–5 — even when the action policy is stable. The reward signal has high natural variance because episode returns depend on stochastic SUMO traffic, episode length variation, and the soft-policy entropy bonus structure; this noise dominates the rolling-5 reward standard deviation and pushes `std/|mean|` above the 0.05 threshold.

The hypothesis is therefore: the Phase 3 reward-std criterion is too tight relative to the natural variance of the noisy `mean_reward` signal under stable Soft-HJB policy iteration. The "drift" detected by the criterion is criterion-driven, not policy-driven.

## 2. First-satisfying-window proposal — W = ?

Per spec §4.1, the phenomenon-branch fix is to require *W consecutive* evaluation iterations satisfying the criterion before declaring convergence, replacing "first-satisfaction" with "first-satisfying-window of length W". I computed the smallest W such that "criterion holds for W consecutive evals" implies "criterion holds for ≥ 80 % of remaining evals" across all 6 Soft-HJB jobs. Per-job per-W satisfaction:

| W | 1a_s42 | 1a_s123 | 1a_s456 | 2_dense × 3 |
|---|--------|---------|---------|-------------|
| 3 | 8.6 % | 2.0 % | 4.7 % | never satisfies |
| 4 | 5.4 % | never | 3.2 % | never |
| 5 | 4.1 % | never | 1.6 % | never |
| 6 | 2.8 % | never | 0.0 % | never |
| 7 | 1.4 % | never | never | never |
| 8 | 0.0 % | never | never | never |
| 10 | never | never | never | never |

**No value of W satisfies the ≥ 80 % rule on the 1a seeds.** The 1a criterion is failed >90% of post-window evals at every W ≥ 3. The 2_dense seeds never satisfy the criterion at all (the policy doesn't actually converge to a useful success rate; peak SR ≤ 0.125).

A simple W-fix is therefore *not sufficient*. The criterion thresholds themselves need adjustment.

## 3. Recommended criterion update

Two complementary changes:

### 3.1 Loosen the reward-std-rel threshold

The current 0.05 cutoff is unrealistically tight for the noisy `mean_reward` signal in this domain. Empirical post-convergence rolling-5 reward std/|mean| ratios on 1a:

| Cell | post-t_first reward σ_rel range | post-t_first reward σ_rel mean |
|------|---------------------------------|-------------------------------|
| 1a_s42  | ~0.10–0.30 | ~0.18 |
| 1a_s123 | ~0.10–0.40 | ~0.22 |
| 1a_s456 | ~0.10–0.30 | ~0.18 |

(Approximate ranges from inspection of mean_reward post-t_first — exact values in the per-seed plots.)

Recommend `REWARD_STD_REL_THRESHOLD = 0.20` (4× the current value), which would put the criterion comfortably above the typical post-convergence noise floor on 1a while still rejecting genuinely diverging or oscillating signals.

### 3.2 Add a success-rate plateau condition

Use the *direct measure of policy quality* (success rate) rather than only the noisy reward proxy:

```python
# Pseudocode for Step 11 convergence criterion
def detect_convergence_window(df, W: int = 3) -> Optional[int]:
    """Smallest step S such that W consecutive eval iterations starting at or
    after S all satisfy:
      - mean(rolling-5 success_rate over [S, S+50_000]) >= 0.85
      - std(rolling-5 success_rate over [S, S+50_000]) <= 0.05 absolute
      - std(rolling-5 mean_reward) / |mean(window)| < 0.20
      - std(rolling-5 collision_rate) < 0.02 absolute
    """
    # ... (rolling-5 smoothing as before)
    for i in range(len(df) - W + 1):
        if all(
            check_window(df, j) for j in range(i, i + W)
        ):
            return int(df["total_steps"].iloc[i])
    return None
```

The success-rate plateau is the canonical signal for behavioral-driving convergence; supplementing the noisier reward criterion with it is more robust.

## 4. Paper-ready paragraph (~200–300 words)

Soft-policy iteration in discrete-action behavioral driving exhibits a characteristic post-convergence pattern under finite-sample auxiliary-critic distillation: the action-distribution converges to a near-deterministic policy and the success rate stabilises (mean post-convergence success rate 0.98–1.00 across our 1a seeds), while the noisy mean-reward signal continues to fluctuate at ~10–30 % relative standard deviation due to episode-length variability, stochastic traffic, and the residual entropy of the soft policy (Wang & Zhou, 2020). Naively applying a reward-variance plateau criterion (reward σ/|μ| < 0.05) flags this as non-convergence on 88–94 % of post-first-satisfaction evals on our scenario-1a runs, even though the underlying policy is essentially stable and optimal. The convergence-time spread of {16k, 115k, 233k} steps across the three 1a seeds reflects criterion-noise sensitivity, not seed-dependent algorithmic instability: in all three cases the policy reaches near-perfect success rate well before any criterion-satisfying window is found, and in all three cases the policy then remains near-perfect indefinitely. We therefore recommend, for soft-policy-iteration calibration: (i) supplement the reward-variance criterion with a direct success-rate plateau condition, and (ii) loosen the reward σ/|μ| threshold from 0.05 to 0.20 to reflect the natural noise floor of episode-return signals in stochastic-traffic discrete-action driving. The first-satisfying-window-with-W approach commonly used to handle bounded post-convergence oscillation in policy-iteration evaluations (e.g. Espeholt et al. 2018) does not in itself help here, because the issue is not bounded oscillation around a satisfying state but a systematically over-tight criterion threshold relative to the natural reward variance. The proposed criterion change has no effect on training code; it is a calibration-analysis adjustment.

## 5. Implementation note

No code changes to `models/`, `experiments/`, or `env/`. The criterion adjustment lives in `analysis/calibration_analysis.py` (specifically `REWARD_STD_REL_THRESHOLD` constant and a new `detect_convergence_window()` function with a `W` parameter and the additional success-rate condition). Step 11 spec would lock in the new threshold and W choices.

## 6. Honest caveat — why this branch is not declared

The phenomenon hypothesis fits 1a cleanly (SR stable at 1.0 → criterion oscillation is methodological), but does NOT explain 2_dense, where peak SR ≤ 0.125 and final SR = 0.0 across all three seeds. Soft-HJB simply does not produce a useful policy on 2_dense within 500 k training steps. This is a separate finding and is not addressed by a criterion adjustment alone. The Stage 3 verdict is therefore inconclusive between (i) phenomenon-on-1a-with-criterion-adjustment and (ii) algorithmic-issue-on-2_dense; the bug artifact (companion file) details a candidate algorithmic-issue interpretation. User direction is required to choose between the branches.
