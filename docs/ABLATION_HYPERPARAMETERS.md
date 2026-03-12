# Ablation Study: Hyperparameter Sensitivity

Ablation results are **conditional on the chosen hyperparameters**. A term that appears useless at one setting may be helpful at another. This document describes the experimental design.

## Variants

| Variant | pinn_placement | use_l_ego | use_safety_filter | Description |
|---------|---------------|-----------|-------------------|-------------|
| `nopinn` | none | False | False | Baseline PPO (no physics) |
| `pinn_critic` | critic | False | False | Design A: physics on critic |
| `pinn_actor` | actor | False | False | Design B: physics on actor |
| `pinn_both` | both | False | False | Designs A+B combined |
| `pinn_ego` | critic | True | False | Design A + L_ego dynamics |
| `pinn_no_ttc` | critic | False | False | Design A without TTC residual |
| `pinn_no_stop` | critic | False | False | Design A without stopping-distance |
| `pinn_no_fric` | critic | False | False | Design A without friction-circle |
| `safety_filter` | none | False | True | No physics loss, safety filter only |
| `pinn_critic_sf` | critic | False | True | Design A + safety filter |

## Sensitivity Sweep

The main ablation uses the canonical config (`lambda_physics_critic=0.5`). The sensitivity sweep tests:

- `lambda_physics_critic` in {0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}

Run with:

```bash
make ablation-sensitivity
```

Or manually:

```bash
python3 experiments/run_ablation.py \
  --seeds 42 123 456 789 999 \
  --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 \
  --variants nopinn pinn_critic pinn_ego
```

## Seeds

All ablations use 5 seeds (42, 123, 456, 789, 999) and report mean +/- std.

## Canonical Hyperparameters

From `configs/residuals/default.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda_physics_critic` | 0.5 | Overall physics loss weight |
| `lambda_physics_ttc` | 1.0 | TTC violation weight |
| `lambda_physics_stop` | 1.0 | Stopping-distance violation weight |
| `lambda_physics_fric` | 1.0 | Friction-circle violation weight |
| `lambda_physics_ego` | 0.1 | L_ego weight |
| `physics_ttc_thr` | 3.0 | TTC threshold (seconds) |
| `physics_tau` | 0.5 | Reaction time (seconds) |
| `a_max` | 5.0 | Max deceleration (m/s^2) |
| `mu` | 0.8 | Friction coefficient |

## Interpreting Results

- If conclusions hold across λ_phys values, they are robust.
- If a term helps at λ=0.5 but hurts at λ=1.0, report that the benefit is sensitive to weight tuning.
- If a drop-one variant (e.g. `pinn_no_ttc`) performs the same as `pinn`, the dropped term contributes little at this config.
- Always check violation statistics: if a term fires <1% of steps, it may not be active enough to matter.

## Output

Results are saved to `results/ablation/`:

- `ablation_results.csv` — per-seed eval metrics
- `ablation_train_log.csv` — per-step training metrics (losses, violations)

View with:

```bash
make dashboard
```
