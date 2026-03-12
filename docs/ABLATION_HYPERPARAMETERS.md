# Ablation Study: Hyperparameter Sensitivity

**Important**: Ablation results (which variant "wins") can depend strongly on hyperparameters.
The same model may perform better or worse under different reward scales, learning rates, or physics weights.

## Key Parameters That Affect Ablation

| Parameter | Location | Effect |
|-----------|----------|--------|
| `w_risk`, `w_coll` | `configs/reward/default.yaml` | Stronger penalties → policy stops more, lower collision rate. If too weak, no variant may learn to stop. |
| `lambda_physics_*` | `configs/residuals/default.yaml` | PINN strength. Too high → over-regularization; too low → no benefit over nopinn. |
| `lr`, `n_steps`, `batch_size` | `configs/algo/default.yaml` | Learning dynamics. Different combos can flip which variant converges best. |

## Recommendation

For fair ablation comparison:

1. **Fix hyperparameters** across all variants (nopinn, pinn, pinn_ego) when comparing.
2. **Optional**: Run a small hyperparameter sweep (e.g. `w_risk` in [-1, -3, -5], `lambda_physics_critic` in [0.3, 0.5, 0.7]) and report whether the ranking of variants is robust across configs.
3. **Report** the exact config used for each ablation run.
