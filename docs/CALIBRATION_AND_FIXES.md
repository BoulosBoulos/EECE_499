# Calibration Pipeline and Bug Fixes (SPEC A5)

## Bug Fixes

### 1. `--lambda_aux` Override (Tier 2 Lambda Sweep)

All 4 PDE training scripts now accept `--lambda_aux <float>` to override the PDE residual weight from the YAML config. This enables the Tier 2 lambda sensitivity sweep in `run_full_ablation.py`.

```bash
# Override lambda_hjb from 0.2 (YAML default) to 0.05
python experiments/pde/train_hjb_aux.py --lambda_aux 0.05 --total_steps 50000
```

The provenance `meta.json` records both the override value and the effective lambda.

### 2. Missing Ablation Tiers

`run_full_ablation.py` now generates all tiers:

| Tier | Jobs | Formula |
|------|------|---------|
| 1 | 600 | 12 combos x 5 methods x 5 seeds x 2 intent |
| 2 | 108 | 48 lambda (1x4x4x3) + 60 occlusion (4x5x3) |
| 3 | 165 | 60 state (4x5x3) + 60 behavioral (4x5x3) + 45 dense (3x5x3) |
| supp | 126 | 42 combos (7x6) x 1 method x 3 seeds |
| all | 873 | Tier 1 + 2 + 3 (excludes supp) |

### 3. Deleted `experiments/run_single_job.py`

Obsolete heuristic ablation runner removed. Replaced by `run_full_ablation.py`.

## Calibration Pipeline

### Purpose

Determine the correct `total_steps` before launching the full ~873-run ablation. Running at too few steps produces unconverged results; too many wastes compute.

### How to Run

```bash
# Step 1: Run calibration (trains all 5 methods for 200k steps)
python experiments/pde/run_calibration.py --scenario 1a --ego_maneuver stem_right --steps 200000

# Step 2: Look at plots
ls results/calibration/figures/
# calibration_return.png       <- MOST IMPORTANT
# calibration_collision.png
# calibration_pde_residual.png
# calibration_overhead.png

# Step 3: Re-analyze without re-training
python experiments/pde/run_calibration.py --analyze_only
```

### Interpreting Results

The script prints convergence analysis for each method:

```
  hjb_aux      : last 20% mean=+12.34, improvement=2.1%, est. convergence=120000, [CONVERGED]
  drppo        : last 20% mean=+8.90, improvement=6.1%, est. convergence=180000, [STILL IMPROVING]
```

- **CONVERGED**: last 20% vs last 40% differ by < 5% -- safe to stop training
- **STILL IMPROVING**: > 5% improvement in last 40% of training -- needs more steps

### Setting TOTAL_STEPS

| Observation | Action |
|------------|--------|
| All methods converge by 100k | Use `--total_steps 100000` (cheap ablation) |
| PDE methods need 200k, DRPPO needs 100k | Use `--total_steps 200000` (PDE methods need more time for physics learning) |
| Nothing converges by 200k | Check for bugs: reward scale, learning rate, scenario complexity |
| Only one method doesn't converge | That method may be fundamentally weaker (worth reporting) |

### Launch Full Ablation

```bash
# After determining TOTAL_STEPS from calibration:
python experiments/pde/run_full_ablation.py --tier all --total_steps 150000 --max_parallel 32
```

## Files Changed

| File | Change |
|------|--------|
| `experiments/pde/train_hjb_aux.py` | `--lambda_aux` arg + provenance |
| `experiments/pde/train_soft_hjb_aux.py` | Same |
| `experiments/pde/train_eikonal_aux.py` | Same |
| `experiments/pde/train_cbf_aux.py` | Same |
| `experiments/pde/run_full_ablation.py` | Complete Tier 2 occ + Tier 3 state/behav/dense |
| `experiments/pde/run_calibration.py` | New: calibration pipeline |
| `experiments/run_single_job.py` | Deleted |
| `Makefile` | `calibrate` and `calibrate-analyze` targets |
