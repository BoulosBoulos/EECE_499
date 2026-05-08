# Training and Evaluation Infrastructure

## Training Scripts (5 methods)

| Script | Method | PDE Critic | Config |
|--------|--------|-----------|--------|
| `experiments/pde/train_hjb_aux.py` | Hard-HJB | HJBAuxCritic | `configs/pde/hjb_aux.yaml` |
| `experiments/pde/train_soft_hjb_aux.py` | Soft-HJB | SoftHJBAuxCritic | `configs/pde/soft_hjb_aux.yaml` |
| `experiments/pde/train_eikonal_aux.py` | Eikonal | EikonalAuxCritic | `configs/pde/eikonal_aux.yaml` |
| `experiments/pde/train_cbf_aux.py` | CBF-PDE | CBFAuxCritic | `configs/pde/cbf_aux.yaml` |
| `experiments/pde/train_drppo_baseline.py` | DRPPO | None (baseline) | `configs/algo/default.yaml` |

## Running a Single Training Job

```bash
# HJB (default scenario 1a, stem_right maneuver)
python experiments/pde/train_hjb_aux.py --scenario 1a --ego_maneuver stem_right --total_steps 50000 --seed 42

# Eikonal on stem_left
python experiments/pde/train_eikonal_aux.py --scenario 2 --ego_maneuver stem_left --seed 123

# DRPPO baseline
python experiments/pde/train_drppo_baseline.py --scenario 1a --total_steps 50000

# With intent features
python experiments/pde/train_hjb_aux.py --use_intent --scenario 3

# With SUMO GUI
python experiments/pde/train_hjb_aux.py --sumo_gui --total_steps 2048
```

## Evaluation

```bash
python experiments/pde/eval.py \
    --checkpoint results/pde/model_hjb_aux_1a_stem_right.pt \
    --method hjb_aux \
    --scenario 1a --ego_maneuver stem_right \
    --episodes 50 --save_failures
```

Supported `--method` values: `hjb_aux`, `soft_hjb_aux`, `eikonal_aux`, `cbf_aux`, `drppo`

### Failure Recording
Add `--save_failures` to save per-step CSV trajectories for collision episodes to `{out_dir}/failures/`.

## Serial Ablation

```bash
python experiments/pde/run_ablation.py \
    --scenarios 1a 2 \
    --variants hjb_aux soft_hjb_aux eikonal_aux cbf_aux drppo \
    --lambda_aux 0.1 0.2 0.5 \
    --seeds 42 123 456 \
    --ego_maneuver stem_right
```

## Parallel Ablation (Tiered)

```bash
# Tier 1: Main 5-method comparison across key scenario/maneuver combos
python experiments/pde/run_full_ablation.py --tier 1 --max_parallel 32

# Tier 2: Lambda sensitivity sweep
python experiments/pde/run_full_ablation.py --tier 2 --max_parallel 16

# Tier 3: Full 7-scenario x 6-maneuver grid (best method from Tier 1)
python experiments/pde/run_full_ablation.py --tier 3 --max_parallel 16

# Dry run (print jobs without executing)
python experiments/pde/run_full_ablation.py --tier all --dry_run
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make train-hjb-aux` | Train HJB method |
| `make train-soft-hjb-aux` | Train Soft-HJB method |
| `make train-eikonal-aux` | Train Eikonal method |
| `make train-cbf-aux` | Train CBF-PDE method |
| `make train-drppo` | Train DRPPO baseline |
| `make eval-pde CHECKPOINT=... METHOD=...` | Evaluate a checkpoint |
| `make ablation-tier1` | Launch Tier 1 parallel ablation |
| `make ablation-tier2` | Launch Tier 2 parallel ablation |
| `make ablation-all` | Launch all tiers |
| `make ablation-serial` | Run serial ablation |
| `make visualize-pde-gui CHECKPOINT=... METHOD=...` | GUI visualization |
| `make regen-scenarios` | Regenerate SUMO scenarios |
| `make clean` | Remove ablation results |

Override variables: `SCENARIO=2 MANEUVER=stem_left STEPS=10000 SEED=123`

## Output Directory Structure

```
results/
  pde/                          # Single-run outputs
    train_{method}_{scen}_{man}.csv
    model_{method}_{scen}_{man}.pt
    meta_{method}_{scen}_{man}.json
    failures/                   # Collision trajectory CSVs
      fail_{method}_{scen}_{man}_s{seed}_{mode}_ep{N}.csv
  pde_ablation/                 # Serial ablation outputs
    ablation_results.csv
    ablation_train_log.csv
  ablation/                     # Parallel ablation outputs
    tier1/{scen}_{man}_{method}_{intent}_s{seed}/
    tier2_lambda/{scen}_{man}_{method}_l{lambda}_s{seed}/
    tier3_full/{scen}_{man}_{method}_s{seed}/
```

## Provenance Metadata (meta.json)

Each training run produces a `meta_*.json` file containing:
- `method`, `scenario`, `ego_maneuver`, `seed`
- `total_steps`, `use_intent`
- `config_file`, `algo_config_file`
- `pde_config` (full PDE hyperparameters)
- `algo_config` (full PPO hyperparameters)
- `device`, `git_hash`
- `start_time`, `wall_time_seconds`

## Timing

All training scripts record `train_time_per_iter` in the CSV — wall-clock seconds per collect+train iteration. Use this for computational overhead comparison.
