# Setup, Train, Eval, Visualize

Everything runs on SUMO. Source of truth for commands.

## Prerequisites

- Python 3.10+
- SUMO: `sudo apt install sumo sumo-tools` and `export SUMO_HOME=/usr/share/sumo`

## Setup

```bash
make setup
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export SUMO_HOME=/usr/share/sumo
```

## Pipeline

### 1. Train (SUMO, both PINN and non-PINN)

```bash
make train
```

Trains both PINN-augmented and plain DRPPO on SUMO, 50k steps each. Use `--no-compare` to train only one.

**Outputs (default scenario=1a):**
- `results/train_pinn_1a.csv`, `results/train_nopinn_1a.csv`
- `results/model_pinn_1a.pt`, `results/model_nopinn_1a.pt`

Use `--scenario 1b|1c|1d|2|3|4` for other scenarios. Add `--use_intent` for intent LSTM.

### 2. Eval (SUMO)

```bash
make eval
```

Evaluates `model_pinn_1a.pt` on 100 SUMO episodes. Use `--scenario 1b|1c|1d|2|3|4` for other scenarios. Add `--gui` to watch.

**Outputs:**
- `results/eval_metrics.csv`
- Printed summary: mean return, collision rate, TTC

### 3. Plot (compare algorithms)

```bash
make plot
```

Plots return, losses, collision rate, TTC for PINN vs non-PINN.

**Outputs:**
- `results/comparison.png`
- `results/comparison_loss.png`
- `results/comparison_all.png`

### 4. Visualize (watch episodes)

**Headless (log only):**
```bash
make visualize
```

**With SUMO GUI (watch ego drive):**
```bash
make visualize-gui
```

**With trained model:**
```bash
python3 experiments/run_visualize_sumo.py --gui --policy checkpoint --checkpoint results/model_pinn_1a.pt --episodes 5
```

### 5. Ablation Study (10 variants, multi-seed)

Compares **10 plug-and-play variants** across all scenarios (1a–1d, 2–4), with **5 seeds** each:

| Variant | pinn_placement | use_l_ego | use_safety_filter | Description |
|---------|---------------|-----------|-------------------|-------------|
| `nopinn` | none | False | False | Baseline PPO, no physics |
| `pinn_critic` | critic | False | False | Design A: physics on critic |
| `pinn_actor` | actor | False | False | Design B: physics on actor |
| `pinn_both` | both | False | False | Designs A+B: both |
| `pinn_ego` | critic | True | False | Design A + L_ego dynamics |
| `pinn_no_ttc` | critic | False | False | Design A without TTC residual |
| `pinn_no_stop` | critic | False | False | Design A without stopping-distance |
| `pinn_no_fric` | critic | False | False | Design A without friction-circle |
| `safety_filter` | none | False | True | No physics loss, safety filter only |
| `pinn_critic_sf` | critic | False | True | Design A + safety filter |

**Run full ablation (5 seeds, all variants):**
```bash
make ablation
```

**Custom seeds / variants / eval only:**
```bash
python experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --seeds 42 123 456 789 999
python experiments/run_ablation.py --skip_train --eval_episodes 50   # eval existing checkpoints only
python experiments/run_ablation.py --variants nopinn pinn_critic pinn_ego   # subset of variants
```

**Outputs:**
- `results/ablation/ablation_{scenario}_{variant}_lp{lambda}_s{seed}.pt` — checkpoints
- `results/ablation/ablation_results.csv` — per-seed eval metrics
- `results/ablation/ablation_train_log.csv` — per-step training metrics (losses, violations)

### 6. Hyperparameter Sensitivity Sweep

Tests `lambda_physics_critic` across 7 values (0.001 to 1.0) for main variants:

```bash
make ablation-sensitivity
```

Or manually:
```bash
python experiments/run_ablation.py --seeds 42 123 456 789 999 --lambda_phys 0.001 0.01 0.05 0.1 0.2 0.5 1.0 --variants nopinn pinn_critic pinn_ego
```

See `docs/ABLATION_HYPERPARAMETERS.md` for full design and interpretation guidance.

### 7. Dashboard (localhost)

Interactive web dashboard showing training curves, ablation results, sensitivity, violations, and more:

```bash
make dashboard
```

Opens at `http://localhost:8501`. Tabs:
- **Training Curves** — return, losses, collisions, TTC, entropy, L_physics, L_ego
- **Ablation Summary** — bar charts per variant/scenario with error bars
- **Sensitivity** — return and collision rate vs λ_phys
- **Violations** — per-term violation rates and magnitudes over training
- **Intent Model** — LSTM train/val loss and accuracy
- **Raw Tables** — browse any CSV from results/

### 8. Running on 16 GPUs (parallel ablation)

To distribute the full ablation across 16 GPUs so each GPU runs a disjoint subset of jobs:

**Step 1 — Generate job manifest** (same scenarios/variants/seeds/lambda as desired):

```bash
make jobs-manifest
```

This writes `results/ablation/job_manifest.json` and `results/ablation/job_manifest.csv`. Customize with:

```bash
python experiments/generate_jobs.py --out_dir results/ablation --seeds 42 123 456 --lambda_phys 0.1 0.5 1.0 --variants nopinn pinn_critic pinn_ego
```

**Step 2 — Launch 16 workers** (one process per GPU, jobs split by index modulo 16):

```bash
make ablation-16gpu
```

Or with a custom manifest or number of GPUs:

```bash
./scripts/launch_parallel_16gpu.sh --manifest results/ablation/job_manifest.json --num_gpus 8
```

Worker logs: `results/ablation/jobs/worker_0.log` … `worker_15.log`. Per-job outputs: `results/ablation/jobs/eval_000000.csv`, `train_000000.csv`, etc. Checkpoints are written to `results/ablation/` as usual (`ablation_{scenario}_{variant}_lp{lp}_s{seed}.pt`).

**Step 3 — Aggregate results** for the dashboard and plotting:

```bash
make ablation-aggregate
```

This merges all `jobs/eval_*.csv` into `ablation_results.csv` and all `jobs/train_*.csv` into `ablation_train_log.csv`, so `make dashboard` and `make plot-ablation` work unchanged. **Fail-fast:** if any worker exited with an error or any job is missing/incomplete (no eval or train data), the launcher and/or the aggregator will exit with a non-zero code so CI or scripts can detect failure.

**Single-job run** (e.g. for debugging one job on GPU 0):

```bash
make jobs-run-one
# or:
CUDA_VISIBLE_DEVICES=0 python experiments/run_single_job.py --manifest results/ablation/job_manifest.json --job_id 0 --out_dir results/ablation
```

## Command summary

| Command | What it does |
|---------|--------------|
| `make train` | Train both PINN and non-PINN on SUMO (50k steps) |
| `make train-1a` ... `make train-4` | Train on scenario 1a–4 |
| `make eval` | Evaluate model on SUMO (100 episodes) |
| `make eval-multiseed` | Multi-seed eval (4 seeds × 50 episodes) |
| `make eval-1a` ... `make eval-4` | Eval on scenario |
| `make ablation` | Ablation study (7 scenarios × 10 variants × 5 seeds) |
| `make ablation-sensitivity` | Sensitivity sweep (λ_phys: 0.001–1.0, 4 variants × 5 seeds) |
| `make jobs-manifest` | Generate job list for 16-GPU parallel ablation |
| `make ablation-16gpu` | Run ablation in parallel across 16 GPUs |
| `make ablation-aggregate` | Merge per-job CSVs into ablation_results.csv and train log |
| `make dashboard` | Launch Streamlit dashboard at localhost:8501 |
| `make jobs-manifest-batch1` / `make jobs-manifest-batch2` | Default: full ablation **with and without LSTM** for every combo (run `make train-intent` first). |
| `make jobs-manifest-batch1-no-intent` / `make jobs-manifest-batch2-no-intent` | Manifests with only `use_intent=False`. |
| `make jobs-manifest-batch1-intent` / `make jobs-manifest-batch2-intent` | Manifests with only `use_intent=True`. |
| `make hpo` | Bayesian HPO (Optuna) |
| `make plot` | Plot training curves (return, loss, collision, TTC) |
| `make plot-ablation` | Plot ablation bar charts |
| `make visualize` | Run 3 SUMO episodes, log to CSV |
| `make visualize-gui` | Run 3 SUMO episodes with GUI |
| `make train-intent` | Train intent/style LSTM |

## Config

- `configs/reward/default.yaml` — reward weights
- `configs/algo/default.yaml` — PPO hyperparameters
- `configs/residuals/default.yaml` — physics-informed critic (Design A) lambdas
- `configs/scenario/default.yaml` — T-intersection layout
- `configs/state/default.yaml` — state builder parameters

## Documentation

- `docs/PHYSICS_INFORMED.md` — Physics-informed critic (Design A) explained
- `docs/ABLATION_HYPERPARAMETERS.md` — Ablation design, sensitivity sweep, interpretation
- `docs/HYPERPARAMETERS.md` — All hyperparameter values
- `docs/STATE.md` — Full state vector specification
- `docs/FRAMEWORK.md` — Complete A-to-Z reference
- `docs/SLURM.md` — SLURM sbatch: 1 job per GPU, 8 concurrent, 16 total
