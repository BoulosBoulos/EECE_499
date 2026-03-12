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

### 5. Ablation Study (L_ego and physics variants)

Compares **3 variants** across all scenarios (1a–1d, 2–4):

| Variant | use_pinn | use_l_ego | Description |
|---------|----------|-----------|-------------|
| `nopinn` | False | False | Standard PPO, no physics |
| `pinn` | True | False | Design A: TTC + stop + friction only |
| `pinn_ego` | True | True | Design A + L_ego (ego dynamics prediction error) |

**Run full ablation (train + eval):**
```bash
make ablation
```

**Custom steps / eval only:**
```bash
python experiments/run_ablation.py --total_steps 50000 --eval_episodes 50 --out_dir results/ablation
python experiments/run_ablation.py --skip_train --eval_episodes 50   # eval existing checkpoints only
```

**Outputs:**
- `results/ablation/ablation_{scenario}_{variant}.pt` — checkpoints
- `results/ablation/ablation_results.csv` — mean_return, std_return, collision_rate, pothole_hits_mean
- `results/ablation/ablation_results.json` — full results

See `docs/PHYSICS_INFORMED.md` for L_ego and Design A details.

## Command summary

| Command | What it does |
|---------|--------------|
| `make train` | Train both PINN and non-PINN on SUMO (50k steps) |
| `make train-1a` ... `make train-4` | Train on scenario 1a–4 |
| `make eval` | Evaluate model on SUMO (100 episodes) |
| `make eval-1a` ... `make eval-4` | Eval on scenario |
| `make ablation` | Run ablation study (all scenarios × 3 variants: nopinn, pinn, pinn_ego) |
| `make hpo` | Bayesian HPO (Optuna) |
| `make plot` | Plot training curves (return, loss, collision, TTC) |
| `make visualize` | Run 3 SUMO episodes, log to CSV |
| `make visualize-gui` | Run 3 SUMO episodes with GUI |

## Config

- `configs/reward/default.yaml` — reward weights
- `configs/algo/default.yaml` — PPO hyperparameters
- `configs/residuals/default.yaml` — physics-informed critic (Design A) lambdas
- `configs/scenario/default.yaml` — T-intersection layout

## Documentation

- `docs/PHYSICS_INFORMED.md` — Physics-informed critic (Design A) explained
- `docs/HYPERPARAMETERS.md` — All hyperparameter values
- `docs/STATE.md` — Full state vector specification
