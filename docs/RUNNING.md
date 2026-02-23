# Setup, Train, and Eval

Source of truth for running the project.

## Prerequisites

- Python 3.10+
- (Optional) SUMO for full sim; synthetic env works without SUMO

## Setup

```bash
make setup
```

Then activate the venv and set PYTHONPATH:

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

## Train

```bash
make train
```

Compare PINN vs non-PINN (trains both and saves separate CSVs):

```bash
make train-compare
```

Or:

```bash
export PYTHONPATH=$(pwd)
python3 experiments/run_train.py --config configs/algo/default.yaml
python3 experiments/run_train.py --compare --total_steps 50000
```

### Expected Output

- CSV metrics: `results/train_pinn.csv`, `results/train_nopinn.csv`
- Model checkpoints: `results/model_pinn.pt`, `results/model_nopinn.pt`

## Eval

```bash
make eval
```

Or:

```bash
python3 experiments/run_eval.py --checkpoint results/model_pinn.pt --episodes 100
```

### Expected Output

- CSV metrics: `results/eval_metrics.csv`

## Import Verification

```bash
python3 -c "import env.sumo_env"
python3 -c "import state.builder"
python3 -c "import rl.interface"
```

All three should run without import errors.
