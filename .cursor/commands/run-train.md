# Run Training

Train the behavioral DM at the T-intersection.

## Command
See `docs/RUNNING.md` for the exact train command. Example:

```bash
make train
# or: python experiments/run_train.py --config configs/algo/default.yaml
```

## Expected Output
- CSV metrics (e.g., `results/train_metrics.csv`)
- Plots (if configured)
- Model checkpoints in `runs/` or `results/`
