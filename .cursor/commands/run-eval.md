# Run Evaluation

Evaluate the trained behavioral DM.

## Command
See `docs/RUNNING.md` for the exact eval command. Example:

```bash
make eval
# or: python experiments/run_eval.py --checkpoint runs/best.pt --episodes 100
```

## Expected Output
- CSV metrics (e.g., `results/eval_metrics.csv`)
- Plots (collision rate, behavior distribution, etc.)
