# Guide: How to Run the 16-GPU Ablation & What to Expect

This guide walks you through running the full ablation (all variants, sensitivity sweep, deterministic + stochastic eval) in two 48-hour batches, then viewing results.

---

## Prerequisites

- **Python 3.10+**
- **SUMO** installed (e.g. `sudo apt install sumo sumo-tools`)
- **16 GPUs** (or set `--num_gpus` to your count)
- Terminal at your **project root** (folder with `Makefile`, `experiments/`, `env/`)

---

## Step-by-step commands

Run these **in order**. Do not skip the one-time setup.

### One-time setup

| Step | Command | What to expect |
|------|---------|----------------|
| 1 | `cd /path/to/your/project/root` | You are in the project directory. |
| 2 | `make setup` | Creates `.venv`, installs dependencies. May take 1–2 minutes. |
| 3 | `source .venv/bin/activate` | Prompt may show `(.venv)`. Required for later steps. |
| 4 | `export PYTHONPATH=$(pwd)` | No output. Needed so Python finds your modules. |
| 5 | `export SUMO_HOME=/usr/share/sumo` | No output. Use your SUMO path if different. |
| 6 | `make regen-scenarios` | Generates `scenarios/sumo_1a` … `sumo_4`. Needed if `scenarios/` is missing. |

---

### Batch 1 (first 48-hour run) — scenarios 1a, 1b, 1c, 1d

| Step | Command | What to expect |
|------|---------|----------------|
| 7 | `make jobs-manifest-batch1` | Prints "Wrote … job_manifest.json with N jobs." Creates `results/ablation_batch1/`. |
| 8 | `make ablation-16gpu-batch1` | Starts 16 background workers. Logs go to `results/ablation_batch1/jobs/worker_*.log`. Run for up to ~48 hours. |
| 9 | *(Wait until all workers finish.)* | Script prints "All 16 workers finished successfully." or "WARNING: N workers failed." Check logs if any failed. |
| 10 | `make ablation-aggregate-batch1` | Merges per-job CSVs into `ablation_results.csv` and `ablation_train_log.csv` in `results/ablation_batch1/`. Exits 1 if any job is missing. |

---

### Batch 2 (second 48-hour run) — scenarios 2, 3, 4

| Step | Command | What to expect |
|------|---------|----------------|
| 11 | `make jobs-manifest-batch2` | Same as step 7, for batch 2. Creates `results/ablation_batch2/`. |
| 12 | `make ablation-16gpu-batch2` | Same as step 8, for batch 2. Again up to ~48 hours. |
| 13 | *(Wait until all workers finish.)* | Same as step 9. |
| 14 | `make ablation-aggregate-batch2` | Same as step 10, for batch 2. |

---

### Merge and view results

| Step | Command | What to expect |
|------|---------|----------------|
| 15 | `make ablation-merge` | Merges batch1 and batch2 into `results/ablation/`. Prints "Merged N eval rows and M train rows -> results/ablation/". |
| 16 | `make dashboard` | Starts Streamlit. Open **http://localhost:8501** in a browser. Use "Ablation Summary" and "Raw Tables" to inspect results. |

---

## What gets run

- **10 variants:** nopinn, pinn_critic, pinn_actor, pinn_both, pinn_ego, pinn_no_ttc, pinn_no_stop, pinn_no_fric, safety_filter, pinn_critic_sf  
- **7 scenarios:** 1a, 1b, 1c, 1d (batch 1); 2, 3, 4 (batch 2)  
- **5 seeds:** 42, 123, 456, 789, 999  
- **λ sweep:** 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 (for PINN variants)  
- **Eval:** 50 episodes **deterministic** + 50 episodes **stochastic** per job  

---

## If something fails

- **Launcher exits with code 1:** Some workers failed. Check `results/ablation_batchN/jobs/worker_*.log` for the failing GPU index and error.  
- **Aggregate exits with code 1:** Some jobs did not produce valid eval or train CSVs. Fix or re-run those jobs, then run the aggregate again.  
- **Merge fails:** Ensure both `make ablation-aggregate-batch1` and `make ablation-aggregate-batch2` finished with exit 0 and produced `ablation_results.csv` and `ablation_train_log.csv` in each batch folder.  

---

## Optional: plot without dashboard

```bash
make plot-ablation
```

Uses `results/ablation/ablation_results.csv` and writes plots into `results/ablation/`.

---

## Quick copy-paste block

*(After one-time setup: venv active, PYTHONPATH and SUMO_HOME set.)*

```bash
make jobs-manifest-batch1
make ablation-16gpu-batch1
# wait until done (~48h)
make ablation-aggregate-batch1

make jobs-manifest-batch2
make ablation-16gpu-batch2
# wait until done (~48h)
make ablation-aggregate-batch2

make ablation-merge
make dashboard
```
