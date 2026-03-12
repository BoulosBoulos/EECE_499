# How to Run the Setup — A to Z

Exact steps to run the 16-GPU ablation in two batches (scenarios 1a–1d, then 2–3–4), each within a 48-hour run.

---

## A. Prerequisites

- **Python 3.10+**
- **SUMO** installed (e.g. `sudo apt install sumo sumo-tools` on Ubuntu)
- **16 GPUs** available on the machine (or fewer; adjust `--num_gpus`)

---

## B. One-Time Setup

From the **project root** (the directory that contains `Makefile`, `experiments/`, `env/`, etc.):

```bash
cd /path/to/your/project/root

make setup
source .venv/bin/activate
export PYTHONPATH=$(pwd)
export SUMO_HOME=/usr/share/sumo
```

*(Set `SUMO_HOME` to your SUMO install path if different.)*

---

## C. Generate SUMO Scenarios (If Not Already Present)

If the `scenarios/` directory with `sumo_1a`, `sumo_1b`, … is missing:

```bash
make regen-scenarios
```

---

## D. (Optional) Train Intent LSTM (for use_intent runs)

If you plan to use the **intent LSTM** (state includes VRU intent/style distributions), train it once:

```bash
make train-intent
```

This writes `results/intent_model.pt`, which `env.SumoEnv` loads when `use_intent=True`.

---

## E. Run 16-GPU Ablation in Two Batches (48h Each, no intent)

### Run 1 — Batch 1 (scenarios 1a, 1b, 1c, 1d)

**Step 1.** Generate the job manifest for batch 1:

```bash
make jobs-manifest-batch1
```

**Step 2.** Launch 16 workers (run for up to 48 hours). Runs both deterministic and stochastic evaluation per job.

```bash
make ablation-16gpu-batch1
```

**Step 3.** After all workers finish, aggregate batch 1 results:

```bash
make ablation-aggregate-batch1
```

*(If any worker failed, the launcher will have exited with code 1; check `results/ablation_batch1/jobs/worker_*.log`.)*

---

### Run 2 — Batch 2 (scenarios 2, 3, 4)

**Step 1.** Generate the job manifest for batch 2:

```bash
make jobs-manifest-batch2
```

**Step 2.** Launch 16 workers (again, up to 48 hours). Runs both deterministic and stochastic evaluation per job.

```bash
make ablation-16gpu-batch2
```

**Step 3.** After all workers finish, aggregate batch 2 results:

```bash
make ablation-aggregate-batch2
```

---

### After Both Batches — Merge into One Result Set

**Step 4.** Merge batch 1 and batch 2 CSVs into `results/ablation/`:

```bash
make ablation-merge
```

This writes:
- `results/ablation/ablation_results.csv`
- `results/ablation/ablation_train_log.csv`

---

## F. (Optional) Intent LSTM Ablation on 16 GPUs

To compare **with vs without intent LSTM** for every combo (scenario × variant × λ × seed) in the 16‑GPU split pipeline:

1. Make sure `results/intent_model.pt` exists:

   ```bash
   make train-intent
   ```

2. Generate manifests that include **both** `use_intent=False` and `use_intent=True` jobs:

   ```bash
   make jobs-manifest-batch1-intent-ablation   # scenarios 1a–1d
   make jobs-manifest-batch2-intent-ablation   # scenarios 2–4
   ```

3. Launch, aggregate, and merge exactly as in sections E.1–E.3:

   ```bash
   make ablation-16gpu-batch1
   make ablation-aggregate-batch1

   make ablation-16gpu-batch2
   make ablation-aggregate-batch2

   make ablation-merge
   ```

The merged `results/ablation/ablation_results.csv` will contain a `use_intent` column so the dashboard and plots can separate LSTM vs non‑LSTM runs.

---

## G. View Results

**Dashboard (interactive):**

```bash
make dashboard
```

Then open **http://localhost:8501** in a browser.

**Ablation plots:**

```bash
make plot-ablation
```

*(Uses `results/ablation/ablation_results.csv`; outputs in `results/ablation/`.)*

---

## H. Quick Reference — Copy-Paste Block

Assumes you are in the project root, venv is activated, and `PYTHONPATH` and `SUMO_HOME` are set.

**One-time:**
```bash
make setup && source .venv/bin/activate
export PYTHONPATH=$(pwd) SUMO_HOME=/usr/share/sumo
make regen-scenarios   # if scenarios/ missing
```

**Batch 1 (first 48h run; deterministic + stochastic eval):**
```bash
make jobs-manifest-batch1
make ablation-16gpu-batch1
make ablation-aggregate-batch1
```

**Batch 2 (second 48h run; deterministic + stochastic eval):**
```bash
make jobs-manifest-batch2
make ablation-16gpu-batch2
make ablation-aggregate-batch2
```

**Merge and view:**
```bash
make ablation-merge
make dashboard
```

---

## I. Single-Run (Full Manifest, No Split)

If you have no 48h limit and want one big run with all 7 scenarios:

```bash
make jobs-manifest
make ablation-16gpu
make ablation-aggregate
```

Then `make dashboard` or `make plot-ablation` (results in `results/ablation/`).

---

## J. Test One Job (Debug)

To run a single job on GPU 0 before launching the full batch:

```bash
make jobs-manifest-batch1
CUDA_VISIBLE_DEVICES=0 python3 experiments/run_single_job.py --manifest results/ablation_batch1/job_manifest.json --job_id 0 --out_dir results/ablation_batch1
```

---

## K. Failure Handling

- **Launcher exits 1:** At least one worker failed. Check `results/ablation_batchN/jobs/worker_*.log`.
- **Aggregate exits 1:** Some jobs are missing or incomplete (no eval/train CSV or empty). Re-run failed jobs or fix the cause, then aggregate again.
- **Merge:** Run `make ablation-merge` only after both `ablation-aggregate-batch1` and `ablation-aggregate-batch2` have succeeded (exit 0).
