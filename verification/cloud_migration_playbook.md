# Tier 1 Cloud Migration Playbook

**Phase**: Pre-Tier-1 cloud-fallback staging while local Tier 1 runs.
**Purpose**: If DigitalOcean (or any cloud provider) approves CPU-Optimized rentals while the local Tier 1 is in flight, follow this playbook to decide whether to migrate, then how to migrate cleanly.
**No execution by this playbook itself** — staging document only.

---

## 0. Local launch reference (kill-clean target)

The local Tier 1 was launched as:

```bash
cd /home/boulosboulos/Desktop/EECE_499-main
PATH=/home/boulosboulos/.venvs/eece499/bin:$PATH \
PYTHONPATH=/home/boulosboulos/Desktop/EECE_499-main \
nohup setsid /home/boulosboulos/.venvs/eece499/bin/python \
  experiments/pde/run_full_ablation.py \
  --tier 1 --total_steps 400000 --max_parallel 22 \
  > /tmp/tier1_local.log 2>&1 < /dev/null &
echo $! > /tmp/tier1_local.pid
```

Outputs land at `results/ablation/tier1/<job_tag>/` (single-machine, no `--machine_id`).

The orchestrator PID is in `/tmp/tier1_local.pid`. Health snapshots every 30 min in `/tmp/tier1_health.log` (monitor PID `$(pgrep -f tier1_health_monitor.sh | grep -v claude | head -1)`).

---

## 1. Killing the local run cleanly

```bash
# Step 1: SIGTERM the orchestrator first (gives it a chance to log final state)
ORCH_PID=$(cat /tmp/tier1_local.pid)
kill -TERM "$ORCH_PID" 2>/dev/null

# Step 2: Wait 30 sec for graceful shutdown; if subprocesses persist, escalate
sleep 30

# Step 3: SIGKILL anything still attached to this launch
pkill -KILL -f "experiments/pde/run_full_ablation"
pkill -KILL -f "experiments/pde/train_"
pkill -KILL -f "experiments/pde/eval.py"
pkill -KILL -x sumo

# Step 4: Stop the health monitor (it'll otherwise keep logging the dead launch)
pkill -KILL -f "tier1_health_monitor.sh"

# Step 5: Verify zero stragglers
pgrep -f "experiments/pde/" | wc -l   # expect 0
pgrep -x sumo | wc -l                  # expect 0
```

**Do NOT delete `results/ablation/tier1/`** — the per-job dirs hold partial-completion state used by Step 2 below.

---

## 2. Computing partial-completion state

Before deciding migration vs continue-local, count how many jobs are already done.

```bash
cd /home/boulosboulos/Desktop/EECE_499-main

# Total target = 1,680 (1,440 trainable + 240 rule_based eval-only)
TOTAL_TARGET=1680

# Count training-complete: meta.json with non-null end_time_iso
# (rule_based jobs have no meta.json since they're eval-only)
N_TRAIN_DONE=$(find results/ablation/tier1 -maxdepth 2 -name meta.json \
    -exec sh -c 'grep -q "\"end_time_iso\": \"[^n]" "$1"' _ {} \; -print 2>/dev/null \
    | wc -l)

# Count eval-complete: eval_metrics.csv with > 1 line (header + ≥1 episode)
N_EVAL_DONE=$(find results/ablation/tier1 -maxdepth 2 -name eval_metrics.csv \
    -exec sh -c '[ "$(wc -l < "$1")" -gt 1 ]' _ {} \; -print 2>/dev/null \
    | wc -l)

# Distinct (combo, method, seed, intent) tuples seen
N_DIRS=$(ls -1 results/ablation/tier1 2>/dev/null | wc -l)

echo "Result dirs: $N_DIRS"
echo "Training complete: $N_TRAIN_DONE"
echo "Eval complete: $N_EVAL_DONE  (job fully done = eval rows present)"
echo "Total target: $TOTAL_TARGET"
```

Each result dir is a unique `(scenario, ego_maneuver, method, intent_tag, seed)` tuple. The tuple is encoded in the dir name:
`<scenario>_<ego_maneuver>_<method>_<{intent|nointent}>_s<seed>`

To enumerate which tuples are NOT yet done, generate the full manifest and diff:

```bash
PYTHONPATH=. /home/boulosboulos/.venvs/eece499/bin/python <<'EOF'
import os, json
from experiments.pde.run_full_ablation import (
    generate_jobs, _apply_filters, _partition_balance_jobs,
)
from types import SimpleNamespace
jobs = generate_jobs(tier="1", total_steps=400000)
jobs = _apply_filters(jobs, SimpleNamespace(
    seeds=None, methods=None, scenarios=None, maneuvers=None,
    intents=None, include_rule_based=True,
))
jobs = _partition_balance_jobs(jobs)
print(f"Manifest size: {len(jobs)}")
done_tags = set()
results_dir = "results/ablation/tier1"
if os.path.isdir(results_dir):
    for d in os.listdir(results_dir):
        meta_path = os.path.join(results_dir, d, "meta.json")
        if os.path.isfile(meta_path):
            try:
                m = json.load(open(meta_path))
                if m.get("end_time_iso") not in (None, "null"):
                    done_tags.add(d)
            except Exception:
                pass
        # rule_based has no meta.json — use eval_metrics.csv presence
        eval_path = os.path.join(results_dir, d, "eval_metrics.csv")
        if "rule_based" in d and os.path.isfile(eval_path):
            try:
                with open(eval_path) as f:
                    if sum(1 for _ in f) > 1:
                        done_tags.add(d)
            except Exception:
                pass
remaining = []
for j in jobs:
    intent_tag = "intent" if j.get("intent_on") else "nointent"
    tag = f"{j['scenario']}_{j['ego_maneuver']}_{j['method']}_{intent_tag}_s{j['seed']}"
    if tag not in done_tags:
        remaining.append((j, tag))
print(f"Done: {len(done_tags)}")
print(f"Remaining: {len(remaining)}")
# write to disk for migration step
with open("/tmp/tier1_remaining_manifest.json", "w") as f:
    json.dump([{**j, "tag": tag} for j, tag in remaining], f, indent=2, default=str)
print(f"Remaining manifest -> /tmp/tier1_remaining_manifest.json")
EOF
```

After this, `/tmp/tier1_remaining_manifest.json` holds every (combo, method, seed, intent) yet to be done.

---

## 3. Decision tree: migrate vs continue local

Local Tier 1 runs at ~22-parallel and is projected to take ~6.4 days end-to-end. Cloud (4 × CCX53) runs at 4 × 22 = 88-parallel and would finish in ~1.5–2 days from a clean start.

| Local progress when cloud opens | Recommendation |
|---|---|
| **< 25% complete** (< 420 jobs done) | **Migrate**. Saved time worth the migration cost; cloud finishes the rest in ~1 day vs ~5 days local. |
| **25–60% complete** (420–1008 jobs done) | **Migrate the remaining**. Cloud picks up from `/tmp/tier1_remaining_manifest.json`; local is killed. Cloud finishes in ~12–24 h. |
| **> 60% complete** (> 1008 jobs done) | **Stay local**. Migration cost (rsync + cloud spin-up + per-machine boot ~1 h) eats most of the saved wall time. Local finishes in ~2.5 days. |
| **> 85% complete** | **Stay local definitively**. Don't touch anything. Migration would only delay. |

Plus: cloud rentals cost ~$0.10–0.20/hour per Droplet; 4 × 24 h ≈ $10–20 rental burn. The local PC is already paid; only marginal electricity cost. Factor cost into decisions for borderline cases.

---

## 4. Multi-machine launch — 4-way and 10-way splits

### 4.1 Determinism contract

`preview_tier_1_split.py` and `run_full_ablation.py` BOTH import the same `manifest_sort_key` and `_partition_balance_jobs` helpers — slice indices on different machines correspond to identical job sets by construction. Determinism re-verified during this playbook:

```
4-way:  sha256(preview output) = 8e8378f2ada7b96234c26780dfb9b25938c16e3113f0bed721b9f319eb9485d3   (byte-identical across 2 runs)
10-way: sha256(preview output) = b96f142730728275c5a728acb3a89a8534762b7a32f985926ef5c2048f8d0a5c   (byte-identical across 2 runs)
```

### 4.2 4-way split

Per-machine launch indices (each slice = 360 trainable + 60 rule_based = 420 jobs):

| Machine | `--job_index_start` | `--job_index_end` | First job | Last job |
|---|---:|---:|---|---|
| 1 | 0    | 420  | cbf_aux × 1a × right_stem × s42 × nointent | rule_based × 1a × stem_right × s5555 × intent |
| 2 | 420  | 840  | drppo × 2 × stem_right × s42 × nointent | rule_based × 2 × stem_left × s5555 × intent |
| 3 | 840  | 1260 | fusion_aux × 1a × right_stem × s42 × nointent | rule_based × 3 × left_right × s5555 × intent |
| 4 | 1260 | 1680 | hjb_aux × 2 × stem_right × s42 × nointent | soft_hjb_aux × 4 × stem_right × s5555 × intent |

Per-rental launch command (substitute `<N>` ∈ {1, 2, 3, 4} and the matching `<start>`, `<end>` values):

```bash
ssh root@<rental-N>
cd /opt/EECE_499  # cloned from github.com/BoulosBoulos/EECE_499 @ tag v2_phase_3F_step_12-tier_1_ready
PATH=$VENV_BIN:$PATH PYTHONPATH=$(pwd) nohup setsid python3 \
  experiments/pde/run_full_ablation.py \
  --tier 1 --total_steps 400000 --max_parallel 32 \
  --job_index_start <start> --job_index_end <end> \
  --machine_id <N> \
  > /tmp/tier1_machine_<N>.log 2>&1 < /dev/null &
echo $! > /tmp/tier1_machine_<N>.pid
```

CCX53 has 32 dedicated vCPUs → `--max_parallel 32` (vs 22 locally for the 24-core/32-thread i9).

### 4.3 10-way split

Per-machine launch indices (each slice = 144 trainable + 24 rule_based = 168 jobs):

| Machine | `--job_index_start` | `--job_index_end` | First | Last |
|---|---:|---:|---|---|
| 1  | 0    | 168  | cbf_aux × 1a × right_stem × s42 × nointent       | rule_based × 1a × stem_left × s123 × intent |
| 2  | 168  | 336  | cbf_aux × 2_dense × right_left × s456 × nointent | rule_based × 1a × stem_right × s789 × intent |
| 3  | 336  | 504  | drppo × 1a × stem_right × s999 × nointent        | rule_based × 1b × stem_left × s1111 × intent |
| 4  | 504  | 672  | drppo × 3 × right_left × s2222 × nointent        | rule_based × 1b × stem_right × s3333 × intent |
| 5  | 672  | 840  | eikonal_aux × 1b × stem_right × s4444 × nointent | rule_based × 2 × stem_left × s5555 × intent |
| 6  | 840  | 1008 | fusion_aux × 1a × right_stem × s42 × nointent    | rule_based × 2_dense × right_left × s123 × intent |
| 7  | 1008 | 1176 | fusion_aux × 2_dense × right_left × s456 × nointent | rule_based × 3 × left_right × s789 × intent |
| 8  | 1176 | 1344 | hjb_aux × 1a × stem_right × s999 × nointent      | rule_based × 3 × right_left × s1111 × intent |
| 9  | 1344 | 1512 | hjb_aux × 3 × right_left × s2222 × nointent      | soft_hjb_aux × 1b × stem_right × s3333 × intent |
| 10 | 1512 | 1680 | rule_based × 4 × stem_left × s4444 × nointent    | soft_hjb_aux × 4 × stem_right × s5555 × intent |

### 4.4 Re-running the preview to refresh indices

Indices are deterministic but if the manifest ever changes (new combos, new methods), re-run the preview:

```bash
cd /home/boulosboulos/Desktop/EECE_499-main
PYTHONPATH=. /home/boulosboulos/.venvs/eece499/bin/python \
  experiments/pde/preview_tier_1_split.py --n_machines 4
PYTHONPATH=. /home/boulosboulos/.venvs/eece499/bin/python \
  experiments/pde/preview_tier_1_split.py --n_machines 10
```

Always sanity-check with two consecutive runs and `cmp -s` / `sha256sum` — if outputs differ, do NOT launch (the sort key is broken).

---

## 5. Result aggregation

Each cloud Droplet writes to `results/tier_1_machine_<N>/tier1/<job_tag>/`. After all rentals complete:

### 5.1 Pull results back to local

```bash
# On local workstation
cd /home/boulosboulos/Desktop/EECE_499-main
mkdir -p results
for N in 1 2 3 4; do
    rsync -avz --partial --progress \
        root@<rental-N>:/opt/EECE_499/results/tier_1_machine_${N}/ \
        results/tier_1_machine_${N}/
done
```

For 10-way, `for N in $(seq 1 10)`. Use `--partial --progress` so an interrupted rsync resumes.

### 5.2 Merge into unified tree

```bash
PYTHONPATH=. /home/boulosboulos/.venvs/eece499/bin/python \
  experiments/pde/aggregate_tier_1_results.py \
    --source_dirs results/tier_1_machine_1/tier1 \
                  results/tier_1_machine_2/tier1 \
                  results/tier_1_machine_3/tier1 \
                  results/tier_1_machine_4/tier1 \
    --target_dir results/tier_1
```

For 10-way, list all 10 source dirs.

The aggregator skips duplicates (job_dir name collision) — these would only happen if two machines somehow ran the same slice, which is impossible if `preview_tier_1_split.py` was used. Duplicates are reported but never overwritten; manual reconciliation if any.

### 5.3 Completeness check

```bash
N_FINAL=$(find results/tier_1 -maxdepth 2 -name metrics.csv | wc -l)
echo "Trainable jobs in results/tier_1: $N_FINAL  (expect 1440)"
N_RB=$(find results/tier_1 -maxdepth 2 -name eval_metrics.csv -path "*rule_based*" | wc -l)
echo "Rule-based eval-only jobs: $N_RB  (expect 240)"
echo "Total jobs: $((N_FINAL + N_RB))  (expect 1680)"
```

If counts are off, re-rsync from the relevant machine — likely an in-flight job at the time of original rsync.

---

## 6. Hybrid mode (continue local + add cloud machines)

If only some cloud rentals come online (e.g. 2 of 4 approved), you can shard the remaining manifest across whatever's available. Generate the remaining manifest per §2, then split it instead of the full 1,680. Example: if 800 done locally and 880 remain, with 2 cloud Droplets:

```bash
# Synthesize a "shifted preview" by passing the remaining count as a tier override
# (or hand-compute: 2-way of 880 = 440 each; rebalance trainable/RB if needed).
# Conservative: use the original preview indices but add --job_index_start
# and --job_index_end that correspond to the not-yet-done portion of each slice.
```

This is a manual exercise — there's no built-in tool for partial-resume cloud sharding. Recommended approach: **don't** hybrid; pick local OR cloud, not both. Hybrid coordination is fragile (clock skew, partial config_lock drift, intent encoder version mismatch).

---

## 7. Stop conditions during cloud run

Mirror the local stop conditions (Phase 4 of `PromptF.txt`):

1. Any rental's orchestrator dies unexpectedly (PID gone, `[Done:]` not in log) → SSH to that rental, inspect `/tmp/tier1_machine_<N>.log` tail.
2. Disk free on any rental drops below 30 GB → check `df -h /` on each.
3. Load average drops below 5 for > 30 min on any rental → orchestrator may have crashed.
4. SUMO process count drops to 0 for > 5 min on any rental.
5. NaN / Inf in any metrics.csv on any rental.

Roll up a per-rental health monitor analogous to `/home/boulosboulos/tier1_health_monitor.sh` (modify `RESULTS_DIR` + log paths) and run it on each rental.

---

## 8. Provenance after migration

After aggregation completes, record the migration in `verification/`:

```bash
PYTHONPATH=. /home/boulosboulos/.venvs/eece499/bin/python <<EOF
import json, datetime
record = {
    "migration_phase": "tier_1_local_to_cloud",
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit_at_migration": "$(git rev-parse HEAD)",
    "git_tag": "v2_phase_3F_step_12-tier_1_ready",
    "n_machines_used": 4,  # or 10
    "local_partial_completion_at_migration": {
        "n_dirs": <fill>,
        "n_train_done": <fill>,
        "n_eval_done": <fill>,
    },
    "cloud_completion": {
        "n_dirs": <fill>,
        "n_train_done": <fill>,
        "n_eval_done": <fill>,
    },
    "aggregated_into": "results/tier_1",
    "expected_total": 1680,
}
json.dump(record, open("verification/tier_1_migration_record.json", "w"), indent=2)
EOF
```

Commit this record (`git add verification/tier_1_migration_record.json && git commit -m "Record Tier 1 cloud-migration provenance"`) so the experimental result chain is reproducible end-to-end from the source-code lock at `4c5ead7` (tag `v2_phase_3F_step_12-tier_1_ready`) through the migrated run.

---

## 9. What NOT to do during migration

- Do NOT delete local `results/ablation/tier1/` until cloud aggregation is verified complete.
- Do NOT change `config_frozen_v1.yaml`, `config_lock.json`, or any code in `experiments/pde/`, `models/`, `env/`, `scenario/` between local and cloud runs — slice indices and the result-merge logic depend on bit-identical code.
- Do NOT run two orchestrators on the same machine_id (would cause job_dir collisions).
- Do NOT mix 4-way and 10-way slices in the same aggregation (slice boundaries differ; would cause gaps).
- Do NOT push to GitHub from the cloud Droplets (auth + commit graph divergence). Only the local workstation pushes.
- Do NOT force-sync the local clock; SUMO time is internal to each Droplet.

---

## 10. Quick reference

| Action | Command |
|---|---|
| Kill local Tier 1 cleanly | See §1 |
| Show partial-completion state | See §2 |
| Refresh per-machine indices (4-way) | `python experiments/pde/preview_tier_1_split.py --n_machines 4` |
| Refresh per-machine indices (10-way) | `python experiments/pde/preview_tier_1_split.py --n_machines 10` |
| Per-rental launch | See §4.2 / §4.3 |
| Pull results back | See §5.1 |
| Merge results | `python experiments/pde/aggregate_tier_1_results.py --source_dirs … --target_dir results/tier_1` |
| Health monitor | `/home/boulosboulos/tier1_health_monitor.sh --once` (one-shot) or detached loop already running |

---

**Generated**: 2026-05-09 by Phase 3B of `PromptF.txt`.
**Determinism verified**: 4-way sha256 `8e8378f2…`, 10-way sha256 `b96f1427…`, byte-identical re-runs.
**Source lock**: tag `v2_phase_3F_step_12-tier_1_ready` at commit `4c5ead7`.
