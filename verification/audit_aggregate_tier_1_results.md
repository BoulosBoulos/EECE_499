# Audit — `experiments/pde/aggregate_tier_1_results.py`

**Type**: read-only audit (PromptF Phase 3C).
**Date**: 2026-05-09.
**Author**: PromptF audit pass; no fixes applied. All findings need user approval before any code change.
**File**: `experiments/pde/aggregate_tier_1_results.py` @ HEAD `b648cda` (102 lines).
**Source for the script**: PromptC §3.3 + PromptD spec confirmation that aggregator was unchanged.

The script merges per-machine `tier_1_machine_<N>/tier1/<job_tag>/` subtrees into one `tier_1/` tree, skipping name collisions, and prints a count of `metrics.csv` files for a completeness signal. Below are 11 edge cases / improvement notes. Severity: **B** = potentially blocks correctness on the production aggregation; **W** = warning / quality-of-life; **I** = informational.

---

## Findings

### F1 (B) — Atomicity: a killed `shutil.copytree` leaves a partial target dir
- **Where**: line 76 `shutil.copytree(job_dir, target_job)`.
- **Issue**: `copytree` is not atomic; it builds the target directory file-by-file. If the user CTRL-Cs mid-copy, the target dir contains a partial subset of the source files. The next aggregation run will see `target_job.exists()` and skip it (line 70-72 → `duplicates.append`), so the partial target is permanently broken.
- **Production impact**: a single CTRL-C during a 200–500 GB Tier 1 aggregation produces an unrecoverable corrupt result for one job dir, with no warning to the user. The completeness count (line 97) sees the metrics.csv and reports the job as "done" even though e.g. the model checkpoint is missing.
- **Suggested fix (DO NOT APPLY without approval)**: copy to `target_job.with_suffix(".tmp")`, then `os.rename(...)` to the final name. `os.rename` is atomic within a filesystem.

### F2 (B) — `--source_dirs` accepts paths that are still being written to (in-flight runs)
- **Where**: line 57-64 (loop over `args.source_dirs`).
- **Issue**: no check that the source machine has actually FINISHED. If the user runs aggregation while a rental's orchestrator is still active, in-flight job dirs (those without final `metrics.csv` or with a partial last row) get copied. Subsequent reruns will skip them as duplicates even after the rental finishes.
- **Production impact**: silent partial aggregation. The user has no warning that they aggregated an active source.
- **Suggested fix**: optional `--require_complete` flag that checks each source for the orchestrator's "Done:" sentinel in its log, OR checks each job dir's meta.json for `end_time_iso != None`. Refuse to aggregate sources that look in-flight unless `--force`.

### F3 (W) — `shutil.copytree` is slow + memory-hungry for large model checkpoints
- **Where**: line 76.
- **Issue**: each Tier 1 job dir contains 3 intermediate model checkpoints (`model_*_step{200k,300k,400k}.pt`) plus a final checkpoint, totaling ~50–200 MB per job. For 1,440 trainable jobs that's ~100–300 GB to copy. `shutil.copytree` is single-threaded and copies file-by-file with default 64 KB chunks — a typical aggregation will take 10–60 minutes wall.
- **No progress reporting** during the copy means the user can't tell if it's progressing.
- **Suggested fix**: log a per-job-dir progress line; OR fall back to `os.rename` if same filesystem (zero-copy); OR offer an `--rsync` mode that shells out to `rsync` with `--progress`.

### F4 (W) — Cross-filesystem move is sub-optimal compared to `mv` / hardlinks
- **Where**: line 76.
- **Issue**: `copytree` always physically duplicates bytes. If the user's source dirs are on the same filesystem as the target, `os.rename` is instant + atomic. The aggregator never tries this.
- **Suggested fix**: `try: os.rename(job_dir, target_job)` first; on `OSError` (cross-device), fall back to copytree.

### F5 (B) — Symlink handling unspecified; default is to preserve as symlinks
- **Where**: line 76.
- **Issue**: `shutil.copytree` defaults to `symlinks=False`, which means symlinks in the source are FOLLOWED and the targets are copied. But the project doesn't currently use symlinks anywhere I can see, so behavior is consistent. WORST case: if any future per-job dir has a symlink (e.g., to a shared `intent_model_v9_member0.pt`), it would be physically copied, massively bloating storage. Or if `symlinks=True`, the symlink would be preserved pointing to the source path — broken after source machine is destroyed.
- **Suggested fix**: explicit `symlinks=False` and document the behavior in the docstring.

### F6 (W) — Duplicate detection by `Path.name` only — first-write wins
- **Where**: line 69-72.
- **Issue**: detection is `target_job.exists()`. If two source dirs have the same `job_dir.name` (e.g., both have `1a_stem_right_hjb_aux_intent_s42`), only the first is copied. The duplicate is reported and skipped — but the SKIPPED one might be the more complete copy.
- **Production impact**: with deterministic `preview_tier_1_split.py` indices this shouldn't happen — every machine runs a distinct slice. But IF the user manually overrides indices and accidentally double-runs a slice, the aggregator silently picks the first machine's result.
- **Suggested fix**: when a duplicate is detected, compare the two for completeness (number of metrics.csv rows, whether eval_metrics.csv exists) and keep the more complete one. Or fail loudly with `--strict` flag.

### F7 (W) — Completeness signal counts `metrics.csv` but ignores rule_based eval-only dirs
- **Where**: line 97 `target.glob("**/metrics.csv")`.
- **Issue**: rule_based is eval-only (no metrics.csv). For Tier 1, the canonical breakdown is 1,440 trainable + 240 rule_based eval-only = 1,680 total. The completeness count only sees 1,440 (at most), so the user reading "Final: X jobs with metrics.csv" thinks they're missing 240 jobs when they're actually complete.
- **Production impact**: misleading completeness signal for any Tier where rule_based is included.
- **Suggested fix**: also count `eval_metrics.csv` (rule_based jobs have those), and report:
  ```
  Final: <X> trainable jobs with metrics.csv + <Y> eval-only jobs with eval_metrics.csv = <X+Y> total
  ```

### F8 (W) — No NaN/Inf scanning post-merge
- **Where**: nowhere — never added.
- **Issue**: a job that NaN'd during training still produces metrics.csv (with NaN values). The aggregator copies it as "complete". The user has no signal that some merged jobs are silently broken.
- **Suggested fix**: optional `--scan_nan` post-merge mode that opens each metrics.csv, checks for any NaN/Inf in numeric columns, and reports a per-job pass/fail summary.

### F9 (W) — `--target_dir` not protected against being one of `--source_dirs`
- **Where**: lines 47-72.
- **Issue**: if the user accidentally passes `--source_dirs results/tier_1/tier1 ... --target_dir results/tier_1`, the script would walk the existing target as source, see all dirs already exist, and report "1,680 duplicates skipped". Confusing but non-destructive.
- **Suggested fix**: refuse the operation when source ∈ target hierarchy, or assert source != target.

### F10 (I) — `--dry_run` does not preview duplicate detection
- **Where**: line 73-77.
- **Issue**: in `--dry_run`, `target.exists()` returns False for non-existing target trees, so all copies are reported as new. If the user runs aggregation, sees 1680 copies, then re-runs aggregation expecting "0 duplicates", they're surprised. Not a correctness bug — just a UX confusion.

### F11 (I) — Per-source job_dir count not reported
- **Where**: end of run (line 82).
- **Issue**: only the total_copied count is reported. The user can't tell how many came from each machine. For debugging "machine 3 finished early", this is opaque.
- **Suggested fix**: print per-source breakdown:
  ```
  Source results/tier_1_machine_1/tier1: 360 dirs copied, 0 skipped
  Source results/tier_1_machine_2/tier1: 360 dirs copied, 0 skipped
  ...
  ```

---

## Severity summary

| ID | Severity | Issue |
|---|:---:|---|
| F1 | B | Mid-copy CTRL-C corrupts target |
| F2 | B | Aggregating in-flight sources |
| F5 | B | Symlink preservation semantics unspecified |
| F3 | W | No progress reporting on slow copies |
| F4 | W | Misses fast `os.rename` for same-FS case |
| F6 | W | Duplicate handling first-write-wins |
| F7 | W | Completeness count misses rule_based jobs |
| F8 | W | No post-merge NaN/Inf scan |
| F9 | W | source ∈ target not validated |
| F10 | I | --dry_run does not show duplicates |
| F11 | I | No per-source breakdown |

---

## Recommendation

Three findings (F1, F2, F7) plausibly affect production Tier 1 aggregation correctness. F1 (atomicity) is the most acute — **a single CTRL-C during the 10–60 minute aggregation produces a silently corrupt result that survives subsequent runs**. F2 (in-flight sources) affects user workflow but is recoverable. F7 (rule_based count) is misleading but easy to read around.

**Suggested patch sequence (NOT APPLIED)**: F1 + F2 + F7 in one focused PR; F3-F6, F8-F11 as a follow-up if time allows. Total: ~30 lines of diff. None block Tier 1 launch *now* — they only matter when aggregation actually happens later. The local Tier 1 currently in flight will write to `results/ablation/tier1/` directly (single-machine path, no aggregation needed).
