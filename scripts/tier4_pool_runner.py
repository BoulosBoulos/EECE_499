#!/usr/bin/env python3
"""
Distributed T4 eval worker.

Pools remaining (not-done) eval jobs from all 8 source machines, takes this
worker's balanced slice, and runs them with --max_parallel concurrency.

Designed to be called by run_tier4_rebalance.sh as an 8-way SLURM array.
Skip-if-done is checked both at scan time (to partition only real remaining
work) and at run time (re-checked before each job in case another worker just
finished it).
"""
import argparse
import os
import subprocess
import sys
import time


def _get_out_dir(job: dict) -> str:
    cmd = job["cmd_eval"]
    idx = cmd.index("--out_dir") + 1
    return cmd[idx]


def _is_done(job: dict) -> bool:
    return os.path.exists(os.path.join(_get_out_dir(job), "eval_metrics.csv"))


def main():
    p = argparse.ArgumentParser(description="T4 pool eval worker")
    p.add_argument("--worker_id",    type=int, required=True, help="1-indexed worker ID")
    p.add_argument("--n_workers",    type=int, default=8)
    p.add_argument("--total_steps",  type=int, default=400000)
    p.add_argument("--max_parallel", type=int, default=30)
    p.add_argument("--repo",         type=str, required=True, help="Absolute path to repo root")
    args = p.parse_args()

    # Stagger filesystem scan to avoid NFS thundering herd at job start
    stagger = (args.worker_id - 1) * 20
    if stagger:
        print(f"[w{args.worker_id}] staggering {stagger}s to spread NFS load …")
        time.sleep(stagger)

    sys.path.insert(0, args.repo)
    from experiments.pde.run_full_ablation import generate_tier4_jobs

    print(f"[w{args.worker_id}/{args.n_workers}] Scanning T4 jobs across all 8 source machines …")
    all_jobs = []
    for m in range(1, 9):
        mid = f"cmu{m}"
        src = {
            "tier1":       os.path.join(args.repo, f"results/tier_1_machine_{mid}_p2/tier1"),
            "tier2_noocc": os.path.join(args.repo, f"results/tier_2_machine_{mid}/tier2/2b_occlusion_sweep"),
            "tier3_behav": os.path.join(args.repo, f"results/tier_3_machine_{mid}/tier3_behav"),
        }
        outroot = os.path.join(args.repo, f"results/tier_4_machine_{mid}")
        jobs = generate_tier4_jobs(args.total_steps, outroot, src)
        done_m = sum(1 for j in jobs if _is_done(j))
        print(f"  cmu{m}: {done_m}/{len(jobs)} done, {len(jobs)-done_m} remaining")
        all_jobs.extend(jobs)

    remaining = [j for j in all_jobs if not _is_done(j)]
    n = len(remaining)
    per = (n + args.n_workers - 1) // args.n_workers
    start = (args.worker_id - 1) * per
    end   = min(start + per, n)
    my_jobs = remaining[start:end]
    print(
        f"[w{args.worker_id}/{args.n_workers}] {n} remaining across all machines → "
        f"my slice [{start}:{end}] = {len(my_jobs)} jobs  (max_parallel={args.max_parallel})"
    )

    completed = failed = skipped = 0
    t0 = time.time()
    active = []  # (proc, tag, log_f, out_dir)

    def _drain(wait_for_slot: bool):
        nonlocal completed, failed
        while True:
            still = []
            for proc, tag, lf, odir in active:
                ret = proc.poll()
                if ret is None:
                    still.append((proc, tag, lf, odir))
                else:
                    lf.close()
                    if ret == 0:
                        completed += 1
                        elapsed = (time.time() - t0) / 60
                        print(f"  [OK] {tag} ({completed+skipped}/{len(my_jobs)}, {elapsed:.0f}m)")
                    else:
                        failed += 1
                        print(f"  [FAIL] {tag} (exit {ret})")
            active[:] = still
            if not wait_for_slot or len(active) < args.max_parallel:
                break
            time.sleep(5)

    for job in my_jobs:
        _drain(wait_for_slot=True)

        out_dir = _get_out_dir(job)
        # Re-check at run time in case another worker just finished this job
        if os.path.exists(os.path.join(out_dir, "eval_metrics.csv")):
            skipped += 1
            print(f"  [SKIP] {job['tag']}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "stdout.log")
        eval_str = " ".join(f"'{c}'" for c in job["cmd_eval"])
        lf = open(log_path, "a")
        proc = subprocess.Popen(
            f"({eval_str})",
            stdout=lf, stderr=subprocess.STDOUT, shell=True,
        )
        active.append((proc, job["tag"], lf, out_dir))
        print(f"  [START] {job['tag']} (pid {proc.pid})")

    # Drain remaining
    for proc, tag, lf, odir in active:
        ret = proc.wait()
        lf.close()
        if ret == 0:
            completed += 1
        else:
            failed += 1
            print(f"  [FAIL] {tag} (exit {ret})")

    elapsed_h = (time.time() - t0) / 3600
    print(
        f"\n[w{args.worker_id}] Done in {elapsed_h:.1f}h: "
        f"{completed} OK, {skipped} skipped (already done), {failed} FAIL"
    )


if __name__ == "__main__":
    main()
