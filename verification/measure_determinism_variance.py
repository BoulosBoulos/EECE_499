"""Phase 29 followup — measure across-run variance with identical config.

Runs 5 consecutive train_hjb_aux invocations with the same seed=42 / scenario=1a /
maneuver=stem_right / total_steps=5000, then computes the sample std across the 5
runs of the metrics named in Prompt11. Used to set 3*std tolerances for the
relaxed determinism tests 29.8 / 29.9.
"""
from __future__ import annotations
import os, sys, json, subprocess, shutil, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = 5
RUN_PREFIX = "/tmp/det_variance_run"


def _run_one(idx: int) -> dict:
    out_dir = Path(f"{RUN_PREFIX}{idx}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "experiments/pde/train_hjb_aux.py",
        "--total_steps", "5000",
        "--scenario", "1a", "--ego_maneuver", "stem_right",
        "--seed", "42",
        "--output_dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["SUMO_HOME"] = "/usr/share/sumo"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    return {"idx": idx, "exit": p.returncode, "wall_s": round(wall, 1),
            "out_dir": str(out_dir),
            "stderr_tail": "\n".join(p.stderr.splitlines()[-5:])}


def _extract_metrics(run_idx: int) -> dict:
    csv_path = Path(f"{RUN_PREFIX}{run_idx}") / "metrics.csv"
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    n_eps = float(last["n_episodes"]) if last["n_episodes"] else 1.0
    final_collision_rate = float(last["n_collisions"]) / n_eps
    final_success_rate   = float(last["n_successes"])  / n_eps
    return {
        "run_idx": run_idx,
        "n_iterations": int(len(df)),
        "n_collisions_sum": int(df["n_collisions"].sum()),
        "n_successes_sum":  int(df["n_successes"].sum()),
        "n_episodes_sum":   int(df["n_episodes"].sum()),
        "final_collision_rate": final_collision_rate,
        "final_success_rate":   final_success_rate,
        "final_mean_reward":    float(last["mean_reward"]),
    }


def main() -> int:
    print(f"[det-variance] Running {RUNS} consecutive trainings (seed=42, 5000 steps each)…")
    runs = []
    for i in range(1, RUNS + 1):
        r = _run_one(i)
        if r["exit"] != 0:
            print(f"  run {i}: FAILED (exit {r['exit']}); stderr: {r['stderr_tail']}")
            return 1
        print(f"  run {i}: OK ({r['wall_s']}s)")
        runs.append(r)

    # Extract metrics
    metrics = [_extract_metrics(i) for i in range(1, RUNS + 1)]
    df = pd.DataFrame(metrics)

    # Sample std (ddof=1) and mean across the 5 runs
    cols = ["final_collision_rate", "final_success_rate", "final_mean_reward",
            "n_collisions_sum", "n_successes_sum"]
    summary = {}
    for col in cols:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std":  float(df[col].std(ddof=1)),
            "min":  float(df[col].min()),
            "max":  float(df[col].max()),
            "values": df[col].tolist(),
        }

    out = {
        "phase": "29-determinism-variance",
        "n_runs": RUNS,
        "per_run": metrics,
        "summary": summary,
        "tolerance_3sigma": {col: 3.0 * summary[col]["std"] for col in cols},
    }
    Path("verification/phase29_determinism_variance.json").write_text(json.dumps(out, indent=2))

    print()
    print("=" * 70)
    print("Across-5-run statistics:")
    print(f"  {'metric':28s}  {'mean':>14s} {'std':>14s} {'3*std':>14s}")
    for col in cols:
        s = summary[col]
        print(f"  {col:28s}  {s['mean']:>14.6f} {s['std']:>14.6f} {3.0*s['std']:>14.6f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
