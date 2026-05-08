"""Phase 29.8 / 29.9 (relaxed) — determinism check at 3*sigma tolerance.

Per Prompt11:
  - Bit-reproducibility is unachievable with SUMO-coupled training.
  - Tolerance is set at 3*sigma where sigma is the measured across-run
    sample std for identical config (verification/phase29_determinism_variance.json).
  - 29.8 (GPU) and 29.9 (CPU) become the same test with the same tolerance.

Acceptance:
  - Integer columns exact match (still strict — if they differ we have a worse bug).
  - final_collision_rate within 3 * sigma absolute.
  - final_success_rate within 3 * sigma absolute.
  - final_mean_reward within 3 * sigma absolute.
"""
from __future__ import annotations
import os, sys, json, subprocess, shutil, time, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INT_COLS = {"iteration", "total_steps", "n_episodes",
            "n_collisions", "n_successes", "n_timeouts", "n_aborts"}


def _load_tolerance() -> dict:
    p = Path("verification/phase29_determinism_variance.json")
    data = json.loads(p.read_text())
    return data["tolerance_3sigma"]


def _train(out_dir: Path, force_cpu: bool) -> tuple[int, float]:
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
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTHONHASHSEED"] = "0"
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    return p.returncode, round(time.time() - t0, 1)


def _final_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    n_eps = float(last["n_episodes"]) if last["n_episodes"] else 1.0
    return {
        "final_collision_rate": float(last["n_collisions"]) / n_eps,
        "final_success_rate":   float(last["n_successes"])  / n_eps,
        "final_mean_reward":    float(last["mean_reward"]),
        "n_collisions_sum":     int(df["n_collisions"].sum()),
        "n_successes_sum":      int(df["n_successes"].sum()),
        "shape":                list(df.shape),
    }


def _ints_match(csv_a: Path, csv_b: Path) -> tuple[bool, list[str]]:
    a = pd.read_csv(csv_a); b = pd.read_csv(csv_b)
    if a.shape != b.shape:
        return False, [f"shape mismatch {a.shape} vs {b.shape}"]
    bad = []
    for col in a.columns:
        if col in INT_COLS:
            if not (a[col].astype(int) == b[col].astype(int)).all():
                bad.append(col)
    return (not bad), bad


def determinism_relaxed(mode: str) -> dict:
    print(f"\n=== Phase 29.{8 if mode == 'gpu' else 9} — relaxed determinism ({mode.upper()}) ===")
    tol = _load_tolerance()
    out_a = Path(f"/tmp/phase2_det_relaxed_{mode}_run1")
    out_b = Path(f"/tmp/phase2_det_relaxed_{mode}_run2")
    rc_a, w_a = _train(out_a, force_cpu=(mode == "cpu"))
    rc_b, w_b = _train(out_b, force_cpu=(mode == "cpu"))
    res = {
        "phase": "29.8" if mode == "gpu" else "29.9",
        "mode": mode,
        "wall_a": w_a, "wall_b": w_b, "exit_a": rc_a, "exit_b": rc_b,
        "tolerance_3sigma": tol,
    }
    if rc_a != 0 or rc_b != 0:
        res["pass"] = False; res["reason"] = "training failed"; return res
    csv_a = out_a / "metrics.csv"; csv_b = out_b / "metrics.csv"
    m_a = _final_metrics(csv_a); m_b = _final_metrics(csv_b)
    res["metrics_a"] = m_a; res["metrics_b"] = m_b

    ints_ok, bad_int_cols = _ints_match(csv_a, csv_b)
    res["int_cols_match_exactly"] = ints_ok
    res["bad_int_cols"] = bad_int_cols

    deltas = {}
    within = {}
    for k in ("final_collision_rate", "final_success_rate", "final_mean_reward"):
        d = abs(m_a[k] - m_b[k])
        deltas[k] = d
        within[k] = d <= tol[k]
    res["abs_deltas"] = deltas
    res["within_3sigma"] = within
    res["pass"] = ints_ok and all(within.values())
    return res


def main(mode: str) -> int:
    if mode == "gpu":
        r = determinism_relaxed("gpu")
        out = Path("verification/phase29_determinism_gpu.json")
    elif mode == "cpu":
        r = determinism_relaxed("cpu")
        out = Path("verification/phase29_determinism_cpu.json")
    else:
        print(f"Unknown mode {mode!r}"); return 1
    out.write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps(r, indent=2))
    return 0 if r.get("pass") else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_determinism_relaxed.py {gpu|cpu}"); sys.exit(1)
    sys.exit(main(sys.argv[1]))
