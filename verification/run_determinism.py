"""Phase 29.8/29.9 — Determinism verification.

Two consecutive 5,000-step train_hjb_aux runs with same seed.

GPU mode (relaxed, 1e-5 tolerance):
  - All integer columns exactly equal
  - All float columns within 1e-5 absolute tolerance

CPU strict mode (byte-identical with --device cpu, torch.set_num_threads(1),
CUBLAS_WORKSPACE_CONFIG=:4096:8):
  - SHA256 hash match between the two metrics.csv files
"""
from __future__ import annotations
import os, sys, json, hashlib, subprocess, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INT_COLS = {"iteration", "total_steps", "n_episodes",
            "n_collisions", "n_successes", "n_timeouts", "n_aborts"}


def _run_one(out_dir: Path, env_overrides: dict, force_cpu: bool) -> tuple[int, float]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "experiments/pde/train_hjb_aux.py",
        "--scenario", "1a", "--ego_maneuver", "stem_right",
        "--seed", "42", "--total_steps", "5000",
        "--out_dir", str(out_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["SUMO_HOME"] = "/usr/share/sumo"
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.update(env_overrides)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    if p.returncode != 0:
        print(f"[determinism] training failed in {out_dir}: exit={p.returncode}")
        print(p.stderr[-2000:])
    return p.returncode, wall


def _find_metrics_csv(out_dir: Path) -> Path | None:
    cands = list(out_dir.rglob("metrics.csv"))
    if cands:
        return cands[0]
    cands = list(out_dir.rglob("train_hjb_aux_*.csv"))
    return cands[0] if cands else None


def _compare_gpu(csv1: Path, csv2: Path, tol: float = 1e-5) -> dict:
    import pandas as pd
    a = pd.read_csv(csv1)
    b = pd.read_csv(csv2)
    summary = {"shape_a": a.shape, "shape_b": b.shape, "tolerances": {}, "max_abs_delta": 0.0}
    if a.shape != b.shape:
        summary["pass"] = False
        summary["reason"] = "shape mismatch"
        return summary
    if list(a.columns) != list(b.columns):
        summary["pass"] = False
        summary["reason"] = "column mismatch"
        return summary
    bad_int_cols = []
    max_float_delta = 0.0
    worst_col = None
    for col in a.columns:
        if col in INT_COLS:
            if not (a[col].astype(int) == b[col].astype(int)).all():
                bad_int_cols.append(col)
        else:
            try:
                af = a[col].astype(float).fillna(0.0).to_numpy()
                bf = b[col].astype(float).fillna(0.0).to_numpy()
                d = float((abs(af - bf)).max() if len(af) else 0.0)
                summary["tolerances"][col] = d
                if d > max_float_delta:
                    max_float_delta = d
                    worst_col = col
            except Exception:
                summary["tolerances"][col] = "non-numeric"
    summary["max_abs_delta"] = max_float_delta
    summary["worst_col"] = worst_col
    summary["bad_int_cols"] = bad_int_cols
    summary["pass"] = (not bad_int_cols) and (max_float_delta <= tol)
    return summary


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def determinism_gpu() -> dict:
    print("\n=== Phase 29.8 — GPU determinism (1e-5 tolerance) ===")
    out_a = Path("/tmp/phase2_det_gpu_run1")
    out_b = Path("/tmp/phase2_det_gpu_run2")
    rc_a, w_a = _run_one(out_a, {}, force_cpu=False)
    rc_b, w_b = _run_one(out_b, {}, force_cpu=False)
    res = {"phase": "29.8", "wall_a": round(w_a, 1), "wall_b": round(w_b, 1),
           "exit_a": rc_a, "exit_b": rc_b}
    if rc_a != 0 or rc_b != 0:
        res["pass"] = False
        res["reason"] = "training failed"
        return res
    csv_a = _find_metrics_csv(out_a)
    csv_b = _find_metrics_csv(out_b)
    if csv_a is None or csv_b is None:
        res["pass"] = False
        res["reason"] = f"metrics.csv not found (a={csv_a}, b={csv_b})"
        return res
    res["csv_a"] = str(csv_a); res["csv_b"] = str(csv_b)
    res.update(_compare_gpu(csv_a, csv_b, tol=1e-5))
    return res


def determinism_cpu_strict() -> dict:
    print("\n=== Phase 29.9 — CPU strict determinism (byte-identical) ===")
    out_a = Path("/tmp/phase2_det_cpu_run1")
    out_b = Path("/tmp/phase2_det_cpu_run2")
    env_overrides = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }
    rc_a, w_a = _run_one(out_a, env_overrides, force_cpu=True)
    rc_b, w_b = _run_one(out_b, env_overrides, force_cpu=True)
    res = {"phase": "29.9", "wall_a": round(w_a, 1), "wall_b": round(w_b, 1),
           "exit_a": rc_a, "exit_b": rc_b}
    if rc_a != 0 or rc_b != 0:
        res["pass"] = False
        res["reason"] = "training failed"
        return res
    csv_a = _find_metrics_csv(out_a)
    csv_b = _find_metrics_csv(out_b)
    if csv_a is None or csv_b is None:
        res["pass"] = False
        res["reason"] = f"metrics.csv not found (a={csv_a}, b={csv_b})"
        return res
    h_a = _sha256(csv_a); h_b = _sha256(csv_b)
    res.update({
        "csv_a": str(csv_a), "csv_b": str(csv_b),
        "sha256_a": h_a, "sha256_b": h_b,
        "byte_identical": h_a == h_b,
    })
    res["pass"] = (h_a == h_b)
    if not res["pass"]:
        # Add a softer comparison so we can see how close it is
        res.update({"_softer_compare": _compare_gpu(csv_a, csv_b, tol=1e-5)})
    return res


def main(mode: str) -> int:
    if mode == "gpu":
        r = determinism_gpu()
        out = Path("verification/phase29_determinism_gpu.json")
    elif mode == "cpu":
        r = determinism_cpu_strict()
        out = Path("verification/phase29_determinism_cpu.json")
    else:
        print(f"Unknown mode {mode!r}")
        return 1
    out.write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "tolerances"}, indent=2))
    return 0 if r.get("pass") else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_determinism.py {gpu|cpu}")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
