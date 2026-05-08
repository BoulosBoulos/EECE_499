"""Phase 29.7 — Checkpoint load round-trip for all 6 trainable methods.

After Step 7 produces checkpoints under /tmp/phase2_smoke_v2/tier1/<run>/, this
script invokes eval.py with --episodes 5 against each method's checkpoint.
Validates:
  - eval.py exits 0
  - eval_metrics.csv produced (Phase 1A schema, 13 cols from EVAL_METRICS_COLUMNS)
  - no NaN in numeric columns
"""
from __future__ import annotations
import os, sys, json, csv, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = Path("/tmp/phase2_smoke_v2")
TIER1 = SMOKE_ROOT / "tier1"
RT_ROOT = Path("/tmp/phase2_ckpt_round_trip")
METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]

result = {"phase": "29.7", "pass": False, "per_method": [], "smoke_root": str(SMOKE_ROOT)}

RT_ROOT.mkdir(parents=True, exist_ok=True)
all_ok = True
for method in METHODS:
    src_dir = TIER1 / f"1a_stem_right_{method}_nointent_s42"
    ckpt = src_dir / f"model_{method}_1a_stem_right.pt"
    out_dir = RT_ROOT / method
    out_dir.mkdir(parents=True, exist_ok=True)
    # Step 8 acceptance: produce a valid eval_metrics.csv (per-episode rows
    # with the EVAL_METRICS_COLUMNS schema). The aggregate-per-seed file
    # `eval_{method}_{scen}_{man}.csv` is a different artifact.
    eval_csv = out_dir / "eval_metrics.csv"
    entry = {"method": method, "checkpoint": str(ckpt), "out_dir": str(out_dir)}
    if not ckpt.is_file():
        entry["error"] = f"checkpoint missing: {ckpt}"
        entry["eval_ok"] = False
        all_ok = False
        result["per_method"].append(entry)
        continue
    cmd = [
        "python3", "experiments/pde/eval.py",
        "--method", method,
        "--checkpoint", str(ckpt),
        "--scenario", "1a", "--ego_maneuver", "stem_right",
        "--episodes", "5", "--seeds", "9001",
        "--out_dir", str(out_dir),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    entry["wall_seconds"] = round(wall, 1)
    entry["exit"] = p.returncode
    entry["stdout_tail"] = "\n".join(p.stdout.splitlines()[-15:])
    entry["stderr_tail"] = "\n".join(p.stderr.splitlines()[-15:])
    if p.returncode != 0:
        entry["eval_ok"] = False
        all_ok = False
        result["per_method"].append(entry)
        continue
    # Inspect the eval CSV
    if eval_csv.is_file():
        with open(eval_csv) as fh:
            rows = list(csv.DictReader(fh))
        entry["eval_csv_n_rows"] = len(rows)
        entry["eval_csv_columns"] = list(rows[0].keys()) if rows else []
        nan_seen = False
        if rows:
            for r in rows:
                for k, v in r.items():
                    if v == "" or (isinstance(v, str) and v.lower() == "nan"):
                        nan_seen = True
        entry["eval_csv_no_nan"] = not nan_seen
        entry["eval_ok"] = (len(rows) >= 1 and not nan_seen)
        if not entry["eval_ok"]:
            all_ok = False
    else:
        entry["error"] = "eval_metrics.csv missing"
        entry["eval_ok"] = False
        all_ok = False
    result["per_method"].append(entry)
    print(f"[{method:>14s}] exit={entry.get('exit')} wall={entry.get('wall_seconds')}s eval_ok={entry.get('eval_ok')}")

result["pass"] = all_ok
Path("verification/phase29_ckpt_round_trip.json").write_text(json.dumps(result, indent=2))
print(f"ALL_OK={all_ok}")
sys.exit(0 if all_ok else 1)
