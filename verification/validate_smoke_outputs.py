"""Phase 29.6 — Validate the 6-job orchestrator smoke outputs."""
from __future__ import annotations
import os, sys, json
from pathlib import Path

SMOKE_ROOT = Path("/tmp/phase2_smoke_v2")
TIER1 = SMOKE_ROOT / "tier1"
METHODS = ["drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]
LOG = SMOKE_ROOT / "orchestrator.log"

result = {"phase": "29.6", "pass": False, "per_method": [], "smoke_root": str(SMOKE_ROOT)}

if not LOG.is_file():
    result["error"] = "orchestrator.log missing"
    Path("verification/phase29_e2e_smoke.json").write_text(json.dumps(result, indent=2))
    sys.exit(1)

log_text = LOG.read_text()
result["log_size_bytes"] = len(log_text)
result["log_tail"] = log_text.splitlines()[-3:] if log_text else []
n_ok = log_text.count("[OK]")
n_fail = log_text.count("[FAIL]")
done_line = ""
for line in log_text.splitlines():
    if line.startswith("Done:"):
        done_line = line
result["n_ok_tag_in_log"] = n_ok
result["n_fail_tag_in_log"] = n_fail
result["done_line"] = done_line

# Per-method file checks
for method in METHODS:
    rd = TIER1 / f"1a_stem_right_{method}_nointent_s42"
    metrics_csv = rd / "metrics.csv"
    meta_json = rd / "meta.json"
    eval_csv = rd / "eval_metrics.csv"
    ckpt = rd / f"model_{method}_1a_stem_right.pt"
    entry = {
        "method": method,
        "run_dir": str(rd),
        "exists_run_dir": rd.is_dir(),
        "exists_metrics_csv": metrics_csv.is_file(),
        "exists_meta_json": meta_json.is_file(),
        "exists_eval_csv": eval_csv.is_file(),
        "exists_checkpoint": ckpt.is_file(),
    }
    if metrics_csv.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            entry["metrics_n_rows"] = int(len(df))
            entry["metrics_n_cols"] = int(len(df.columns))
            entry["metrics_has_nan"] = bool(df.isna().any().any())
        except Exception as e:
            entry["metrics_csv_error"] = str(e)
    if meta_json.is_file():
        try:
            m = json.loads(meta_json.read_text())
            entry["meta_has_result_summary"] = isinstance(m.get("result_summary"), dict) and bool(m["result_summary"])
            entry["meta_method"] = m.get("method")
        except Exception as e:
            entry["meta_json_error"] = str(e)
    result["per_method"].append(entry)

# Pass rules
n_dirs_with_required = sum(
    1 for m in result["per_method"]
    if m.get("exists_metrics_csv") and m.get("exists_meta_json")
)
n_completed = sum(1 for m in result["per_method"] if m.get("exists_checkpoint"))
result["n_jobs_completed"] = int(n_completed)
result["n_jobs_failed"] = int(len(METHODS) - n_completed)
result["pass"] = bool(
    len(result["per_method"]) == 6
    and n_dirs_with_required == 6
    and n_completed == 6
    and n_fail == 0
    and "Done:" in log_text
)
Path("verification/phase29_e2e_smoke.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k not in ("log_tail",)}, indent=2)[:2000])
sys.exit(0 if result["pass"] else 1)
