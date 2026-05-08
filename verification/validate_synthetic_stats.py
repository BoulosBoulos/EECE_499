"""Phase 29.5 validator: confirm synthetic 60-run dataset produces Decision-E-aligned stats."""
import sys, json, csv

CSV_PATH = "/tmp/synthetic_60run_analysis/statistical_tests/tier1_family_A_final_collision_rate.csv"

# Decision E predictions
PREDICTED = {
    "hjb_aux":      {"d": -2.4, "sig": True},
    "soft_hjb_aux": {"d": -2.6, "sig": True},
    "eikonal_aux":  {"d": -0.4, "sig": False},
    "cbf_aux":      {"d": -3.9, "sig": True},
    "fusion_aux":   {"d": -4.2, "sig": True},
}
TOL_FRAC = 0.30  # |actual - predicted| / |predicted| <= 30%

import os
result = {"phase": "29.5", "pass": False, "csv_path": CSV_PATH, "checks": []}

if not os.path.isfile(CSV_PATH):
    result["error"] = f"CSV not found at {CSV_PATH}"
    json.dump(result, open("verification/phase29_synthetic_stats.json", "w"), indent=2)
    sys.exit(1)

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

# 5 rows expected (5 PDE methods compared to DRPPO)
expected_n = 5
result["n_rows"] = len(rows)
result["checks"].append({"check": "n_rows == 5", "pass": len(rows) == expected_n,
                          "actual": len(rows)})

# All required columns must be populated (no NaN/empty) and insufficient_n must be False.
required_cols = ["t_stat", "p_raw", "p_holm", "cohens_d"]
fully_populated = True
for r in rows:
    for c in required_cols:
        v = r.get(c, "")
        if v == "" or v.lower() == "nan":
            fully_populated = False
result["checks"].append({"check": "all required columns populated (no NaN)", "pass": fully_populated})

all_n_sufficient = all(r.get("insufficient_n", "False").lower() == "false" for r in rows)
result["checks"].append({"check": "insufficient_n is False for all rows (n=10)", "pass": all_n_sufficient})

# Per-method significance and effect size match
mismatches = []
for r in rows:
    m = r["method_test"]
    if m not in PREDICTED:
        continue
    pred = PREDICTED[m]
    actual_d = float(r["cohens_d"])
    actual_sig = (r["significant_holm"].lower() == "true")
    sig_ok = (actual_sig == pred["sig"])
    pred_d = pred["d"]
    d_dev = abs(actual_d - pred_d) / abs(pred_d) if pred_d != 0 else 0.0
    d_ok = d_dev <= TOL_FRAC
    if not (sig_ok and d_ok):
        mismatches.append({
            "method": m, "predicted_d": pred_d, "actual_d": actual_d,
            "d_deviation_frac": d_dev,
            "predicted_sig": pred["sig"], "actual_sig": actual_sig,
            "sig_match": sig_ok, "d_within_30pct": d_ok,
        })
    result["checks"].append({
        "check": f"{m}: sig {pred['sig']}, d~{pred_d}",
        "pass": sig_ok and d_ok,
        "actual_d": actual_d, "actual_sig": actual_sig, "d_dev_frac": d_dev,
    })

result["mismatches"] = mismatches
result["pass"] = (
    len(rows) == expected_n
    and fully_populated
    and all_n_sufficient
    and not mismatches
)

with open("verification/phase29_synthetic_stats.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
sys.exit(0 if result["pass"] else 1)
