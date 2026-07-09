"""Aggregate Tier-2 and Tier-3 results into summary tables.

One row per training run (N = number of seeds, typically 5). Statistical unit
is the run directory — same design as the corrected Tier-1 aggregation.

Outputs (all written to --out, default results/tables/):
  tier2a_lambda_sensitivity.csv
  tier2b_occlusion_impact.csv
  tier2c_fusion_weights.csv
  tier3_state_ablation.csv
  tier3_behavioral_robustness.csv
  tier3_dense_stress.csv

Usage:
  python experiments/pde/analysis/generate_tier2_tier3_tables.py \
      --repo /path/to/repo --out results/tables/
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

# ---------------------------------------------------------------------------
KNOWN_METHODS = [
    "soft_hjb_aux", "hjb_aux", "eikonal_aux", "fusion_aux", "cbf_aux", "drppo", "rule_based",
]
KNOWN_SCENARIOS = [
    "2_dense", "3_dense", "4_dense", "1a", "1b", "1c", "1d", "2", "3", "4",
]
KNOWN_MANEUVERS = [
    "stem_right", "stem_left", "right_left", "right_stem", "left_right", "left_stem",
]
METRICS = ["SR", "CR", "mean_return", "mean_ttc", "min_ttc"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_scen_man(s):
    """Parse leading '{scenario}_{maneuver}' from string s using known sets."""
    for scen in sorted(KNOWN_SCENARIOS, key=len, reverse=True):
        if s.startswith(scen + "_"):
            rest = s[len(scen) + 1:]
            for man in sorted(KNOWN_MANEUVERS, key=len, reverse=True):
                if rest.startswith(man):
                    return scen, man, rest[len(man):]
    return None, None, s


def _extract_method(s):
    """Find the first known method in s and return (method, before, after)."""
    for m in sorted(KNOWN_METHODS, key=len, reverse=True):
        idx = s.find(m)
        if idx >= 0:
            return m, s[:idx].rstrip("_"), s[idx + len(m):].lstrip("_")
    return None, s, ""


def _training_seed(dirname):
    """Extract training seed from directory name (_s\\d+ suffix)."""
    m = re.search(r"_s(\d+)$", dirname)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Episode-level → run-level aggregation
# ---------------------------------------------------------------------------

def _run_stats(path):
    """Load eval_metrics.csv and return one-row stats dict."""
    csv_path = os.path.join(path, "eval_metrics.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if df.empty:
        return None
    n_ep = len(df)
    ts = df["terminal_state"].str.lower() if "terminal_state" in df.columns else None
    sr = float((ts == "success").sum() / n_ep) if ts is not None else float("nan")
    cr = float((ts == "collision").sum() / n_ep) if ts is not None else float("nan")
    ret = float(df["return_total"].mean()) if "return_total" in df.columns else float("nan")
    mttc = float(df["mean_ttc"].mean()) if "mean_ttc" in df.columns else float("nan")
    minttc = float(df["min_ttc"].min()) if "min_ttc" in df.columns else float("nan")
    return {"SR": sr, "CR": cr, "mean_return": ret, "mean_ttc": mttc, "min_ttc": minttc,
            "n_episodes": n_ep}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _welch_p(a, b):
    if sp_stats is None or len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        _, p = sp_stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        return float("nan")


def _holm(p_values, alpha=0.05):
    arr = np.array(p_values, dtype=float)
    n = len(arr)
    valid = ~np.isnan(arr)
    nv = int(valid.sum())
    if nv == 0:
        return [float("nan")] * n
    vi = np.where(valid)[0]
    vp = arr[valid]
    order = np.argsort(vp)
    corr = np.zeros(nv)
    for i, p in enumerate(vp[order]):
        corr[i] = min(p * (nv - i), 1.0)
    for i in range(1, nv):
        corr[i] = max(corr[i], corr[i - 1])
    unsorted = np.argsort(order)
    result = [float("nan")] * n
    for j, vi_j in enumerate(vi):
        result[vi_j] = float(corr[unsorted[j]])
    return result


def _ci95(vals):
    """Bootstrap 95% CI for the mean."""
    v = np.asarray([x for x in vals if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(2000)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _sig(p):
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# T2a: Lambda sensitivity
# ---------------------------------------------------------------------------

def aggregate_t2a(machines_root, out_dir):
    """Scan all per-machine T2a dirs and write tier2a_lambda_sensitivity.csv."""
    rows = []  # one per training run
    for m in range(1, 9):
        d = os.path.join(machines_root, f"tier_2_machine_cmu{m}", "tier2", "2a_lambda_sweep")
        if not os.path.isdir(d):
            continue
        for run in sorted(os.listdir(d)):
            if not run.startswith("T2a_"):
                continue
            seed = _training_seed(run)
            if seed is None:
                continue
            # strip prefix and trailing seed
            body = re.sub(r"^T2a_", "", run)
            body = re.sub(r"_s\d+$", "", body)
            # extract lambda
            lam_m = re.search(r"_lam([0-9.]+)$", body)
            if not lam_m:
                continue
            lam = lam_m.group(1)
            body = body[:lam_m.start()]
            # extract method and scenario/maneuver
            method, before, _ = _extract_method(body)
            if method is None:
                continue
            scen, man, _ = _parse_scen_man(before)
            stats = _run_stats(os.path.join(d, run))
            if stats is None:
                continue
            rows.append({
                "scenario": scen, "maneuver": man, "method": method,
                "lambda": lam, "seed": seed, **stats,
            })

    if not rows:
        print("T2a: no data found")
        return 0

    # Group and write
    keys = ["scenario", "maneuver", "method", "lambda"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, grp in sorted(groups.items()):
        n = len(grp)
        for metric in ["SR", "CR", "mean_return", "mean_ttc"]:
            vals = [g[metric] for g in grp]
            mean = float(np.nanmean(vals))
            lo, hi = _ci95(vals)
            out_rows.append({
                "scenario": k[0], "maneuver": k[1], "method": k[2], "lambda": k[3],
                "metric": metric, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi,
            })

    out_path = os.path.join(out_dir, "tier2a_lambda_sensitivity.csv")
    _write_csv(out_rows, out_path)
    print(f"T2a: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# T2b: Occlusion impact
# ---------------------------------------------------------------------------

def aggregate_t2b(machines_root, out_dir):
    rows = []
    for m in range(1, 9):
        d = os.path.join(machines_root, f"tier_2_machine_cmu{m}", "tier2", "2b_occlusion_sweep")
        if not os.path.isdir(d):
            continue
        for run in sorted(os.listdir(d)):
            if not run.startswith("T2b_"):
                continue
            seed = _training_seed(run)
            if seed is None:
                continue
            body = re.sub(r"^T2b_", "", run)
            body = re.sub(r"_s\d+$", "", body)
            # extract occ setting
            occ_m = re.search(r"_(occ(?:ON|OFF))$", body)
            if not occ_m:
                continue
            occ = occ_m.group(1)
            body = body[:occ_m.start()]
            method, before, _ = _extract_method(body)
            if method is None:
                continue
            scen, man, _ = _parse_scen_man(before)
            stats = _run_stats(os.path.join(d, run))
            if stats is None:
                continue
            rows.append({
                "scenario": scen, "maneuver": man, "method": method,
                "occ_setting": occ, "seed": seed, **stats,
            })

    if not rows:
        print("T2b: no data found")
        return 0

    # Group and apply Welch per (scenario, maneuver, method, metric) ON vs OFF
    cell_keys = ["scenario", "maneuver", "method"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in cell_keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    welch_ps = {}  # (k, metric) → raw p
    for k, grp in sorted(groups.items()):
        on = [g for g in grp if g["occ_setting"] == "occON"]
        off = [g for g in grp if g["occ_setting"] == "occOFF"]
        for metric in ["SR", "CR", "mean_return"]:
            on_vals = [g[metric] for g in on]
            off_vals = [g[metric] for g in off]
            p = _welch_p(on_vals, off_vals)
            welch_ps[(k, metric)] = p
            for occ, vals, grp_list in [("occON", on_vals, on), ("occOFF", off_vals, off)]:
                mean = float(np.nanmean(vals)) if vals else float("nan")
                lo, hi = _ci95(vals) if len(vals) >= 2 else (float("nan"), float("nan"))
                out_rows.append({
                    "scenario": k[0], "maneuver": k[1], "method": k[2],
                    "occ_setting": occ, "metric": metric,
                    "n": len(grp_list), "mean": mean, "ci_low": lo, "ci_high": hi,
                    "raw_p_welch_on_vs_off": p, "corrected_p_welch": float("nan"),
                    "marker": "pending",
                })

    # Holm correction per metric
    for metric in ["SR", "CR", "mean_return"]:
        metric_rows = [r for r in out_rows if r["metric"] == metric and r["occ_setting"] == "occON"]
        ps = [r["raw_p_welch_on_vs_off"] for r in metric_rows]
        corrected = _holm(ps)
        for r, cp in zip(metric_rows, corrected):
            # Update all rows with same key
            key = (r["scenario"], r["maneuver"], r["method"], r["metric"])
            for row in out_rows:
                if (row["scenario"], row["maneuver"], row["method"], row["metric"]) == key:
                    row["corrected_p_welch"] = cp
                    row["marker"] = _sig(cp)

    out_path = os.path.join(out_dir, "tier2b_occlusion_impact.csv")
    _write_csv(out_rows, out_path)
    print(f"T2b: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# T2c: Fusion weights
# ---------------------------------------------------------------------------

def aggregate_t2c(machines_root, out_dir):
    rows = []
    for m in range(1, 9):
        d = os.path.join(machines_root, f"tier_2_machine_cmu{m}", "tier2", "2c_fusion_weights")
        if not os.path.isdir(d):
            continue
        for run in sorted(os.listdir(d)):
            if not run.startswith("T2c_"):
                continue
            seed = _training_seed(run)
            if seed is None:
                continue
            body = re.sub(r"^T2c_", "", run)
            body = re.sub(r"_s\d+$", "", body)
            # extract weight pair: w{a}_{b}
            w_m = re.search(r"_(w[0-9.]+_[0-9.]+)$", body)
            if not w_m:
                continue
            weights = w_m.group(1)
            body = body[:w_m.start()]
            method, before, _ = _extract_method(body)
            if method is None:
                continue
            scen, man, _ = _parse_scen_man(before)
            stats = _run_stats(os.path.join(d, run))
            if stats is None:
                continue
            rows.append({
                "scenario": scen, "maneuver": man, "method": method,
                "weights": weights, "seed": seed, **stats,
            })

    if not rows:
        print("T2c: no data found")
        return 0

    keys = ["scenario", "maneuver", "method", "weights"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, grp in sorted(groups.items()):
        n = len(grp)
        for metric in ["SR", "CR", "mean_return"]:
            vals = [g[metric] for g in grp]
            mean = float(np.nanmean(vals))
            lo, hi = _ci95(vals)
            out_rows.append({
                "scenario": k[0], "maneuver": k[1], "method": k[2], "weights": k[3],
                "metric": metric, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi,
            })

    out_path = os.path.join(out_dir, "tier2c_fusion_weights.csv")
    _write_csv(out_rows, out_path)
    print(f"T2c: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# T3: state ablation (novis)
# ---------------------------------------------------------------------------

def aggregate_t3_state(t3_full, out_dir):
    d = os.path.join(t3_full, "tier3_state")
    if not os.path.isdir(d):
        print("T3 state: dir not found")
        return 0
    rows = []
    for run in sorted(os.listdir(d)):
        seed = _training_seed(run)
        if seed is None:
            continue
        body = re.sub(r"_s\d+$", "", run)
        # state setting is the last token after method
        method, before, after = _extract_method(body)
        if method is None:
            continue
        state_setting = after.strip("_") if after else "unknown"
        scen, man, _ = _parse_scen_man(before)
        stats = _run_stats(os.path.join(d, run))
        if stats is None:
            continue
        rows.append({
            "scenario": scen, "maneuver": man, "method": method,
            "state_setting": state_setting, "seed": seed, **stats,
        })

    if not rows:
        print("T3 state: no data found")
        return 0

    keys = ["scenario", "maneuver", "method", "state_setting"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, grp in sorted(groups.items()):
        n = len(grp)
        for metric in ["SR", "CR", "mean_return", "mean_ttc"]:
            vals = [g[metric] for g in grp]
            mean = float(np.nanmean(vals))
            lo, hi = _ci95(vals)
            out_rows.append({
                "scenario": k[0], "maneuver": k[1], "method": k[2], "state_setting": k[3],
                "metric": metric, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi,
            })

    out_path = os.path.join(out_dir, "tier3_state_ablation.csv")
    _write_csv(out_rows, out_path)
    print(f"T3 state: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# T3: behavioral robustness
# ---------------------------------------------------------------------------

def aggregate_t3_behav(t3_full, out_dir):
    d = os.path.join(t3_full, "tier3_behav")
    if not os.path.isdir(d):
        print("T3 behav: dir not found")
        return 0
    rows = []
    for run in sorted(os.listdir(d)):
        seed = _training_seed(run)
        if seed is None:
            continue
        body = re.sub(r"_s\d+$", "", run)
        method, before, after = _extract_method(body)
        if method is None:
            continue
        behavior = after.strip("_") if after else "unknown"
        scen, man, _ = _parse_scen_man(before)
        stats = _run_stats(os.path.join(d, run))
        if stats is None:
            continue
        rows.append({
            "scenario": scen, "maneuver": man, "method": method,
            "behavior": behavior, "seed": seed, **stats,
        })

    if not rows:
        print("T3 behav: no data found")
        return 0

    keys = ["scenario", "maneuver", "method", "behavior"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, grp in sorted(groups.items()):
        n = len(grp)
        for metric in ["SR", "CR", "mean_return", "mean_ttc"]:
            vals = [g[metric] for g in grp]
            mean = float(np.nanmean(vals))
            lo, hi = _ci95(vals)
            out_rows.append({
                "scenario": k[0], "maneuver": k[1], "method": k[2], "behavior": k[3],
                "metric": metric, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi,
            })

    out_path = os.path.join(out_dir, "tier3_behavioral_robustness.csv")
    _write_csv(out_rows, out_path)
    print(f"T3 behav: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# T3: dense traffic stress
# ---------------------------------------------------------------------------

def aggregate_t3_dense(t3_full, out_dir):
    d = os.path.join(t3_full, "tier3_dense")
    if not os.path.isdir(d):
        print("T3 dense: dir not found")
        return 0
    rows = []
    for run in sorted(os.listdir(d)):
        seed = _training_seed(run)
        if seed is None:
            continue
        body = re.sub(r"_s\d+$", "", run)
        method, before, after = _extract_method(body)
        if method is None:
            continue
        scen, man, _ = _parse_scen_man(before)
        stats = _run_stats(os.path.join(d, run))
        if stats is None:
            continue
        rows.append({
            "scenario": scen, "maneuver": man, "method": method,
            "seed": seed, **stats,
        })

    if not rows:
        print("T3 dense: no data found")
        return 0

    keys = ["scenario", "maneuver", "method"]
    groups = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        groups.setdefault(k, []).append(r)

    out_rows = []
    for k, grp in sorted(groups.items()):
        n = len(grp)
        for metric in ["SR", "CR", "mean_return", "mean_ttc"]:
            vals = [g[metric] for g in grp]
            mean = float(np.nanmean(vals))
            lo, hi = _ci95(vals)
            out_rows.append({
                "scenario": k[0], "maneuver": k[1], "method": k[2],
                "metric": metric, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi,
            })

    out_path = os.path.join(out_dir, "tier3_dense_stress.csv")
    _write_csv(out_rows, out_path)
    print(f"T3 dense: {len(rows)} runs → {len(out_rows)} rows → {out_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="results/tables")
    args = parser.parse_args()

    if pd is None:
        print("pandas required")
        return

    os.makedirs(args.out, exist_ok=True)
    results_root = os.path.join(args.repo, "results")
    t3_full = os.path.join(results_root, "tier_3_full")

    print("=== Tier 2 ===")
    n2a = aggregate_t2a(results_root, args.out)
    n2b = aggregate_t2b(results_root, args.out)
    n2c = aggregate_t2c(results_root, args.out)

    print("\n=== Tier 3 ===")
    n3s = aggregate_t3_state(t3_full, args.out)
    n3b = aggregate_t3_behav(t3_full, args.out)
    n3d = aggregate_t3_dense(t3_full, args.out)

    print(f"\nDone. T2a:{n2a} T2b:{n2b} T2c:{n2c} | T3state:{n3s} T3behav:{n3b} T3dense:{n3d}")


if __name__ == "__main__":
    main()
