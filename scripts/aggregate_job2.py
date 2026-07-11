"""Aggregate Job 2 Actor-KL ablation results.

Produces results/tables/tier2d_actor_kl_ablation.csv with one row per training run
(per-run file) plus aggregated rows:
    scenario, maneuver, method, kl_setting, metric, n, mean, ci_low, ci_high

KL-on comparison arm pulled from existing Tier-2a lam0.2 and Tier-2b occON runs.
KL-off (nokl) arm pulled from the new Job 2 output directory.

Welch t-test + Holm correction across each (scenario, maneuver, method) group.

Usage:
    python scripts/aggregate_job2.py \
        --nokl_root results/tier_2_machine_job2/tier2/2d_actor_kl_ablation \
        --kl_on_roots results/tier_2_machine_cmu8/tier2/2a_lambda_sweep \
                      results/tier_2_machine_cmu3/tier2/2a_lambda_sweep \
                      results/tier_2_machine_cmu8/tier2/2b_occlusion_sweep \
                      results/tier_2_machine_cmu3/tier2/2b_occlusion_sweep \
        --out results/tables/tier2d_actor_kl_ablation.csv
"""

from __future__ import annotations
import argparse, csv, math, os, re, sys
import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── target cells ──────────────────────────────────────────────────────────────
TARGET_CELLS = [
    ("2_dense", "right_left"),
    ("2_dense", "stem_right"),
    ("1b",      "stem_right"),
]
METHODS = ["soft_hjb_aux", "eikonal_aux"]
METRICS = ["SR", "CR", "mean_return"]


def _read_eval_metrics(run_dir: str) -> dict | None:
    """Read eval_metrics.csv, return per-run aggregate stats dict or None."""
    path = os.path.join(run_dir, "eval_metrics.csv")
    if not os.path.isfile(path):
        return None
    episodes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(row)
    if not episodes:
        return None
    terminals  = [r.get("terminal_state", r.get("terminal", "")) for r in episodes]
    returns    = [float(r["return_total"]) for r in episodes if r.get("return_total", "")]
    n          = len(episodes)
    sr         = sum(1 for t in terminals if t == "success") / n if n else 0.0
    cr         = sum(1 for t in terminals if t == "collision") / n if n else 0.0
    mean_ret   = float(np.mean(returns)) if returns else float("nan")
    return {"SR": sr, "CR": cr, "mean_return": mean_ret, "n_episodes": n}


def _ci95(values: list[float]) -> tuple[float, float]:
    """Bootstrap 95 % CI (2000 resamples)."""
    if len(values) < 2:
        m = float(np.nanmean(values)) if values else float("nan")
        return m, m
    arr  = np.array(values, dtype=float)
    boot = [np.nanmean(np.random.choice(arr, len(arr), replace=True)) for _ in range(2000)]
    lo   = float(np.percentile(boot, 2.5))
    hi   = float(np.percentile(boot, 97.5))
    return lo, hi


def _welch_p(a: list[float], b: list[float]) -> float:
    if not HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(p)


def _holm(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


# ── load nokl runs ─────────────────────────────────────────────────────────────

def _parse_nokl_tag(dirname: str):
    """Parse T2d_{scen}_{man}_{method}_nokl_s{seed} → (scen, man, method, seed) or None."""
    m = re.match(r"T2d_(.+)_(soft_hjb_aux|eikonal_aux)_nokl_s(\d+)$", dirname)
    if not m:
        return None
    cell_part, method, seed = m.group(1), m.group(2), int(m.group(3))
    # Split cell_part into (scenario, maneuver)
    for scen, man in TARGET_CELLS:
        tag_cell = f"{scen}_{man}"
        if cell_part == tag_cell:
            return scen, man, method, seed
    return None


def load_nokl(nokl_root: str) -> list[dict]:
    rows = []
    if not os.path.isdir(nokl_root):
        print(f"[agg_job2] WARNING: nokl_root not found: {nokl_root}")
        return rows
    for d in sorted(os.listdir(nokl_root)):
        parsed = _parse_nokl_tag(d)
        if parsed is None:
            continue
        scen, man, method, seed = parsed
        stats = _read_eval_metrics(os.path.join(nokl_root, d))
        if stats is None:
            print(f"[agg_job2] missing eval_metrics: {d}")
            continue
        rows.append({"scenario": scen, "maneuver": man, "method": method,
                     "seed": seed, "kl_setting": "off", "run_dir": d, **stats})
    print(f"[agg_job2] loaded {len(rows)} nokl runs")
    return rows


# ── load kl_on runs (from existing 2a lam0.2 and 2b occON) ───────────────────

def _parse_kl_on_tag(dirname: str):
    """
    Matches:
      T2a_{scen}_{man}_{method}_lam0.2_s{seed}  (2a)
      T2b_{scen}_{man}_{method}_occON_s{seed}    (2b)
    Returns (scen, man, method, seed) or None.
    """
    # 2a pattern with lam0.2
    m = re.match(r"T2a_(.+)_(soft_hjb_aux|eikonal_aux)_lam0\.2_s(\d+)$", dirname)
    if m:
        cell_part, method, seed = m.group(1), m.group(2), int(m.group(3))
        for scen, man in TARGET_CELLS:
            if cell_part == f"{scen}_{man}":
                return scen, man, method, seed
    # 2b occON pattern
    m = re.match(r"T2b_(.+)_(soft_hjb_aux|eikonal_aux)_occON_s(\d+)$", dirname)
    if m:
        cell_part, method, seed = m.group(1), m.group(2), int(m.group(3))
        for scen, man in TARGET_CELLS:
            if cell_part == f"{scen}_{man}":
                return scen, man, method, seed
    return None


def load_kl_on(roots: list[str]) -> list[dict]:
    seen = set()
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            parsed = _parse_kl_on_tag(d)
            if parsed is None:
                continue
            scen, man, method, seed = parsed
            key = (scen, man, method, seed)
            if key in seen:
                continue  # deduplicate across roots
            seen.add(key)
            stats = _read_eval_metrics(os.path.join(root, d))
            if stats is None:
                continue
            rows.append({"scenario": scen, "maneuver": man, "method": method,
                         "seed": seed, "kl_setting": "on", "run_dir": d, **stats})
    print(f"[agg_job2] loaded {len(rows)} kl_on runs")
    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(0)

    ap = argparse.ArgumentParser()
    ap.add_argument("--nokl_root",   required=True)
    ap.add_argument("--kl_on_roots", nargs="+", required=True)
    ap.add_argument("--out",         required=True)
    args = ap.parse_args()

    nokl_rows  = load_nokl(args.nokl_root)
    kl_on_rows = load_kl_on(args.kl_on_roots)
    all_rows   = nokl_rows + kl_on_rows

    if not all_rows:
        print("[agg_job2] ERROR: no data loaded.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── per-run CSV ──────────────────────────────────────────────────────────
    per_run_path = args.out.replace(".csv", "_per_run.csv")
    per_run_cols = ["scenario", "maneuver", "method", "kl_setting", "seed",
                    "SR", "CR", "mean_return", "n_episodes", "run_dir"]
    with open(per_run_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_run_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(all_rows, key=lambda r: (r["scenario"], r["maneuver"],
                                                    r["method"], r["kl_setting"], r["seed"])))
    print(f"[agg_job2] per-run CSV → {per_run_path}  ({len(all_rows)} rows)")

    # ── aggregated CSV with Welch + Holm ─────────────────────────────────────
    groups = {}
    for r in all_rows:
        key = (r["scenario"], r["maneuver"], r["method"], r["kl_setting"])
        groups.setdefault(key, []).append(r)

    # Collect Welch p-values for Holm correction
    # One test per (scenario, maneuver, method, metric)
    test_keys    = []
    raw_p_values = []

    for scen, man in TARGET_CELLS:
        for method in METHODS:
            for metric in METRICS:
                on_vals  = [r[metric] for r in groups.get((scen, man, method, "on"),  [])
                            if not math.isnan(r[metric])]
                off_vals = [r[metric] for r in groups.get((scen, man, method, "off"), [])
                            if not math.isnan(r[metric])]
                p = _welch_p(on_vals, off_vals)
                test_keys.append((scen, man, method, metric))
                raw_p_values.append(p)

    adj_p = _holm(raw_p_values)
    welch_table = {k: (raw_p_values[i], adj_p[i]) for i, k in enumerate(test_keys)}

    agg_rows = []
    for scen, man in TARGET_CELLS:
        for method in METHODS:
            for kl_setting in ("on", "off"):
                grp = groups.get((scen, man, method, kl_setting), [])
                n   = len(grp)
                for metric in METRICS:
                    vals  = [r[metric] for r in grp if not math.isnan(r[metric])]
                    mean  = float(np.nanmean(vals)) if vals else float("nan")
                    lo, hi = _ci95(vals) if len(vals) >= 2 else (float("nan"), float("nan"))
                    p_raw, p_adj = welch_table.get((scen, man, method, metric), (float("nan"), float("nan")))
                    agg_rows.append({
                        "scenario":    scen,
                        "maneuver":    man,
                        "method":      method,
                        "kl_setting":  kl_setting,
                        "metric":      metric,
                        "n":           n,
                        "mean":        round(mean,  4),
                        "ci_low":      round(lo,    4),
                        "ci_high":     round(hi,    4),
                        "welch_p":     round(p_raw, 4) if not math.isnan(p_raw) else "nan",
                        "welch_p_adj": round(p_adj, 4) if not math.isnan(p_adj) else "nan",
                        "sig_05":      (not math.isnan(p_adj)) and p_adj < 0.05,
                    })

    agg_cols = ["scenario", "maneuver", "method", "kl_setting", "metric",
                "n", "mean", "ci_low", "ci_high", "welch_p", "welch_p_adj", "sig_05"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_cols)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"[agg_job2] aggregated CSV → {args.out}  ({len(agg_rows)} rows)")

    # ── brief console report ──────────────────────────────────────────────────
    print("\n=== ACTOR-KL ABLATION SUMMARY (SR) ===")
    for scen, man in TARGET_CELLS:
        for method in METHODS:
            on_sr  = [r["SR"] for r in groups.get((scen, man, method, "on"),  [])]
            off_sr = [r["SR"] for r in groups.get((scen, man, method, "off"), [])]
            p_raw, p_adj = welch_table.get((scen, man, method, "SR"), (float("nan"), float("nan")))
            on_m   = f"{np.mean(on_sr):.3f}" if on_sr  else "n/a"
            off_m  = f"{np.mean(off_sr):.3f}" if off_sr else "n/a"
            sig    = "*" if not math.isnan(p_adj) and p_adj < 0.05 else ""
            print(f"  {scen}/{man} {method:15s}  KL-on={on_m} (n={len(on_sr)})  "
                  f"KL-off={off_m} (n={len(off_sr)})  p_adj={p_adj:.3f}{sig}")


if __name__ == "__main__":
    main()
