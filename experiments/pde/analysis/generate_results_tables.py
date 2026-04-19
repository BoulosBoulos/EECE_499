"""Generate LaTeX results tables with rigorous statistical analysis.

Features:
- Robust filename parser using known method/scenario sets for longest-match
- Holm-Bonferroni correction for multiple comparisons (per metric family)
- Bootstrap 95% confidence intervals
- Cohen's d effect sizes with bootstrap CI
- Mann-Whitney U as robustness secondary test
- Paired t-test for matched-seed comparisons
- PDE-vs-PDE head-to-head comparisons
- Paradigm comparison (optimality-PDE vs safety-PDE)
- Per-scenario + aggregated reporting

Usage:
    python experiments/pde/analysis/generate_results_tables.py \
        --eval_dir results/ablation/tier1 --out results/tables/
"""

import argparse
import os
import csv
import glob
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
# Known sets for filename parsing
# ---------------------------------------------------------------------------
KNOWN_METHODS = [
    "soft_hjb_aux", "hjb_aux", "eikonal_aux", "cbf_aux", "drppo", "rule_based",
]
KNOWN_SCENARIOS = [
    "1a", "1b", "1c", "1d", "2_dense", "3_dense", "4_dense", "2", "3", "4",
]
KNOWN_MANEUVERS = [
    "stem_right", "stem_left", "right_left", "right_stem", "left_right", "left_stem",
]

ALL_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
PDE_METHODS = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux"]
BASELINE = "drppo"

METRICS = [
    "mean_return", "collision_rate", "success_rate", "mean_ttc", "min_ttc",
    "ttc_p10_mean", "action_entropy_mean", "hard_brakes_per_ep_mean",
    "row_violations_per_ep_mean", "action_go_frac", "action_yield_frac",
    "switching_rate_mean", "decision_latency_mean",
]

METRIC_LABELS = {
    "mean_return": "Return",
    "collision_rate": "Collision",
    "success_rate": "Success",
    "mean_ttc": "Mean TTC",
    "min_ttc": "Min TTC",
    "ttc_p10_mean": "TTC p10",
    "action_entropy_mean": "Act. Entropy",
    "hard_brakes_per_ep_mean": "Hard Brakes",
    "row_violations_per_ep_mean": "ROW Viol.",
    "action_go_frac": "Go Frac.",
    "action_yield_frac": "Yield Frac.",
    "switching_rate_mean": "Switch Rate",
    "decision_latency_mean": "Latency",
}

METHOD_LABELS = {
    "drppo": "DRPPO",
    "hjb_aux": "Hard-HJB",
    "soft_hjb_aux": "Soft-HJB",
    "eikonal_aux": "Eikonal",
    "cbf_aux": "CBF-PDE",
    "rule_based": "Rule-Based",
}


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------
def parse_eval_filename(filename):
    """Parse an eval CSV filename using longest-match against known sets.

    Expected pattern: eval_{method}_{scenario}_{maneuver}.csv
    Returns (method, scenario, maneuver) or (None, None, None) on failure.
    """
    base = os.path.basename(filename)
    if not base.startswith("eval_") or not base.endswith(".csv"):
        return None, None, None
    stem = base[len("eval_"):-len(".csv")]  # e.g. "soft_hjb_aux_1a_stem_right"

    # Try each method (longest first, since KNOWN_METHODS is ordered that way)
    for method in sorted(KNOWN_METHODS, key=len, reverse=True):
        if stem.startswith(method + "_"):
            rest = stem[len(method) + 1:]
            # Try each scenario (longest first)
            for scenario in sorted(KNOWN_SCENARIOS, key=len, reverse=True):
                if rest.startswith(scenario + "_"):
                    maneuver = rest[len(scenario) + 1:]
                    if maneuver in KNOWN_MANEUVERS:
                        return method, scenario, maneuver
                elif rest == scenario:
                    return method, scenario, None
            # If no scenario matched, maybe rest is just maneuver
            if rest in KNOWN_MANEUVERS:
                return method, None, rest
            return method, None, None
    return None, None, None


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def compute_single_p(values_method, values_baseline):
    """Welch's t-test p-value only (corrected later in batch)."""
    if sp_stats is None:
        return float("nan")
    if len(values_method) < 2 or len(values_baseline) < 2:
        return float("nan")
    _, p_val = sp_stats.ttest_ind(values_method, values_baseline, equal_var=False)
    return float(p_val)


def compute_mannwhitney_p(x, y):
    """Mann-Whitney U two-sided p-value."""
    if sp_stats is None:
        return float("nan")
    x = [v for v in x if v == v]
    y = [v for v in y if v == v]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    try:
        _, p_val = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(p_val)
    except Exception:
        return float("nan")


def compute_paired_p(x, y):
    """Paired t-test p-value for matched-seed comparisons.

    Uses scipy.stats.ttest_rel. Requires equal-length, matched arrays.
    """
    if sp_stats is None:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Drop pairs where either is NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return float("nan")
    try:
        _, p_val = sp_stats.ttest_rel(x, y)
        return float(p_val)
    except Exception:
        return float("nan")


def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns list of (raw_p, corrected_p, significant) tuples in original order.
    """
    p_arr = np.array(p_values, dtype=float)
    n_tests = len(p_arr)
    valid_mask = ~np.isnan(p_arr)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        return [(p, float("nan"), False) for p in p_values]

    valid_indices = np.where(valid_mask)[0]
    valid_p = p_arr[valid_mask]
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]

    corrected = np.zeros(n_valid)
    for i, p in enumerate(sorted_p):
        corrected[i] = min(p * (n_valid - i), 1.0)
    for i in range(1, n_valid):
        corrected[i] = max(corrected[i], corrected[i - 1])

    unsort_order = np.argsort(sort_order)
    corrected_in_valid_order = corrected[unsort_order]

    results = []
    v_idx = 0
    for i in range(n_tests):
        if valid_mask[i]:
            cp = float(corrected_in_valid_order[v_idx])
            results.append((float(p_arr[i]), cp, cp < alpha))
            v_idx += 1
        else:
            results.append((float("nan"), float("nan"), False))
    return results


def significance_marker(corrected_p):
    """Return significance marker based on corrected p-value."""
    if corrected_p != corrected_p:
        return "n/a"
    if corrected_p < 0.001:
        return "***"
    if corrected_p < 0.01:
        return "**"
    if corrected_p < 0.05:
        return "*"
    return "ns"


def bootstrap_ci(values, n_resamples=10000, confidence=0.95, seed=42):
    """Nonparametric bootstrap confidence interval for the mean."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(vals, size=len(vals), replace=True))
             for _ in range(n_resamples)]
    alpha_half = (1 - confidence) / 2
    return (float(np.percentile(boots, 100 * alpha_half)),
            float(np.percentile(boots, 100 * (1 - alpha_half))))


def cohens_d(x, y):
    """Cohen's d effect size: (mean_x - mean_y) / pooled_std."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cohens_d_ci(x, y, n_resamples=5000, confidence=0.95, seed=42):
    """Bootstrap CI for Cohen's d effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_resamples):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        d = cohens_d(bx, by)
        if d == d:  # not NaN
            ds.append(d)
    if len(ds) < 10:
        return float("nan"), float("nan")
    alpha_half = (1 - confidence) / 2
    return (float(np.percentile(ds, 100 * alpha_half)),
            float(np.percentile(ds, 100 * (1 - alpha_half))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if pd is None:
        print("pandas required: pip install pandas")
        return

    os.makedirs(args.out, exist_ok=True)

    # ------------------------------------------------------------------
    # Load eval CSVs
    # ------------------------------------------------------------------
    pattern = os.path.join(args.eval_dir, "**", "eval_*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        pattern = os.path.join(args.eval_dir, "ablation_results.csv")
        csv_files = glob.glob(pattern)
    if not csv_files:
        print(f"No eval CSVs found in {args.eval_dir}")
        return

    dfs = []
    for f in csv_files:
        try:
            tmp = pd.read_csv(f)
            method, scenario, maneuver = parse_eval_filename(f)
            if method:
                tmp["_parsed_method"] = method
            if scenario:
                tmp["_parsed_scenario"] = scenario
            if maneuver:
                tmp["_parsed_maneuver"] = maneuver
            dfs.append(tmp)
        except Exception:
            continue
    if not dfs:
        print("No valid CSVs loaded")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(dfs)} eval files")

    # Determine method column
    if "_parsed_method" in df.columns and df["_parsed_method"].notna().any():
        method_col = "_parsed_method"
    elif "variant" in df.columns:
        method_col = "variant"
    elif "method" in df.columns:
        method_col = "method"
    else:
        method_col = None

    # Scenario / maneuver columns
    scen_col = "_parsed_scenario" if "_parsed_scenario" in df.columns else (
        "scenario" if "scenario" in df.columns else None)
    man_col = "_parsed_maneuver" if "_parsed_maneuver" in df.columns else (
        "ego_maneuver" if "ego_maneuver" in df.columns else None)

    # Enumerate unique (scenario, maneuver) combos present
    if scen_col and man_col:
        combos = df.groupby([scen_col, man_col]).size().reset_index()[[scen_col, man_col]]
        combos = list(zip(combos[scen_col], combos[man_col]))
    elif scen_col:
        combos = [(s, None) for s in df[scen_col].dropna().unique()]
    else:
        combos = [(None, None)]

    # Filter to available metrics
    available_metrics = [m for m in METRICS if m in df.columns]
    if not available_metrics:
        print("No recognized metrics found in data columns:", list(df.columns))
        return

    # ------------------------------------------------------------------
    # PASS 1: collect all comparisons (scenario, maneuver, method, metric)
    # ------------------------------------------------------------------
    results = []
    for scenario, maneuver in combos:
        for metric in available_metrics:
            for method in PDE_METHODS:
                if method_col is None:
                    m_vals = np.array([])
                    b_vals = np.array([])
                else:
                    mask_m = df[method_col] == method
                    mask_b = df[method_col] == BASELINE
                    if scen_col and scenario is not None:
                        mask_m = mask_m & (df[scen_col] == scenario)
                        mask_b = mask_b & (df[scen_col] == scenario)
                    if man_col and maneuver is not None:
                        mask_m = mask_m & (df[man_col] == maneuver)
                        mask_b = mask_b & (df[man_col] == maneuver)
                    m_vals = df.loc[mask_m, metric].dropna().values
                    b_vals = df.loc[mask_b, metric].dropna().values

                p_welch = compute_single_p(m_vals, b_vals)
                p_mwu = compute_mannwhitney_p(m_vals, b_vals)

                # Paired test: match by seed+eval_mode if available
                p_paired = float("nan")
                if "seed" in df.columns and "eval_mode" in df.columns and method_col:
                    mask_m_base = df[method_col] == method
                    mask_b_base = df[method_col] == BASELINE
                    if scen_col and scenario is not None:
                        mask_m_base = mask_m_base & (df[scen_col] == scenario)
                        mask_b_base = mask_b_base & (df[scen_col] == scenario)
                    if man_col and maneuver is not None:
                        mask_m_base = mask_m_base & (df[man_col] == maneuver)
                        mask_b_base = mask_b_base & (df[man_col] == maneuver)
                    df_m = df.loc[mask_m_base].set_index(["seed", "eval_mode"])
                    df_b = df.loc[mask_b_base].set_index(["seed", "eval_mode"])
                    common = df_m.index.intersection(df_b.index)
                    if len(common) >= 2:
                        paired_m = df_m.loc[common, metric].dropna()
                        paired_b = df_b.loc[common, metric].dropna()
                        shared = paired_m.index.intersection(paired_b.index)
                        if len(shared) >= 2:
                            p_paired = compute_paired_p(
                                paired_m.loc[shared].values,
                                paired_b.loc[shared].values)

                d = cohens_d(m_vals, b_vals)
                d_ci_lo, d_ci_hi = cohens_d_ci(m_vals, b_vals)
                m_mean = float(np.mean(m_vals)) if len(m_vals) > 0 else float("nan")
                ci_lo, ci_hi = bootstrap_ci(m_vals)

                results.append({
                    "scenario": scenario if scenario else "all",
                    "maneuver": maneuver if maneuver else "all",
                    "method": method,
                    "metric": metric,
                    "n_method": len(m_vals),
                    "n_baseline": len(b_vals),
                    "mean": m_mean,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "cohens_d": d,
                    "cohens_d_ci_low": d_ci_lo,
                    "cohens_d_ci_high": d_ci_hi,
                    "raw_p_welch": p_welch,
                    "raw_p_mwu": p_mwu,
                    "raw_p_paired": p_paired,
                })

    # ------------------------------------------------------------------
    # PASS 2: Holm-Bonferroni per metric across ALL (scenario, maneuver, method)
    #         Family size = n_combos x n_pde_methods
    #         Apply separately to Welch, MWU, and paired p-values
    # ------------------------------------------------------------------
    for metric in available_metrics:
        metric_results = [r for r in results if r["metric"] == metric]
        # Welch
        ps_welch = [r["raw_p_welch"] for r in metric_results]
        corrected_welch = holm_bonferroni(ps_welch, alpha=args.alpha)
        # MWU
        ps_mwu = [r["raw_p_mwu"] for r in metric_results]
        corrected_mwu = holm_bonferroni(ps_mwu, alpha=args.alpha)
        # Paired
        ps_paired = [r["raw_p_paired"] for r in metric_results]
        corrected_paired = holm_bonferroni(ps_paired, alpha=args.alpha)

        for r, (_, cw, sw), (_, cm, sm), (_, cp, sp) in zip(
                metric_results, corrected_welch, corrected_mwu, corrected_paired):
            r["corrected_p_welch"] = cw
            r["significant_welch"] = sw
            r["marker_welch"] = significance_marker(cw)
            r["corrected_p_mwu"] = cm
            r["significant_mwu"] = sm
            r["marker_mwu"] = significance_marker(cm)
            r["corrected_p_paired"] = cp
            r["significant_paired"] = sp
            r["marker_paired"] = significance_marker(cp)

    # ------------------------------------------------------------------
    # Output: all_comparisons.csv
    # ------------------------------------------------------------------
    out_csv = os.path.join(args.out, "all_comparisons.csv")
    fieldnames = [
        "scenario", "maneuver", "metric", "method",
        "n_method", "n_baseline", "mean", "ci_low", "ci_high",
        "cohens_d", "cohens_d_ci_low", "cohens_d_ci_high",
        "raw_p_welch", "corrected_p_welch", "significant_welch", "marker_welch",
        "raw_p_mwu", "corrected_p_mwu", "significant_mwu", "marker_mwu",
        "raw_p_paired", "corrected_p_paired", "significant_paired", "marker_paired",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved {out_csv} ({len(results)} rows)")

    # ------------------------------------------------------------------
    # Output: main_results.tex
    # ------------------------------------------------------------------
    out_tex = os.path.join(args.out, "main_results.tex")
    with open(out_tex, "w") as f:
        f.write("% Main results: mean [95\\% bootstrap CI] "
                "with Holm-Bonferroni significance vs DRPPO\n")
        f.write("\\begin{table*}[ht]\n\\centering\n")
        f.write("\\caption{Main comparison. Significance vs.\\ "
                "DRPPO with Holm-Bonferroni correction.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{l" + "c" * len(available_metrics) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(
            METRIC_LABELS.get(m, m) for m in available_metrics) + " \\\\\n\\midrule\n")

        for method in ALL_METHODS:
            row = METHOD_LABELS.get(method, method)
            for metric in available_metrics:
                if method == BASELINE:
                    if method_col:
                        vals = df[df[method_col] == method][metric].dropna().values
                    else:
                        vals = np.array([])
                    if len(vals) > 0:
                        m = np.mean(vals)
                        lo, hi = bootstrap_ci(vals)
                        row += f" & ${m:.2f}$ [${lo:.2f}, {hi:.2f}$]"
                    else:
                        row += " & --"
                else:
                    matching = [r for r in results if r["method"] == method
                                and r["metric"] == metric]
                    if matching:
                        # Aggregate across scenarios: use first or average
                        r0 = matching[0]
                        marker = r0.get("marker_welch", "")
                        row += (f" & ${r0['mean']:.2f}$ "
                                f"[${r0['ci_low']:.2f}, {r0['ci_high']:.2f}$] {marker}")
                    else:
                        row += " & --"
            row += " \\\\\n"
            f.write(row)

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved {out_tex}")

    # ------------------------------------------------------------------
    # Output: effect_sizes.tex (with Cohen's d CI)
    # ------------------------------------------------------------------
    out_effects = os.path.join(args.out, "effect_sizes.tex")
    with open(out_effects, "w") as f:
        f.write("% Cohen's d effect sizes with bootstrap CI: PDE methods vs DRPPO\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Effect sizes (Cohen's $d$) vs.\\ DRPPO "
                "with 95\\% bootstrap CI.}\n")
        f.write("\\label{tab:effect_sizes}\n")
        f.write("\\begin{tabular}{l" + "c" * len(PDE_METHODS) + "}\n\\toprule\n")
        f.write("Metric & " + " & ".join(
            METHOD_LABELS.get(m, m) for m in PDE_METHODS) + " \\\\\n\\midrule\n")
        for metric in available_metrics:
            row = METRIC_LABELS.get(metric, metric)
            for method in PDE_METHODS:
                matching = [r for r in results
                            if r["method"] == method and r["metric"] == metric]
                if matching and not np.isnan(matching[0]["cohens_d"]):
                    r0 = matching[0]
                    d = r0["cohens_d"]
                    d_lo = r0["cohens_d_ci_low"]
                    d_hi = r0["cohens_d_ci_high"]
                    tag = ""
                    if abs(d) < 0.2:
                        tag = ""
                    elif abs(d) < 0.5:
                        tag = " (S)"
                    elif abs(d) < 0.8:
                        tag = " (M)"
                    else:
                        tag = " (L)"
                    if np.isnan(d_lo) or np.isnan(d_hi):
                        row += f" & ${d:+.2f}${tag}"
                    else:
                        row += f" & ${d:+.2f}$ [{d_lo:+.2f}, {d_hi:+.2f}]{tag}"
                else:
                    row += " & --"
            row += " \\\\\n"
            f.write(row)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {out_effects}")

    # ------------------------------------------------------------------
    # Output: per_scenario_{metric}.tex for each metric
    # ------------------------------------------------------------------
    for metric in available_metrics:
        out_per = os.path.join(args.out, f"per_scenario_{metric}.tex")
        metric_results = [r for r in results if r["metric"] == metric]
        if not metric_results:
            continue
        # Collect unique (scenario, maneuver) tuples
        sm_pairs = sorted(set((r["scenario"], r["maneuver"]) for r in metric_results))
        with open(out_per, "w") as f:
            f.write(f"% Per-scenario breakdown: {METRIC_LABELS.get(metric, metric)}\n")
            f.write("\\begin{table*}[ht]\n\\centering\n")
            f.write(f"\\caption{{Per-scenario {METRIC_LABELS.get(metric, metric)} "
                    f"vs.\\ DRPPO.}}\n")
            f.write(f"\\label{{tab:per_scenario_{metric}}}\n")
            f.write("\\begin{tabular}{ll" + "c" * len(PDE_METHODS) + "}\n\\toprule\n")
            f.write("Scenario & Maneuver & " + " & ".join(
                METHOD_LABELS.get(m, m) for m in PDE_METHODS) + " \\\\\n\\midrule\n")
            for scen, man in sm_pairs:
                row = f"{scen} & {man}"
                for method in PDE_METHODS:
                    match = [r for r in metric_results
                             if r["method"] == method
                             and r["scenario"] == scen
                             and r["maneuver"] == man]
                    if match:
                        r0 = match[0]
                        marker = r0.get("marker_welch", "")
                        row += (f" & ${r0['mean']:.2f}$ "
                                f"[${r0['ci_low']:.2f}, {r0['ci_high']:.2f}$] {marker}")
                    else:
                        row += " & --"
                row += " \\\\\n"
                f.write(row)
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
        print(f"Saved {out_per}")

    # ------------------------------------------------------------------
    # PDE-vs-PDE comparisons: all 6 ordered pairs
    # ------------------------------------------------------------------
    pde_vs_pde = []
    for metric in available_metrics:
        for i, ma in enumerate(PDE_METHODS):
            for j, mb in enumerate(PDE_METHODS):
                if i == j:
                    continue
                for scenario, maneuver in combos:
                    if method_col is None:
                        va = np.array([])
                        vb = np.array([])
                    else:
                        mask_a = df[method_col] == ma
                        mask_b = df[method_col] == mb
                        if scen_col and scenario is not None:
                            mask_a = mask_a & (df[scen_col] == scenario)
                            mask_b = mask_b & (df[scen_col] == scenario)
                        if man_col and maneuver is not None:
                            mask_a = mask_a & (df[man_col] == maneuver)
                            mask_b = mask_b & (df[man_col] == maneuver)
                        va = df.loc[mask_a, metric].dropna().values
                        vb = df.loc[mask_b, metric].dropna().values

                    p_pair = float("nan")
                    if "seed" in df.columns and "eval_mode" in df.columns and method_col:
                        mask_a2 = df[method_col] == ma
                        mask_b2 = df[method_col] == mb
                        if scen_col and scenario is not None:
                            mask_a2 = mask_a2 & (df[scen_col] == scenario)
                            mask_b2 = mask_b2 & (df[scen_col] == scenario)
                        if man_col and maneuver is not None:
                            mask_a2 = mask_a2 & (df[man_col] == maneuver)
                            mask_b2 = mask_b2 & (df[man_col] == maneuver)
                        dfa = df.loc[mask_a2].set_index(["seed", "eval_mode"])
                        dfb = df.loc[mask_b2].set_index(["seed", "eval_mode"])
                        common = dfa.index.intersection(dfb.index)
                        if len(common) >= 2 and metric in dfa.columns and metric in dfb.columns:
                            pa = dfa.loc[common, metric].dropna()
                            pb = dfb.loc[common, metric].dropna()
                            shared = pa.index.intersection(pb.index)
                            if len(shared) >= 2:
                                p_pair = compute_paired_p(
                                    pa.loc[shared].values, pb.loc[shared].values)

                    d_val = cohens_d(va, vb)

                    pde_vs_pde.append({
                        "scenario": scenario if scenario else "all",
                        "maneuver": maneuver if maneuver else "all",
                        "metric": metric,
                        "method_a": ma,
                        "method_b": mb,
                        "raw_p_paired": p_pair,
                        "cohens_d": d_val,
                    })

    # Apply Holm correction per metric across all PDE-vs-PDE tests
    for metric in available_metrics:
        subset = [r for r in pde_vs_pde if r["metric"] == metric]
        ps = [r["raw_p_paired"] for r in subset]
        corrected = holm_bonferroni(ps, alpha=args.alpha)
        for r, (_, cp, sig) in zip(subset, corrected):
            r["corrected_p_paired"] = cp
            r["significant_paired"] = sig
            r["marker_paired"] = significance_marker(cp)

    out_pvp = os.path.join(args.out, "pde_vs_pde_comparisons.csv")
    pvp_fields = [
        "scenario", "maneuver", "metric", "method_a", "method_b",
        "raw_p_paired", "corrected_p_paired", "significant_paired",
        "marker_paired", "cohens_d",
    ]
    with open(out_pvp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pvp_fields)
        writer.writeheader()
        for r in pde_vs_pde:
            writer.writerow({k: r.get(k, "") for k in pvp_fields})
    print(f"Saved {out_pvp} ({len(pde_vs_pde)} rows)")

    # ------------------------------------------------------------------
    # Paradigm comparison: optimality-PDE vs safety-PDE
    # ------------------------------------------------------------------
    optimality_methods = ["hjb_aux", "soft_hjb_aux"]
    safety_methods = ["eikonal_aux", "cbf_aux"]
    out_paradigm = os.path.join(args.out, "paradigm_comparison.tex")
    with open(out_paradigm, "w") as f:
        f.write("% Paradigm comparison: Optimality-PDE "
                "(Hard-HJB + Soft-HJB) vs Safety-PDE (Eikonal + CBF)\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Paradigm comparison: pooled optimality-PDE "
                "vs.\\ safety-PDE.}\n")
        f.write("\\label{tab:paradigm_comparison}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Metric & Opt.~mean & Safety~mean & Cohen's $d$ & "
                "$p_{\\mathrm{Welch}}$ & Sig. \\\\\n\\midrule\n")
        for metric in available_metrics:
            if method_col is None:
                opt_vals = np.array([])
                saf_vals = np.array([])
            else:
                opt_vals = df[df[method_col].isin(optimality_methods)][metric].dropna().values
                saf_vals = df[df[method_col].isin(safety_methods)][metric].dropna().values
            if len(opt_vals) < 2 or len(saf_vals) < 2:
                f.write(f"{METRIC_LABELS.get(metric, metric)} & -- & -- & -- & -- & -- \\\\\n")
                continue
            opt_mean = float(np.mean(opt_vals))
            saf_mean = float(np.mean(saf_vals))
            d_val = cohens_d(opt_vals, saf_vals)
            p_val = compute_single_p(opt_vals, saf_vals)
            sig = significance_marker(p_val) if p_val == p_val else "n/a"
            f.write(f"{METRIC_LABELS.get(metric, metric)} & "
                    f"${opt_mean:.3f}$ & ${saf_mean:.3f}$ & "
                    f"${d_val:+.2f}$ & ${p_val:.4f}$ & {sig} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {out_paradigm}")

    # Held-out analysis (Tier 4) — only runs if Tier 4 results exist
    _analyze_heldout(args, df, METRICS, METRIC_LABELS, METHOD_LABELS, BASELINE)

    print(f"\nAll tables saved to {args.out}/")


def _analyze_heldout(args, df, metrics, metric_labels, method_labels, baseline):
    """Generate held-out (Tier 4) comparison tables if data exists."""
    import glob as _glob

    HO_CONFIGS = [
        ("HO1_occ_to_noocc", "tier1", "tier4_HO1_occ_to_noocc"),
        ("HO2_noocc_to_occ", "tier2_noocc", "tier4_HO2_noocc_to_occ"),
        ("HO3_full_to_adversarial", "tier1", "tier4_HO3_full_to_adversarial"),
        ("HO4_nominal_to_adversarial", "tier3_behav", "tier4_HO4_nominal_to_adversarial"),
        ("HO5_vis_to_novis", "tier1", "tier4_HO5_vis_to_novis"),
    ]

    base = os.path.dirname(args.eval_dir) if args.eval_dir else "results/ablation"
    summary_rows = []
    methods = list(method_labels.keys())

    for ho_name, source_dir_name, ho_dir_name in HO_CONFIGS:
        source_path = os.path.join(base, source_dir_name)
        ho_path = os.path.join(base, ho_dir_name)
        if not os.path.isdir(source_path) or not os.path.isdir(ho_path):
            continue

        source_csvs = _glob.glob(os.path.join(source_path, "**", "eval_*.csv"), recursive=True)
        ho_csvs = _glob.glob(os.path.join(ho_path, "**", "eval_*.csv"), recursive=True)
        if not source_csvs or not ho_csvs:
            continue

        for method in methods:
            method_src = [f for f in source_csvs if f"eval_{method}_" in f]
            method_ho = [f for f in ho_csvs if f"eval_{method}_" in f]
            for metric in metrics:
                try:
                    if pd is None:
                        continue
                    src_dfs = [pd.read_csv(f) for f in method_src]
                    ho_dfs = [pd.read_csv(f) for f in method_ho]
                    src_vals = pd.concat(src_dfs)[metric].dropna().values if src_dfs else np.array([])
                    ho_vals = pd.concat(ho_dfs)[metric].dropna().values if ho_dfs else np.array([])
                    if len(src_vals) < 2 or len(ho_vals) < 2:
                        continue
                    summary_rows.append({
                        "ho_name": ho_name, "method": method, "metric": metric,
                        "source_mean": float(np.mean(src_vals)),
                        "ho_mean": float(np.mean(ho_vals)),
                        "delta": float(np.mean(ho_vals) - np.mean(src_vals)),
                        "cohens_d": cohens_d(ho_vals, src_vals),
                    })
                except Exception:
                    continue

    if not summary_rows:
        return

    out_csv = os.path.join(args.out, "heldout_comparisons.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {out_csv} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
