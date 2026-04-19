"""Generate LaTeX results tables with rigorous statistical analysis.

Features:
- Holm-Bonferroni correction for multiple comparisons (per metric family)
- Bootstrap 95% confidence intervals
- Cohen's d effect sizes
- Mann-Whitney U as robustness secondary test
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


def compute_single_p(values_method, values_baseline):
    """Welch's t-test p-value only (corrected later in batch)."""
    try:
        from scipy import stats
    except ImportError:
        return float("nan")
    if len(values_method) < 2 or len(values_baseline) < 2:
        return float("nan")
    _, p_val = stats.ttest_ind(values_method, values_baseline, equal_var=False)
    return float(p_val)


def compute_mannwhitney_p(x, y):
    """Mann-Whitney U two-sided p-value."""
    try:
        from scipy import stats
    except ImportError:
        return float("nan")
    x = [v for v in x if v == v]
    y = [v for v in y if v == v]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    try:
        _, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
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
    alpha = (1 - confidence) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas")
        return

    os.makedirs(args.out, exist_ok=True)

    # Load eval CSVs
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
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        print("No valid CSVs loaded")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(dfs)} eval files")

    methods = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "drppo"]
    pde_methods = ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux"]
    baseline = "drppo"
    metrics = ["mean_return", "collision_rate", "success_rate", "mean_ttc", "min_ttc"]
    metric_labels = {"mean_return": "Return", "collision_rate": "Collision",
                     "success_rate": "Success", "mean_ttc": "Mean TTC", "min_ttc": "Min TTC"}
    method_labels = {"drppo": "DRPPO", "hjb_aux": "Hard-HJB", "soft_hjb_aux": "Soft-HJB",
                     "eikonal_aux": "Eikonal", "cbf_aux": "CBF-PDE"}

    # Determine method column
    method_col = "variant" if "variant" in df.columns else "method" if "method" in df.columns else None

    # PASS 1: collect all comparisons
    results = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for method in pde_methods:
            if method_col:
                m_vals = df[df[method_col] == method][metric].dropna().values
                b_vals = df[df[method_col] == baseline][metric].dropna().values
            else:
                m_vals = np.array([])
                b_vals = np.array([])
            p_welch = compute_single_p(m_vals, b_vals)
            p_mwu = compute_mannwhitney_p(m_vals, b_vals)
            d = cohens_d(m_vals, b_vals)
            m_mean = float(np.mean(m_vals)) if len(m_vals) > 0 else float("nan")
            ci_lo, ci_hi = bootstrap_ci(m_vals)
            results.append({
                "method": method, "metric": metric,
                "n_method": len(m_vals), "n_baseline": len(b_vals),
                "mean": m_mean, "ci_low": ci_lo, "ci_high": ci_hi,
                "cohens_d": d, "raw_p_welch": p_welch, "raw_p_mwu": p_mwu,
            })

    # PASS 2: Holm-Bonferroni per metric
    for metric in metrics:
        metric_results = [r for r in results if r["metric"] == metric]
        ps = [r["raw_p_welch"] for r in metric_results]
        corrected = holm_bonferroni(ps, alpha=args.alpha)
        for r, (raw, corr, sig) in zip(metric_results, corrected):
            r["corrected_p"] = corr
            r["significant"] = sig
            r["marker"] = significance_marker(corr)

    # Save all_comparisons.csv
    out_csv = os.path.join(args.out, "all_comparisons.csv")
    fieldnames = ["metric", "method", "n_method", "n_baseline",
                  "mean", "ci_low", "ci_high", "cohens_d",
                  "raw_p_welch", "raw_p_mwu", "corrected_p", "significant", "marker"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved {out_csv} ({len(results)} rows)")

    # Main results LaTeX table
    out_tex = os.path.join(args.out, "main_results.tex")
    with open(out_tex, "w") as f:
        f.write("% Main results: mean [95% bootstrap CI] with Holm-Bonferroni significance vs DRPPO\n")
        f.write("\\begin{table*}[ht]\n\\centering\n")
        f.write("\\caption{Main comparison. Significance vs.\\ DRPPO with Holm-Bonferroni correction.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{l" + "c" * len(metrics) + "}\n\\toprule\n")
        f.write("Method & " + " & ".join(metric_labels.get(m, m) for m in metrics) + " \\\\\n\\midrule\n")

        for method in methods:
            row = method_labels.get(method, method)
            for metric in metrics:
                if method == baseline:
                    # Baseline: just show mean [CI]
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
                    matching = [r for r in results if r["method"] == method and r["metric"] == metric]
                    if matching:
                        r = matching[0]
                        marker = r.get("marker", "")
                        row += f" & ${r['mean']:.2f}$ [${r['ci_low']:.2f}, {r['ci_high']:.2f}$] {marker}"
                    else:
                        row += " & --"
            row += " \\\\\n"
            f.write(row)

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    print(f"Saved {out_tex}")

    # Effect sizes table
    out_effects = os.path.join(args.out, "effect_sizes.tex")
    with open(out_effects, "w") as f:
        f.write("% Cohen's d effect sizes: PDE methods vs DRPPO\n")
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Effect sizes (Cohen's $d$) vs.\\ DRPPO.}\n")
        f.write("\\label{tab:effect_sizes}\n")
        f.write("\\begin{tabular}{l" + "c" * len(pde_methods) + "}\n\\toprule\n")
        f.write("Metric & " + " & ".join(method_labels.get(m, m) for m in pde_methods) + " \\\\\n\\midrule\n")
        for metric in metrics:
            row = metric_labels.get(metric, metric)
            for method in pde_methods:
                matching = [r for r in results if r["method"] == method and r["metric"] == metric]
                if matching and not np.isnan(matching[0]["cohens_d"]):
                    d = matching[0]["cohens_d"]
                    tag = ""
                    if abs(d) < 0.2: tag = ""
                    elif abs(d) < 0.5: tag = " (S)"
                    elif abs(d) < 0.8: tag = " (M)"
                    else: tag = " (L)"
                    row += f" & ${d:+.2f}${tag}"
                else:
                    row += " & --"
            row += " \\\\\n"
            f.write(row)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {out_effects}")

    print(f"\nAll tables saved to {args.out}/")


if __name__ == "__main__":
    main()
