"""HO2 (noocc->occ transfer) for all 5 PDE methods, run-level, self-contained.

The stock _analyze_heldout only sees cbf_aux for HO2 because the occOFF *source*
named-eval files live only in the per-machine Tier-2 dirs, not in the
tier2_noocc symlink target. This aggregator sources occOFF (occlusion-OFF
training, the baseline) from the per-machine 2b dirs and the occ-ON transfer
from results/tier4_HO2_noocc_to_occ, both via each run's episode-level
eval_metrics.csv, aggregated to one value per training run (seed).

Outputs:
  results/tables/tier4_ho2_extended.csv   (all 5 methods, n + CIs + Cohen's d)
  and updates the HO2 rows of results/tables/heldout_comparisons.csv to the
  same 7-column (ho_name, method, metric, source_mean, ho_mean, delta, cohens_d)
  format used by the other HO rows.

Usage:
  python scripts/aggregate_ho2_extended.py --repo .
"""
import argparse, csv, glob, os, re
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

METHODS = ["soft_hjb_aux","hjb_aux","eikonal_aux","fusion_aux","cbf_aux"]
HO_NAME = "HO2_noocc_to_occ"
METRICS = ["mean_return","collision_rate","success_rate","mean_ttc"]


def m_of(name):
    for m in sorted(METHODS, key=len, reverse=True):
        if f"_{m}_" in name:
            return m
    return None


def seed_of(name):
    m = re.search(r"_s(\d+)$", name)
    return m.group(1) if m else None


def run_stats(d):
    p = os.path.join(d, "eval_metrics.csv")
    if not os.path.isfile(p):
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty or "terminal_state" not in df.columns:
        return None
    n = len(df); ts = df["terminal_state"].str.lower()
    return {
        "success_rate": float((ts == "success").sum() / n),
        "collision_rate": float((ts == "collision").sum() / n),
        "mean_return": float(df["return_total"].mean()) if "return_total" in df else float("nan"),
        "mean_ttc": float(df["mean_ttc"].mean()) if "mean_ttc" in df else float("nan"),
    }


def cohens_d(x, y):
    x = np.asarray([v for v in x if v == v], float); y = np.asarray([v for v in y if v == v], float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pv = ((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1)) / (len(x)+len(y)-2)
    s = np.sqrt(pv)
    return float((np.mean(x)-np.mean(y))/s) if s > 0 else float("nan")


def ci95(vals):
    x = np.asarray([v for v in vals if v == v], float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    b = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(2000)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/tables/tier4_ho2_extended.csv")
    ap.add_argument("--heldout", default="results/tables/heldout_comparisons.csv")
    a = ap.parse_args()
    if pd is None:
        print("pandas required"); return

    # source (occOFF) per-machine, transfer (occ-ON) from tier4_HO2
    src_dirs = glob.glob(os.path.join(a.repo, "results/tier_2_machine_cmu*/tier2/2b_occlusion_sweep/*occOFF*"))
    tr_dirs  = glob.glob(os.path.join(a.repo, "results/tier4_HO2_noocc_to_occ/*"))

    src = {m: [] for m in METHODS}   # method -> list of run stats
    for d in src_dirs:
        m = m_of(os.path.basename(d))
        if m and os.path.isdir(d):
            st = run_stats(d)
            if st: src[m].append(st)
    tr = {m: [] for m in METHODS}
    for d in tr_dirs:
        m = m_of(os.path.basename(d))
        if m and os.path.isdir(d):
            st = run_stats(d)
            if st: tr[m].append(st)

    ext_rows = []; held_rows = []
    for m in METHODS:
        if len(src[m]) < 2 or len(tr[m]) < 2:
            print(f"[ho2] skip {m}: src={len(src[m])} transfer={len(tr[m])}")
            continue
        for metric in METRICS:
            sv = [s[metric] for s in src[m]]; tv = [s[metric] for s in tr[m]]
            sm = float(np.nanmean(sv)); tm = float(np.nanmean(tv))
            slo, shi = ci95(sv); tlo, thi = ci95(tv)
            d = cohens_d(tv, sv)
            ext_rows.append({"ho_name": HO_NAME, "method": m, "metric": metric,
                "n_source": len(sv), "n_transfer": len(tv),
                "source_mean": round(sm, 5), "source_ci_low": round(slo, 5), "source_ci_high": round(shi, 5),
                "transfer_mean": round(tm, 5), "transfer_ci_low": round(tlo, 5), "transfer_ci_high": round(thi, 5),
                "delta": round(tm - sm, 5), "cohens_d": round(d, 4)})
            held_rows.append({"ho_name": HO_NAME, "method": m, "metric": metric,
                "source_mean": sm, "ho_mean": tm, "delta": tm - sm, "cohens_d": d})

    # standalone extended table
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ext_rows[0].keys())); w.writeheader(); w.writerows(ext_rows)
    print(f"Saved {a.out} ({len(ext_rows)} rows, methods={sorted(set(r['method'] for r in ext_rows))})")

    # merge into heldout_comparisons.csv: drop old HO2 rows, append new
    if os.path.isfile(a.heldout):
        with open(a.heldout) as f:
            rd = csv.DictReader(f); fields = rd.fieldnames; existing = list(rd)
        kept = [r for r in existing if not r["ho_name"].startswith("HO2")]
        merged = kept + [{k: r.get(k, "") for k in fields} for r in held_rows]
        with open(a.heldout, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(merged)
        methods_ho2 = sorted({r["method"] for r in merged if r["ho_name"].startswith("HO2")})
        print(f"Updated {a.heldout}: HO2 now covers {methods_ho2}")
    else:
        print(f"WARNING: {a.heldout} not found — only standalone table written")


if __name__ == "__main__":
    main()
