"""Four Tier-1 exports from results/tier1 (tier_1_full), run-level only.

Statistical unit is ONE TRAINING RUN (one directory). Seed is parsed from the
directory name. Per-run metrics come from the named per-run summary
eval_{method}_{scenario}_{maneuver}.csv (its seed x eval_mode rows are averaged
to a single value per run); eval_metrics.csv episode logs are used only as a
fallback. Any per-cell n that is a multiple of 3 or 6 beyond the seed count
would indicate episode-level pooling and is reported as a BUG.

  Percentile bootstrap: B=1000, seed 12345
  Welch two-sided + Holm within family-metric

Outputs:
  results/tables/baseline_per_cell_tier1.csv
  results/tables/baseline_per_run_tier1.csv
  results/tables/pde_vs_pde_comparisons.csv   (regenerated with Welch+Holm)
  results/tables/pde_diagnostics.csv
  results/tables/tier1_intent_effect.csv
"""
import argparse, csv, json, os, re
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

B_BOOT   = 1000
BOOT_SEED = 12345
ALPHA     = 0.05

METHODS = ["soft_hjb_aux","hjb_aux","eikonal_aux","fusion_aux","cbf_aux","drppo","rule_based"]
PDE_METHODS = ["hjb_aux","soft_hjb_aux","eikonal_aux","fusion_aux","cbf_aux","rule_based"]
BASELINE = "drppo"
NO_TEST = {"rule_based"}
SCENS = ["4_dense","3_dense","2_dense","1a","1b","1c","1d","2","3","4"]
MANS  = ["stem_right","stem_left","right_left","right_stem","left_right","left_stem"]

BASE_METRICS = ["success_rate","collision_rate","mean_return","mean_ttc"]
ALL_METRICS  = ["mean_return","collision_rate","success_rate","mean_ttc","min_ttc",
                "ttc_p10_mean","action_entropy_mean","hard_brakes_per_ep_mean",
                "row_violations_per_ep_mean","action_go_frac","action_yield_frac",
                "switching_rate_mean","decision_latency_mean"]
INTENT_METRICS = ["success_rate","collision_rate","mean_return"]


# ---------------------------------------------------------------- parsing
def psm(rest):
    for s in SCENS:
        if rest.startswith(s + "_"):
            m = rest[len(s)+1:]
            if m in MANS:
                return s, m
    return None, None


def parse_dir(name):
    meth = next((m for m in sorted(METHODS, key=len, reverse=True) if f"_{m}_" in name), None)
    if meth is None:
        return None
    sm = re.search(r"_s(\d+)$", name)
    if not sm:
        return None
    intent = ("_intent_" in name and "_nointent_" not in name)
    scen, man = psm(name.split(f"_{meth}_")[0])
    if scen is None:
        return None
    return dict(method=meth, scenario=scen, maneuver=man, seed=sm.group(1), intent=intent)


# ---------------------------------------------------------------- loaders
def load_run_metrics(d, meta):
    """Run-level metrics: average the named summary's seed x mode rows."""
    named = os.path.join(d, f"eval_{meta['method']}_{meta['scenario']}_{meta['maneuver']}.csv")
    if os.path.isfile(named):
        try:
            df = pd.read_csv(named)
            if not df.empty:
                out = {c: float(df[c].mean()) for c in ALL_METRICS if c in df.columns}
                out["_source"] = "named_summary"; out["_n_rows"] = len(df)
                return out
        except Exception:
            pass
    ep = os.path.join(d, "eval_metrics.csv")           # fallback
    if os.path.isfile(ep):
        try:
            df = pd.read_csv(ep)
            if not df.empty and "terminal_state" in df.columns:
                n = len(df); ts = df["terminal_state"].str.lower()
                out = {"success_rate": float((ts=="success").sum()/n),
                       "collision_rate": float((ts=="collision").sum()/n),
                       "mean_return": float(df["return_total"].mean()) if "return_total" in df else np.nan,
                       "mean_ttc": float(df["mean_ttc"].mean()) if "mean_ttc" in df else np.nan,
                       "min_ttc": float(df["min_ttc"].min()) if "min_ttc" in df else np.nan}
                out["_source"] = "episode_fallback"; out["_n_rows"] = n
                return out
        except Exception:
            pass
    return None


def load_machine(d, meta):
    """Provenance machine from the eval meta's checkpoint_path."""
    mj = os.path.join(d, f"meta_eval_{meta['method']}_{meta['scenario']}_{meta['maneuver']}.json")
    for cand in (mj, os.path.join(d, "meta.json")):
        if os.path.isfile(cand):
            try:
                j = json.load(open(cand))
            except Exception:
                continue
            cp = j.get("checkpoint_path", "") or ""
            m = re.search(r"tier_1_machine_(cmu\d+(?:_p2)?|local[a-z_]*)", cp)
            if m:
                return m.group(1)
            if j.get("hostname"):
                return j["hostname"]
    return "unknown"


def load_loss_windows(d):
    """Mean PDE residual / distillation over first and last 10% of iterations."""
    p = os.path.join(d, "metrics.csv")
    if not os.path.isfile(p):
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty:
        return None
    if "iteration" in df.columns:
        df = df.sort_values("iteration")
    n = len(df)
    if n < 4:
        return None
    k = max(1, int(round(0.10 * n)))
    ropt = df["L_residual_optimality"] if "L_residual_optimality" in df else None
    rsaf = df["L_residual_safety"]     if "L_residual_safety"     in df else None
    if ropt is None and rsaf is None:
        return None
    res = (ropt.fillna(0) if ropt is not None else 0) + (rsaf.fillna(0) if rsaf is not None else 0)
    dis = df["L_distill"].fillna(0) if "L_distill" in df.columns else None
    early_r = float(np.mean(res.iloc[:k])); late_r = float(np.mean(res.iloc[-k:]))
    out = {"n_iters": n, "window_iters": k,
           "residual_early": early_r, "residual_late": late_r,
           "residual_ratio": (late_r/early_r) if early_r not in (0.0,) and np.isfinite(early_r) else np.nan,
           "distill_early": float(np.mean(dis.iloc[:k])) if dis is not None else np.nan,
           "distill_late":  float(np.mean(dis.iloc[-k:])) if dis is not None else np.nan}
    out["has_trace"] = bool(np.nanmax([abs(early_r), abs(late_r)]) > 0)
    return out


# ---------------------------------------------------------------- stats
def boot_ci(vals, b=B_BOOT, seed=BOOT_SEED):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(b)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def welch_p(a, b):
    a = [x for x in a if x == x]; b = [x for x in b if x == x]
    if sp_stats is None or len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        p = float(sp_stats.ttest_ind(a, b, equal_var=False)[1])
        return p if p == p else float("nan")
    except Exception:
        return float("nan")


def holm(ps, alpha=ALPHA):
    arr = np.array(ps, float); n = len(arr)
    valid = ~np.isnan(arr); nv = int(valid.sum())
    if nv == 0:
        return [float("nan")]*n
    idx = np.where(valid)[0]; vp = arr[valid]
    order = np.argsort(vp); corr = np.zeros(nv)
    for i, p in enumerate(vp[order]):
        corr[i] = min(p*(nv-i), 1.0)
    for i in range(1, nv):
        corr[i] = max(corr[i], corr[i-1])
    un = np.argsort(order); out = [float("nan")]*n
    for j, ii in enumerate(idx):
        out[ii] = float(corr[un[j]])
    return out


def cohens_d(x, y):
    x = np.asarray([v for v in x if v == v], float); y = np.asarray([v for v in y if v == v], float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pv = ((len(x)-1)*np.var(x, ddof=1) + (len(y)-1)*np.var(y, ddof=1))/(len(x)+len(y)-2)
    s = np.sqrt(pv)
    return float((np.mean(x)-np.mean(y))/s) if s > 0 else float("nan")


def marker(p):
    if p != p:
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def wcsv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    if pd is None:
        print("pandas required"); return

    root = os.path.join(a.repo, "results/tier_1_full")
    runs = []
    print("Scanning Tier-1 tree ...")
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        meta = parse_dir(name)
        if meta is None:
            continue
        met = load_run_metrics(d, meta)
        rec = dict(meta); rec["dir"] = name
        rec["machine"] = load_machine(d, meta)
        rec["metrics"] = met
        rec["loss"] = load_loss_windows(d)
        runs.append(rec)
    print(f"  {len(runs)} run dirs parsed; {sum(1 for r in runs if r['metrics'])} with metrics")

    # ---------------- (1) DRPPO baseline -------------------------------
    print("\n(1) DRPPO baseline")
    base = [r for r in runs if r["method"] == BASELINE and not r["intent"] and r["metrics"]]
    per_run = []
    for r in sorted(base, key=lambda x: (x["scenario"], x["maneuver"], int(x["seed"]))):
        row = {"scenario": r["scenario"], "maneuver": r["maneuver"], "seed": r["seed"],
               "machine": r["machine"]}
        for m in BASE_METRICS:
            row[m] = round(r["metrics"].get(m, float("nan")), 6)
        per_run.append(row)
    wcsv(os.path.join(a.out, "baseline_per_run_tier1.csv"), per_run,
         ["scenario","maneuver","seed","machine"] + BASE_METRICS)

    cells = {}
    for r in base:
        cells.setdefault((r["scenario"], r["maneuver"]), []).append(r)
    per_cell = []; bug = []
    for (s, mv), grp in sorted(cells.items()):
        n_seeds = len({g["seed"] for g in grp})
        if len(grp) != n_seeds:
            bug.append(f"{s}/{mv}: {len(grp)} rows vs {n_seeds} distinct seeds")
        for met in BASE_METRICS:
            vals = [g["metrics"].get(met, float("nan")) for g in grp]
            lo, hi = boot_ci(vals)
            per_cell.append({"scenario": s, "maneuver": mv, "metric": met, "n": len(grp),
                             "mean": round(float(np.nanmean(vals)), 6),
                             "ci_low": round(lo, 6), "ci_high": round(hi, 6)})
    wcsv(os.path.join(a.out, "baseline_per_cell_tier1.csv"), per_cell,
         ["scenario","maneuver","metric","n","mean","ci_low","ci_high"])
    ns = sorted({r["n"] for r in per_cell})
    print(f"  distinct per-cell n: {ns}  cells={len(cells)}  runs={len(base)}")
    print("  PSEUDOREPLICATION BUG: " + ("; ".join(bug) if bug else "none (n == distinct seeds)"))

    # ---------------- (2) Family B: pde_vs_pde -------------------------
    print("\n(2) pde_vs_pde (run-level Welch + Holm)")
    noint = [r for r in runs if not r["intent"] and r["metrics"]]
    idx = {}
    for r in noint:
        idx.setdefault((r["scenario"], r["maneuver"], r["method"]), []).append(r)
    combos = sorted({(r["scenario"], r["maneuver"]) for r in noint})
    pvp = []
    for met in ALL_METRICS:
        for ma in PDE_METHODS:
            for mb in PDE_METHODS:
                if ma == mb:
                    continue
                for (s, mv) in combos:
                    va = [x["metrics"].get(met, float("nan")) for x in idx.get((s,mv,ma), [])]
                    vb = [x["metrics"].get(met, float("nan")) for x in idx.get((s,mv,mb), [])]
                    pvp.append({"scenario": s, "maneuver": mv, "metric": met,
                                "method_a": ma, "method_b": mb,
                                "n_a": len(va), "n_b": len(vb),
                                "raw_p_welch": welch_p(va, vb),
                                "cohens_d": cohens_d(va, vb)})
    for met in ALL_METRICS:                      # Holm within family-metric
        sub = [r for r in pvp if r["metric"] == met and
               r["method_a"] not in NO_TEST and r["method_b"] not in NO_TEST]
        for r, cp in zip(sub, holm([r["raw_p_welch"] for r in sub])):
            r["corrected_p_welch"] = cp
            r["significant_welch"] = bool(cp == cp and cp < ALPHA)
            r["marker_welch"] = marker(cp)
    for r in pvp:
        if "corrected_p_welch" not in r:         # rule_based rows: descriptive
            r["corrected_p_welch"] = float("nan")
            r["significant_welch"] = False
            r["marker_welch"] = "desc"
    wcsv(os.path.join(a.out, "pde_vs_pde_comparisons.csv"), pvp,
         ["scenario","maneuver","metric","method_a","method_b","n_a","n_b",
          "raw_p_welch","corrected_p_welch","significant_welch","marker_welch","cohens_d"])
    nn = sum(1 for r in pvp if r["corrected_p_welch"] == r["corrected_p_welch"])
    print(f"  non-null corrected_p_welch: {nn}/{len(pvp)} = {nn/len(pvp):.3f}")

    # ---------------- (3) pde_diagnostics ------------------------------
    print("\n(3) pde_diagnostics")
    diag = []
    for r in sorted(runs, key=lambda x: (x["method"], x["scenario"], x["maneuver"], int(x["seed"]))):
        if r["method"] in (BASELINE, "rule_based") or r["loss"] is None or not r["loss"]["has_trace"]:
            continue
        L = r["loss"]
        diag.append({"method": r["method"], "scenario": r["scenario"], "maneuver": r["maneuver"],
                     "seed": r["seed"], "machine": r["machine"], "intent": r["intent"],
                     "n_iters": L["n_iters"], "window_iters": L["window_iters"],
                     "residual_early": round(L["residual_early"], 8),
                     "residual_late": round(L["residual_late"], 8),
                     "residual_ratio": round(L["residual_ratio"], 6) if L["residual_ratio"] == L["residual_ratio"] else "",
                     "distill_early": round(L["distill_early"], 8) if L["distill_early"] == L["distill_early"] else "",
                     "distill_late": round(L["distill_late"], 8) if L["distill_late"] == L["distill_late"] else ""})
    wcsv(os.path.join(a.out, "pde_diagnostics.csv"), diag,
         ["method","scenario","maneuver","seed","machine","intent","n_iters","window_iters",
          "residual_early","residual_late","residual_ratio","distill_early","distill_late"])
    summ = []
    for meth in sorted({d["method"] for d in diag}):
        g = [d for d in diag if d["method"] == meth]
        ratios = [d["residual_ratio"] for d in g if d["residual_ratio"] != ""]
        grew = [d for d in g if d["residual_ratio"] != "" and d["residual_ratio"] > 1.0]
        summ.append({"method": meth, "n_runs_with_trace": len(g),
                     "n_residual_grew": len(grew),
                     "rate_residual_grew": round(len(grew)/len(g), 4) if g else "",
                     "median_late_over_early": round(float(np.median(ratios)), 6) if ratios else "",
                     "median_residual_late": round(float(np.median([d["residual_late"] for d in g])), 8) if g else ""})
    wcsv(os.path.join(a.out, "pde_diagnostics_summary.csv"), summ,
         ["method","n_runs_with_trace","n_residual_grew","rate_residual_grew",
          "median_late_over_early","median_residual_late"])
    tot_pde = sum(1 for r in runs if r["method"] not in (BASELINE, "rule_based"))
    print(f"  runs WITH retained loss traces: {len(diag)} / {tot_pde} PDE-method runs")
    for s in summ:
        print(f"    {s['method']:14s} n={s['n_runs_with_trace']:4d} grew={s['n_residual_grew']:4d} "
              f"({s['rate_residual_grew']}) med_ratio={s['median_late_over_early']} "
              f"med_late={s['median_residual_late']}")

    # ---------------- (4) intent effect --------------------------------
    print("\n(4) tier1_intent_effect")
    cellm = {}
    for r in runs:
        if not r["metrics"]:
            continue
        cellm.setdefault((r["scenario"], r["maneuver"], r["method"]),
                         {"intent": [], "nointent": []})["intent" if r["intent"] else "nointent"].append(r)
    ie = []
    for (s, mv, meth), arms in sorted(cellm.items()):
        ni, nn_ = len(arms["intent"]), len(arms["nointent"])
        if ni < 2 or nn_ < 2:
            continue
        for met in INTENT_METRICS:
            iv = [x["metrics"].get(met, float("nan")) for x in arms["intent"]]
            nv = [x["metrics"].get(met, float("nan")) for x in arms["nointent"]]
            ilo, ihi = boot_ci(iv); nlo, nhi = boot_ci(nv)
            ie.append({"scenario": s, "maneuver": mv, "method": meth, "metric": met,
                       "n_intent": ni, "n_nointent": nn_,
                       "intent_mean": round(float(np.nanmean(iv)), 6),
                       "intent_ci_low": round(ilo, 6), "intent_ci_high": round(ihi, 6),
                       "nointent_mean": round(float(np.nanmean(nv)), 6),
                       "nointent_ci_low": round(nlo, 6), "nointent_ci_high": round(nhi, 6),
                       "delta": round(float(np.nanmean(iv))-float(np.nanmean(nv)), 6),
                       "cohens_d": round(cohens_d(iv, nv), 4),
                       "raw_p_welch": welch_p(iv, nv) if meth not in NO_TEST else float("nan")})
    for met in INTENT_METRICS:
        sub = [r for r in ie if r["metric"] == met and r["method"] not in NO_TEST]
        for r, cp in zip(sub, holm([r["raw_p_welch"] for r in sub])):
            r["corrected_p_welch"] = cp
            r["significant_welch"] = bool(cp == cp and cp < ALPHA)
            r["marker_welch"] = marker(cp)
    for r in ie:
        if "corrected_p_welch" not in r:
            r["corrected_p_welch"] = float("nan"); r["significant_welch"] = False
            r["marker_welch"] = "desc"
    wcsv(os.path.join(a.out, "tier1_intent_effect.csv"), ie,
         ["scenario","maneuver","method","metric","n_intent","n_nointent",
          "intent_mean","intent_ci_low","intent_ci_high",
          "nointent_mean","nointent_ci_low","nointent_ci_high",
          "delta","cohens_d","raw_p_welch","corrected_p_welch","significant_welch","marker_welch"])
    if ie:
        print(f"  n_intent distinct: {sorted({r['n_intent'] for r in ie})}")
        print(f"  n_nointent distinct: {sorted({r['n_nointent'] for r in ie})}")
        sg = [r for r in ie if r.get("marker_welch") in ("*","**","***")]
        print(f"  significant intent effects: {len(sg)}")
        for r in sg[:12]:
            print(f"    {r['scenario']}/{r['maneuver']} {r['method']:13s} {r['metric']:14s} "
                  f"int={r['intent_mean']:.3f} noint={r['nointent_mean']:.3f} "
                  f"d={r['cohens_d']:+.2f} p_adj={r['corrected_p_welch']:.4f} {r['marker_welch']}")

    # ---------------- run census ---------------------------------------
    print("\n=== Tier-1 run census ===")
    print(f"  total run dirs: {len(runs)}")
    print(f"  with run-level metrics: {sum(1 for r in runs if r['metrics'])}")
    src = {}
    for r in runs:
        if r["metrics"]:
            src[r["metrics"]["_source"]] = src.get(r["metrics"]["_source"], 0) + 1
    print(f"  metric source: {src}")
    print(f"  no-intent with metrics: {sum(1 for r in runs if r['metrics'] and not r['intent'])}")
    print(f"  intent with metrics:    {sum(1 for r in runs if r['metrics'] and r['intent'])}")


if __name__ == "__main__":
    main()
