"""Tier-1 intent effect: intent vs nointent, one row per training run.

Reads Tier-1 run dirs (results/tier_1_full), computes per-run SR/CR/return from
each run's episode-level eval_metrics.csv, then per (scenario, maneuver, method)
compares intent vs nointent at the run level (Welch + Holm within metric family,
bootstrap 95% CI, Cohen's d).

Does NOT touch the frozen main Tier-1 tables — output is standalone.

Usage:
  python scripts/aggregate_intent_effect.py --repo . \
      --out results/tables/tier1_intent_effect.csv
"""
import argparse, csv, os, re
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

METHODS = ["soft_hjb_aux","hjb_aux","eikonal_aux","fusion_aux","cbf_aux","drppo","rule_based"]
SCENS = ["4_dense","3_dense","2_dense","1a","1b","1c","1d","2","3","4"]
MANS  = ["stem_right","stem_left","right_left","right_stem","left_right","left_stem"]
METRICS = ["SR","CR","mean_return","mean_ttc"]
NO_TEST = {"rule_based"}


def parse_scen_man(rest):
    for s in SCENS:
        if rest.startswith(s+"_"):
            m = rest[len(s)+1:]
            if m in MANS: return s, m
    return None, None


def run_stats(d):
    p = os.path.join(d, "eval_metrics.csv")
    if not os.path.isfile(p): return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty or "terminal_state" not in df.columns: return None
    n = len(df); ts = df["terminal_state"].str.lower()
    return {"SR": float((ts=="success").sum()/n), "CR": float((ts=="collision").sum()/n),
            "mean_return": float(df["return_total"].mean()) if "return_total" in df else float("nan"),
            "mean_ttc": float(df["mean_ttc"].mean()) if "mean_ttc" in df else float("nan")}


def welch_p(a, b):
    if sp_stats is None or len(a)<2 or len(b)<2: return float("nan")
    try: return float(sp_stats.ttest_ind(a, b, equal_var=False)[1])
    except Exception: return float("nan")


def holm(ps, alpha=0.05):
    arr=np.array(ps,float); n=len(arr); v=~np.isnan(arr); nv=int(v.sum())
    if nv==0: return [float("nan")]*n
    idx=np.where(v)[0]; vp=arr[v]; order=np.argsort(vp); corr=np.zeros(nv)
    for i,p in enumerate(vp[order]): corr[i]=min(p*(nv-i),1.0)
    for i in range(1,nv): corr[i]=max(corr[i],corr[i-1])
    out=[float("nan")]*n; un=np.argsort(order)
    for j,ii in enumerate(idx): out[ii]=float(corr[un[j]])
    return out


def ci95(vals):
    x=np.asarray([v for v in vals if v==v],float)
    if len(x)<2: return float("nan"),float("nan")
    rng=np.random.default_rng(42)
    b=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(2000)]
    return float(np.percentile(b,2.5)), float(np.percentile(b,97.5))


def cohens_d(x, y):
    x=np.asarray([v for v in x if v==v],float); y=np.asarray([v for v in y if v==v],float)
    if len(x)<2 or len(y)<2: return float("nan")
    pv=((len(x)-1)*np.var(x,ddof=1)+(len(y)-1)*np.var(y,ddof=1))/(len(x)+len(y)-2)
    s=np.sqrt(pv)
    return float((np.mean(x)-np.mean(y))/s) if s>0 else float("nan")


def sig(p):
    if p!=p: return "n/a"
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="results/tables/tier1_intent_effect.csv")
    a=ap.parse_args()
    if pd is None: print("pandas required"); return
    root=os.path.join(a.repo,"results/tier_1_full")

    # cell -> {"intent":[stats...], "nointent":[...]}
    cells={}
    for base in sorted(os.listdir(root)):
        d=os.path.join(root,base)
        if not os.path.isdir(d): continue
        method=next((m for m in sorted(METHODS,key=len,reverse=True) if f"_{m}_" in base),None)
        if method is None: continue
        arm = "intent" if ("_intent_" in base and "_nointent_" not in base) else \
              ("nointent" if "_nointent_" in base else None)
        if arm is None: continue
        scen,man=parse_scen_man(base.split(f"_{method}_")[0])
        if scen is None: continue
        st=run_stats(d)
        if st is None: continue
        cells.setdefault((scen,man,method),{"intent":[],"nointent":[]})[arm].append(st)

    rows=[]
    for (scen,man,method),arms in sorted(cells.items()):
        if len(arms["intent"])<2 or len(arms["nointent"])<2: continue
        for metric in METRICS:
            iv=[s[metric] for s in arms["intent"]]; nv=[s[metric] for s in arms["nointent"]]
            im=float(np.nanmean(iv)); nm=float(np.nanmean(nv))
            ilo,ihi=ci95(iv)
            rows.append({"scenario":scen,"maneuver":man,"method":method,"metric":metric,
                "n_intent":len(iv),"n_nointent":len(nv),
                "intent_mean":round(im,5),"intent_ci_low":round(ilo,5),"intent_ci_high":round(ihi,5),
                "nointent_mean":round(nm,5),"delta":round(im-nm,5),
                "cohens_d":round(cohens_d(iv,nv),4),
                "welch_p":welch_p(iv,nv) if method not in NO_TEST else float("nan")})

    # Holm within metric family (exclude rule_based from the family)
    for metric in METRICS:
        sub=[r for r in rows if r["metric"]==metric and r["method"] not in NO_TEST]
        for r,cp in zip(sub, holm([r["welch_p"] for r in sub])):
            r["welch_p_adj"]=cp; r["sig"]=sig(cp)
    for r in rows:
        if r["method"] in NO_TEST:
            r["welch_p_adj"]=float("nan"); r["sig"]="desc"
        r.setdefault("welch_p_adj",float("nan")); r.setdefault("sig","n/a")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields=["scenario","maneuver","method","metric","n_intent","n_nointent",
            "intent_mean","intent_ci_low","intent_ci_high","nointent_mean","delta",
            "cohens_d","welch_p","welch_p_adj","sig"]
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})
    print(f"Saved {a.out} ({len(rows)} rows, {len(cells)} cells)")
    sig_rows=[r for r in rows if r.get("sig") in ("*","**","***")]
    print(f"Significant intent effects: {len(sig_rows)}")
    for r in sig_rows[:15]:
        print(f"  {r['scenario']}/{r['maneuver']} {r['method']:12s} {r['metric']:11s} "
              f"intent={r['intent_mean']:.3f} nointent={r['nointent_mean']:.3f} "
              f"d={r['cohens_d']:+.2f} p_adj={r['welch_p_adj']:.4f} {r['sig']}")


if __name__=="__main__":
    main()
