"""Step 7E analysis: produce phase3F_A_step7E_status.json + plots.

Run AFTER both Step 7E training jobs complete cleanly.

Reads:
  - results/calibration/CAL_eikonal_aux_1a_s42/{train_eikonal_aux_1a_stem_right.csv, model_*.pt, meta_*.json}
  - results/calibration/CAL_eikonal_aux_2_dense_s42/{...}
  - verification/phase3F_A_step8_archive_s42/CAL_eikonal_aux_*_s42/* (for overlay)

Writes:
  - verification/phase3F_A_step7E_status.json
  - verification/phase3F_A_step7E_plots/*.png
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verification.phase3F_A_step7E_eval_rollout import evaluate_cell

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CELLS = [
    {"name": "1a_s42", "scenario": "1a", "seed": 42, "maneuver": "stem_right"},
    {"name": "2_dense_s42", "scenario": "2_dense", "seed": 42, "maneuver": "stem_right"},
]
CAL_ROOT = PROJECT_ROOT / "results/calibration"
ARCHIVE_ROOT = PROJECT_ROOT / "verification/phase3F_A_step8_archive_s42"
PLOTS_DIR = PROJECT_ROOT / "verification/phase3F_A_step7E_plots"
STATUS_OUT = PROJECT_ROOT / "verification/phase3F_A_step7E_status.json"

# Strict 7D margins (same set the 7C/7D verifications used).
TH_RHO_MAX = 0.4
TH_T_SUCC_1A_MAX = 3.0
TH_T_COLL_2DENSE_MIN = 90.0
TH_PEARSON_MIN = 0.75
TH_A_EIK_AGREE_MIN = 0.70
TH_KL_MAX = 0.4


def load_cell_csv(cell: dict, archive: bool = False) -> list[dict]:
    base = ARCHIVE_ROOT if archive else CAL_ROOT
    csv_path = base / f"CAL_eikonal_aux_{cell['scenario']}_s{cell['seed']}" / f"train_eikonal_aux_{cell['scenario']}_{cell['maneuver']}.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({k: (float(v) if v not in ("", None) else None) for k, v in row.items()})
            except Exception:
                continue
    return rows


def head_tail_block(rows: list[dict], col: str, frac: float = 0.10) -> tuple[float, float]:
    vals = [r[col] for r in rows if r.get(col) is not None]
    if len(vals) < 2:
        return float("nan"), float("nan")
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[:n])), float(np.mean(vals[-n:]))


def lambda_slope_rel_per_20k(rows: list[dict]) -> float:
    """Slope of λ over the last 20k steps, normalized by final λ. Falls back to last
    half of training when logging is too sparse (Step 7E logs ~3 rows at 100k total)."""
    if not rows:
        return float("nan")
    final_step = rows[-1]["step"]
    pairs = [(r["step"], r["alm_lambda"]) for r in rows
             if r.get("step") is not None and r.get("alm_lambda") is not None]
    if len(pairs) < 2:
        return float("nan")
    last = [p for p in pairs if p[0] >= max(0, final_step - 20_000)]
    if len(last) < 2:
        last = [p for p in pairs if p[0] >= max(0, final_step // 2)]
        if len(last) < 2:
            last = pairs[-2:]
    xs = np.array([t[0] for t in last]); ys = np.array([t[1] for t in last])
    if abs(ys[-1]) < 1e-12:
        return 0.0
    span = max(xs[-1] - xs[0], 1.0)
    slope = (ys[-1] - ys[0]) / span
    return float(slope * 20_000 / abs(ys[-1]))


def lambda_growth_factor_second_half(rows: list[dict]) -> float:
    """λ_final / λ at the earliest available POSITIVE λ datapoint after init.
    Robust to sparse logs and λ_init=0. Returns the multiplicative growth factor."""
    pairs = [(r["step"], r["alm_lambda"]) for r in rows
             if r.get("step") is not None and r.get("alm_lambda") is not None]
    if len(pairs) < 2:
        return float("nan")
    final_step, lam_final = pairs[-1]
    # Earliest positive λ row strictly before the final.
    pre = [p for p in pairs[:-1] if p[1] is not None and p[1] > 1e-9]
    if not pre or lam_final <= 0:
        return float("nan")
    base = pre[0][1]
    return float(lam_final / base)


def find_checkpoint(out_dir: Path, scenario: str, maneuver: str) -> Path | None:
    candidates = [
        out_dir / f"model_eikonal_aux_{scenario}_{maneuver}.pt",
        out_dir / f"model_eikonal_aux_{scenario}_{maneuver}_step100000.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    cands = sorted(out_dir.glob("model_*.pt"))
    return cands[-1] if cands else None


def per_cell_summary(cell: dict) -> dict:
    rows = load_cell_csv(cell, archive=False)
    if not rows:
        return {"cell": cell["name"], "error": "no_csv"}

    head_rho, tail_rho = head_tail_block(rows, "mean_rho", frac=0.10)
    rho_ratio = float(tail_rho / head_rho) if head_rho and head_rho > 1e-12 else float("nan")
    head_L_eik, tail_L_eik = head_tail_block(rows, "L_eik", frac=0.10)
    lam_slope = lambda_slope_rel_per_20k(rows)
    lam_growth_2h = lambda_growth_factor_second_half(rows)
    final = rows[-1]
    sigma_bc_final = float(final.get("sigma_bc", float("nan")))
    sigma_ground_final = float(final.get("sigma_ground", float("nan")))
    alm_lambda_final = float(final.get("alm_lambda", float("nan")))
    alm_mu_final = float(final.get("alm_mu", float("nan")))
    n_succ_buf = int(final.get("n_success_buffer", 0))
    n_coll_buf = int(final.get("n_collision_buffer", 0))
    n_int_buf = int(final.get("n_intermediate_buffer", 0))
    mean_T_succ = final.get("mean_T_succ")
    mean_T_coll = final.get("mean_T_coll")

    out_dir = CAL_ROOT / f"CAL_eikonal_aux_{cell['scenario']}_s{cell['seed']}"
    ckpt = find_checkpoint(out_dir, cell["scenario"], cell["maneuver"])
    diag = (evaluate_cell(scenario=cell["scenario"], seed=cell["seed"], maneuver=cell["maneuver"],
                          ckpt_path=str(ckpt), n_episodes=30, max_steps=500)
            if ckpt is not None else {"error": "no_checkpoint"})

    return {
        "cell": cell["name"],
        "scenario": cell["scenario"], "seed": cell["seed"], "maneuver": cell["maneuver"],
        "head_rho": head_rho, "tail_rho": tail_rho, "rho_ratio": rho_ratio,
        "L_eik_head": head_L_eik, "L_eik_tail": tail_L_eik,
        "T_succ_mean_csv": mean_T_succ if mean_T_succ is not None else None,
        "T_coll_mean_csv": mean_T_coll if mean_T_coll is not None else None,
        "alm_lambda_final": alm_lambda_final,
        "alm_mu_final": alm_mu_final,
        "alm_mu_hit_max": bool(alm_mu_final >= 9999.0),
        "lambda_slope_rel_per_20k": lam_slope,
        "lambda_growth_factor_second_half": lam_growth_2h,
        "sigma_bc_final": sigma_bc_final,
        "sigma_ground_final": sigma_ground_final,
        "n_success_buffer": n_succ_buf,
        "n_collision_buffer": n_coll_buf,
        "n_intermediate_buffer": n_int_buf,
        "trajectories_n_rows": len(rows),
        "checkpoint_path": str(ckpt) if ckpt else None,
        "diagnostics": diag,
    }


def evaluate_criteria(per_cell: list[dict]) -> dict:
    results = {}
    for c in per_cell:
        cell = c["cell"]
        diag = c.get("diagnostics", {})
        rho = c.get("rho_ratio")
        crit1 = (rho is not None and not np.isnan(rho) and rho < TH_RHO_MAX)

        if c["scenario"] == "1a":
            T_succ = c.get("T_succ_mean_csv")
            crit2 = (T_succ is not None and not np.isnan(T_succ) and T_succ < TH_T_SUCC_1A_MAX)
        else:
            T_coll = c.get("T_coll_mean_csv")
            crit2 = (T_coll is not None and not np.isnan(T_coll) and T_coll > TH_T_COLL_2DENSE_MIN)

        pearson = diag.get("pearson", float("nan"))
        a_eik = diag.get("a_eik_agreement", float("nan"))
        kl = diag.get("kl", float("nan"))
        crit3 = (isinstance(pearson, (int, float)) and not np.isnan(pearson)) and pearson > TH_PEARSON_MIN
        crit4 = (isinstance(a_eik, (int, float)) and not np.isnan(a_eik)) and a_eik > TH_A_EIK_AGREE_MIN
        crit5 = (isinstance(kl, (int, float)) and not np.isnan(kl)) and kl < TH_KL_MAX

        results[cell] = {
            "crit1_rho_ratio_lt_0p4": bool(crit1),
            "crit2_terminal_anchor": bool(crit2),
            "crit3_pearson_gt_0p75": bool(crit3),
            "crit4_a_eik_agree_gt_70": bool(crit4),
            "crit5_kl_lt_0p4": bool(crit5),
            "all_pass": bool(crit1 and crit2 and crit3 and crit4 and crit5),
        }
    return results


def make_plots(per_cell: list[dict]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        rows = load_cell_csv(cell, archive=False)
        if not rows: continue
        steps = np.array([r["step"] for r in rows], dtype=float)
        L_eik = np.array([r.get("L_eik") if r.get("L_eik") is not None else np.nan for r in rows])
        lam = np.array([r.get("alm_lambda") if r.get("alm_lambda") is not None else np.nan for r in rows])
        mu = np.array([r.get("alm_mu") if r.get("alm_mu") is not None else np.nan for r in rows])
        rho = np.array([r.get("mean_rho") if r.get("mean_rho") is not None else np.nan for r in rows])

        # residual ratio (using mean_rho for spec-aligned definition)
        head_rho = float(np.nanmean(rho[:max(1, len(rho)//10)]))
        plt.figure(figsize=(6,3.5))
        plt.plot(steps, rho / max(head_rho, 1e-12), lw=1.0, label='ρ / ρ(head)')
        plt.axhline(0.4, color='r', linestyle='--', alpha=0.7, label='strict 7D <0.4')
        plt.xlabel("step"); plt.ylabel("residual ratio (ρ_t / ρ_head)")
        plt.title(f"residual ratio — {cell['name']} (hidden_dim 512)")
        plt.grid(alpha=0.3); plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"residual_{cell['name']}.png", dpi=120); plt.close()

        # lambda + mu
        fig, ax1 = plt.subplots(figsize=(6,3.5))
        ax1.plot(steps, lam, color='C0', label='λ')
        ax1.set_xlabel("step"); ax1.set_ylabel("λ", color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.set_yscale('log' if np.nanmax(lam) > 1e3 else 'linear')
        ax2 = ax1.twinx()
        ax2.plot(steps, mu, color='C3', label='μ')
        ax2.set_ylabel("μ", color='C3'); ax2.tick_params(axis='y', labelcolor='C3')
        ax2.axhline(10000, color='C3', linestyle='--', alpha=0.5)
        plt.title(f"ALM λ, μ — {cell['name']} (hidden_dim 512)")
        plt.tight_layout(); plt.savefig(PLOTS_DIR / f"lambda_mu_{cell['name']}.png", dpi=120); plt.close()

        # L_eik
        plt.figure(figsize=(6,3.5))
        plt.plot(steps, L_eik, lw=1.0)
        plt.xlabel("step"); plt.ylabel("L_eik")
        plt.title(f"L_eik — {cell['name']} (hidden_dim 512)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"L_eik_{cell['name']}.png", dpi=120); plt.close()

        # Pearson proxy: T_succ countdown vs L_ground proxy (just plot mean_T_succ if available)
        T_succ = np.array([r.get("mean_T_succ") if r.get("mean_T_succ") is not None else np.nan for r in rows])
        plt.figure(figsize=(6,3.5))
        plt.plot(steps, T_succ, lw=1.0, label='mean_T_succ (≈ T_φ on success terminals)')
        plt.axhline(0.0, color='g', linestyle='--', alpha=0.5, label='ideal T_succ=0')
        plt.xlabel("step"); plt.ylabel("mean T_φ at s_succ")
        plt.title(f"T_φ at success terminals — {cell['name']}")
        plt.grid(alpha=0.3); plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"T_succ_{cell['name']}.png", dpi=120); plt.close()

    # Overlay 7E (hidden_dim 512) vs Step 8 archive (hidden_dim 256), capped to 100k.
    for cell in CELLS:
        r7 = load_cell_csv(cell, archive=False)
        r8 = load_cell_csv(cell, archive=True)
        if not (r7 and r8): continue
        cap = max(r["step"] for r in r7) if r7 else 100_000
        r8c = [r for r in r8 if r["step"] <= cap]

        s7 = np.array([r["step"] for r in r7]); s8 = np.array([r["step"] for r in r8c])
        rho7 = np.array([r.get("mean_rho") if r.get("mean_rho") is not None else np.nan for r in r7])
        rho8 = np.array([r.get("mean_rho") if r.get("mean_rho") is not None else np.nan for r in r8c])
        h7 = float(np.nanmean(rho7[:max(1, len(rho7)//10)]))
        h8 = float(np.nanmean(rho8[:max(1, len(rho8)//10)]))
        plt.figure(figsize=(6,3.5))
        plt.plot(s8, rho8 / max(h8, 1e-12), label='Step 8 hidden_dim 256', alpha=0.9)
        plt.plot(s7, rho7 / max(h7, 1e-12), label='Step 7E hidden_dim 512', alpha=0.9)
        plt.axhline(0.4, color='r', linestyle='--', alpha=0.7, label='strict 7D <0.4')
        plt.xlabel("step"); plt.ylabel("ρ_t / ρ_head")
        plt.title(f"7E vs Step 8 residual ratio — {cell['name']}")
        plt.grid(alpha=0.3); plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"overlay_residual_{cell['name']}.png", dpi=120); plt.close()

        l7 = np.array([r.get("alm_lambda") if r.get("alm_lambda") is not None else np.nan for r in r7])
        l8 = np.array([r.get("alm_lambda") if r.get("alm_lambda") is not None else np.nan for r in r8c])
        plt.figure(figsize=(6,3.5))
        plt.plot(s8, l8, label='Step 8 hidden_dim 256', alpha=0.9)
        plt.plot(s7, l7, label='Step 7E hidden_dim 512', alpha=0.9)
        plt.xlabel("step"); plt.ylabel("λ")
        plt.yscale('log' if (np.nanmax(l7) > 1e3 or np.nanmax(l8) > 1e3) else 'linear')
        plt.title(f"7E vs Step 8 λ — {cell['name']}")
        plt.grid(alpha=0.3); plt.legend(loc='best'); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"overlay_lambda_{cell['name']}.png", dpi=120); plt.close()


def main():
    per_cell = [per_cell_summary(c) for c in CELLS]
    crit = evaluate_criteria(per_cell)
    both_pass = all(crit[c["name"]]["all_pass"] for c in CELLS)
    outcome = "outcome_1_close_success_hidden_dim_512" if both_pass else "outcome_2_close_with_structural_limitation"

    out = {
        "phase": "3F-A", "step": "7E",
        "predecessor": "Step 8 (6-cell × 500k, all 4 Decision F conditions failed)",
        "config_change": "aux_hidden_dim 256 -> 512 (eikonal critic T_phi only). All other hyperparameters frozen from Step 7D/8.",
        "cells_run": [c["name"] for c in CELLS],
        "criteria_thresholds": {
            "rho_ratio_max": TH_RHO_MAX,
            "T_succ_1a_max": TH_T_SUCC_1A_MAX,
            "T_coll_2dense_min": TH_T_COLL_2DENSE_MIN,
            "pearson_min": TH_PEARSON_MIN,
            "a_eik_agreement_min": TH_A_EIK_AGREE_MIN,
            "kl_max": TH_KL_MAX,
        },
        "per_cell": per_cell,
        "criteria_results": crit,
        "outcome_classification": outcome,
        "outcome_text": ("All 5 criteria passed on both cells. Phase 3F-A closes successfully; Eikonal proceeds to Tier 1 with hidden_dim 512."
                         if both_pass else
                         "At least one criterion failed on at least one cell. Phase 3F-A closes with documented structural limitation; Eikonal proceeds to Tier 1 with hidden_dim 256 and paper-honest framing."),
    }
    STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[7E] wrote {STATUS_OUT}")

    make_plots(per_cell)
    print(f"[7E] plots in {PLOTS_DIR}")
    return out


if __name__ == "__main__":
    main()
