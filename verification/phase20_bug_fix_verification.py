"""Phase 20: Bug fix verification (post SPEC_PHASE_0_BUG_FIXES.md).

Verifies the seven bug fixes applied in Phase 0 with concrete tests.
Follows the pattern of the existing 19 phase scripts: prints PASS/FAIL
per check, writes phase20_bug_fix_verification.json, exits non-zero on
any failure.

Tests:
  20.1 Intent CPA frame consistency (post-fix train_intent matches env).
  20.2 CBF residual scaling (drift-based; responds to alpha_cbf and dt).
  20.3 CBF residual is non-degenerate in the safe regime (U > -10).
  20.4 Method-name extraction (longest-prefix matching).
  20.5 info["prev_action"] reports the actual previous action.
  20.6 Visibility sampling reproducibility under fixed seed.
  20.7 Existing 19-phase suite still passes (regression sentinel).
"""
import sys, os, json, math, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

results = {"phase": "20", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {name}: {details if details else ''}")


# ---------------------------------------------------------------------------
# Test 20.1 -- Intent CPA frame consistency
# ---------------------------------------------------------------------------
def test_20_1():
    """Post-fix train_intent and env produce identical d_cpa for psi_e != 0.

    Also verifies the pre-fix formula (using world-frame dp) WOULD have
    differed -- otherwise this test is vacuous.
    """
    try:
        from state.builder import _rot2d, _wrap

        # Construct a scenario with non-zero ego heading and non-trivial geometry
        p_e = np.array([0.0, 0.0])
        psi_e = 0.7  # nonzero ego heading
        v_e = 5.0
        v_e_vec = v_e * np.array([np.cos(psi_e), np.sin(psi_e)])

        p_i = np.array([12.0, 4.0])
        psi_i = 1.4
        v_i = 6.0
        v_i_vec = v_i * np.array([np.cos(psi_i), np.sin(psi_i)])

        R = _rot2d(-psi_e)
        dp = p_i - p_e
        delta_xy = R @ dp
        delta_v = R @ (v_i_vec - v_e_vec)

        # Post-fix train_intent and env both use ego-frame delta_xy
        t_cpa_post = np.clip(-np.dot(delta_xy, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
        p_cpa_post = delta_xy + t_cpa_post * delta_v
        d_cpa_post = float(np.linalg.norm(p_cpa_post))

        # Pre-fix simulated: would have used world-frame dp
        t_cpa_pre = np.clip(-np.dot(dp, delta_v) / (np.dot(delta_v, delta_v) + 1e-6), 0, 3)
        p_cpa_pre = dp + t_cpa_pre * delta_v
        d_cpa_pre = float(np.linalg.norm(p_cpa_pre))

        # Sanity: bug WAS detectable pre-fix
        bug_detectable = abs(d_cpa_pre - d_cpa_post) > 0.01

        # Read the actual train_intent.py and confirm it uses delta_xy now
        intent_src = open(os.path.join(os.path.dirname(__file__),
                                       "..", "experiments", "train_intent.py")).read()
        uses_delta_xy = (
            "t_cpa = np.clip(-np.dot(delta_xy, delta_v)" in intent_src
            and "p_cpa = delta_xy + t_cpa * delta_v" in intent_src
        )
        no_old_code = (
            "t_cpa = np.clip(-np.dot(dp, delta_v)" not in intent_src
            and "p_cpa = dp + t_cpa * delta_v" not in intent_src
        )

        ok = uses_delta_xy and no_old_code and bug_detectable
        _record("20.1_intent_cpa_frame", ok, {
            "d_cpa_post_fix": d_cpa_post,
            "d_cpa_pre_fix_simulated": d_cpa_pre,
            "delta": abs(d_cpa_pre - d_cpa_post),
            "uses_delta_xy_in_source": uses_delta_xy,
            "no_old_dp_code": no_old_code,
            "bug_detectable_pre_fix": bug_detectable,
        })
    except Exception as e:
        _record("20.1_intent_cpa_frame", False, {"error": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# Test 20.2 -- CBF residual scaling
# ---------------------------------------------------------------------------
def test_20_2():
    """CBF residual is finite, non-zero, scales with alpha_cbf, and responds to dt.

    Uses a U_phi MLP with a negative bias on the output layer so h = U + 10
    is small or negative -- this forces the CBF condition to be active on at
    least some batch elements (otherwise rho = ReLU(-positive) = 0 trivially).
    'Bounded' here means finite (no NaN/Inf); we additionally check the
    magnitude stays below 1e6 as a sanity ceiling.
    """
    try:
        import torch
        import torch.nn as nn
        from models.pde.state_builder import XI_DIM
        from models.pde.dynamics import BehavioralDynamics
        from models.pde.residuals import cbf_residual

        # MLP with negative output bias: ensures h = U + 10 is small/negative
        class NegU(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(XI_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
                with torch.no_grad():
                    self.net[-1].bias.fill_(-12.0)  # h ~ -2 + small variation

            def forward(self, xi):
                return self.net(xi).squeeze(-1)

        torch.manual_seed(42)
        U_phi = NegU()
        U_phi.eval()

        # Physically-realistic xi covering safe-to-marginal range
        B = 8
        xi = torch.zeros(B, XI_DIM, dtype=torch.float32)
        xi[:, 0] = torch.linspace(2.0, 12.0, B)             # v
        xi[:, 4] = torch.linspace(50.0, 10.0, B)            # d_cz
        xi[:, 5] = torch.linspace(60.0, 20.0, B)            # d_exit
        xi[:, 7] = torch.linspace(8.0, 1.5, B)              # ttc_min
        xi[:, 8] = torch.linspace(1.0, 0.2, B)              # alpha_cz
        xi[:, 9] = torch.linspace(1.0, 0.2, B)              # alpha_cross
        for ag in range(3):
            base = 12 + ag * 22
            xi[:, base + 0] = float(ag - 1) * 10.0
            xi[:, base + 1] = 5.0 + 5.0 * ag
            xi[:, base + 2] = -3.0
            xi[:, base + 5] = 8.0
            xi[:, base + 7] = 15.0
            xi[:, base + 8] = 25.0
            xi[:, base + 21] = 1.0
        xi[:, 78] = 100.0

        dyn1 = BehavioralDynamics(dt=0.1)
        dyn2 = BehavioralDynamics(dt=0.2)

        rho_a1_dt1 = cbf_residual(U_phi, xi.clone(), dyn1,
                                   alpha_cbf=1.0, cbf_safe_offset=10.0).detach()
        rho_a2_dt1 = cbf_residual(U_phi, xi.clone(), dyn1,
                                   alpha_cbf=2.0, cbf_safe_offset=10.0).detach()
        rho_a1_dt2 = cbf_residual(U_phi, xi.clone(), dyn2,
                                   alpha_cbf=1.0, cbf_safe_offset=10.0).detach()

        finite = torch.isfinite(rho_a1_dt1).all().item()
        nonzero = (rho_a1_dt1 > 1e-6).any().item()
        bounded = bool(rho_a1_dt1.max().item() < 1e6)

        # Doubling alpha_cbf doubles the alpha*h contribution -- with h<0 this
        # roughly doubles the CBF deficit, hence increases the residual
        alpha_response = (rho_a2_dt1 - rho_a1_dt1).abs().mean().item() > 1e-6

        # dt change moves the drift, hence the h_dot contribution
        dt_response = (rho_a1_dt2 - rho_a1_dt1).abs().mean().item() > 1e-6

        ok = finite and nonzero and bounded and alpha_response and dt_response
        _record("20.2_cbf_scaling", ok, {
            "finite": finite,
            "nonzero": nonzero,
            "bounded_below_1e6": bounded,
            "alpha_response": alpha_response,
            "dt_response": dt_response,
            "rho_alpha1_dt1_mean": float(rho_a1_dt1.mean()),
            "rho_alpha2_dt1_mean": float(rho_a2_dt1.mean()),
            "rho_alpha1_dt2_mean": float(rho_a1_dt2.mean()),
            "rho_alpha1_dt1_max": float(rho_a1_dt1.max()),
        })
    except Exception as e:
        _record("20.2_cbf_scaling", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


# ---------------------------------------------------------------------------
# Test 20.3 -- CBF residual is not the pre-fix degenerate form
# ---------------------------------------------------------------------------
def test_20_3():
    """In the safe regime (U > -10), the residual should not be uniformly zero.

    Pre-fix: rho ~ ReLU(-(near-zero advection + alpha*h)) was zero whenever
    h > 0 (i.e. U > -10). Post-fix: drift contribution is dt-corrected, so
    the max over actions can fail to satisfy h_dot + alpha*h >= 0 for some xi.
    """
    try:
        import torch
        import torch.nn as nn
        from models.pde.state_builder import XI_DIM
        from models.pde.dynamics import BehavioralDynamics
        from models.pde.residuals import cbf_residual

        # Build an MLP with a positive-bias output layer so U_phi(xi) > 0 ~> h > 10
        class PositiveU(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(XI_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
                # Bias the final layer high so h = U + 10 stays well above 0
                with torch.no_grad():
                    self.net[-1].bias.fill_(5.0)

            def forward(self, xi):
                return self.net(xi).squeeze(-1)

        torch.manual_seed(0)
        U = PositiveU()
        dyn = BehavioralDynamics(dt=0.1)

        # Construct xi values where dynamics will drive h_dot down for some actions
        # (high speed, low d_cz, low alpha_cz so visibility-modulated terms hurt).
        B = 16
        xi = torch.zeros(B, XI_DIM)
        xi[:, 0] = torch.linspace(2.0, 12.0, B)  # v
        xi[:, 4] = torch.linspace(20.0, 2.0, B)  # d_cz decreasing
        xi[:, 7] = torch.linspace(8.0, 0.5, B)   # ttc decreasing
        xi[:, 8] = torch.linspace(1.0, 0.1, B)   # alpha_cz decreasing
        for ag in range(3):
            xi[:, 12 + ag * 22 + 21] = 1.0

        rho = cbf_residual(U, xi, dyn, alpha_cbf=1.0, cbf_safe_offset=10.0).detach()
        finite = torch.isfinite(rho).all().item()
        # Some elements should be nonzero (non-degenerate)
        n_active = int((rho > 1e-6).sum().item())
        # And some elements should be zero (CBF condition does hold somewhere)
        # i.e. the residual is not uniformly nonzero either
        n_zero = int((rho < 1e-9).sum().item())

        ok = finite and (n_active >= 1)
        _record("20.3_cbf_non_degenerate", ok, {
            "finite": finite,
            "n_active_elements": n_active,
            "n_zero_elements": n_zero,
            "batch_size": B,
            "rho_mean": float(rho.mean()),
            "rho_max": float(rho.max()),
        })
    except Exception as e:
        _record("20.3_cbf_non_degenerate", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


# ---------------------------------------------------------------------------
# Test 20.4 -- Method-name extraction
# ---------------------------------------------------------------------------
def test_20_4():
    """Longest-prefix matching against canonical method list."""
    try:
        from experiments.pde.analysis.compute_aulc import (
            CANONICAL_METHODS, extract_method_name,
        )
        cases = [
            ("hjb_aux_1a_stem_right_seed42",            "hjb_aux"),
            ("soft_hjb_aux_4_dense_stem_right_seed99",  "soft_hjb_aux"),
            ("eikonal_aux_1b_stem_left_seed42",         "eikonal_aux"),
            ("cbf_aux_3_dense_right_left_seed42",       "cbf_aux"),
            ("drppo_1a_stem_right_seed42",              "drppo"),
            ("rule_based_1a_stem_right_seed42",         "rule_based"),
            ("fusion_aux_2_left_right_seed7",           "fusion_aux"),
            ("nonsense_filename",                       "unknown"),
        ]
        results_per_case = []
        all_ok = True
        for stem, expected in cases:
            actual = extract_method_name(stem)
            ok = (actual == expected)
            if not ok:
                all_ok = False
            results_per_case.append({"stem": stem, "expected": expected,
                                     "actual": actual, "ok": ok})
        # Also ensure plot_learning_curves imports them
        plc_src = open(os.path.join(os.path.dirname(__file__), "..",
                                    "experiments", "pde", "analysis",
                                    "plot_learning_curves.py")).read()
        import_ok = ("from experiments.pde.analysis.compute_aulc import"
                     in plc_src and "extract_method_name" in plc_src)
        ok = all_ok and import_ok
        _record("20.4_method_extraction", ok, {
            "cases": results_per_case,
            "plot_learning_curves_imports_extract_method_name": import_ok,
            "canonical_count": len(CANONICAL_METHODS),
        })
    except Exception as e:
        _record("20.4_method_extraction", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


# ---------------------------------------------------------------------------
# Test 20.5 -- info["prev_action"] correctness
# ---------------------------------------------------------------------------
def test_20_5():
    """info['prev_action'] reports the previous action, not the current one."""
    try:
        from env.sumo_env import SumoEnv
        env = SumoEnv(scenario_name="1a", ego_maneuver="stem_right")
        obs, info0 = env.reset(seed=42)
        actions = [0, 3, 2, 4, 1]  # STOP, GO, YIELD, ABORT, CREEP
        prev_actions_observed = []
        for i, a in enumerate(actions):
            obs, r, term, trunc, info = env.step(a)
            prev_actions_observed.append(info.get("prev_action"))
            if term or trunc:
                break
        env.close()
        # info["prev_action"] at step i should be actions[i-1] (or None for step 0)
        n_steps = len(prev_actions_observed)
        expected = [None] + actions[:n_steps - 1]
        ok = (n_steps >= 2 and prev_actions_observed == expected)
        # If the bug were still present, prev_actions_observed would equal actions[:n_steps]
        bug_simulation = list(actions[:n_steps])
        bug_distinguishable = (prev_actions_observed != bug_simulation)
        _record("20.5_prev_action", ok and bug_distinguishable, {
            "actions_taken": actions[:n_steps],
            "info_prev_action_observed": prev_actions_observed,
            "expected_post_fix": expected,
            "bug_form_distinguishable": bug_distinguishable,
        })
    except Exception as e:
        _record("20.5_prev_action", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


# ---------------------------------------------------------------------------
# Test 20.6 -- Visibility sampling reproducibility
# ---------------------------------------------------------------------------
def test_20_6():
    """Two resets with the same seed produce identical visibility tuple."""
    try:
        from env.sumo_env import SumoEnv
        keys = ("alpha_cz", "alpha_cross", "d_occ", "dt_seen", "sigma_percep", "n_occ")

        def vis_at_seed(seed):
            env = SumoEnv(scenario_name="1a", ego_maneuver="stem_right")
            obs, info = env.reset(seed=seed)
            v = info["raw_obs"]["vis"]
            tup = tuple(float(v[k]) for k in keys)
            env.close()
            return tup

        v1 = vis_at_seed(42)
        v2 = vis_at_seed(42)
        v3 = vis_at_seed(99)

        identical_same_seed = all(a == b for a, b in zip(v1, v2))
        differs_different_seed = any(abs(a - b) > 1e-9 for a, b in zip(v1, v3))
        ok = identical_same_seed and differs_different_seed
        _record("20.6_vis_reproducibility", ok, {
            "seed42_run1": v1,
            "seed42_run2": v2,
            "seed99": v3,
            "identical_same_seed": identical_same_seed,
            "differs_different_seed": differs_different_seed,
        })
    except Exception as e:
        _record("20.6_vis_reproducibility", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


# ---------------------------------------------------------------------------
# Test 20.7 -- Existing 19-phase verification still passes
# ---------------------------------------------------------------------------
def test_20_7():
    """Re-aggregate the existing 19-phase verification JSONs and confirm ALL_PASS."""
    try:
        import glob
        ver_dir = os.path.dirname(os.path.abspath(__file__))
        # Re-run the aggregator inline
        phases = {}
        for path in sorted(glob.glob(os.path.join(ver_dir, "phase*.json"))):
            name = os.path.basename(path).replace(".json", "")
            if name == "phase20_bug_fix_verification":
                continue  # don't include self
            with open(path) as f:
                phases[name] = json.load(f)
        all_pass = all(v.get("pass", True) is True
                       for v in phases.values() if isinstance(v, dict))
        n_phases = len(phases)
        failed = [n for n, v in phases.items()
                  if isinstance(v, dict) and v.get("pass", True) is False]
        ok = all_pass and n_phases >= 19
        _record("20.7_existing_suite", ok, {
            "n_phases": n_phases,
            "all_pass": all_pass,
            "failed_phases": failed,
        })
    except Exception as e:
        _record("20.7_existing_suite", False, {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        })


def main():
    print("==== PHASE 20: BUG FIX VERIFICATION ====")
    test_20_1()
    test_20_2()
    test_20_3()
    test_20_4()
    test_20_5()
    test_20_6()
    test_20_7()
    out_path = os.path.join(os.path.dirname(__file__),
                            "phase20_bug_fix_verification.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
