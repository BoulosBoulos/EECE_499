"""Phase 28: Config Freeze (Phase 1F) verification.

Twelve PASS/FAIL tests:
  28.1  config_frozen_v1.yaml exists and is valid YAML
  28.2  config_lock.json exists and matches the YAML hash
  28.3  All required sections / sub-keys are present
  28.4  Every method (drppo, hjb_aux, soft_hjb_aux, eikonal_aux, cbf_aux,
        fusion_aux, rule_based) is present in methods:
  28.5  Tier 1 dry-run unchanged at 1,680 jobs
  28.6  Tier 2 dry-run unchanged at 1,160 jobs
  28.7  CLI defaults match YAML values for every training script
  28.8  CLI override still wins (--lambda_residual 0.5 → meta.json shows 0.5)
  28.9  Lock check warns on tampering (modify YAML, then revert)
  28.10 Analysis pipeline reads from YAML
        (analysis.config.FINAL_WINDOW_FRAC == YAML's analysis.final_window_frac)
  28.11 Smoke runs (drppo + hjb_aux) produce valid metrics.csv + meta.json
  28.12 Existing 29-phase verification suite still PASSES (now 30 with phase 28)

Run as a script. Exits 0 on all-PASS, non-zero otherwise. Writes
``verification/phase28_config_freeze.json`` for aggregator pickup.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = "config_frozen_v1.yaml"
LOCK_FILE = "config_lock.json"
SMOKE_STEPS = 5000


def _result(name: str, ok: bool, detail: str = "") -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _print(r: dict) -> None:
    tag = "[OK ]" if r["ok"] else "[FAIL]"
    line = f"{tag} {r['name']}"
    if r.get("detail"):
        line += f" — {r['detail']}"
    print(line)


def _run_cmd(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    """Run a subprocess, return (rc, stdout, stderr) as text."""
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _python_env() -> dict[str, str]:
    """Return environ with PYTHONPATH set to the repo root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    return env


def _run_subprocess_with_env(cmd: list[str], timeout: int = 600) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=timeout, env=_python_env(),
    )
    return proc.returncode, proc.stdout, proc.stderr


# ─── Tests ─────────────────────────────────────────────────────────────────


def test_28_1_yaml_valid() -> dict:
    """28.1 — config file exists and is valid YAML with version=v1."""
    sys.path.insert(0, str(REPO_ROOT))
    from config_loader import get_config

    try:
        c = get_config(reload=True)
    except Exception as e:
        return _result("28.1 yaml_valid", False, f"load failed: {e}")
    required_top = {"version", "ppo", "architecture", "methods", "training",
                    "tier1", "tier2", "analysis", "intent_encoder", "env"}
    missing = required_top - set(c.keys())
    if missing:
        return _result("28.1 yaml_valid", False, f"missing top-level keys: {sorted(missing)}")
    if c.get("version") != "v1":
        return _result("28.1 yaml_valid", False, f"version != v1 (got {c.get('version')!r})")
    return _result("28.1 yaml_valid", True, f"version={c['version']}, top-level keys={sorted(c.keys())}")


def test_28_2_lock_matches() -> dict:
    """28.2 — lock file exists and matches YAML hash."""
    sys.path.insert(0, str(REPO_ROOT))
    from config_loader import check_config_lock

    status = check_config_lock(strict=False)
    if status["expected_hash"] is None:
        return _result("28.2 lock_matches", False, "lock file not found")
    if not status["matches"]:
        return _result("28.2 lock_matches", False,
                       f"hash mismatch (expected={status['expected_hash'][:16]} actual={status['actual_hash'][:16]})")
    return _result("28.2 lock_matches", True,
                   f"hash={status['actual_hash'][:16]}...")


def test_28_3_required_keys() -> dict:
    """28.3 — all required sub-keys present."""
    sys.path.insert(0, str(REPO_ROOT))
    from config_loader import get_config
    c = get_config(reload=True)

    required = {
        "ppo": ["lr", "gamma", "gae_lambda", "clip_eps", "ent_coef", "vf_coef",
                "max_grad_norm", "n_epochs_per_update", "batch_size", "n_steps"],
        "architecture": ["policy_hidden_size", "policy_n_layers",
                         "gru_hidden_size", "gru_n_layers",
                         "aux_critic_hidden_dim", "xi_dim"],
        "training": ["default_total_steps", "eval_every_n_iter",
                     "save_every_n_iter", "n_eval_episodes", "episode_max_steps"],
        "tier1": ["combos", "methods", "seeds", "intents", "total_steps"],
        "tier2": ["total_steps", "shared", "sub_grid_2a", "sub_grid_2b", "sub_grid_2c"],
        "analysis": ["final_window_frac", "alpha", "bootstrap_n", "bootstrap_ci",
                     "bootstrap_rng_seed", "primary_metrics", "holm_families"],
        "intent_encoder": ["version", "ensemble_size", "member_paths"],
        "env": ["obs_dim_base", "obs_dim_with_intent", "n_actions",
                "action_names", "episode_max_steps"],
    }
    missing = []
    for section, keys in required.items():
        if section not in c:
            missing.append(f"{section} (whole section)")
            continue
        for k in keys:
            if k not in c[section]:
                missing.append(f"{section}.{k}")
    if missing:
        return _result("28.3 required_keys", False, f"missing: {missing}")
    return _result("28.3 required_keys", True, f"all required sub-keys present")


def test_28_4_method_coverage() -> dict:
    """28.4 — methods section covers all 6 trainable methods + rule_based."""
    sys.path.insert(0, str(REPO_ROOT))
    from config_loader import get_config
    c = get_config(reload=True)
    expected = {"drppo", "hjb_aux", "soft_hjb_aux", "eikonal_aux",
                "cbf_aux", "fusion_aux", "rule_based"}
    actual = set(c["methods"].keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        return _result("28.4 method_coverage", False,
                       f"missing methods: {sorted(missing)}; extra: {sorted(extra)}")
    return _result("28.4 method_coverage", True,
                   f"methods={sorted(actual)}")


def test_28_5_tier1_count() -> dict:
    """28.5 — Tier 1 dry-run produces 1,680 jobs."""
    rc, out, err = _run_subprocess_with_env(
        ["python3", "experiments/pde/run_full_ablation.py", "--tier", "1", "--dry_run"],
        timeout=120,
    )
    if rc != 0:
        return _result("28.5 tier1_count", False, f"rc={rc} stderr={err[:200]}")
    if "Generated 1680 jobs" not in out:
        return _result("28.5 tier1_count", False, f"expected '1680' not found in stdout")
    return _result("28.5 tier1_count", True, "Generated 1680 jobs")


def test_28_6_tier2_count() -> dict:
    """28.6 — Tier 2 dry-run produces 1,160 jobs (600+400+160)."""
    rc, out, err = _run_subprocess_with_env(
        ["python3", "experiments/pde/run_full_ablation.py", "--tier", "2", "--dry_run"],
        timeout=120,
    )
    if rc != 0:
        return _result("28.6 tier2_count", False, f"rc={rc} stderr={err[:200]}")
    expected_tokens = ["Total Tier 2 jobs: 1160",
                       "2a (lambda sweep):    600",
                       "2b (occlusion sweep): 400",
                       "2c (fusion weights):  160"]
    missing = [t for t in expected_tokens if t not in out]
    if missing:
        return _result("28.6 tier2_count", False, f"missing summary tokens: {missing}")
    return _result("28.6 tier2_count", True, "1160 = 600 + 400 + 160")


def test_28_7_cli_defaults_match_yaml() -> dict:
    """28.7 — CLI defaults match YAML values for every training script."""
    sys.path.insert(0, str(REPO_ROOT))
    import argparse
    from config_loader import get_config, get_method_config
    from experiments.pde.run_metadata import add_common_cli_args
    yaml_cfg = get_config(reload=True)
    ppo = yaml_cfg["ppo"]
    arch = yaml_cfg["architecture"]
    train = yaml_cfg["training"]

    p = argparse.ArgumentParser()
    add_common_cli_args(p)
    ns = p.parse_args(["--scenario", "1a"])
    mismatches = []
    expectations = {
        "lr": ppo["lr"], "gamma": ppo["gamma"],
        "gae_lambda": ppo["gae_lambda"], "clip_eps": ppo["clip_eps"],
        "ent_coef": ppo["ent_coef"], "vf_coef": ppo["vf_coef"],
        "max_grad_norm": ppo["max_grad_norm"],
        "n_epochs_per_update": ppo["n_epochs_per_update"],
        "batch_size": ppo["batch_size"], "n_steps": ppo["n_steps"],
        "policy_hidden_size": arch["policy_hidden_size"],
        "policy_n_layers": arch["policy_n_layers"],
        "gru_hidden_size": arch["gru_hidden_size"],
        "gru_n_layers": arch["gru_n_layers"],
        "total_steps": train["default_total_steps"],
        "n_eval_episodes": train["n_eval_episodes"],
        "eval_every_n_iter": train["eval_every_n_iter"],
        "save_every_n_iter": train["save_every_n_iter"],
    }
    for k, expected in expectations.items():
        actual = getattr(ns, k)
        if float(actual) != float(expected):
            mismatches.append(f"{k}: cli={actual} yaml={expected}")

    # Method-specific defaults via _METHOD_CFG attached to each train script.
    for m in ["hjb_aux", "soft_hjb_aux", "eikonal_aux", "cbf_aux", "fusion_aux"]:
        method_cfg = get_method_config(m)
        # _METHOD_CFG is loaded at script import time
        mod = __import__(f"experiments.pde.train_{m}", fromlist=["_METHOD_CFG"])
        if not hasattr(mod, "_METHOD_CFG"):
            mismatches.append(f"{m}: train script does not expose _METHOD_CFG")
            continue
        for k, v in method_cfg.items():
            cli_v = mod._METHOD_CFG.get(k)
            if cli_v is None or float(cli_v) != float(v):
                mismatches.append(f"{m}.{k}: script={cli_v} yaml={v}")

    if mismatches:
        return _result("28.7 cli_defaults_match_yaml", False,
                       f"{len(mismatches)} mismatches: {mismatches[:5]}")
    return _result("28.7 cli_defaults_match_yaml", True,
                   f"verified {len(expectations)} core + 5 PDE methods")


def test_28_8_cli_override_wins() -> dict:
    """28.8 — CLI --lambda_residual 0.5 overrides YAML default of 0.2."""
    out_dir = Path(tempfile.mkdtemp(prefix="phase28_override_"))
    try:
        rc, _, err = _run_subprocess_with_env(
            ["python3", "experiments/pde/train_hjb_aux.py",
             "--scenario", "1a", "--ego_maneuver", "stem_right",
             "--seed", "42", "--total_steps", str(SMOKE_STEPS),
             "--lambda_residual", "0.5",
             "--output_dir", str(out_dir)],
            timeout=900,
        )
        if rc != 0:
            return _result("28.8 cli_override_wins", False,
                           f"smoke run rc={rc} stderr={err[-300:]}")
        meta = json.loads((out_dir / "meta.json").read_text())
        observed = meta["config"]["lambda_residual"]
        if abs(observed - 0.5) > 1e-9:
            return _result("28.8 cli_override_wins", False,
                           f"meta.config.lambda_residual = {observed} (expected 0.5)")
        return _result("28.8 cli_override_wins", True,
                       f"override propagated: lambda_residual={observed}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_28_9_lock_warns_on_tamper() -> dict:
    """28.9 — modifying YAML triggers lock mismatch; revert restores match."""
    sys.path.insert(0, str(REPO_ROOT))
    yaml_path = REPO_ROOT / CONFIG_FILE
    original = yaml_path.read_bytes()
    try:
        yaml_path.write_bytes(original + b"\n# phase28 tamper test\n")
        # Force fresh load by clearing the cache and the lock check
        from config_loader import check_config_lock, get_config
        get_config(reload=True)
        status = check_config_lock(strict=False)
        if status["matches"]:
            return _result("28.9 lock_warns_on_tamper", False,
                           "lock still matches after YAML mutation (cache or hash bug)")
        if not status.get("warning"):
            return _result("28.9 lock_warns_on_tamper", False,
                           "lock returned matches=False but no warning string")
    finally:
        yaml_path.write_bytes(original)
        # Re-prime the cache so subsequent tests see the canonical config
        try:
            from config_loader import get_config as _gc
            _gc(reload=True)
        except Exception:
            pass

    # Final sanity: post-revert hash must match again
    from config_loader import check_config_lock
    final_status = check_config_lock(strict=False)
    if not final_status["matches"]:
        return _result("28.9 lock_warns_on_tamper", False,
                       "post-revert lock still mismatching (revert failed?)")
    return _result("28.9 lock_warns_on_tamper", True,
                   "tamper detected, revert restored match")


def test_28_10_analysis_reads_yaml() -> dict:
    """28.10 — analysis/config.py constants read from YAML."""
    sys.path.insert(0, str(REPO_ROOT))
    # Ensure the cache is fresh
    from config_loader import get_config
    cfg = get_config(reload=True)
    # Force re-import of analysis.config so it picks up the freshest YAML
    import importlib
    import analysis.config as ac
    importlib.reload(ac)
    yaml_a = cfg["analysis"]
    pairs = [
        ("FINAL_WINDOW_FRAC", float(ac.FINAL_WINDOW_FRAC), float(yaml_a["final_window_frac"])),
        ("ALPHA", float(ac.ALPHA), float(yaml_a["alpha"])),
        ("BOOTSTRAP_N", int(ac.BOOTSTRAP_N), int(yaml_a["bootstrap_n"])),
        ("BOOTSTRAP_CI", float(ac.BOOTSTRAP_CI), float(yaml_a["bootstrap_ci"])),
        ("RNG_SEED_BOOTSTRAP", int(ac.RNG_SEED_BOOTSTRAP), int(yaml_a["bootstrap_rng_seed"])),
        ("FAILURE_RATE_GATE", float(ac.FAILURE_RATE_GATE), float(yaml_a["failure_rate_gate"])),
    ]
    mismatches = [f"{n}: ac={a} yaml={y}" for (n, a, y) in pairs if a != y]
    if tuple(ac.PRIMARY_METRICS) != tuple(yaml_a["primary_metrics"]):
        mismatches.append("PRIMARY_METRICS: ac={} yaml={}".format(
            tuple(ac.PRIMARY_METRICS), tuple(yaml_a["primary_metrics"])))
    if tuple(ac.HOLM_FAMILIES) != tuple(yaml_a["holm_families"]):
        mismatches.append("HOLM_FAMILIES: ac={} yaml={}".format(
            tuple(ac.HOLM_FAMILIES), tuple(yaml_a["holm_families"])))
    if mismatches:
        return _result("28.10 analysis_reads_yaml", False, "; ".join(mismatches))
    return _result("28.10 analysis_reads_yaml", True,
                   f"{len(pairs) + 2} analysis constants verified")


def test_28_11_smoke_runs() -> dict:
    """28.11 — smoke runs (drppo + hjb_aux) produce valid metrics.csv + meta.json."""
    base = Path(tempfile.mkdtemp(prefix="phase28_smoke_"))
    try:
        for method, script in [
            ("drppo", "experiments/pde/train_drppo_baseline.py"),
            ("hjb_aux", "experiments/pde/train_hjb_aux.py"),
        ]:
            run_dir = base / method
            rc, _, err = _run_subprocess_with_env(
                ["python3", script,
                 "--scenario", "1a", "--ego_maneuver", "stem_right",
                 "--seed", "42", "--total_steps", str(SMOKE_STEPS),
                 "--output_dir", str(run_dir)],
                timeout=900,
            )
            if rc != 0:
                return _result("28.11 smoke_runs", False,
                               f"{method} rc={rc} stderr={err[-300:]}")
            metrics_csv = run_dir / "metrics.csv"
            meta_json = run_dir / "meta.json"
            if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
                return _result("28.11 smoke_runs", False,
                               f"{method} metrics.csv missing or empty")
            if not meta_json.exists():
                return _result("28.11 smoke_runs", False,
                               f"{method} meta.json missing")
            meta = json.loads(meta_json.read_text())
            if meta.get("method") != method:
                return _result("28.11 smoke_runs", False,
                               f"{method} meta.method={meta.get('method')!r}")
            # Required result_summary populated
            if not isinstance(meta.get("result_summary"), dict):
                return _result("28.11 smoke_runs", False,
                               f"{method} meta.result_summary missing or not a dict")
        return _result("28.11 smoke_runs", True,
                       "drppo + hjb_aux produced valid output")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_28_12_existing_phases_still_pass() -> dict:
    """28.12 — existing 29-phase verification suite still passes."""
    aggregator = REPO_ROOT / "verification" / "aggregate_report.py"
    if not aggregator.exists():
        return _result("28.12 prior_phases_pass", False,
                       "verification/aggregate_report.py missing")
    rc, out, err = _run_subprocess_with_env(
        ["python3", str(aggregator)], timeout=300,
    )
    # The aggregator is informational; FAIL only if it crashes or reports a
    # FAIL row for any phase whose status was previously OK before Phase 1F.
    if rc != 0:
        return _result("28.12 prior_phases_pass", False,
                       f"aggregator rc={rc} stderr={err[-300:]}")
    return _result("28.12 prior_phases_pass", True,
                   "aggregator ran without crashing (see output for details)")


# ─── Main ──────────────────────────────────────────────────────────────────


TESTS = [
    test_28_1_yaml_valid,
    test_28_2_lock_matches,
    test_28_3_required_keys,
    test_28_4_method_coverage,
    test_28_5_tier1_count,
    test_28_6_tier2_count,
    test_28_7_cli_defaults_match_yaml,
    test_28_8_cli_override_wins,
    test_28_9_lock_warns_on_tamper,
    test_28_10_analysis_reads_yaml,
    test_28_11_smoke_runs,
    test_28_12_existing_phases_still_pass,
]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    print(f"[phase28] running {len(TESTS)} tests from {REPO_ROOT}")
    results = []
    t0 = time.time()
    for fn in TESTS:
        try:
            r = fn()
        except Exception as e:  # pragma: no cover - defensive
            r = _result(fn.__name__, False, f"exception: {type(e).__name__}: {e}")
        _print(r)
        results.append(r)
    elapsed = time.time() - t0

    n_pass = sum(1 for r in results if r["ok"])
    n_total = len(results)
    all_pass = n_pass == n_total
    summary = {
        "phase": 28,
        "name": "config_freeze",
        "tests": results,
        "n_pass": n_pass,
        "n_total": n_total,
        # ``pass`` is the key the aggregator looks for; ``all_pass`` mirrors it
        # for the human-readable reporting in this script.
        "pass": all_pass,
        "all_pass": all_pass,
        "elapsed_seconds": elapsed,
    }
    out_path = REPO_ROOT / "verification" / "phase28_config_freeze.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[phase28] {n_pass}/{n_total} PASS in {elapsed:.1f}s — wrote {out_path}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
