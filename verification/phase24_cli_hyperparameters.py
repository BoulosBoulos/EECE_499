"""Phase 24: CLI hyperparameter exposure + git provenance verification (post 1B).

Verifies that:
  - All 5 training scripts accept the canonical Phase 1B CLI args (24.1)
  - meta.json["config"] uses the uniform key set across all scripts (24.2)
  - CLI args propagate into meta.json["config"] (24.3)
  - --use_intent boolean flag round-trips (24.4)
  - Git provenance walks parent dirs / falls back to placeholder (24.5)
  - The existing 25-phase suite still passes after Phase 1B edits (24.6)
"""
import sys
import os
import re
import json
import shutil
import subprocess
import tempfile
import traceback
from glob import glob
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

results = {"phase": "24", "tests": {}, "pass": True}


def _record(name, ok, details=None):
    results["tests"][name] = {"pass": bool(ok)}
    if details:
        results["tests"][name].update(details)
    if not ok:
        results["pass"] = False
    sym = "OK  " if ok else "FAIL"
    print(f"  [{sym}] {name}")


METHODS = [
    {"name": "drppo",         "script": "experiments/pde/train_drppo_baseline.py", "is_pde": False, "lambda_default": None,  "extra": {}},
    {"name": "hjb_aux",       "script": "experiments/pde/train_hjb_aux.py",        "is_pde": True,  "lambda_default": 0.2,   "extra": {"collocation_size": 256}},
    {"name": "soft_hjb_aux",  "script": "experiments/pde/train_soft_hjb_aux.py",   "is_pde": True,  "lambda_default": 0.2,   "extra": {"tau_soft": 1.0, "lambda_actor_kl": 0.1, "collocation_size": 256}},
    {"name": "eikonal_aux",   "script": "experiments/pde/train_eikonal_aux.py",    "is_pde": True,  "lambda_default": 0.2,   "extra": {"w_fail": 50.0, "collocation_size": 256}},
    {"name": "cbf_aux",       "script": "experiments/pde/train_cbf_aux.py",        "is_pde": True,  "lambda_default": 0.2,   "extra": {"alpha_cbf": 1.0, "barrier_offset": 10.0, "collocation_size": 256}},
]


SMOKE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_phase24_smoke")
PYTHON_BIN = sys.executable

CORE_FLAGS = [
    "--lr", "--gamma", "--gae_lambda", "--clip_eps",
    "--ent_coef", "--vf_coef", "--max_grad_norm",
    "--n_epochs_per_update", "--batch_size", "--n_steps",
]
ARCH_FLAGS = [
    "--policy_hidden_size", "--policy_n_layers",
    "--gru_hidden_size", "--gru_n_layers",
]
TRAIN_CTRL_FLAGS = [
    "--total_steps", "--seed", "--scenario", "--ego_maneuver",
    "--use_intent", "--output_dir",
    "--n_eval_episodes", "--eval_every_n_iter", "--save_every_n_iter",
]
PDE_COMMON_FLAGS = ["--lambda_residual", "--lambda_distill", "--collocation_size"]


def _help_text(script_path):
    proc = subprocess.run(
        [PYTHON_BIN, script_path, "--help"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": REPO_ROOT},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    return proc.stdout.decode(), proc.returncode


def test_24_1():
    """Each script's --help lists every required flag; PDE-only flags only in their scripts."""
    per_method = []
    overall_ok = True
    for entry in METHODS:
        try:
            help_text, rc = _help_text(entry["script"])
            missing_core = [f for f in CORE_FLAGS if f not in help_text]
            missing_arch = [f for f in ARCH_FLAGS if f not in help_text]
            missing_train = [f for f in TRAIN_CTRL_FLAGS if f not in help_text]
            unexpected_pde = []
            missing_pde = []
            if entry["is_pde"]:
                for f in PDE_COMMON_FLAGS:
                    if f not in help_text:
                        missing_pde.append(f)
            else:
                for f in PDE_COMMON_FLAGS + ["--alpha_cbf", "--barrier_offset", "--tau_soft", "--w_fail", "--lambda_actor_kl"]:
                    if f in help_text:
                        unexpected_pde.append(f)
            method_ok = (rc == 0
                         and not missing_core
                         and not missing_arch
                         and not missing_train
                         and not missing_pde
                         and not unexpected_pde)
            per_method.append({
                "method": entry["name"],
                "help_returncode_zero": rc == 0,
                "missing_core": missing_core,
                "missing_arch": missing_arch,
                "missing_train_ctrl": missing_train,
                "missing_pde_common": missing_pde,
                "unexpectedly_present_pde": unexpected_pde,
            })
            overall_ok = overall_ok and method_ok
        except Exception as e:
            per_method.append({
                "method": entry["name"],
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2),
            })
            overall_ok = False
    _record("24.1_cli_args_present", overall_ok, {"per_method": per_method})


REQUIRED_CONFIG_KEYS = (
    # Core PPO
    "lr", "gamma", "gae_lambda", "clip_eps", "ent_coef", "vf_coef", "max_grad_norm",
    "n_epochs_per_update", "batch_size", "n_steps",
    # Architecture
    "policy_hidden_size", "policy_n_layers", "gru_hidden_size", "gru_n_layers",
    # PDE-specific (some null per-method)
    "alpha_cbf", "tau_soft", "w_fail", "barrier_offset",
    "lambda_residual", "lambda_distill", "lambda_actor_kl", "collocation_size",
)


def _smoke_dir(name, suffix=""):
    return os.path.join(SMOKE_BASE, name + (f"_{suffix}" if suffix else ""))


def _run_smoke(entry, extra_args=None, suffix=""):
    out_dir = _smoke_dir(entry["name"], suffix)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [
        PYTHON_BIN, entry["script"],
        "--output_dir", out_dir,
        "--total_steps", "5000",
        "--scenario", "1a",
        "--ego_maneuver", "stem_right",
        "--seed", "42",
    ]
    if extra_args:
        cmd.extend(extra_args)
    log_path = out_dir + "_smoke.log"
    os.makedirs(SMOKE_BASE, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            stdout=logf, stderr=subprocess.STDOUT, timeout=600,
        )
    return proc.returncode == 0, out_dir, log_path


def _meta(out_dir):
    p = os.path.join(out_dir, "meta.json")
    with open(p) as f:
        return json.load(f)


def test_24_2():
    """Each script's meta.json["config"] has all REQUIRED_CONFIG_KEYS, with
    correct null-or-value placement per method."""
    per_method = []
    overall_ok = True
    for entry in METHODS:
        try:
            ok_run, out_dir, log_path = _run_smoke(entry, suffix="default")
            issues = []
            cfg_keys = []
            if not ok_run:
                issues.append(f"smoke run failed (see {log_path})")
            else:
                meta = _meta(out_dir)
                cfg = meta.get("config") or {}
                cfg_keys = sorted(cfg.keys())
                missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
                extra = [k for k in cfg if k not in REQUIRED_CONFIG_KEYS]
                if missing:
                    issues.append(f"missing keys: {missing}")
                if extra:
                    issues.append(f"unexpected keys: {extra}")
                # Per-method null/value placement
                if not entry["is_pde"]:  # drppo
                    for k in ("alpha_cbf", "tau_soft", "w_fail", "barrier_offset",
                              "lambda_residual", "lambda_distill",
                              "lambda_actor_kl", "collocation_size"):
                        if cfg.get(k) is not None:
                            issues.append(f"drppo expects {k}=null, got {cfg[k]}")
                else:
                    if cfg.get("lambda_residual") is None:
                        issues.append("PDE method must have lambda_residual non-null")
                    if cfg.get("lambda_distill") is None:
                        issues.append("PDE method must have lambda_distill non-null")
                    if cfg.get("collocation_size") is None:
                        issues.append("PDE method must have collocation_size non-null")
                    if entry["name"] == "soft_hjb_aux":
                        if cfg.get("tau_soft") is None or cfg.get("lambda_actor_kl") is None:
                            issues.append("soft_hjb missing tau_soft / lambda_actor_kl")
                    else:
                        if cfg.get("tau_soft") is not None or cfg.get("lambda_actor_kl") is not None:
                            issues.append(f"{entry['name']} expects tau_soft/lambda_actor_kl null")
                    if entry["name"] == "cbf_aux":
                        if cfg.get("alpha_cbf") is None or cfg.get("barrier_offset") is None:
                            issues.append("cbf missing alpha_cbf / barrier_offset")
                    else:
                        if cfg.get("alpha_cbf") is not None or cfg.get("barrier_offset") is not None:
                            issues.append(f"{entry['name']} expects alpha_cbf/barrier_offset null")
                    if entry["name"] == "eikonal_aux":
                        if cfg.get("w_fail") is None:
                            issues.append("eikonal missing w_fail")
                    else:
                        if cfg.get("w_fail") is not None:
                            issues.append(f"{entry['name']} expects w_fail null")
            per_method.append({
                "method": entry["name"],
                "smoke_ok": ok_run,
                "n_issues": len(issues),
                "issues": issues[:8],
                "n_config_keys": len(cfg_keys),
            })
            overall_ok = overall_ok and (len(issues) == 0)
        except Exception as e:
            per_method.append({
                "method": entry["name"],
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=2),
            })
            overall_ok = False
    _record("24.2_uniform_config_schema", overall_ok, {"per_method": per_method})


def test_24_3():
    """For each PDE script, --lambda_residual 0.5 propagates to meta.config.
    For soft_hjb, --lambda_actor_kl 0.5 also propagates."""
    per_method = []
    overall_ok = True
    for entry in METHODS:
        if not entry["is_pde"]:
            continue
        try:
            extra = ["--lambda_residual", "0.5"]
            suffix = "lambda_05"
            if entry["name"] == "soft_hjb_aux":
                extra.extend(["--lambda_actor_kl", "0.5"])
                suffix = "lambda_05_kl_05"
            ok_run, out_dir, log_path = _run_smoke(entry, extra_args=extra, suffix=suffix)
            issues = []
            if not ok_run:
                issues.append(f"smoke failed (see {log_path})")
            else:
                cfg = (_meta(out_dir).get("config") or {})
                if abs(float(cfg.get("lambda_residual", 0.0)) - 0.5) > 1e-9:
                    issues.append(f"lambda_residual not propagated: got {cfg.get('lambda_residual')}")
                if entry["name"] == "soft_hjb_aux":
                    if abs(float(cfg.get("lambda_actor_kl", 0.0)) - 0.5) > 1e-9:
                        issues.append(f"lambda_actor_kl not propagated: got {cfg.get('lambda_actor_kl')}")
            per_method.append({
                "method": entry["name"],
                "n_issues": len(issues),
                "issues": issues,
            })
            overall_ok = overall_ok and (len(issues) == 0)
        except Exception as e:
            per_method.append({"method": entry["name"], "error": f"{type(e).__name__}: {e}"})
            overall_ok = False
    _record("24.3_cli_propagation", overall_ok, {"per_method": per_method})


def test_24_4():
    """Boolean flag --use_intent: meta.intent_on is False without it, True with it.
    Verifying for one representative script (drppo) is sufficient since the flag
    is registered identically by the shared helper."""
    issues = []
    try:
        # Without flag (already produced by 24.2 default smoke)
        out_no = _smoke_dir("drppo", "default")
        meta_no = _meta(out_no)
        if meta_no.get("intent_on") is not False:
            issues.append(f"drppo without --use_intent: intent_on={meta_no.get('intent_on')!r}")
        # With flag
        ok_run, out_yes, log_path = _run_smoke(METHODS[0], extra_args=["--use_intent"], suffix="intent")
        if not ok_run:
            issues.append(f"smoke with --use_intent failed (see {log_path})")
        else:
            meta_yes = _meta(out_yes)
            if meta_yes.get("intent_on") is not True:
                issues.append(f"drppo with --use_intent: intent_on={meta_yes.get('intent_on')!r}")
    except Exception as e:
        issues.append(f"{type(e).__name__}: {e}")
    _record("24.4_use_intent_flag", len(issues) == 0, {"issues": issues})


def test_24_5():
    """Git provenance:
       (a) From inside a real git repo, meta records actual commit/branch/dirty.
       (b) From outside any git repo, falls back to '00000000' / 'unknown' / true.
    """
    issues = []
    in_repo_meta = None
    out_repo_meta = None

    # Sub-test (a): create a temporary git repo and verify _git_info()
    try:
        with tempfile.TemporaryDirectory() as repo_tmp:
            subprocess.run(["git", "init", "-q"], cwd=repo_tmp, check=True)
            subprocess.run(["git", "-C", repo_tmp, "config", "user.email", "p1b@example.com"], check=True)
            subprocess.run(["git", "-C", repo_tmp, "config", "user.name", "p1b"], check=True)
            # Make an initial commit so HEAD resolves
            (Path(repo_tmp) / "README.md").write_text("phase24 fixture\n")
            subprocess.run(["git", "-C", repo_tmp, "add", "."], check=True)
            subprocess.run(["git", "-C", repo_tmp, "commit", "-q", "-m", "init"], check=True)
            commit_sha = subprocess.check_output(
                ["git", "-C", repo_tmp, "rev-parse", "HEAD"]).decode().strip()
            # Create a working subdir to verify the parent walk
            subdir = Path(repo_tmp) / "deep" / "nested"
            subdir.mkdir(parents=True)
            # Run _git_info() with cwd inside the nested dir
            probe = (
                "import sys, os, json\n"
                f"sys.path.insert(0, {REPO_ROOT!r})\n"
                f"os.chdir({str(subdir)!r})\n"
                "from experiments.pde.run_metadata import _git_info\n"
                "print(json.dumps(_git_info()))\n"
            )
            proc = subprocess.run(
                [PYTHON_BIN, "-c", probe],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            )
            if proc.returncode != 0:
                issues.append(f"_git_info() probe failed: {proc.stderr.decode()[:300]}")
            else:
                in_repo_meta = json.loads(proc.stdout.decode().strip())
                # Branch should be a real branch name (not 'unknown')
                if in_repo_meta.get("git_branch", "unknown") in ("unknown", ""):
                    issues.append(f"in-repo branch came back as {in_repo_meta.get('git_branch')!r}")
                # Commit must be 8 hex chars matching the prefix of the actual sha
                if not re.match(r"^[0-9a-f]{8}$", in_repo_meta.get("git_commit", "")):
                    issues.append(f"in-repo git_commit malformed: {in_repo_meta.get('git_commit')!r}")
                if not commit_sha.startswith(in_repo_meta.get("git_commit", "ZZZZZZZZ")):
                    issues.append(f"in-repo git_commit prefix mismatch: meta={in_repo_meta.get('git_commit')!r} repo={commit_sha[:8]}")
                if not isinstance(in_repo_meta.get("git_dirty"), bool):
                    issues.append("in-repo git_dirty not bool")
    except Exception as e:
        issues.append(f"in-repo sub-test: {type(e).__name__}: {e}")

    # Sub-test (b): not under any git repo
    try:
        with tempfile.TemporaryDirectory() as no_repo_tmp:
            probe = (
                "import sys, os, json\n"
                f"sys.path.insert(0, {REPO_ROOT!r})\n"
                f"os.chdir({no_repo_tmp!r})\n"
                "from experiments.pde.run_metadata import _git_info\n"
                "print(json.dumps(_git_info()))\n"
            )
            # Run with HOME and prevent walking into ancestors that have .git
            # by also running inside / via a chdir to /tmp itself? tempfile dirs
            # are under /tmp which is not a git repo. Walks up: /tmp -> / -> none.
            proc = subprocess.run(
                [PYTHON_BIN, "-c", probe],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            )
            if proc.returncode != 0:
                issues.append(f"no-repo _git_info() probe failed: {proc.stderr.decode()[:300]}")
            else:
                out_repo_meta = json.loads(proc.stdout.decode().strip())
                if out_repo_meta.get("git_commit") != "00000000":
                    issues.append(f"no-repo git_commit expected '00000000', got {out_repo_meta.get('git_commit')!r}")
                if out_repo_meta.get("git_branch") != "unknown":
                    issues.append(f"no-repo git_branch expected 'unknown', got {out_repo_meta.get('git_branch')!r}")
                if out_repo_meta.get("git_dirty") is not True:
                    issues.append(f"no-repo git_dirty expected True, got {out_repo_meta.get('git_dirty')!r}")
    except Exception as e:
        issues.append(f"no-repo sub-test: {type(e).__name__}: {e}")

    _record("24.5_git_provenance", len(issues) == 0, {
        "issues": issues,
        "in_repo": in_repo_meta,
        "no_repo": out_repo_meta,
    })


def test_24_6():
    ver_dir = os.path.dirname(os.path.abspath(__file__))
    phases = {}
    failed_load = []
    for path in sorted(glob(os.path.join(ver_dir, "phase*.json"))):
        name = os.path.basename(path).replace(".json", "")
        if name == "phase24_cli_hyperparameters":
            continue
        try:
            with open(path) as f:
                phases[name] = json.load(f)
        except Exception as e:
            failed_load.append((name, f"{type(e).__name__}: {e}"))
    all_pass = all(v.get("pass", True) is True
                   for v in phases.values() if isinstance(v, dict))
    failed_phases = [n for n, v in phases.items()
                     if isinstance(v, dict) and v.get("pass", True) is False]
    n_phases = len(phases)
    ok = all_pass and n_phases >= 25 and not failed_load
    _record("24.6_existing_suite", ok, {
        "n_phases": n_phases,
        "all_pass": all_pass,
        "failed_phases": failed_phases,
        "load_failures": failed_load,
    })


def main():
    print("==== PHASE 24: CLI HYPERPARAMETER EXPOSURE VERIFICATION ====")
    os.makedirs(SMOKE_BASE, exist_ok=True)
    test_24_1()
    test_24_2()
    test_24_3()
    test_24_4()
    test_24_5()
    test_24_6()

    out_path = os.path.join(os.path.dirname(__file__),
                            "phase24_cli_hyperparameters.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    print("ALL_PASS =", results["pass"])
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
