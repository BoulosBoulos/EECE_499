"""Runtime verification of known interaction-benchmark issues (debug session).

Writes NDJSON to the Cursor debug log. Run from repo root:
  python3 experiments/interaction/debug_verify_benchmark_issues.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time

import numpy as np

# #region agent log
_DEBUG_LOG = "/home/vboxuser/Downloads/EECE 499/.cursor/debug-1ffab4.log"
_SESSION = "1ffab4"


def _dbg(hypothesis_id: str, message: str, data: dict | None = None, run_id: str = "pre-fix") -> None:
    rec = {
        "sessionId": _SESSION,
        "timestamp": int(time.time() * 1000),
        "hypothesisId": hypothesis_id,
        "location": "debug_verify_benchmark_issues.py",
        "message": message,
        "data": data or {},
        "runId": run_id,
    }
    os.makedirs(os.path.dirname(_DEBUG_LOG), exist_ok=True)
    with open(_DEBUG_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# #endregion


def _h1_soft_hjb_kwargs() -> None:
    """H1: interaction train_soft_hjb_aux passes kwargs SoftHJBAuxAgent does not define."""
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base, "..", "..")
    agent_path = os.path.join(root, "models", "pde", "soft_hjb_aux_agent.py")
    tree = ast.parse(open(agent_path).read())
    init_params: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SoftHJBAuxAgent":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_params = {a.arg for a in item.args.args} - {"self"}
    script = open(os.path.join(base, "train_soft_hjb_aux.py")).read()
    st = ast.parse(script)
    call_keywords: set[str] = set()
    for node in ast.walk(st):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "SoftHJBAuxAgent":
                for kw in node.keywords:
                    if kw.arg:
                        call_keywords.add(kw.arg)
    unknown = sorted(call_keywords - init_params)
    post = "--post-fix" in sys.argv
    if post:
        _dbg(
            "H1",
            "POSTFIX_OK: no invalid SoftHJBAuxAgent kwargs in train_soft_hjb_aux"
            if not unknown
            else "POSTFIX_FAIL: still invalid kwargs",
            {"undefined_keywords": unknown},
            run_id="post-fix",
        )
    else:
        _dbg(
            "H1",
            "CONFIRMED: train_soft_hjb_aux passes undefined SoftHJBAuxAgent __init__ keywords"
            if unknown
            else "REJECTED: all keywords valid",
            {"undefined_keywords": unknown, "agent_accepts": sorted(init_params)},
        )


def _h2_run_train_drops_extra() -> None:
    """H2: run_train main unpack discards extra and train_step omits extra kwarg."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "run_train.py")
    src = open(path).read()
    discards_extra = "advantages, _, hidden_arr" in src
    pos = src.find("policy.train_step(")
    chunk = src[pos : pos + 500] if pos >= 0 else ""
    passes_extra = "extra=" in chunk
    post = "--post-fix" in sys.argv
    bad = discards_extra or not passes_extra
    if post:
        _dbg(
            "H2",
            "POSTFIX_OK: run_train passes extra into train_step"
            if not bad
            else "POSTFIX_FAIL: extra still dropped or missing kwarg",
            {
                "unpack_uses_underscore_for_extra": discards_extra,
                "train_step_block_has_extra_kwarg": passes_extra,
            },
            run_id="post-fix",
        )
    else:
        _dbg(
            "H2",
            "CONFIRMED: run_train discards extra and/or omits train_step(extra=...)"
            if bad
            else "REJECTED: extra wired",
            {
                "unpack_uses_underscore_for_extra": discards_extra,
                "train_step_block_has_extra_kwarg": passes_extra,
            },
        )


def _h3_extra_lacks_xi() -> None:
    """H3: interaction collect_rollouts extra has no xi_curr (HJB block skipped)."""
    from experiments.interaction.run_train import collect_rollouts

    class _Inner:
        hidden_dim = 8

    class _P:
        hidden_dim = 8
        device = "cpu"
        policy = _Inner()
        _hidden = None

        def reset_hidden(self) -> None:
            pass

        def get_action(self, obs):
            return 0, None, 0.0, 0.0

    class _E:
        def __init__(self):
            self._t = 0
            self._ep = 0

        def reset(self):
            self._t = 0
            self._ep += 1
            o = np.zeros(91, dtype=np.float32)
            return o, {
                "ego_speed": 8.0,
                "actors": [{"eta_enter": 3.0}],
                "events": {},
                "step": 0,
                "reward": 0.0,
            }

        def step(self, a):
            self._t += 1
            o = np.zeros(91, dtype=np.float32)
            term = self._t >= 2
            ev = {"collision": False, "success": term}
            return o, -0.1, term, False, {
                "reward": -0.1,
                "events": ev,
                "step": self._t,
            }

    pol = _P()
    env = _E()
    out = collect_rollouts(env, pol, n_steps=6, gamma=0.99, gae_lambda=0.95)
    _, _, _, _, _, extra, _, _ = out
    has_xi = extra is not None and "xi_curr" in extra
    post = "--post-fix" in sys.argv
    if post:
        _dbg(
            "H3",
            "POSTFIX_OK: collect_rollouts includes xi_curr"
            if has_xi
            else "POSTFIX_FAIL: xi_curr missing",
            {"extra_keys": sorted(extra.keys()) if extra else []},
            run_id="post-fix",
        )
    else:
        _dbg(
            "H3",
            "CONFIRMED: collect_rollouts extra omits xi_curr (before adapter fix)"
            if not has_xi
            else "REJECTED: xi_curr present",
            {"extra_keys": sorted(extra.keys()) if extra else []},
        )


def _h4_sticky_events_overcount() -> None:
    """H4: counting events each timestep inflates vs per-episode any()."""
    infos = []
    for t in range(1, 15):
        infos.append({"step": t, "events": {"success": True, "collision": False}})
    per_step = sum(1 for inf in infos if inf["events"].get("success"))
    per_ep = 1
    inflated = per_step > per_ep
    post = "--post-fix" in sys.argv
    if post:
        from experiments.interaction.metrics_util import episode_event_rates

        rates, n_eps = episode_event_rates(infos)
        ok = abs(rates["success"] - 1.0) < 1e-6 and n_eps == 1
        _dbg(
            "H4",
            "POSTFIX_OK: episode_event_rates reports 1.0 success (not 14/1 steps)"
            if ok
            else "POSTFIX_FAIL: episode rates wrong",
            {
                "per_step_success_hits": per_step,
                "episode_event_rates_success": rates["success"],
                "n_episodes_parsed": n_eps,
            },
            run_id="post-fix",
        )
    else:
        _dbg(
            "H4",
            "CONFIRMED: sticky success True for 14 steps yields step-count 14 vs episode-count 1"
            if inflated
            else "REJECTED",
            {"per_step_success_hits": per_step, "episodes": 1, "per_episode_cap": per_ep},
        )


def _h5_extra_zeros() -> None:
    """H5: a_lon and d_cz arrays in extra are all zero (pre-fill fix)."""
    from experiments.interaction.run_train import collect_rollouts

    class _Inner:
        hidden_dim = 8

    class _P:
        hidden_dim = 8
        device = "cpu"
        policy = _Inner()
        _hidden = None

        def reset_hidden(self) -> None:
            pass

        def get_action(self, obs):
            return 0, None, 0.0, 0.0

    class _E:
        def __init__(self):
            self._t = 0

        def reset(self):
            self._t = 0
            o = np.zeros(91, dtype=np.float32)
            return o, {
                "ego_speed": 8.0,
                "ego_accel": -0.5,
                "d_conflict_entry": 22.0,
                "actors": [{"eta_enter": 3.0}],
                "events": {},
                "step": 0,
                "reward": 0.0,
            }

        def step(self, a):
            self._t += 1
            o = np.zeros(91, dtype=np.float32)
            term = self._t >= 4
            return o, -0.1, term, False, {
                "reward": -0.1,
                "events": {},
                "step": self._t,
            }

    pol = _P()
    env = _E()
    out = collect_rollouts(env, pol, n_steps=5, gamma=0.99, gae_lambda=0.95)
    _, _, _, _, _, extra, _, _ = out
    a0 = float(np.max(np.abs(extra.get("a_lon", np.array([0.0])))))
    d0 = float(np.max(np.abs(extra.get("d_cz", np.array([0.0])))))
    post = "--post-fix" in sys.argv
    if post:
        ok = a0 > 1e-6 and d0 > 1e-6
        _dbg(
            "H5",
            "POSTFIX_OK: a_lon and d_cz populated from env info"
            if ok
            else "POSTFIX_FAIL: physics arrays still near zero",
            {"max_abs_a_lon": a0, "max_abs_d_cz": d0},
            run_id="post-fix",
        )
    else:
        _dbg(
            "H5",
            "CONFIRMED: a_lon and d_cz still zero before fill-from-info fix"
            if a0 == 0.0 and d0 == 0.0
            else "REJECTED: non-zero physics fields",
            {"max_abs_a_lon": a0, "max_abs_d_cz": d0},
        )


def main() -> None:
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    _h1_soft_hjb_kwargs()
    _h2_run_train_drops_extra()
    _h3_extra_lacks_xi()
    _h4_sticky_events_overcount()
    _h5_extra_zeros()
    print("Wrote debug log to", _DEBUG_LOG)


if __name__ == "__main__":
    main()
