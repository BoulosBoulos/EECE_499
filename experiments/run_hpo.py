"""
Optuna Bayesian hyperparameter search across scenarios and all hyperparameters.
Searches: PPO, residuals, reward. Scenarios: 1a, 1b, 1c, 1d, 2, 3, 4.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    optuna = None
    TPESampler = None

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None


def _load_config(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def collect_rollouts(env, policy, n_steps: int, gamma: float, gae_lambda: float):
    """Collect trajectories and compute GAE advantages."""
    from experiments.run_train import collect_rollouts as _collect
    return _collect(env, policy, n_steps, gamma, gae_lambda)


def objective(
    trial,
    total_steps: int,
    n_eval_episodes: int,
    device: str,
    batch_size_min: int = 32,
    fixed_scenario: str | None = None,
) -> float:
    from env.sumo_env import SumoEnv
    from models.drppo import DRPPO

    scenario = fixed_scenario or trial.suggest_categorical("scenario", ["1a", "1b", "1c", "1d", "2", "3", "4"])

    # PPO
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    batch_size = max(batch_size_min, trial.suggest_categorical("batch_size", [32, 64, 128]))
    n_epochs = trial.suggest_int("n_epochs", 3, 10)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.25, 1.0)

    # Residuals
    lambda_ego = trial.suggest_float("lambda_ego", 0.01, 0.5, log=True)
    lambda_stop = trial.suggest_float("lambda_stop", 0.01, 0.5, log=True)
    lambda_fric = trial.suggest_float("lambda_fric", 0.01, 0.3, log=True)
    lambda_risk = trial.suggest_float("lambda_risk", 0.01, 0.5, log=True)

    # Reward
    w_prog = trial.suggest_float("w_prog", 0.5, 2.0)
    w_time = trial.suggest_float("w_time", -0.2, -0.01)
    w_risk = trial.suggest_float("w_risk", -2.0, -0.5)
    w_coll = trial.suggest_float("w_coll", -20.0, -5.0)
    ttc_thr = trial.suggest_float("ttc_thr", 2.0, 4.0)
    d_coll = trial.suggest_float("d_coll", 1.5, 3.0)

    use_pinn = trial.suggest_categorical("use_pinn", [True, False])
    jm_ignore_prob = trial.suggest_categorical("jm_ignore_junction_foe_prob", [0.0, 0.05, 0.1, 0.15, 0.2])

    # Temp reward config
    reward_cfg = {
        "w_prog": w_prog, "w_time": w_time, "w_risk": w_risk,
        "w_coll": w_coll, "ttc_thr": ttc_thr, "d_coll": d_coll,
    }
    fd, reward_path = tempfile.mkstemp(suffix=".yaml", prefix="reward_hpo_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(reward_cfg, f)
        env = SumoEnv(
            scenario_name=scenario,
            use_gui=False,
            reward_config=reward_path,
            jm_ignore_fixed=jm_ignore_prob,
        )
        obs_dim = int(env.observation_space.shape[0])
    finally:
        try:
            os.unlink(reward_path)
        except OSError:
            pass

    policy = DRPPO(
        obs_dim=obs_dim,
        n_actions=5,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        use_pinn=use_pinn,
        lambda_ego=lambda_ego,
        lambda_stop=lambda_stop,
        lambda_fric=lambda_fric,
        lambda_risk=lambda_risk,
        device=device,
    )

    step = 0
    while step < total_steps:
        obs_arr, actions_arr, log_probs_arr, returns_arr, advantages_arr, extra = collect_rollouts(
            env, policy, n_steps, gamma, gae_lambda
        )
        step += n_steps

        n_batches = max(1, (len(obs_arr) + batch_size - 1) // batch_size)
        for _ in range(n_epochs):
            perm = np.random.permutation(len(obs_arr))
            for start in range(0, len(obs_arr), batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) == 0:
                    continue
                o = obs_arr[idx]
                a = actions_arr[idx]
                lp = log_probs_arr[idx]
                ret = returns_arr[idx]
                adv = advantages_arr[idx]
                ex = None
                if extra:
                    ex = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(obs_arr) else v)
                          for k, v in extra.items()}
                policy.train_step(o, a, lp, ret, adv, ex)

    # Eval
    returns_eval = []
    coll_eval = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        coll = 0
        for _ in range(500):
            a, _, _, _ = policy.get_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            if r < -5:
                coll += 1
            if term or trunc:
                break
        returns_eval.append(total_r)
        coll_eval.append(1 if coll > 0 else 0)
    env.close()

    mean_return = float(np.mean(returns_eval))
    coll_rate = float(np.mean(coll_eval))
    return mean_return - 50.0 * coll_rate


def _run_study(
    scenario: str | None,
    n_trials: int,
    total_steps: int,
    n_eval_episodes: int,
    out_dir: str,
    study_name: str,
    device: str,
    seed: int,
) -> dict:
    """Run one Optuna study. If scenario is None, scenario is searched; else fixed."""
    def fn(trial):
        return objective(trial, total_steps, n_eval_episodes, device, fixed_scenario=scenario)

    sampler = TPESampler(n_startup_trials=10, seed=seed)
    name = study_name if scenario is None else f"{study_name}_{scenario}"
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=name)

    study.optimize(
        fn,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    return {
        "best_value": best.value,
        "best_params": best.params,
        "scenario": scenario,
        "n_trials": len(study.trials),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--out_dir", default="results/hpo")
    parser.add_argument("--study_name", default="sumo_hpo")
    parser.add_argument("--scenario", default=None, choices=["1a", "1b", "1c", "1d", "2", "3", "4"],
                        help="Fix scenario; if not set, scenario is searched")
    parser.add_argument("--per_scenario", action="store_true",
                        help="Run separate HPO for each scenario (base, ped, bike, moto)")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if optuna is None:
        raise RuntimeError("Install optuna: pip install optuna")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"

    if args.per_scenario:
        all_results = {}
        for scen in ["1a", "1b", "1c", "1d", "2", "3", "4"]:
            print(f"\n=== HPO for scenario: {scen} ===")
            out = _run_study(
                scenario=scen,
                n_trials=args.n_trials,
                total_steps=args.total_steps,
                n_eval_episodes=args.n_eval_episodes,
                out_dir=args.out_dir,
                study_name=args.study_name,
                device=device,
                seed=args.seed,
            )
            all_results[scen] = out
            path = os.path.join(args.out_dir, f"hpo_{scen}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Best for {scen}: value={out['best_value']:.2f}")
            print(f"Saved to {path}")
        summary = {s: {"best_value": r["best_value"], "best_params": r["best_params"]}
                   for s, r in all_results.items()}
        path = os.path.join(args.out_dir, "hpo_per_scenario.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {path}")
    else:
        out = _run_study(
            scenario=args.scenario,
            n_trials=args.n_trials,
            total_steps=args.total_steps,
            n_eval_episodes=args.n_eval_episodes,
            out_dir=args.out_dir,
            study_name=args.study_name,
            device=device,
            seed=args.seed,
        )
        path = os.path.join(args.out_dir, "hpo_results.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nBest trial: value={out['best_value']:.2f}")
        print("Best params:", out["best_params"])
        print(f"Results saved to {path}")


if __name__ == "__main__":
    main()
