"""Frozen evaluation manifest for the interaction benchmark.

Generates and stores a fixed list of (template_id, scenario, seed) tuples
so every method is evaluated on exactly the same interaction cases.
The manifest is saved as a JSON file and loaded by the eval script.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
from typing import List, Dict

from scenario.template_sampler import TemplateSampler


def _load_benchmark_cfg(config_path: str | None = None) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = config_path or os.path.join(root, "configs", "interaction", "benchmark.yaml")
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def generate_manifest(
    scenarios: List[str],
    episodes_per_scenario: int = 50,
    seeds: List[int] | None = None,
    out_path: str = "results/interaction/eval_manifest.json",
    config_path: str | None = None,
) -> List[dict]:
    """Generate a reproducible manifest of evaluation episodes.

    Each entry specifies scenario, seed, template_family, and episode index
    so that results are deterministic across method comparisons.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    cfg = _load_benchmark_cfg(config_path)
    tp = cfg.get("template_probs")
    eb = cfg.get("eta_bands")
    bar_len = float(cfg.get("bar_half_length", 50.0))
    stem_len = float(cfg.get("stem_length", 50.0))

    manifest = []
    episode_id = 0

    for scenario in scenarios:
        for seed in seeds:
            rng = np.random.RandomState(seed)
            sampler = TemplateSampler(template_probs=tp, eta_bands=eb, rng=rng)

            for ep in range(episodes_per_scenario):
                template = sampler.sample(
                    scenario_id=scenario,
                    has_pothole=(scenario in ("1d", "4")),
                    bar_len=bar_len,
                    stem_len=stem_len,
                )
                manifest.append({
                    "episode_id": episode_id,
                    "scenario": scenario,
                    "seed": seed,
                    "episode_in_seed": ep,
                    "template_family": template.template_family,
                    "n_actors": len(template.actors),
                    "ego_target_eta": round(template.ego_target_eta_enter, 3),
                })
                episode_id += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_manifest(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def manifest_summary(manifest: List[dict]) -> Dict:
    """Print a summary of template distribution in the manifest."""
    from collections import Counter
    scenarios = Counter(e["scenario"] for e in manifest)
    templates = Counter(e["template_family"] for e in manifest)
    seeds = sorted(set(e["seed"] for e in manifest))
    return {
        "total_episodes": len(manifest),
        "scenarios": dict(scenarios),
        "templates": dict(templates),
        "seeds": seeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate frozen eval manifest")
    parser.add_argument("--scenarios", nargs="+",
                        default=["1a", "1b", "1c", "1d", "2", "3", "4"])
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per scenario per seed")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 123, 456, 789, 999])
    parser.add_argument("--out", default="results/interaction/eval_manifest.json")
    args = parser.parse_args()

    manifest = generate_manifest(
        scenarios=args.scenarios,
        episodes_per_scenario=args.episodes,
        seeds=args.seeds,
        out_path=args.out,
    )

    summary = manifest_summary(manifest)
    print(f"Generated manifest with {summary['total_episodes']} episodes")
    print(f"  Scenarios: {summary['scenarios']}")
    print(f"  Templates: {summary['templates']}")
    print(f"  Seeds: {summary['seeds']}")
    print(f"  Saved to: {args.out}")


if __name__ == "__main__":
    main()
