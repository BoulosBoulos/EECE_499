"""Generate visibility progression figure for the paper.

Shows alpha_cz (conflict zone visibility fraction) as a function of
ego distance from the junction. Demonstrates that the static occlusion
model creates a meaningful partial-observability gradient.

Output: results/figures/visibility_progression.png

Usage:
    python experiments/pde/plot_visibility_progression.py
"""

import numpy as np
import os


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required. Install with: pip install matplotlib")
        return

    from env.sumo_env import SumoEnv

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # With buildings
    env = SumoEnv(scenario_name="1a", ego_maneuver="stem_right", buildings=True)
    obs, info = env.reset()
    distances, alphas_on = [], []
    for _ in range(300):
        obs, r, term, trunc, info = env.step(3)  # GO action
        d_cz = info["built"]["s_geom"][1]
        alpha = info["raw_obs"]["vis"]["alpha_cz"]
        distances.append(d_cz)
        alphas_on.append(alpha)
        if term or trunc:
            break
    env.close()

    axes[0].plot(distances, alphas_on, 'b-', linewidth=2)
    axes[0].set_xlabel("Distance to Conflict Zone (m)")
    axes[0].set_ylabel(r"$\alpha_{cz}$ (Visibility Fraction)")
    axes[0].set_title("With Occlusion Buildings")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    # Without buildings
    env = SumoEnv(scenario_name="1a", ego_maneuver="stem_right", buildings=False)
    obs, info = env.reset()
    distances2, alphas_off = [], []
    for _ in range(300):
        obs, r, term, trunc, info = env.step(3)
        d_cz = info["built"]["s_geom"][1]
        alpha = info["raw_obs"]["vis"]["alpha_cz"]
        distances2.append(d_cz)
        alphas_off.append(alpha)
        if term or trunc:
            break
    env.close()

    axes[1].plot(distances2, alphas_off, 'r-', linewidth=2)
    axes[1].set_xlabel("Distance to Conflict Zone (m)")
    axes[1].set_ylabel(r"$\alpha_{cz}$ (Visibility Fraction)")
    axes[1].set_title("Without Occlusion Buildings")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/visibility_progression.png", dpi=200, bbox_inches="tight")
    print("Saved results/figures/visibility_progression.png")


if __name__ == "__main__":
    main()
