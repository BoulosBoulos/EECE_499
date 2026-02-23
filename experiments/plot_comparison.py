"""Plot PINN vs non-PINN training comparison."""

from __future__ import annotations

import argparse
import csv
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results")
    parser.add_argument("--out", default="results/comparison.png")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    data = {}
    for name in ["pinn", "nopinn"]:
        path = os.path.join(args.dir, f"train_{name}.csv")
        if not os.path.exists(path):
            continue
        steps, returns = [], []
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                steps.append(int(row["step"]))
                returns.append(float(row["episode_return"]))
        data[name] = (steps, returns)

    if not data:
        print("No training CSVs found")
        return

    plt.figure(figsize=(8, 5))
    for name, (steps, rets) in data.items():
        plt.plot(steps, rets, label=name, alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Episode return (mean)")
    plt.legend()
    plt.title("PINN vs non-PINN DRPPO")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
