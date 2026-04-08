"""W1 and W2 distance comparison: langevin_potential regression variants vs reference.

Usage:
    python scripts/w2_comparison_langevin.py
"""

import numpy as np

from tpflow.w2 import compare_w1, compare_w2, plot_w2_comparison

RUNS = {
    "reg (model1)": "multirun/2026-03-26/13-54-19/0/1100/eval/rollout.zarr",
    "reg (OT)": "multirun/2026-03-27/10-05-10/0/600/eval/rollout.zarr",
    "DICE": "data/dice_rollout.zarr",
}

# Target physical times for the summary markdown tables
TABLE_TIMES = [0.0, 0.05, 0.20, 0.50, 1.0]

STRIDE = 10
EPSILON = 1e-3
MAX_SAMPLES = 512  # set e.g. 256 for a quick order-of-magnitude run
CLIP_VALUE = 1.0  # clip coordinates to [-CLIP_VALUE, CLIP_VALUE]; set None to disable


def print_markdown_table(results: dict, target_times: list[float]) -> None:
    labels = list(results.keys())
    print("| Time | " + " | ".join(labels) + " |")
    print("|-----:|" + "|:------:|" * len(labels))
    for t in target_times:
        row = f"| {t:.2f} |"
        for _, (_, dists, times) in results.items():
            i = int(np.argmin(np.abs(times - t)))
            row += f" {dists[i]:.3f} |"
        print(row)


if __name__ == "__main__":
    kw = dict(
        stride=STRIDE, epsilon=EPSILON, max_samples=MAX_SAMPLES, clip_value=CLIP_VALUE
    )

    print("Computing W1...")
    w1_results = compare_w1(RUNS, **kw)
    print("\n## W1 distance (rollout vs. reference)\n")
    print_markdown_table(w1_results, TABLE_TIMES)

    print("\nComputing W2...")
    w2_results = compare_w2(RUNS, **kw)
    print("\n## W2 distance (rollout vs. reference)\n")
    print_markdown_table(w2_results, TABLE_TIMES)

    fig1 = plot_w2_comparison(
        w1_results, ylabel="W1 distance", title="W1: rollout vs. reference"
    )
    fig1.savefig("w1_comparison_langevin.png", dpi=150)
    print("\nSaved w1_comparison_langevin.png")

    fig2 = plot_w2_comparison(
        w2_results, ylabel="W2 distance", title="W2: rollout vs. reference"
    )
    fig2.savefig("w2_comparison_langevin.png", dpi=150)
    print("Saved w2_comparison_langevin.png")
