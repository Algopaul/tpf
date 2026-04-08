"""W1 and W2 distance comparison: holder regression variants vs reference.

Usage:
    python scripts/w2_comparison_holder.py

Add checkpoint paths as they become available (model1, ot, physics variants).
Physical time range is 0–5 (6 time steps: 0, 1, 2, 3, 4, 5).
"""

import numpy as np

from tpflow.w2 import compare_w1, compare_w2, plot_w2_comparison

RUNS = {
    # Fill in checkpoint paths after training each variant:
    # "TPF reg (model1)": "multirun/.../eval/rollout.zarr",
    # "TPF reg (OT)":     "multirun/.../eval/rollout.zarr",
    # "TPF reg (physics)":"multirun/.../eval/rollout.zarr",
    "tpf": "multirun/2026-03-29/21-55-04/1/2000/eval/rollout.zarr",
    "phys": "multirun/2026-03-29/22-42-27/0/2000/eval/rollout.zarr",
}
RUNS = {
    "TPF reg (physics)": "multirun/2026-03-29/22-26-09/0/2000/eval/rollout.zarr",
    "TPF reg (model1)": "multirun/2026-03-29/21-55-04/1/2000/eval/rollout.zarr",
}


# Physical time steps are 0–5
TABLE_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

STRIDE = 1  # only 6 time steps total — evaluate all of them
EPSILON = 1e-3
MAX_SAMPLES = None  # use all particles; 512 caused large sampling bias in 10D
CLIP_VALUE = None
SOLVER = "hungarian"  # "sinkhorn" or "hungarian" (exact, no regularisation bias)


def print_markdown_table(results: dict, target_times: list[float]) -> None:
    labels = list(results.keys())
    headers = [f"{label} (N={results[label][3]})" for label in labels]
    print("| Time | " + " | ".join(headers) + " |")
    print("|-----:|" + "|:------:|" * len(labels))
    for t in target_times:
        row = f"| {t:.0f} |"
        for _, (_, dists, times, _n) in results.items():
            i = int(np.argmin(np.abs(times - t)))
            row += f" {dists[i]:.3f} |"
        print(row)


if __name__ == "__main__":
    if not RUNS:
        print("No checkpoints configured yet. Train regression and add paths to RUNS.")
        print("Example:")
        print("  just holder-regression")
        print("  just export-eval <checkpoint-path>")
        print("  # then add the rollout.zarr path to RUNS in this script")
        raise SystemExit(0)

    kw = dict(
        stride=STRIDE,
        epsilon=EPSILON,
        max_samples=MAX_SAMPLES,
        clip_value=CLIP_VALUE,
        solver=SOLVER,
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
        w1_results, ylabel="W1 distance", title="W1: holder rollout vs. reference"
    )
    fig1.savefig("w1_comparison_holder.png", dpi=150)
    print("\nSaved w1_comparison_holder.png")

    fig2 = plot_w2_comparison(
        w2_results, ylabel="W2 distance", title="W2: holder rollout vs. reference"
    )
    fig2.savefig("w2_comparison_holder.png", dpi=150)
    print("Saved w2_comparison_holder.png")
