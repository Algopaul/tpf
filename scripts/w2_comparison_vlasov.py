"""W1 and W2 distance comparison: vlasov regression variants vs reference.

Usage:
    python scripts/w2_comparison_vlasov.py
"""

import numpy as np

from tpflow.w2 import compare_w1, compare_w2, plot_w2_comparison

RUNS = {
    # "Terpin OT": "data/jkonet_star_vlasov_time.zarr",
    # "TPF reg (OT)": "multirun/2026-03-30/12-37-28/0/700/eval/rollout.zarr",
    # "TPF reg (OT) 1-step": (
    #     "multirun/2026-03-30/12-37-28/0/700/eval/rollout.zarr",
    #     "one_step_ahead",
    # ),
    "OT bump": "multirun/2026-03-30/15-08-42/0/950/eval/rollout.zarr",
    "TPF bump (OT) 1-step": (
        "multirun/2026-03-30/15-08-42/0/950/eval/rollout.zarr",
        "one_step_ahead",
    ),
    # Add model1 / physics once trained:
    # "TPF reg (model1)": "multirun/.../eval/rollout.zarr",
}

# TPF OT rollout uses ot_trajectories (time_stride=8): physical times [0,8,16,...,120]
# Use times that exist in that grid; 0/32/64/96/120 cover the full range.
TABLE_TIMES = [0.0, 32.0, 64.0, 96.0, 120.0]

STRIDE = 5
EPSILON = 1e-3
MAX_SAMPLES = None
CLIP_VALUE = 5.0


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
        w1_results, ylabel="W1 distance", title="W1: vlasov rollout vs. reference"
    )
    fig1.savefig("w1_comparison_vlasov.png", dpi=150)
    print("\nSaved w1_comparison_vlasov.png")

    fig2 = plot_w2_comparison(
        w2_results, ylabel="W2 distance", title="W2: vlasov rollout vs. reference"
    )
    fig2.savefig("w2_comparison_vlasov.png", dpi=150)
    print("Saved w2_comparison_vlasov.png")
