"""Re-plot hw2d statistics from saved zarr files (no recomputation).

Usage:
    python scripts/plot_hw2d_statistics.py
    python scripts/plot_hw2d_statistics.py --dataset-root data/datasets/hw2d
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


def load_stats(path: Path) -> dict[str, np.ndarray]:
    store = zarr.open(str(path))
    return {k: np.array(store[k]) for k in store.keys()}


def filter_outlier_trajectories(
    stats: dict[str, np.ndarray],
    ref_key: str = "energy",
    threshold_frac: float = 0.01,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Remove trajectories whose time-mean statistic is near zero.

    Identifies trajectories where the time-averaged ``ref_key`` stat is below
    ``threshold_frac * median(time-mean)``.  These are failed/uninitialized
    simulations whose statistics are identically zero.

    Args:
        stats: dict of ``(n_time, n_traj)`` arrays.
        ref_key: statistic to use for the outlier criterion; falls back to the
            first available key if not present.
        threshold_frac: fraction of the median below which a trajectory is dropped.

    Returns:
        Filtered stats dict and boolean mask of kept trajectories.
    """
    key = ref_key if ref_key in stats else next(iter(stats))
    traj_mean = stats[key].mean(axis=0)
    keep = traj_mean >= threshold_frac * np.median(traj_mean)
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"Filtered {n_dropped} outlier trajectories (zero/{ref_key}) out of {len(keep)}")
    return {k: v[:, keep] for k, v in stats.items()}, keep


def _plot_source(ax, time, vals, color, label, linestyle="-"):
    """Plot mean ± std (shaded) with min/max as dashed lines."""
    m = vals.mean(axis=1)
    s = vals.std(axis=1)
    lo, hi = vals.min(axis=1), vals.max(axis=1)
    ax.fill_between(time, m - s, m + s, alpha=0.2, color=color)
    ax.plot(time, m, color=color, linestyle=linestyle, label=label)
    ax.plot(time, lo, color=color, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.plot(time, hi, color=color, linestyle="--", linewidth=0.7, alpha=0.6)


def plot_comparison(
    physics_stats: dict[str, np.ndarray],
    cfm_stats: dict[str, np.ndarray],
    time_vector: np.ndarray,
    output_path: Path,
    title: str = "hw2d statistics: physics vs CFM (train set)",
) -> None:
    stat_names = list(physics_stats.keys())
    n_stats = len(stat_names)
    fig, axes = plt.subplots(1, n_stats, figsize=(5 * n_stats, 4))
    if n_stats == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        _plot_source(ax, time_vector, physics_stats[name], "tab:blue", "physics")
        _plot_source(ax, time_vector, cfm_stats[name], "tab:orange", "CFM model1", linestyle="--")
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.set_title(name)
        # single legend entry per source (suppress duplicate dashed lines)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_single(
    stats: dict[str, np.ndarray],
    time_vector: np.ndarray,
    output_path: Path,
    label: str,
    title: str,
) -> None:
    stat_names = list(stats.keys())
    n_stats = len(stat_names)
    fig, axes = plt.subplots(1, n_stats, figsize=(5 * n_stats, 4))
    if n_stats == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        _plot_source(ax, time_vector, stats[name], "tab:blue", label)
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.set_title(name)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/datasets/hw2d")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    stats_root = root / "stats"

    time_vector = np.array(
        zarr.open(str(root / "raw_trajectories" / "train.zarr"))["time"],
        dtype=np.float32,
    )
    test_time = np.array(
        zarr.open(str(root / "raw_trajectories" / "test.zarr"))["time"],
        dtype=np.float32,
    )

    phys_train, _ = filter_outlier_trajectories(load_stats(stats_root / "train" / "physics_statistics.zarr"))
    cfm_train      = load_stats(stats_root / "train" / "cfm_statistics.zarr")
    phys_test, _  = filter_outlier_trajectories(load_stats(stats_root / "test"  / "physics_statistics.zarr"))

    plot_comparison(
        phys_train, cfm_train, time_vector,
        stats_root / "plots" / "hw2d_statistics_comparison.png",
    )
    plot_single(
        phys_test, test_time,
        stats_root / "plots" / "hw2d_test_statistics.png",
        label="physics (test)",
        title="hw2d statistics: physics test set",
    )


if __name__ == "__main__":
    main()
