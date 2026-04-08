"""Extract and compare hw2d statistics between physics and CFM trajectories.

Loads raw_trajectories (physics) and cfm_trajectories/model1.zarr, unnormalises
the CFM data using per-time-step stats from cfm_train_data, computes hw2d
statistics (gamma_n, gamma_c, energy, hw2d_enstrophy) for each source, and
saves them to zarr + produces matplotlib plots.

Usage:
    python scripts/extract_hw2d_statistics.py [--n-traj N] [--batch-size B]

Output:
    data/datasets/hw2d/stats/train/physics_statistics.zarr
    data/datasets/hw2d/stats/train/cfm_statistics.zarr
    data/datasets/hw2d/stats/test/physics_statistics.zarr
    data/datasets/hw2d/stats/plots/hw2d_statistics_comparison.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tpflow.statistics import hw2d_statistics


# ── helpers ─────────────────────────────────────────────────────────────────────

def unnormalize_per_time(data: np.ndarray, per_time_mean: np.ndarray, per_time_std: np.ndarray) -> np.ndarray:
    """Undo per-time-step normalisation.

    Args:
        data: ``(n_rollout, n_time, H, W, C)``
        per_time_mean: ``(n_time, C)``
        per_time_std:  ``(n_time, C)``

    Returns:
        ``(n_rollout, n_time, H, W, C)`` in physical units.
    """
    # Broadcast: (1, n_time, 1, 1, C)
    mean = per_time_mean[None, :, None, None, :]
    std  = per_time_std[None,  :, None, None, :]
    return data * std + mean


def compute_stats_batched(
    zarr_data: zarr.Array,
    per_time_mean: np.ndarray | None,
    per_time_std: np.ndarray | None,
    batch_size: int,
    n_traj: int | None,
) -> dict[str, np.ndarray]:
    """Compute hw2d statistics batch-wise to avoid OOM.

    Args:
        zarr_data: shape ``(N, n_time, H, W, C)``
        per_time_mean: if not None, unnormalise before computing stats.
        per_time_std:  same.
        batch_size: number of trajectories to process at once.
        n_traj: cap on number of trajectories to use; None = all.

    Returns:
        Dict ``{stat_name: (n_time, n_traj)}``
    """
    N = zarr_data.shape[0]
    if n_traj is not None:
        N = min(N, n_traj)

    n_time = zarr_data.shape[1]

    all_stats: dict[str, list[np.ndarray]] = {}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        chunk = zarr_data[start:end].astype(np.float32)  # (B, n_time, H, W, C)

        if per_time_mean is not None:
            chunk = unnormalize_per_time(chunk, per_time_mean.astype(np.float32),
                                         per_time_std.astype(np.float32))

        # hw2d_statistics expects (n_time, n_rollout, H, W, C)
        chunk_t = np.moveaxis(chunk, 0, 1)  # (n_time, B, H, W, C)
        stats = hw2d_statistics(chunk_t)    # each (n_time, B)

        for k, v in stats.items():
            all_stats.setdefault(k, []).append(v)

        print(f"  processed {end}/{N} trajectories", flush=True)

    return {k: np.concatenate(v, axis=1) for k, v in all_stats.items()}


def save_statistics(stats: dict[str, np.ndarray], path: Path) -> None:
    """Save statistics dict to a zarr group."""
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(path), mode="w")
    for name, vals in stats.items():
        store.create_array(name, data=vals.astype(np.float32))
    print(f"Saved statistics to {path}")


def plot_statistics(
    physics_stats: dict[str, np.ndarray],
    cfm_stats: dict[str, np.ndarray],
    time_vector: np.ndarray,
    output_path: Path,
) -> None:
    """Plot mean ± std comparison for all statistics."""
    stat_names = list(physics_stats.keys())
    n_stats = len(stat_names)

    fig, axes = plt.subplots(1, n_stats, figsize=(5 * n_stats, 4))
    if n_stats == 1:
        axes = [axes]

    for ax, name in zip(axes, stat_names):
        phys = physics_stats[name]   # (n_time, n_traj)
        cfm  = cfm_stats[name]       # (n_time, n_traj)

        pm, ps = phys.mean(axis=1), phys.std(axis=1)
        cm, cs = cfm.mean(axis=1),  cfm.std(axis=1)

        ax.fill_between(time_vector, pm - ps, pm + ps, alpha=0.25, color="tab:blue")
        ax.plot(time_vector, pm, color="tab:blue", label="physics")
        ax.fill_between(time_vector, cm - cs, cm + cs, alpha=0.25, color="tab:orange")
        ax.plot(time_vector, cm, color="tab:orange", linestyle="--", label="CFM model1")
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()

    fig.suptitle("hw2d statistics: physics vs CFM (train set)", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract hw2d statistics")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="Max trajectories per source (default: all)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Trajectories per batch (default: 50)")
    parser.add_argument("--dataset-root", type=str,
                        default="data/datasets/hw2d",
                        help="Path to hw2d dataset root")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    stats_root = root / "stats"

    # Load per-time-step normalisation stats
    print("Loading normalisation stats…")
    norm_zarr = zarr.open(str(root / "cfm_train_data" / "train.zarr"))
    per_time_mean = np.array(norm_zarr.attrs["per_time_mean"], dtype=np.float64)  # (n_time, 2)
    per_time_std  = np.array(norm_zarr.attrs["per_time_std"],  dtype=np.float64)  # (n_time, 2)
    print(f"  per_time_mean shape: {per_time_mean.shape}")

    # ── train split: physics ──────────────────────────────────────────────────
    print("\nComputing physics (raw_trajectories/train) statistics…")
    phys_train_z = zarr.open(str(root / "raw_trajectories" / "train.zarr"))
    time_vector  = np.array(phys_train_z["time"], dtype=np.float32)  # (n_time,)
    phys_train_stats = compute_stats_batched(
        phys_train_z["data"], None, None, args.batch_size, args.n_traj
    )
    save_statistics(phys_train_stats, stats_root / "train" / "physics_statistics.zarr")

    # ── train split: CFM model1 ───────────────────────────────────────────────
    print("\nComputing CFM model1 (cfm_trajectories/model1) statistics…")
    cfm_z = zarr.open(str(root / "cfm_trajectories" / "model1.zarr"))
    cfm_stats = compute_stats_batched(
        cfm_z["data"], per_time_mean, per_time_std, args.batch_size, args.n_traj
    )
    save_statistics(cfm_stats, stats_root / "train" / "cfm_statistics.zarr")

    # ── test split: physics ───────────────────────────────────────────────────
    print("\nComputing physics (raw_trajectories/test) statistics…")
    phys_test_z = zarr.open(str(root / "raw_trajectories" / "test.zarr"))
    phys_test_stats = compute_stats_batched(
        phys_test_z["data"], None, None, args.batch_size, args.n_traj
    )
    save_statistics(phys_test_stats, stats_root / "test" / "physics_statistics.zarr")

    # ── plot ─────────────────────────────────────────────────────────────────
    print("\nPlotting…")
    plot_statistics(
        phys_train_stats, cfm_stats, time_vector,
        stats_root / "plots" / "hw2d_statistics_comparison.png",
    )

    # also plot test physics with a single-source figure
    stat_names = list(phys_test_stats.keys())
    n_stats = len(stat_names)
    fig, axes = plt.subplots(1, n_stats, figsize=(5 * n_stats, 4))
    if n_stats == 1:
        axes = [axes]
    test_time = np.array(phys_test_z["time"], dtype=np.float32)
    for ax, name in zip(axes, stat_names):
        vals = phys_test_stats[name]
        m, s = vals.mean(axis=1), vals.std(axis=1)
        ax.fill_between(test_time, m - s, m + s, alpha=0.25, color="tab:blue")
        ax.plot(test_time, m, color="tab:blue", label="physics (test)")
        ax.set_xlabel("time")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
    fig.suptitle("hw2d statistics: physics test set", fontsize=13)
    fig.tight_layout()
    out_test = stats_root / "plots" / "hw2d_test_statistics.png"
    fig.savefig(str(out_test), dpi=150)
    print(f"Saved plot to {out_test}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
