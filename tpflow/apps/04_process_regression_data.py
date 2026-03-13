"""Create regression training data from trajectory zarr files.

Reads a zarr with layout:
    data:  (n_trajectories, n_time, *state_shape)
    param: (n_trajectories,)
    time:  (n_time,)            [optional; defaults to 0, 1, ..., n_time-1]

Writes a zarr with layout:
    data:  (n_samples, *state_shape)   current state
    next:  (n_samples, *state_shape)   next state
    time:  (n_samples,)                time of current state
    param: (n_samples,)                param of trajectory

and a zarr attribute:
    diff_scale: float  std((x_next - x)) over all samples; used to normalise
                       difference-mode regression targets so var(target) ≈ 1.

where n_samples = n_trajectories * (n_time - 1).

Usage:
    python tpflow/apps/04_process_regression_data.py \\
        input=data/datasets/gaurot/raw_trajectories/train.zarr \\
        output=data/datasets/gaurot/regression_train_data/train.zarr \\
        shuffle=true
"""

import logging
import math
from typing import cast

import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import RegressionDataConfig
from tpflow.processing import (
    auto_block_sizes,
    auto_blocks_per_shard,
    extract_regression_pairs,
    open_zarr_array,
)
from tpflow.util import init_wandb, log_duration


def _make_outfile(
    path: str,
    n_samples: int,
    block_size: int,
    state_shape: tuple[int, ...],
) -> zarr.Group:
    outfile = zarr.open_group(path, mode="w")
    # f4 dtype → 4 bytes per scalar; target ~1 GB per shard file.
    # Shard whenever n_samples > block_size (more than one inner chunk).
    # Small test splits produce one shard file; unit-test arrays (n_samples ≤
    # block_size) skip sharding entirely to avoid codec overhead.
    n_bps = auto_blocks_per_shard(block_size, state_shape, dtype_itemsize=4)
    shard_size = n_bps * block_size
    use_shards = n_samples > block_size
    logging.info(
        "shard_size=%d samples%s",
        shard_size,
        "" if use_shards else " — skipped, dataset fits in one chunk",
    )
    data_shape: tuple[int, ...] = (n_samples, *state_shape)
    data_chunks: tuple[int, ...] = (block_size, *state_shape)
    if use_shards:
        data_shard: tuple[int, ...] = (shard_size, *state_shape)
        scalar_shard: tuple[int, ...] = (shard_size,)
        outfile.create_array("data", shape=data_shape, chunks=data_chunks, dtype="f4", shards=data_shard)
        outfile.create_array("next", shape=data_shape, chunks=data_chunks, dtype="f4", shards=data_shard)
        outfile.create_array("time", shape=(n_samples,), chunks=(block_size,), dtype="f8", shards=scalar_shard)
        outfile.create_array("param", shape=(n_samples,), chunks=(block_size,), dtype="f8", shards=scalar_shard)
    else:
        outfile.create_array("data", shape=data_shape, chunks=data_chunks, dtype="f4")
        outfile.create_array("next", shape=data_shape, chunks=data_chunks, dtype="f4")
        outfile.create_array("time", shape=(n_samples,), chunks=(block_size,), dtype="f8")
        outfile.create_array("param", shape=(n_samples,), chunks=(block_size,), dtype="f8")
    return outfile


def _process(
    input: str,
    output: str,
    block_size: int,
    trajectory_block_size: int,
) -> int:
    infile = cast(zarr.Group, zarr.open(input, mode="r"))
    indata = open_zarr_array(infile, "data")
    inparam = open_zarr_array(infile, "param", indata.shape[0])

    n_traj, n_time = indata.shape[:2]
    state_shape: tuple[int, ...] = indata.shape[2:]
    n_steps = n_time - 1
    n_samples = n_traj * n_steps

    auto_bs, auto_tbs = auto_block_sizes(state_shape, n_time)
    block_size = block_size or auto_bs
    trajectory_block_size = trajectory_block_size or auto_tbs

    # Clamp trajectory_block_size up so each write covers at least one full shard.
    # Without this, small trajectory blocks cause zarr to read-modify-write the
    # entire shard file (potentially ~1 GB) for every iteration.
    n_bps = auto_blocks_per_shard(block_size, state_shape, dtype_itemsize=4)
    shard_size = n_bps * block_size
    min_tbs_for_alignment = math.ceil(shard_size / n_steps)
    trajectory_block_size = max(trajectory_block_size, min_tbs_for_alignment)

    logging.info(
        "block_size=%d, trajectory_block_size=%d (shard=%d samples)",
        block_size, trajectory_block_size, shard_size,
    )

    if "time" in infile:
        time_vector = np.array(infile["time"])
    elif "conditioning" in infile:
        time_vector = np.array(infile["conditioning"])
    else:
        time_vector = np.arange(n_time, dtype=np.float64)

    outfile = _make_outfile(output, n_samples, block_size, state_shape)

    sum_diff = 0.0
    sum_sq_diff = 0.0
    count_diff = 0

    for traj_start in tqdm(
        range(0, n_traj, trajectory_block_size),
        desc="Processing trajectories",
    ):
        traj_end = min(traj_start + trajectory_block_size, n_traj)

        data_block = np.array(
            indata[traj_start:traj_end]
        )  # (block, n_time, *state_shape)
        param_block = np.array(inparam[traj_start:traj_end])  # (block,)

        cur, nxt, time_flat, param_flat = extract_regression_pairs(
            data_block, time_vector, param_block
        )

        diff = nxt - cur
        sum_diff += float(np.sum(diff))
        sum_sq_diff += float(np.sum(diff**2))
        count_diff += diff.size

        sample_start = traj_start * n_steps
        sample_end = traj_end * n_steps
        outfile["data"][sample_start:sample_end] = cur  # pyright: ignore[reportArgumentType]
        outfile["next"][sample_start:sample_end] = nxt  # pyright: ignore[reportArgumentType]
        outfile["time"][sample_start:sample_end] = time_flat  # pyright: ignore[reportArgumentType]
        outfile["param"][sample_start:sample_end] = param_flat  # pyright: ignore[reportArgumentType]

    mean_diff = sum_diff / count_diff
    diff_scale = float(np.sqrt(sum_sq_diff / count_diff - mean_diff**2))
    outfile.attrs["diff_scale"] = diff_scale
    logging.info("diff_scale = %.6g", diff_scale)
    return block_size


@hydra.main(version_base=None, config_name="regression_data", config_path="../../conf")
@log_duration()
def main(cfg: RegressionDataConfig) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    with init_wandb(cfg, "regression-data", data_name=cfg.dataset or None):
        _process(
            cfg.input,
            cfg.output,
            cfg.block_size,
            cfg.trajectory_block_size,
        )
        logging.info("Saved regression data to %s", cfg.output)


if __name__ == "__main__":
    main()
