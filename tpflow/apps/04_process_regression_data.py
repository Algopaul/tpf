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
from typing import cast

import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import RegressionDataConfig
from tpflow.processing import (
    auto_block_sizes,
    extract_regression_pairs,
    open_zarr_array,
)
from tpflow.util import init_wandb, log_duration


def _make_outfile(
    path: str,
    n_samples: int,
    block_size: int,
    blocks_per_shard: int,
    state_shape: tuple,
) -> zarr.Group:
    outfile = zarr.open_group(path, mode="w")
    shard_size = blocks_per_shard * block_size
    outfile.create_array(
        "data",
        shape=(n_samples, *state_shape),
        chunks=(block_size, *state_shape),
        shards=(shard_size, *state_shape),
        dtype="f4",
    )
    outfile.create_array(
        "next",
        shape=(n_samples, *state_shape),
        chunks=(block_size, *state_shape),
        shards=(shard_size, *state_shape),
        dtype="f4",
    )
    outfile.create_array(
        "time",
        shape=(n_samples,),
        chunks=(block_size,),
        shards=(shard_size,),
        dtype="f8",
    )
    outfile.create_array(
        "param",
        shape=(n_samples,),
        chunks=(block_size,),
        shards=(shard_size,),
        dtype="f8",
    )
    return outfile


def _process(
    input: str,
    output: str,
    block_size: int,
    blocks_per_shard: int,
    trajectory_block_size: int,
) -> int:
    infile = cast(zarr.Group, zarr.open(input, mode="r"))
    indata = open_zarr_array(infile, "data")
    inparam = open_zarr_array(infile, "param", indata.shape[0])

    n_traj, n_time = indata.shape[:2]
    state_shape = indata.shape[2:]
    n_steps = n_time - 1
    n_samples = n_traj * n_steps

    auto_bs, auto_tbs = auto_block_sizes(state_shape, n_time)
    block_size = block_size or auto_bs
    trajectory_block_size = trajectory_block_size or auto_tbs
    logging.info(
        "block_size=%d, trajectory_block_size=%d", block_size, trajectory_block_size
    )

    if "time" in infile:
        time_vector = np.array(infile["time"])
    else:
        time_vector = np.arange(n_time, dtype=np.float64)

    outfile = _make_outfile(
        output, n_samples, block_size, blocks_per_shard, state_shape
    )

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
    with init_wandb(cfg, "regression-data"):
        _process(
            cfg.input,
            cfg.output,
            cfg.block_size,
            cfg.blocks_per_shard,
            cfg.trajectory_block_size,
        )
        logging.info("Saved regression data to %s", cfg.output)


if __name__ == "__main__":
    main()
