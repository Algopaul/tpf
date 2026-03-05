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
from hdfx.shuffle import zarrshuffle
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import RegressionDataConfig
from tpflow.util import init_wandb, log_duration


def _auto_block_sizes(
    state_shape: tuple, n_time: int, target_mb: float = 2.0
) -> tuple[int, int]:
    """Return (block_size, trajectory_block_size) targeting ~target_mb per chunk."""
    target_bytes = int(target_mb * 1024 * 1024)
    bytes_per_sample = max(1, math.prod(state_shape)) * 4  # f4 output dtype
    block_size = max(1, target_bytes // bytes_per_sample)
    traj_block_size = max(1, target_bytes // (n_time * bytes_per_sample))
    return block_size, traj_block_size


def _get_array(
    group: zarr.Group,
    name: str,
    size: int | None = None,
) -> zarr.Array | np.ndarray:
    if name not in group:
        if size is not None:
            return np.ones((size,))
        raise KeyError(f"'{name}' not found in group and no size provided")
    obj = group[name]
    if not isinstance(obj, zarr.Array):
        raise TypeError(f"'{name}' is not a zarr.Array")
    return obj


def _make_outfile(
    path: str, n_samples: int, block_size: int, state_shape: tuple
) -> zarr.Group:
    outfile = zarr.open_group(path, mode="w")
    outfile.create_array(
        "data",
        shape=(n_samples, *state_shape),
        chunks=(block_size, *state_shape),
        dtype="f4",
    )
    outfile.create_array(
        "next",
        shape=(n_samples, *state_shape),
        chunks=(block_size, *state_shape),
        dtype="f4",
    )
    outfile.create_array("time", shape=(n_samples,), chunks=(block_size,), dtype="f8")
    outfile.create_array("param", shape=(n_samples,), chunks=(block_size,), dtype="f8")
    return outfile


def _process(
    input: str, output: str, block_size: int, trajectory_block_size: int
) -> None:
    infile = cast(zarr.Group, zarr.open(input, mode="r"))
    indata = _get_array(infile, "data")
    inparam = _get_array(infile, "param", indata.shape[0])

    n_traj, n_time = indata.shape[:2]
    state_shape = indata.shape[2:]
    n_steps = n_time - 1
    n_samples = n_traj * n_steps

    auto_bs, auto_tbs = _auto_block_sizes(state_shape, n_time)
    block_size = block_size or auto_bs
    trajectory_block_size = trajectory_block_size or auto_tbs
    logging.info(
        "block_size=%d, trajectory_block_size=%d", block_size, trajectory_block_size
    )

    if "time" in infile:
        time_vector = np.array(infile["time"])
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
        block = traj_end - traj_start

        data_block = np.array(
            indata[traj_start:traj_end]
        )  # (block, n_time, *state_shape)
        param_block = np.array(inparam[traj_start:traj_end])  # (block,)

        cur = data_block[:, :-1].reshape(block * n_steps, *state_shape)
        nxt = data_block[:, 1:].reshape(block * n_steps, *state_shape)
        time_flat = np.tile(time_vector[:-1], block)
        param_flat = np.repeat(param_block, n_steps)

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


@hydra.main(version_base=None, config_name="regression_data", config_path="../../conf")
@log_duration()
def main(cfg: RegressionDataConfig) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    with init_wandb(cfg, "regression-data"):
        _process(cfg.input, cfg.output, cfg.block_size, cfg.trajectory_block_size)

        if cfg.shuffle:
            shuffled_out = cfg.output.replace(".zarr", "_shuffled.zarr")
            logging.info("Shuffling to %s", shuffled_out)
            zarrshuffle(cfg.output, shuffled_out, cfg.block_size, 0)

        logging.info("Saved regression data to %s", cfg.output)


if __name__ == "__main__":
    main()
