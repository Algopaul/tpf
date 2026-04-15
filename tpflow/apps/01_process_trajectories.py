import logging
import math
from os.path import join
from typing import cast

import hydra
import numpy as np
import zarr
from flanch.trajectories import flatten_trajectories
from hdfx.statistics import ds_statistics
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import TrajectoryProcessing
from tpflow.processing import auto_blocks_per_shard, open_zarr_array
from tpflow.util import init_wandb, log_duration

# Input must have fields
# data: (n_samples, n_times, sample_shape)
# param: (n_samples, )  parameter is scalar and stays constant throughout trajectory

# Train data:
#   (a) chunk up and train data into individual time points
# Store test trajectories in one zarr
# Chunk everything correctly.


@hydra.main(version_base=None, config_name="config", config_path="../../conf")
@log_duration()
def main(cfg: TrajectoryProcessing) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    basedir = f"data/datasets/{cfg.data.name}"
    with init_wandb(cfg, "traj-orga"):
        logging.info(cfg.data.name)

        # Compute normalisation stats from the training split only, then reuse
        # for all splits so the test split lives in the same normalised space.
        train_raw = join(basedir, "raw_trajectories", "train.zarr")
        train_infile = cast(zarr.Group, zarr.open(train_raw, mode="r"))
        train_indata = open_zarr_array(train_infile, "data")
        data_mean, data_std = ds_statistics(train_indata)
        logging.info(
            "Normalisation stats (from train): mean=%s std=%s",
            np.asarray(data_mean),
            np.asarray(data_std),
        )

        # Optional per-time normalization: one (mean, std) per (time step, channel),
        # averaging over all trajectories and spatial dimensions.
        # state_shape[-1] is treated as the channel axis.
        pt_mean_bc: np.ndarray | None = None
        pt_std_bc: np.ndarray | None = None
        per_time_mean: np.ndarray | None = None
        per_time_std: np.ndarray | None = None
        if cfg.normalize_per_time:
            n_traj_train = train_indata.shape[0]
            n_time_train = train_indata.shape[1]
            state_shape_train = train_indata.shape[2:]
            n_channels = state_shape_train[-1]
            n_spatial = len(state_shape_train) - 1
            # axes in a (B, T, *state_shape) block to sum over: batch + all spatial
            spatial_sum_axes = (0,) + tuple(range(2, 2 + n_spatial))
            n_per_step = n_traj_train * int(np.prod(state_shape_train[:-1]))

            sum_x = np.zeros((n_time_train, n_channels), dtype=np.float64)
            sum_x2 = np.zeros((n_time_train, n_channels), dtype=np.float64)
            for traj_start in tqdm(
                range(0, n_traj_train, cfg.data.trajectory_block_size),
                desc="Per-time stats",
            ):
                traj_end = min(traj_start + cfg.data.trajectory_block_size, n_traj_train)
                block = np.array(train_indata[traj_start:traj_end]).astype(np.float64)
                sum_x += np.sum(block, axis=spatial_sum_axes)
                sum_x2 += np.sum(block ** 2, axis=spatial_sum_axes)
            per_time_mean = sum_x / n_per_step                           # (n_time, C)
            per_time_std = np.sqrt(sum_x2 / n_per_step - per_time_mean ** 2)
            per_time_std = np.where(per_time_std == 0, 1.0, per_time_std)
            logging.info(
                "Per-time stats: mean range [%.4g, %.4g], std range [%.4g, %.4g]",
                float(per_time_mean.min()), float(per_time_mean.max()),
                float(per_time_std.min()), float(per_time_std.max()),
            )
            # Broadcast shape (1, n_time, 1, ..., 1, C) — same for both splits
            bc_shape = (1, n_time_train) + (1,) * n_spatial + (n_channels,)
            pt_mean_bc = per_time_mean.reshape(bc_shape)
            pt_std_bc = per_time_std.reshape(bc_shape)

        for split in ["train", "test"]:
            in_filename = join(basedir, "raw_trajectories", split + ".zarr")
            out_filename = join(basedir, "cfm_train_data", split + ".zarr")

            infile = cast(zarr.Group, zarr.open(in_filename, mode="r"))
            indata = open_zarr_array(infile, "data")
            inparam = open_zarr_array(infile, "param", indata.shape[0])

            n_traj, n_time = indata.shape[:2]
            sample_shape = indata.shape[2:]
            n_samples = n_traj * n_time
            outfile = make_outfile(
                out_filename,
                n_samples,
                cfg.data.block_size,
                sample_shape,
            )

            if split == "train":
                outfile.attrs["data_mean"] = np.asarray(data_mean).tolist()
                outfile.attrs["data_std"] = np.asarray(data_std).tolist()
                if per_time_mean is not None:
                    outfile.attrs["per_time_mean"] = per_time_mean.tolist()
                    outfile.attrs["per_time_std"] = per_time_std.tolist()  # type: ignore[union-attr]

            time_vector = np.array(infile["time"])
            for traj_start in tqdm(
                range(0, n_traj, cfg.data.trajectory_block_size),
                desc="Flatten trajectories",
            ):
                traj_end = min(traj_start + cfg.data.trajectory_block_size, n_traj)
                raw_block = np.array(indata[traj_start:traj_end])
                if pt_mean_bc is not None and pt_std_bc is not None:
                    data_block = (raw_block - pt_mean_bc) / pt_std_bc
                else:
                    data_block = (raw_block - data_mean) / data_std
                param_block = np.array(inparam[traj_start:traj_end])

                flat_data, flat_time, flat_param = flatten_trajectories(
                    data_block, time_vector, param_block
                )

                sample_start = traj_start * n_time
                sample_end = traj_end * n_time
                outfile["data"][sample_start:sample_end] = flat_data  # pyright: ignore[reportArgumentType]
                outfile["time"][sample_start:sample_end] = flat_time  # pyright: ignore[reportArgumentType]
                outfile["param"][sample_start:sample_end] = flat_param  # pyright: ignore[reportArgumentType]


def make_outfile(name, n_samples, block_size, sample_shape):
    outfile = zarr.create_group(name, overwrite=True)
    # f8 dtype in app 01 → 8 bytes per scalar value.
    # Shard whenever n_samples > block_size (i.e. more than one inner chunk).
    # Small test splits produce one shard file; unit-test arrays (n_samples ≤
    # block_size) skip sharding entirely to avoid codec overhead.
    n_bps = auto_blocks_per_shard(block_size, sample_shape, dtype_itemsize=4)
    shard_size = n_bps * block_size
    use_shards = n_samples > block_size
    logging.info(
        "shard_size=%d samples (%.1f GB)%s",
        shard_size,
        shard_size * max(1, math.prod(sample_shape)) * 4 / 1024**3,
        "" if use_shards else " — skipped, dataset fits in one shard",
    )

    arrays = {
        "data": {"shape": (n_samples, *sample_shape), "chunks": (block_size, *sample_shape)},
        "time": {"shape": (n_samples,), "chunks": (block_size,)},
        "param": {"shape": (n_samples,), "chunks": (block_size,)},
    }
    if use_shards:
        arrays["data"]["shards"] = (shard_size, *sample_shape)
        arrays["time"]["shards"] = (shard_size,)
        arrays["param"]["shards"] = (shard_size,)

    for array_name, config in arrays.items():
        outfile.create_array(array_name, dtype="f4", **config)

    return outfile


if __name__ == "__main__":
    main()
