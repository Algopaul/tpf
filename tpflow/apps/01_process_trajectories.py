import logging
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
from tpflow.processing import open_zarr_array
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
                cfg.data.blocks_per_shard,
                sample_shape,
            )
            data_mean, data_std = ds_statistics(indata)

            time_vector = np.array(infile["time"])
            for traj_start in tqdm(
                range(0, n_traj, cfg.data.trajectory_block_size),
                desc="Flatten trajectories",
            ):
                traj_end = min(traj_start + cfg.data.trajectory_block_size, n_traj)
                data_block = (
                    np.array(indata[traj_start:traj_end]) - data_mean
                ) / data_std
                param_block = np.array(inparam[traj_start:traj_end])

                flat_data, flat_time, flat_param = flatten_trajectories(
                    data_block, time_vector, param_block
                )

                sample_start = traj_start * n_time
                sample_end = traj_end * n_time
                outfile["data"][sample_start:sample_end] = flat_data  # pyright: ignore[reportArgumentType]
                outfile["time"][sample_start:sample_end] = flat_time  # pyright: ignore[reportArgumentType]
                outfile["param"][sample_start:sample_end] = flat_param  # pyright: ignore[reportArgumentType]


def make_outfile(name, n_samples, block_size, blocks_per_shard, sample_shape):
    outfile = zarr.create_group(name, overwrite=True)
    shard_size = blocks_per_shard * block_size

    arrays = {
        "data": {
            "shape": (n_samples, *sample_shape),
            "chunks": (block_size, *sample_shape),
            "shards": (shard_size, *sample_shape),
        },
        "time": {
            "shape": (n_samples,),
            "chunks": (block_size,),
            "shards": (shard_size,),
        },
        "param": {
            "shape": (n_samples,),
            "chunks": (block_size,),
            "shards": (shard_size,),
        },
    }

    for array_name, config in arrays.items():
        outfile.create_array(array_name, dtype="f8", **config)

    return outfile


if __name__ == "__main__":
    main()
