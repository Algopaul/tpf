import logging
from os.path import join
from typing import cast

import hydra
import numpy as np
import zarr
from flanch.trajectories import flatten_trajectories
from hdfx.shuffle import zarrshuffle
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import TrajectoryProcessing
from tpflow.util import init_wandb, log_duration

# Input must have fields
# data: (n_samples, n_times, sample_shape)
# param: (n_samples, )  parameter is scalar and stays constant throughout trajectory

# Train data:
#   (a) chunk up and train data into individual time points
# Store test trajectories in one zarr
# Chunk everything correctly.


@hydra.main(version_base=None, config_name='config', config_path='../../conf')
@log_duration()
def main(cfg: TrajectoryProcessing) -> None:
  logging.info("\n%s", OmegaConf.to_yaml(cfg))
  basedir = f'data/datasets/{cfg.data.name}'
  with init_wandb(cfg, 'traj-orga'):
    logging.info(cfg.data.name)
    for split in ['train', 'test']:
      in_filename = join(basedir, 'raw_trajectories', split + '.zarr')
      out_filename = join(basedir, 'cfm_train_data', split + '.zarr')

      infile = cast(zarr.Group, zarr.open(in_filename, mode='r'))
      indata = get_array(infile, 'data')
      inparam = get_array(infile, 'param')

      n_traj, n_time = indata.shape[:2]
      sample_shape = indata.shape[2:]
      n_samples = n_traj * n_time
      outfile = make_outfile(
          out_filename,
          n_samples,
          cfg.block_size,
          n_time,
          sample_shape,
      )

      time_vector = np.array(infile['time'])
      for traj_start in tqdm(range(0, n_traj, cfg.block_size)):
        traj_end = min(traj_start + cfg.block_size, n_traj)
        data_block = np.array(indata[traj_start:traj_end])
        param_block = np.array(inparam[traj_start:traj_end])

        flat_data, flat_time, flat_param = flatten_trajectories(
            data_block, time_vector, param_block)

        sample_start = traj_start * n_time
        sample_end = traj_end * n_time
        get_array(outfile, 'data')[sample_start:sample_end] = flat_data
        get_array(outfile, 'time')[sample_start:sample_end] = flat_time
        get_array(outfile, 'param')[sample_start:sample_end] = flat_param

      if split == 'train':
        shuffled_out = join(basedir, 'cfm_train_data', 'train_shuffled.zarr')
        zarrshuffle(out_filename, shuffled_out, cfg.data.shuffle_block_size, 0)


def make_outfile(name, n_samples, block_size, n_time, sample_shape):
  outfile = zarr.create_group(name, overwrite=True)
  outfile.create_array(
      'data',
      shape=(n_samples, *sample_shape),
      dtype='f8',
      chunks=(block_size * n_time, *sample_shape),
  )
  outfile.create_array(
      'time',
      chunks=(block_size * n_time,),
      shape=(n_samples,),
      dtype='f8',
  )
  outfile.create_array(
      'param',
      chunks=(block_size * n_time,),
      shape=(n_samples,),
      dtype='f8',
  )
  return outfile


def get_array(group: zarr.Group, name: str) -> zarr.Array:
  obj = group[name]
  if not isinstance(obj, zarr.Array):
    raise TypeError(f"{name} is not an array")
  return obj


if __name__ == "__main__":
  main()
