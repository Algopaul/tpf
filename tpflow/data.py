import os

import numpy as np
import zarr

from tpflow.config import DataConfig


def block_shuffle(arr, block_size, seed):
  rng = np.random.default_rng(seed)
  n = arr.shape[0]
  n_full = (n // block_size) * block_size
  out = np.empty_like(arr)
  blocks = arr[:n_full].reshape(-1, block_size, *arr.shape[1:])
  perm = rng.permutation(blocks.shape[0])
  out[:n_full] = blocks[perm].reshape(n_full, *arr.shape[1:])
  if n_full < n:
    out[n_full:] = arr[n_full:]
  return out


def get_data(cfg: DataConfig):
  basedir = f'data/datasets/{cfg.name}'
  in_filename = os.path.join(basedir, 'cfm_train_data', 'train.zarr')
  file = zarr.open(in_filename, mode='r')
  split_data = {}
  n_batches = None

  for field in cfg.fields:
    d = block_shuffle(np.array(file[field]), cfg.shuffle_block_size, 0)
    batches = len(d) // cfg.batch_size

    if n_batches is None:
      n_batches = batches
    else:
      assert n_batches == batches

    split_data[field] = np.array_split(d, batches)

  assert n_batches is not None
  keys = list(split_data.keys())
  batch_list = [{k: split_data[k][i] for k in keys} for i in range(n_batches)]
  return batch_list
