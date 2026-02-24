import os

import jax
import numpy as np
import zarr

from tpflow.config import DataConfig


def device_prefetch(it, size=2):
  it = iter(it)
  queue = []

  for _ in range(size):
    try:
      queue.append(jax.device_put(next(it)))
    except StopIteration:
      break

  while queue:
    yield queue.pop(0)
    try:
      queue.append(jax.device_put(next(it)))
    except StopIteration:
      pass


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


def get_data(cfg: DataConfig, split):
  basedir = f'data/datasets/{cfg.name}'
  in_filename = os.path.join(basedir, 'cfm_train_data', f'{split}.zarr')
  file = zarr.open(in_filename, mode='r')
  split_data = {}
  n_batches = None

  for field in cfg.fields:
    d = np.array(file[field])
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


class ZarrData:

  def __init__(self, cfg: DataConfig, split: str):
    self.cfg = DataConfig
    self.split = split
    basedir = f'data/datasets/{cfg.name}'
    in_filename = os.path.join(basedir, 'cfm_train_data', f'{split}.zarr')
    file = zarr.open(in_filename, mode='r')
    self.n_blocks = None
    self.n_batches = None
    self.split_data = {}
    self.fields = cfg.fields
    assert cfg.batch_size % cfg.shuffle_block_size == 0

    for field in cfg.fields:
      d = np.array(file[field])
      batches = len(d) // cfg.batch_size
      blocks = (batches * cfg.batch_size) // cfg.shuffle_block_size

      if self.n_blocks is None:
        self.n_blocks = blocks
      else:
        assert self.n_blocks == blocks

      if self.n_batches is None:
        self.n_batches = batches
      else:
        assert self.n_batches == batches

      self.split_data[field] = np.array(
          np.split(
              d[:self.n_batches * cfg.batch_size],
              self.n_blocks,
          ))

  def __len__(self):
    assert self.n_batches is not None
    return self.n_batches

  def iter_batches(self, seed=None):
    rng = np.random.default_rng(seed)
    assert self.n_blocks is not None
    assert self.n_batches is not None
    indices = rng.permutation(self.n_blocks)
    batch_idcs = np.array(np.split(indices, self.n_batches))
    for b in batch_idcs:
      yield {
          f: np.concatenate(self.split_data[f][b], axis=0) for f in self.fields
      }
