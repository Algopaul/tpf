"""Convert zarr cfm-train-data files to WebDataset tar shards.

Each WebDataset sample is one block of `data.block_size` zarr rows. Blocks
are the unit of shuffling in WDSData, matching the ZarrData shuffle granularity.
Shards contain `blocks_per_shard` blocks each.

Output layout:
    data/datasets/{name}/cfm_train_data_wds/{split}/
        metadata.json          # n_samples, block_size, and field shapes
        shard-000000.tar
        shard-000001.tar
        ...

Usage:
    python tpflow/apps/01b_convert_to_wds.py
    python tpflow/apps/01b_convert_to_wds.py data.name=imgrot splits='[train_shuffled]'
    python tpflow/apps/01b_convert_to_wds.py data.block_size=5000 blocks_per_shard=1000
"""

import io
import json
import logging
import os

import hydra
import numpy as np
import webdataset as wds
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import WDSConvertConfig
from tpflow.util import log_duration


def convert_split(name: str, split: str, fields: tuple, block_size: int,
                  blocks_per_shard: int):
  src = f'data/datasets/{name}/cfm_train_data/{split}.zarr'
  dst_dir = f'data/datasets/{name}/cfm_train_data_wds/{split}'
  os.makedirs(dst_dir, exist_ok=True)

  z = zarr.open(src)
  arr0 = z[fields[0]]
  assert isinstance(arr0, zarr.Array)
  n = arr0.shape[0]
  n_blocks = n // block_size

  field_shapes = {}
  for f in fields:
    arr = z[f]
    assert isinstance(arr, zarr.Array)
    field_shapes[f] = list(arr.shape[1:])

  logging.info('%s: %d samples → %d blocks of %d → shards of %d blocks',
               split, n, n_blocks, block_size, blocks_per_shard)

  metadata = {'n_samples': n, 'block_size': block_size, 'fields': field_shapes}
  with open(os.path.join(dst_dir, 'metadata.json'), 'w') as mf:
    json.dump(metadata, mf)

  shard_pattern = os.path.join(dst_dir, 'shard-%06d.tar')
  with wds.ShardWriter(shard_pattern, maxcount=blocks_per_shard) as writer:
    for block_idx in tqdm(range(n_blocks), desc=split):
      start = block_idx * block_size
      end = start + block_size
      sample: dict[str, object] = {'__key__': f'{block_idx:08d}'}
      for f in fields:
        arr = z[f]
        assert isinstance(arr, zarr.Array)
        buf = io.BytesIO()
        np.save(buf, np.array(arr[start:end]))
        sample[f'{f}.npy'] = buf.getvalue()
      writer.write(sample)

  logging.info('Wrote %d blocks to %s', n_blocks, dst_dir)


@hydra.main(
    version_base=None, config_name='wds_convert', config_path='../../conf')
@log_duration()
def main(cfg: WDSConvertConfig) -> None:
  logging.info('\n%s', OmegaConf.to_yaml(cfg))
  for split in cfg.splits:
    convert_split(
        cfg.data.name,
        split,
        cfg.data.fields,
        cfg.data.block_size,
        cfg.blocks_per_shard,
    )


if __name__ == '__main__':
  main()
