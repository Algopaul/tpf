import io
import json
import os

import jax
import numpy as np
import webdataset as wds
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
    basedir = f"data/datasets/{cfg.name}"
    in_filename = os.path.join(basedir, "cfm_train_data", f"{split}.zarr")
    file = zarr.open(in_filename, mode="r")
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
        basedir = f"data/datasets/{cfg.name}"
        in_filename = os.path.join(basedir, "cfm_train_data", f"{split}.zarr")
        file = zarr.open(in_filename, mode="r")
        self.n_blocks = None
        self.n_batches = None
        self.split_data = {}
        self.fields = cfg.fields
        assert cfg.batch_size % cfg.block_size == 0

        for field in cfg.fields:
            d = np.array(file[field])
            batches = len(d) // cfg.batch_size
            blocks = (batches * cfg.batch_size) // cfg.block_size

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
                    d[: self.n_batches * cfg.batch_size],
                    self.n_blocks,
                )
            )

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


class RegressionZarrData:
    """Zarr-backed dataloader for regression training data.

    Expects arrays: data, next, time, param — all at a direct zarr path
    (as written by 04_process_regression_data.py).
    """

    _FIELDS = ("data", "next", "time", "step_size", "param")

    def __init__(self, path: str, batch_size: int, block_size: int):
        assert batch_size % block_size == 0
        file = zarr.open(path, mode="r")
        n_samples = len(file["data"])
        n_batches = n_samples // batch_size
        n_blocks = n_batches * (batch_size // block_size)
        self.n_batches = n_batches
        self.n_blocks = n_blocks
        self.split_data = {}
        for field in self._FIELDS:
            d = np.array(file[field])
            self.split_data[field] = np.array(
                np.split(d[: n_batches * batch_size], n_blocks)
            )

    def __len__(self):
        return self.n_batches

    def iter_batches(self, seed=None):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_blocks)
        batch_idcs = np.array(np.split(indices, self.n_batches))
        for b in batch_idcs:
            yield {
                f: np.concatenate(self.split_data[f][b], axis=0) for f in self._FIELDS
            }


def get_regression_val_data(path: str, batch_size: int) -> list[dict]:
    file = zarr.open(path, mode="r")
    fields = ("data", "next", "time", "step_size", "param")
    split_data = {}
    n_batches = None
    for field in fields:
        d = np.array(file[field])
        n = len(d) // batch_size
        if n_batches is None:
            n_batches = n
        split_data[field] = np.array_split(d[: n * batch_size], n)
    assert n_batches is not None
    return [{f: split_data[f][i] for f in fields} for i in range(n_batches)]


def _decode_npy(key, data):
    if key.endswith(".npy"):
        return np.load(io.BytesIO(data))
    return data


class WDSData:
    """Streaming dataloader backed by WebDataset tar shards.

    Drop-in replacement for ZarrData. Each shard entry is a block of
    `block_size` samples (set at conversion time). Blocks are shuffled and
    concatenated into batches of `cfg.batch_size` samples.

    Args:
      cfg: DataConfig. batch_size and block_size must match the converted shards.
      split: Dataset split name (e.g. 'train_shuffled', 'test').
      shuffle_buffer: Number of blocks buffered for shuffling. Larger values
        give better randomness at the cost of memory. Set to 0 to disable.
    """

    def __init__(self, cfg: DataConfig, split: str, shuffle_buffer: int = 100):
        self.cfg = cfg
        self.fields = cfg.fields

        wds_dir = f"data/datasets/{cfg.name}/cfm_train_data_wds/{split}"
        with open(os.path.join(wds_dir, "metadata.json")) as f:
            meta = json.load(f)

        self.block_size = meta["block_size"]
        assert self.block_size == cfg.block_size, (
            f"WDS block_size {self.block_size} != cfg.block_size {cfg.block_size}"
        )
        assert cfg.batch_size % self.block_size == 0
        self.blocks_per_batch = cfg.batch_size // self.block_size

        n_samples = meta["n_samples"]
        self.n_batches = n_samples // cfg.batch_size
        self.shuffle_buffer = shuffle_buffer

        import glob as _glob

        self._shards = sorted(_glob.glob(os.path.join(wds_dir, "shard-*.tar")))

    def __len__(self):
        return self.n_batches

    def iter_batches(self, seed=None):
        rng = np.random.default_rng(seed)
        shards = list(self._shards)
        rng.shuffle(shards)

        dataset = (
            wds.WebDataset(shards, shardshuffle=False)
            .decode(_decode_npy)
            .to_tuple(*[f"{f}.npy" for f in self.fields])
        )
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)

        blocks: dict[str, list] = {f: [] for f in self.fields}
        batches_yielded = 0

        for sample in dataset:
            for i, f in enumerate(self.fields):
                blocks[f].append(sample[i])
            if len(blocks[self.fields[0]]) == self.blocks_per_batch:
                yield {f: np.concatenate(blocks[f], axis=0) for f in self.fields}
                blocks = {f: [] for f in self.fields}
                batches_yielded += 1
                if batches_yielded >= self.n_batches:
                    break
