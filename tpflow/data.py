import io
import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast

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


def _prefetch_shards(
    arrays: dict[str, zarr.Array],
    shard_order: np.ndarray,
    shard_size: int,
):
    """Iterate shards, loading the next one in a background thread while the
    caller processes the current one.  Yields one shard dict per iteration."""
    fields = list(arrays.keys())

    def _load(idx: int) -> dict[str, np.ndarray]:
        start = idx * shard_size
        end = start + shard_size
        return {f: np.array(arrays[f][start:end]) for f in fields}  # pyright: ignore[reportArgumentType]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future: Future[dict[str, np.ndarray]] | None = pool.submit(_load, int(shard_order[0]))
        for i, shard_idx in enumerate(shard_order):
            shard = future.result()
            # Kick off next load before yielding, so I/O overlaps with training
            if i + 1 < len(shard_order):
                future = pool.submit(_load, int(shard_order[i + 1]))
            yield shard  # type: ignore[misc]


class ZarrData:
    """Block-shuffled dataloader for sharded zarr CFM training data.

    Iterates shard by shard: shuffles shard order, then within each shard
    shuffles blocks before assembling batches. Memory footprint is one shard
    at a time rather than the full dataset.
    """

    def __init__(self, cfg: DataConfig, split: str):
        basedir = f"data/datasets/{cfg.name}"
        in_filename = os.path.join(basedir, "cfm_train_data", f"{split}.zarr")
        group = cast(zarr.Group, zarr.open(in_filename, mode="r"))
        self._fields = cfg.fields
        self._block_size = cfg.block_size
        self._batch_size = cfg.batch_size
        assert cfg.batch_size % cfg.block_size == 0
        self._blocks_per_batch = cfg.batch_size // cfg.block_size

        self._arrays: dict[str, zarr.Array] = {
            f: cast(zarr.Array, group[f]) for f in cfg.fields
        }
        first = self._arrays[cfg.fields[0]]
        n_samples = first.shape[0]
        shard_shape = first.shards
        shard_size = shard_shape[0] if shard_shape is not None else n_samples
        assert shard_size % cfg.block_size == 0, (
            f"shard_size {shard_size} must be divisible by block_size {cfg.block_size}"
        )
        self._shard_size = shard_size
        self._n_shards = n_samples // shard_size
        self._blocks_per_shard = shard_size // cfg.block_size
        self._n_batches = (self._n_shards * shard_size) // cfg.batch_size

    def __len__(self):
        return self._n_batches

    def iter_batches(self, seed=None):
        rng = np.random.default_rng(seed)
        shard_order = rng.permutation(self._n_shards)
        block_buffer: dict[str, list] = {f: [] for f in self._fields}
        n_buffered = 0
        batches_yielded = 0

        for shard in _prefetch_shards(self._arrays, shard_order, self._shard_size):
            for b in rng.permutation(self._blocks_per_shard):
                bs, be = int(b) * self._block_size, (int(b) + 1) * self._block_size
                for f in self._fields:
                    block_buffer[f].append(shard[f][bs:be])
                n_buffered += 1
                if n_buffered == self._blocks_per_batch:
                    yield {
                        f: np.concatenate(block_buffer[f], axis=0) for f in self._fields
                    }
                    block_buffer = {f: [] for f in self._fields}
                    n_buffered = 0
                    batches_yielded += 1
                    if batches_yielded >= self._n_batches:
                        return


class RegressionZarrData:
    """Shard-aware dataloader for regression training data.

    Iterates shard by shard: shuffles shard order, then within each shard
    shuffles blocks before assembling batches. Memory footprint is one shard
    at a time rather than the full dataset.
    """

    _FIELDS = ("data", "next", "time", "param")

    def __init__(self, path: str, batch_size: int, block_size: int):
        assert batch_size % block_size == 0
        group = cast(zarr.Group, zarr.open(path, mode="r"))
        self.diff_scale: np.ndarray = np.array(
            group.attrs.get("diff_scale", 1.0), dtype=np.float32
        )
        self._block_size = block_size
        self._batch_size = batch_size
        self._blocks_per_batch = batch_size // block_size

        self._arrays: dict[str, zarr.Array] = {
            f: cast(zarr.Array, group[f]) for f in self._FIELDS
        }
        first = self._arrays["data"]
        n_samples = first.shape[0]
        shard_shape = first.shards
        shard_size = shard_shape[0] if shard_shape is not None else n_samples
        assert shard_size % block_size == 0, (
            f"shard_size {shard_size} must be divisible by block_size {block_size}"
        )
        self._shard_size = shard_size
        self._n_shards = n_samples // shard_size
        self._blocks_per_shard = shard_size // block_size
        self._n_batches = (self._n_shards * shard_size) // batch_size

    def __len__(self):
        return self._n_batches

    def iter_batches(self, seed=None):
        rng = np.random.default_rng(seed)
        shard_order = rng.permutation(self._n_shards)
        block_buffer: dict[str, list] = {f: [] for f in self._FIELDS}
        n_buffered = 0
        batches_yielded = 0

        for shard in _prefetch_shards(self._arrays, shard_order, self._shard_size):
            for b in rng.permutation(self._blocks_per_shard):
                bs, be = int(b) * self._block_size, (int(b) + 1) * self._block_size
                for f in self._FIELDS:
                    block_buffer[f].append(shard[f][bs:be])
                n_buffered += 1
                if n_buffered == self._blocks_per_batch:
                    yield {
                        f: np.concatenate(block_buffer[f], axis=0) for f in self._FIELDS
                    }
                    block_buffer = {f: [] for f in self._FIELDS}
                    n_buffered = 0
                    batches_yielded += 1
                    if batches_yielded >= self._n_batches:
                        return


def get_regression_val_data(path: str, batch_size: int, max_samples: int = 0) -> list[dict]:
    file = cast(zarr.Group, zarr.open(path, mode="r"))
    fields = ("data", "next", "time", "param")
    n_available = cast(zarr.Array, file["data"]).shape[0]
    n_load = min(n_available, max_samples) if max_samples > 0 else n_available
    split_data = {}
    n_batches = None
    for field in fields:
        d = np.array(cast(zarr.Array, file[field])[:n_load])
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
