"""Benchmark data loading throughput and diagnose GPU utilisation bottlenecks.

Results are logged to wandb so you can compare runs across environments.
Submit to a compute node via +env=torchcpu (no GPU needed).

Usage:
    # Inspect data layout only — safe on login node
    python tpflow/tools/benchmark_dataloader.py dataset=kolflow inspect=true

    # Full benchmark on a compute node
    python tpflow/tools/benchmark_dataloader.py dataset=kolflow --multi +env=torchcpu

    # Simulate a 30 ms GPU step; compare prefetch depths
    python tpflow/tools/benchmark_dataloader.py dataset=kolflow compute_ms=30 compare_prefetch=true --multi +env=torchcpu
"""

from __future__ import annotations

import logging
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

import wandb
from tpflow.config import BenchmarkConfig
from tpflow.util import init_wandb, log_duration


def _histogram(values: np.ndarray, n_bins: int = 12, width: int = 40) -> str:
    lo, hi = values.min(), values.max()
    if lo == hi:
        return f"  all values = {lo*1000:.1f} ms"
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    max_count = max(counts)
    lines = []
    for i, (c, left) in enumerate(zip(counts, edges)):
        bar = "█" * int(round(width * c / max(max_count, 1)))
        right = edges[i + 1]
        lines.append(f"  [{left*1000:6.1f}-{right*1000:6.1f} ms] {bar} {c}")
    return "\n".join(lines)


def _run_and_log(loader_iter, n_batches: int, compute_ms: float, label: str, run) -> np.ndarray:
    times: list[float] = []
    t0 = time.perf_counter()
    for i, _batch in enumerate(loader_iter):
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i + 1 >= n_batches:
            break
        if compute_ms > 0:
            time.sleep(compute_ms / 1000)
        t0 = time.perf_counter()

    arr = np.array(times[1:])  # drop first batch (cold start)
    mean_ms = float(np.mean(arr) * 1000)
    std_ms = float(np.std(arr) * 1000)
    p95_ms = float(np.percentile(arr, 95) * 1000)
    max_ms = float(arr.max() * 1000)

    logging.info(
        "%s  load: mean=%.1f ms  std=%.1f ms  p95=%.1f ms  max=%.1f ms",
        label, mean_ms, std_ms, p95_ms, max_ms,
    )
    logging.info("\n%s", _histogram(arr))

    metrics = {
        f"{label}/load_ms_mean": mean_ms,
        f"{label}/load_ms_std": std_ms,
        f"{label}/load_ms_p95": p95_ms,
        f"{label}/load_ms_max": max_ms,
    }
    if compute_ms > 0:
        idle_frac = float(np.mean(np.maximum(arr - compute_ms / 1000, 0)) / (np.mean(arr) + compute_ms / 1000))
        metrics[f"{label}/gpu_idle_frac"] = idle_frac
        logging.info("%s  estimated GPU idle fraction: %.1f%%", label, idle_frac * 100)

    run.log(metrics)
    run.log({f"{label}/load_ms_hist": wandb.Histogram(arr * 1000)})
    return arr


def _make_iter(cfg: BenchmarkConfig, prefetch: int):
    from collections import deque
    from tpflow.data import RegressionZarrData, ZarrData
    from tpflow.config import DataConfig

    if cfg.loader == "regression":
        path = f"data/datasets/{cfg.dataset}/reg_train_data/{cfg.model}.zarr"
        dl = RegressionZarrData(path, cfg.batch_size, cfg.block_size)
    else:
        dcfg = DataConfig(name=cfg.dataset, batch_size=cfg.batch_size, block_size=cfg.block_size)
        dl = ZarrData(dcfg, "train")

    src = iter(dl.iter_batches(seed=0))
    if prefetch <= 0:
        return dl, src

    q: deque = deque()
    for _ in range(prefetch):
        try:
            q.append(next(src))
        except StopIteration:
            break

    def _prefetched():
        while q:
            yield q.popleft()
            try:
                q.append(next(src))
            except StopIteration:
                pass

    return dl, _prefetched()


def _print_layout(cfg: BenchmarkConfig):
    from tpflow.data import RegressionZarrData, ZarrData
    from tpflow.config import DataConfig

    if cfg.loader == "regression":
        path = f"data/datasets/{cfg.dataset}/reg_train_data/{cfg.model}.zarr"
        dl = RegressionZarrData(path, cfg.batch_size, cfg.block_size)
        state_shape = dl._arrays["data"].shape[1:]
        bytes_per_batch = int(np.prod(state_shape)) * 4 * cfg.batch_size * 2
        logging.info("Dataset: %s", path)
    else:
        dcfg = DataConfig(name=cfg.dataset, batch_size=cfg.batch_size, block_size=cfg.block_size)
        dl = ZarrData(dcfg, "train")
        state_shape = dl._arrays[dcfg.fields[0]].shape[1:]
        bytes_per_batch = int(np.prod(state_shape)) * 4 * cfg.batch_size
        logging.info("Dataset: data/datasets/%s/cfm_train_data/train.zarr", cfg.dataset)

    logging.info(
        "shards=%d  shard_size=%d  blocks/shard=%d  batches/shard=%d  total_batches=%d",
        dl._n_shards, dl._shard_size, dl._blocks_per_shard,
        dl._shard_size // cfg.batch_size, len(dl),
    )
    logging.info("state_shape=%s  batch≈%.1f MB", state_shape, bytes_per_batch / 1e6)
    return dl


@hydra.main(version_base=None, config_name="benchmark", config_path="../../conf")
@log_duration()
def main(cfg: BenchmarkConfig) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    with init_wandb(cfg, "benchmark", data_name=cfg.dataset) as run:
        try:
            dl = _print_layout(cfg)
        except FileNotFoundError as e:
            logging.error("Data not found: %s", e)
            return

        if cfg.inspect:
            return

        prefetch_values = [2, 4, 8] if cfg.compare_prefetch else [cfg.prefetch]
        for pf in prefetch_values:
            _, it = _make_iter(cfg, pf)
            _run_and_log(it, cfg.n_batches, cfg.compute_ms, f"prefetch_{pf}", run)

        logging.info(
            "Rule of thumb: if p95 > 2x mean → shard-boundary stalls; "
            "if mean load > GPU step time → loader is the bottleneck."
        )


if __name__ == "__main__":
    main()
