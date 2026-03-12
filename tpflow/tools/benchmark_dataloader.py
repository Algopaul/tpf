"""Benchmark data loading throughput and diagnose GPU utilisation bottlenecks.

Measures per-batch load times and reports mean/std/p95 plus a histogram
that reveals bimodal distributions (fast in-shard batches vs slow cross-shard
stalls).  Run on the login node or compute node — no GPU required.

Usage:
    # Basic throughput test
    python tpflow/tools/benchmark_dataloader.py kolflow

    # Simulate a 30 ms GPU step to check if the loader keeps up
    python tpflow/tools/benchmark_dataloader.py kolflow --compute-ms 30

    # Test a different prefetch depth
    python tpflow/tools/benchmark_dataloader.py kolflow --prefetch 8

    # Test CFM train data instead of regression data
    python tpflow/tools/benchmark_dataloader.py kolflow --loader cfm
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


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


def _run(
    loader_iter,
    n_batches: int,
    compute_ms: float,
    label: str,
) -> np.ndarray:
    """Iterate *loader_iter* and return per-batch wait times in seconds."""
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

    arr = np.array(times[1:])  # drop first batch (cold start / JIT)
    console.print(f"\n[bold]{label}[/bold]  (n={len(arr)} batches, compute={compute_ms:.0f} ms simulated)")
    console.print(
        f"  load time   mean={np.mean(arr)*1000:6.1f} ms  "
        f"std={np.std(arr)*1000:5.1f} ms  "
        f"p95={np.percentile(arr,95)*1000:6.1f} ms  "
        f"max={arr.max()*1000:6.1f} ms"
    )
    if compute_ms > 0:
        idle_frac = np.mean(np.maximum(arr - compute_ms / 1000, 0)) / (
            np.mean(arr) + compute_ms / 1000
        )
        console.print(f"  estimated GPU idle fraction: [{'red' if idle_frac > 0.1 else 'green'}]{idle_frac:.1%}[/]")
    console.print(_histogram(arr))
    return arr


@app.command()
def main(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. kolflow, hw2d)"),
    loader: str = typer.Option("cfm", help="'cfm' or 'regression'"),
    model: str = typer.Option("model1", help="Model name for regression data"),
    n_batches: int = typer.Option(80, help="Batches to time per run"),
    batch_size: int = typer.Option(512),
    block_size: int = typer.Option(32),
    prefetch: int = typer.Option(2, help="device_prefetch queue depth"),
    compute_ms: float = typer.Option(
        0.0, help="Simulate GPU step of this many ms (0 = measure raw throughput)"
    ),
    compare_prefetch: bool = typer.Option(
        False, help="Run with prefetch=2,4,8 and compare"
    ),
    inspect: bool = typer.Option(
        False, help="Print data layout only — no data loaded, safe on login nodes"
    ),
):
    from tpflow.data import RegressionZarrData, ZarrData
    from tpflow.config import DataConfig

    # Build the dataloader (reads only zarr metadata JSON, no array data).
    try:
        if loader == "regression":
            path = f"data/datasets/{dataset}/reg_train_data/{model}.zarr"
            dl = RegressionZarrData(path, batch_size, block_size)
            state_shape = dl._arrays["data"].shape[1:]
            bytes_per_batch = int(np.prod(state_shape)) * 4 * batch_size * 2  # data + next
            console.print(f"\n[bold]Dataset:[/bold] {path}")
        else:
            cfg = DataConfig(name=dataset, batch_size=batch_size, block_size=block_size)
            dl = ZarrData(cfg, "train")
            state_shape = dl._arrays[cfg.fields[0]].shape[1:]
            bytes_per_batch = int(np.prod(state_shape)) * 4 * batch_size
            console.print(f"\n[bold]Dataset:[/bold] data/datasets/{dataset}/cfm_train_data/train.zarr")
    except FileNotFoundError as e:
        console.print(f"[red]Data not found:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        f"  shards={dl._n_shards}  shard_size={dl._shard_size}  "
        f"blocks/shard={dl._blocks_per_shard}  batches/shard={dl._shard_size // batch_size}  "
        f"total_batches={len(dl)}"
    )
    console.print(f"  state_shape={state_shape}  batch≈{bytes_per_batch / 1e6:.1f} MB")

    if inspect:
        return

    def make_iter(pf: int):
        it = dl.iter_batches(seed=0)
        if pf > 0:
            from collections import deque
            q: deque = deque()
            src = iter(it)
            for _ in range(pf):
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

            return _prefetched()
        return iter(it)

    if compare_prefetch:
        for pf in [2, 4, 8]:
            _run(make_iter(pf), n_batches, compute_ms, f"prefetch={pf}")
    else:
        _run(make_iter(prefetch), n_batches, compute_ms, f"prefetch={prefetch}")

    console.print()
    console.print("[dim]Rule of thumb: if p95 > 2× mean, you have shard-boundary stalls.")
    console.print("If mean load time > GPU step time, the loader is the bottleneck.[/dim]")


if __name__ == "__main__":
    app()
