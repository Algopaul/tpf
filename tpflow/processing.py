"""Pure-numpy helpers for trajectory data reorganisation.

These functions are intentionally free of zarr/Hydra/wandb dependencies so they
can be unit-tested with small synthetic arrays.
"""

import math

import numpy as np
import zarr


def open_zarr_array(
    group: zarr.Group,
    name: str,
    size: int | None = None,
) -> zarr.Array | np.ndarray:
    """Return the named array from *group*, or a ones-vector of length *size*.

    The fallback is used when e.g. 'param' is absent from the input file and
    should be treated as all-ones.
    """
    if name not in group:
        if size is not None:
            return np.ones((size,))
        raise KeyError(f"'{name}' not found in group and no size provided")
    obj = group[name]
    if not isinstance(obj, zarr.Array):
        raise TypeError(f"'{name}' is not a zarr.Array")
    return obj


def auto_block_sizes(
    state_shape: tuple,
    n_time: int,
    target_mb: float = 2.0,
) -> tuple[int, int]:
    """Return ``(block_size, trajectory_block_size)`` targeting *target_mb* per chunk.

    Sizes are computed for f4 (4-byte) samples.
    """
    target_bytes = int(target_mb * 1024 * 1024)
    bytes_per_sample = max(1, math.prod(state_shape)) * 4
    block_size = max(1, target_bytes // bytes_per_sample)
    traj_block_size = max(1, target_bytes // (n_time * bytes_per_sample))
    return block_size, traj_block_size


def extract_regression_pairs(
    data_block: np.ndarray,
    time_vector: np.ndarray,
    param_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract (current, next, time, param) pairs from a block of trajectories.

    Args:
        data_block:  ``(n_traj, n_time, *state_shape)``
        time_vector: ``(n_time,)``
        param_block: ``(n_traj,)``

    Returns:
        cur, nxt, time_flat, param_flat — each with leading dimension
        ``n_traj * (n_time - 1)``.
    """
    n_traj, n_time = data_block.shape[:2]
    state_shape = data_block.shape[2:]
    n_steps = n_time - 1

    cur = data_block[:, :-1].reshape(n_traj * n_steps, *state_shape)
    nxt = data_block[:, 1:].reshape(n_traj * n_steps, *state_shape)
    time_flat = np.tile(time_vector[:-1], n_traj)
    param_flat = np.repeat(param_block, n_steps)

    return cur, nxt, time_flat, param_flat
