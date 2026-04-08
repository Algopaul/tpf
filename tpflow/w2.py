"""Wasserstein-1 and Wasserstein-2 distance utilities for comparing trajectory distributions.

Typical use
-----------
>>> from tpflow.w2 import load_rollout, w2_over_time, w1_over_time, print_w2_table
>>> rollout, reference, time = load_rollout("multirun/.../50/eval/rollout.zarr")
>>> ts, dists = w2_over_time(rollout, reference, stride=10)
>>> print_w2_table(ts, dists, time)

Two solvers are available via the ``solver`` parameter:

* ``"sinkhorn"`` (default) — regularised OT via ott-jax; fast and GPU-friendly.
  Pass ``epsilon`` to control regularisation tightness.
* ``"hungarian"`` — exact OT via scipy's linear_sum_assignment; no regularisation
  bias but scales as O(n²·n) and is CPU-only.  Use for small n or when an exact
  answer is required (e.g. verifying W2=0 at t=0).
"""

import jax.numpy as jnp
import numpy as np
import zarr
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ── I/O ───────────────────────────────────────────────────────────────────────


def load_rollout(
    path: str,
    key: str = "rollout",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a rollout.zarr produced by export_regression_eval.

    Parameters
    ----------
    path : path to rollout.zarr
    key  : which array to load as the prediction (``"rollout"`` or
           ``"one_step_ahead"``)

    Returns
    -------
    rollout   : (n_time, n_particles, *state_shape)  float32
    reference : (n_time, n_particles, *state_shape)  float32
    time      : (n_time,)  float32
    """
    g = zarr.open(path, mode="r")
    rollout   = np.array(g[key])
    reference = np.array(g["reference"])
    if "time" in g:
        time = np.array(g["time"])
    else:
        time = np.linspace(0.0, 1.0, rollout.shape[0], dtype=np.float32)
    return rollout, reference, time


# ── W2 computation ────────────────────────────────────────────────────────────


def _sanitize(a: np.ndarray, clip_value: float | None) -> np.ndarray:
    """Replace nan/inf; optionally clip to [-clip_value, clip_value]."""
    if clip_value is not None:
        a = np.clip(a, -clip_value, clip_value)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _wp_at_time_sinkhorn(
    x: np.ndarray,
    y: np.ndarray,
    p: int,
    epsilon: float,
) -> float:
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)
    cost_fn = None if p == 2 else costs.Euclidean()  # None → SqEuclidean
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon, cost_fn=cost_fn)
    out = sinkhorn.Sinkhorn()(linear_problem.LinearProblem(geom))
    cost = jnp.maximum(out.reg_ot_cost, 0.0)
    return float(jnp.sqrt(cost) if p == 2 else cost)


def _wp_at_time_hungarian(x: np.ndarray, y: np.ndarray, p: int) -> float:
    if p == 2:
        cost = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)   # (n, n)
        _, col = linear_sum_assignment(cost)
        return float(np.sqrt(cost[np.arange(len(x)), col].mean()))
    else:
        cost = np.sqrt(np.sum((x[:, None] - y[None, :]) ** 2, axis=-1))
        _, col = linear_sum_assignment(cost)
        return float(cost[np.arange(len(x)), col].mean())


def _wp_at_time(
    x: np.ndarray,
    y: np.ndarray,
    p: int,
    epsilon: float,
    max_samples: int | None,
    rng: np.random.Generator | None,
    clip_value: float | None,
    solver: str,
) -> float:
    x = _sanitize(x, clip_value)
    y = _sanitize(y, clip_value)
    if max_samples is not None and max_samples < len(x):
        rng = rng or np.random.default_rng()
        idx_x = rng.choice(len(x), max_samples, replace=False)
        idx_y = rng.choice(len(y), max_samples, replace=False)
        x, y = x[idx_x], y[idx_y]
    x = x.reshape(len(x), -1).astype(np.float32)
    y = y.reshape(len(y), -1).astype(np.float32)
    if solver == "hungarian":
        return _wp_at_time_hungarian(x, y, p)
    return _wp_at_time_sinkhorn(x, y, p, epsilon)


def w1_at_time(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> float:
    """W1 (Earth Mover's) distance between two point clouds."""
    return _wp_at_time(x, y, p=1, epsilon=epsilon,
                       max_samples=max_samples, rng=rng, clip_value=clip_value,
                       solver=solver)


def w2_at_time(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    rng: np.random.Generator | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> float:
    """W2 distance between two point clouds."""
    return _wp_at_time(x, y, p=2, epsilon=epsilon,
                       max_samples=max_samples, rng=rng, clip_value=clip_value,
                       solver=solver)


def _wp_over_time(
    rollout: np.ndarray,
    reference: np.ndarray,
    p: int,
    stride: int,
    epsilon: float,
    max_samples: int | None,
    clip_value: float | None,
    solver: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    indices = np.arange(0, rollout.shape[0], stride)
    distances = np.array([
        _wp_at_time(rollout[t], reference[t], p=p, epsilon=epsilon,
                    max_samples=max_samples, rng=rng, clip_value=clip_value,
                    solver=solver)
        for t in tqdm(indices)
    ])
    return indices, distances


def w1_over_time(
    rollout: np.ndarray,
    reference: np.ndarray,
    stride: int = 10,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute W1 between rollout and reference at every ``stride``-th timestep."""
    return _wp_over_time(rollout, reference, p=1, stride=stride,
                         epsilon=epsilon, max_samples=max_samples,
                         clip_value=clip_value, solver=solver)


def w2_over_time(
    rollout: np.ndarray,
    reference: np.ndarray,
    stride: int = 10,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute W2 between rollout and reference distributions at every ``stride``-th timestep."""
    return _wp_over_time(rollout, reference, p=2, stride=stride,
                         epsilon=epsilon, max_samples=max_samples,
                         clip_value=clip_value, solver=solver)


# ── comparison helpers ────────────────────────────────────────────────────────


def _compare_wp(
    runs: dict[str, str],
    p: int,
    stride: int,
    epsilon: float,
    max_samples: int | None,
    clip_value: float | None,
    solver: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    results = {}
    for label, path_spec in runs.items():
        if isinstance(path_spec, tuple):
            path, key = path_spec
        else:
            path, key = path_spec, "rollout"
        rollout, reference, time = load_rollout(path, key=key)
        n_particles = rollout.shape[1]
        n_eval = min(n_particles, max_samples) if max_samples is not None else n_particles
        indices, dists = _wp_over_time(
            rollout, reference, p=p, stride=stride, epsilon=epsilon,
            max_samples=max_samples, clip_value=clip_value, solver=solver,
        )
        results[label] = (indices, dists, time[indices], n_eval)
    return results


def compare_w1(
    runs: dict[str, str],
    stride: int = 10,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Load and compute W1 for multiple rollout.zarr paths.

    Returns {label: (timestep_indices, distances, time_values)}
    """
    return _compare_wp(runs, p=1, stride=stride, epsilon=epsilon,
                       max_samples=max_samples, clip_value=clip_value, solver=solver)


def compare_w2(
    runs: dict[str, str],
    stride: int = 10,
    epsilon: float = 1e-3,
    max_samples: int | None = None,
    clip_value: float | None = None,
    solver: str = "sinkhorn",
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Load and compute W2 for multiple rollout.zarr paths.

    Returns {label: (timestep_indices, distances, time_values)}
    """
    return _compare_wp(runs, p=2, stride=stride, epsilon=epsilon,
                       max_samples=max_samples, clip_value=clip_value, solver=solver)


# ── display ───────────────────────────────────────────────────────────────────


def print_w2_table(
    timestep_indices: np.ndarray,
    distances: np.ndarray,
    time: np.ndarray | None = None,
) -> None:
    """Print a simple table of W2 distances over time."""
    header = (
        f"{'t_idx':>6}  {'time':>8}  {'W2':>10}"
        if time is not None
        else f"{'t_idx':>6}  {'W2':>10}"
    )
    print(header)
    print("-" * len(header))
    for i, (idx, d) in enumerate(zip(timestep_indices, distances)):
        if time is not None:
            print(f"{idx:>6}  {float(time[i]):>8.4f}  {d:>10.6f}")
        else:
            print(f"{idx:>6}  {d:>10.6f}")


def plot_w2_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str = "time",
    ylabel: str = "W2 distance",
    title: str = "W2 distance: rollout vs. reference",
):
    """Plot W2 curves for multiple runs on one axis.

    Parameters
    ----------
    results : output of compare_w2()

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (_, dists, times, n_eval) in results.items():
        ax.plot(times, dists, marker="o", markersize=3, label=f"{label} (N={n_eval})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
