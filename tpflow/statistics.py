"""Trajectory statistics for evaluating marginal distributions.

All functions take trajectories of shape ``(n_time, n_rollout, *state_shape)``
and return ``(n_time, n_rollout)`` — one scalar per sample per time step —
so the caller can compute mean and std over the ensemble.
"""

import numpy as np


def enstrophy(trajectories: np.ndarray) -> np.ndarray:
    """Mean squared state value per sample at each time step.

    For field data this is the spatial mean of x**2, i.e. the L2 energy.
    For particle data it is the mean squared position/velocity component.

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``

    Returns:
        ``(n_time, n_rollout)``
    """
    n_time, n_rollout = trajectories.shape[:2]
    flat = trajectories.reshape(n_time, n_rollout, -1)
    return np.mean(flat**2, axis=-1)


def kurtosis(trajectories: np.ndarray) -> np.ndarray:
    """Excess kurtosis of state values per sample at each time step.

    Flattens the state dimensions and computes the 4th-standardised moment
    minus 3 (so a Gaussian gives 0). Useful for detecting heavy tails in
    particle distributions or field statistics.

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``

    Returns:
        ``(n_time, n_rollout)``
    """
    n_time, n_rollout = trajectories.shape[:2]
    flat = trajectories.reshape(n_time, n_rollout, -1)  # (n_time, n_rollout, n_state)
    mu = np.mean(flat, axis=-1, keepdims=True)
    sigma = np.std(flat, axis=-1, keepdims=True)
    safe_sigma = np.where(sigma == 0, 1.0, sigma)
    z = (flat - mu) / safe_sigma
    return np.mean(z**4, axis=-1) - 3.0


def trajectory_statistics(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all statistics for a trajectory ensemble.

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``

    Returns:
        Dict mapping statistic name to ``(n_time, n_rollout)`` array.
    """
    return {
        "enstrophy": enstrophy(trajectories),
        "kurtosis": kurtosis(trajectories),
    }
