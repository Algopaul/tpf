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


# ── hw2d (Hasegawa-Wakatani) statistics ────────────────────────────────────────
#
# All formulas follow the hw2d README (github.com/the-rccg/hw2d).
# Fields live on a 2-D periodic unit box; <·> denotes the spatial mean.
#
#   Γ_n = -<n · ∂_y φ>       particle flux
#   Γ_c = c1 · <(n - φ)²>    coupling term
#   E   = ½ · <n² + |∇φ|²>   energy (thermal + kinetic)
#   U   = ½ · <(n - ∇²φ)²>   enstrophy
#
# Spatial derivatives are computed via FFT for accuracy on periodic domains.


def _fft_freq_grids(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    """Angular wavenumber grids (ky, kx) for a unit-box domain.

    Returns arrays broadcast-compatible with ``(..., ny, nx//2+1)`` rFFT output.
    """
    ky = (2.0 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny))[:, None]  # (ny, 1)
    kx = (2.0 * np.pi * np.fft.rfftfreq(nx, d=1.0 / nx))[None, :]  # (1, nx//2+1)
    return ky, kx


def _dy(field: np.ndarray) -> np.ndarray:
    """∂_y of *field* via rFFT (y = second-to-last axis).

    Args:
        field: ``(..., H, W)`` real array on a periodic unit-box domain.

    Returns:
        ``(..., H, W)`` real array.
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ky, _ = _fft_freq_grids(ny, nx)
    return np.fft.irfft2(1j * ky * np.fft.rfft2(field), s=(ny, nx))


def _laplacian(field: np.ndarray) -> np.ndarray:
    """∇² of *field* via rFFT (last two axes are y, x).

    Args:
        field: ``(..., H, W)`` real array on a periodic unit-box domain.

    Returns:
        ``(..., H, W)`` real array.
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ky, kx = _fft_freq_grids(ny, nx)
    return np.fft.irfft2(-(ky**2 + kx**2) * np.fft.rfft2(field), s=(ny, nx))


def _grad_sq(field: np.ndarray) -> np.ndarray:
    """Pointwise |∇field|² via rFFT.

    Args:
        field: ``(..., H, W)``

    Returns:
        ``(..., H, W)``
    """
    ny, nx = field.shape[-2], field.shape[-1]
    ky, kx = _fft_freq_grids(ny, nx)
    f_hat = np.fft.rfft2(field)
    dy_f = np.fft.irfft2(1j * ky * f_hat, s=(ny, nx))
    dx_f = np.fft.irfft2(1j * kx * f_hat, s=(ny, nx))
    return dy_f**2 + dx_f**2


def hw2d_gamma_n(n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Particle flux Γ_n = -<n · ∂_y φ>.

    Args:
        n:   ``(..., H, W)`` density field.
        phi: ``(..., H, W)`` electric potential.

    Returns:
        ``(...)`` scalar(s).
    """
    return -np.mean(n * _dy(phi), axis=(-2, -1))


def hw2d_gamma_c(n: np.ndarray, phi: np.ndarray, c1: float = 1.0) -> np.ndarray:
    """Coupling term Γ_c = c1 · <(n - φ)²>.

    Args:
        n:   ``(..., H, W)`` density field.
        phi: ``(..., H, W)`` electric potential.
        c1:  adiabaticity parameter.

    Returns:
        ``(...)`` scalar(s).
    """
    return c1 * np.mean((n - phi) ** 2, axis=(-2, -1))


def hw2d_energy(n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Energy E = ½ · <n² + |∇φ|²>.

    Args:
        n:   ``(..., H, W)`` density field.
        phi: ``(..., H, W)`` electric potential.

    Returns:
        ``(...)`` scalar(s).
    """
    return 0.5 * np.mean(n**2 + _grad_sq(phi), axis=(-2, -1))


def hw2d_enstrophy(n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Enstrophy U = ½ · <(n - ∇²φ)²>.

    Args:
        n:   ``(..., H, W)`` density field.
        phi: ``(..., H, W)`` electric potential.

    Returns:
        ``(...)`` scalar(s).
    """
    omega = _laplacian(phi)
    return 0.5 * np.mean((n - omega) ** 2, axis=(-2, -1))


def hw2d_statistics(
    trajectories: np.ndarray,
    n_channel: int = 0,
    phi_channel: int = 1,
    c1: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute hw2d statistics for a field trajectory ensemble.

    Args:
        trajectories: ``(n_time, n_rollout, H, W, C)`` where the last axis C
            holds at least density (n) and electric potential (phi).
        n_channel:   index of the density channel in the last axis.
        phi_channel: index of the electric-potential channel in the last axis.
        c1:          adiabaticity parameter for Γ_c.

    Returns:
        Dict mapping statistic name to ``(n_time, n_rollout)`` array.
    """
    n = trajectories[..., n_channel].astype(np.float64)    # (n_time, n_rollout, H, W)
    phi = trajectories[..., phi_channel].astype(np.float64)
    return {
        "gamma_n": hw2d_gamma_n(n, phi),
        "gamma_c": hw2d_gamma_c(n, phi, c1),
        "energy": hw2d_energy(n, phi),
        "hw2d_enstrophy": hw2d_enstrophy(n, phi),
    }
