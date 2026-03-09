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


def first_moment(trajectories: np.ndarray) -> np.ndarray:
    """Mean state value per sample at each time step (first moment).

    Flattens all state dimensions and computes the arithmetic mean. For
    vorticity fields on a periodic domain this quantity is conserved and
    should stay near zero; deviations indicate a systematic bias.

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``

    Returns:
        ``(n_time, n_rollout)``
    """
    n_time, n_rollout = trajectories.shape[:2]
    flat = trajectories.reshape(n_time, n_rollout, -1)
    return np.mean(flat, axis=-1)


def energy_spectra(
    trajectories: np.ndarray,
    n_bins: int = 32,
    log_bins: bool = False,
    channel_axis: int | None = None,
    channel_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of a 2-D field trajectory ensemble.

    The state is assumed to be a 2-D periodic field ``(H, W)`` after optional
    channel selection. The discrete Fourier power ``|F̂|²`` is accumulated into
    radial wavenumber bins (linear or logarithmic spacing).

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``
        n_bins: number of wavenumber bins.
        log_bins: if True, bin edges are logarithmically spaced from the
            smallest non-zero wavenumber to the Nyquist limit; otherwise
            linearly spaced from zero.
        channel_axis: axis index within *state_shape* (0-based, negative
            indices allowed) of the channel dimension. When ``None`` the state
            is assumed to be exactly 2-D ``(H, W)``.
        channel_idx: which channel to select when *channel_axis* is not None.

    Returns:
        bin_centers: ``(n_bins,)`` wavenumber bin centres (same units as
            ``np.fft.fftfreq`` with ``d=1/N``, i.e. cycles per grid point
            multiplied by grid size).
        spectra: ``(n_time, n_rollout, n_bins)`` total power in each bin.
    """
    n_time, n_rollout = trajectories.shape[:2]

    if channel_axis is not None:
        ndim_state = trajectories.ndim - 2
        full_axis = 2 + (channel_axis % ndim_state)
        field = np.take(trajectories, channel_idx, axis=full_axis)
    else:
        field = trajectories

    if field.ndim != 4:
        raise ValueError(
            f"After channel selection the field must be 4-D "
            f"(n_time, n_rollout, H, W); got shape {field.shape}. "
            "Pass channel_axis to select a 2-D slice from the state."
        )

    H, W = field.shape[2], field.shape[3]
    field = field.astype(np.float64)

    # 2-D FFT power spectrum, normalised by N² so amplitude is grid-size independent
    f_hat = np.fft.fft2(field)  # (n_time, n_rollout, H, W)
    power = (np.abs(f_hat) ** 2) / (H * W) ** 2

    # Radial wavenumber grid (cycles per grid point × grid size = integer wavenumbers)
    ky = np.fft.fftfreq(H, d=1.0 / H)  # (H,)
    kx = np.fft.fftfreq(W, d=1.0 / W)  # (W,)
    KX, KY = np.meshgrid(kx, ky)        # (H, W)
    K = np.sqrt(KX**2 + KY**2)          # (H, W)

    k_max = float(K.max())
    K_flat = K.ravel()  # (H*W,)

    # Bin edges
    if log_bins:
        k_min = float(K[K > 0].min()) if (K > 0).any() else 1.0
        bin_edges = np.geomspace(k_min, k_max, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, k_max, n_bins + 1)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Map each grid point to a bin; clip so the rightmost edge lands in the last bin
    bin_indices = np.clip(np.digitize(K_flat, bin_edges) - 1, 0, n_bins - 1)

    # Accumulate power into bins
    power_flat = power.reshape(n_time * n_rollout, H * W)
    raw_spectra = np.zeros((n_time * n_rollout, n_bins))
    for i in range(n_time * n_rollout):
        raw_spectra[i] = np.bincount(
            bin_indices, weights=power_flat[i], minlength=n_bins
        )

    return bin_centers, raw_spectra.reshape(n_time, n_rollout, n_bins)


def trajectory_statistics(trajectories: np.ndarray) -> dict[str, np.ndarray]:
    """Compute scalar statistics for a trajectory ensemble.

    Args:
        trajectories: ``(n_time, n_rollout, *state_shape)``

    Returns:
        Dict mapping statistic name to ``(n_time, n_rollout)`` array.
    """
    return {
        "first_moment": first_moment(trajectories),
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
    n = trajectories[..., n_channel].astype(np.float64)  # (n_time, n_rollout, H, W)
    phi = trajectories[..., phi_channel].astype(np.float64)
    return {
        "gamma_n": hw2d_gamma_n(n, phi),
        "gamma_c": hw2d_gamma_c(n, phi, c1),
        "energy": hw2d_energy(n, phi),
        "hw2d_enstrophy": hw2d_enstrophy(n, phi),
    }
