"""Tests for tpflow.statistics.

Generic statistics (enstrophy, kurtosis) and hw2d physics statistics.
Inputs are kept tiny so expected outputs can be verified by hand or with
simple analytical identities.
"""

import numpy as np
import pytest

from tpflow.statistics import (
    _dy,
    _laplacian,
    energy_spectra,
    enstrophy,
    first_moment,
    hw2d_energy,
    hw2d_enstrophy,
    hw2d_gamma_c,
    hw2d_gamma_n,
    hw2d_statistics,
    kurtosis,
    trajectory_statistics,
)

# ---------------------------------------------------------------------------
# Generic statistics
# ---------------------------------------------------------------------------


def test_enstrophy_output_shape():
    traj = np.ones((5, 3, 4, 4))
    out = enstrophy(traj)
    assert out.shape == (5, 3)


def test_enstrophy_constant_field():
    # field = c  →  mean(c²) = c²
    traj = np.full((2, 3, 8, 8), 2.0)
    np.testing.assert_allclose(enstrophy(traj), 4.0)


def test_kurtosis_output_shape():
    traj = np.random.default_rng(0).standard_normal((5, 3, 16))
    assert kurtosis(traj).shape == (5, 3)


def test_kurtosis_gaussian_near_zero():
    # Large sample → excess kurtosis ≈ 0
    rng = np.random.default_rng(42)
    traj = rng.standard_normal((1, 1, 100_000))
    assert abs(float(kurtosis(traj)[0, 0])) < 0.1


def test_kurtosis_constant_field_is_minus_three():
    # All identical values → std = 0, z = 0, kurtosis = 0 - 3 = -3
    traj = np.full((2, 3, 8), 5.0)
    np.testing.assert_allclose(kurtosis(traj), -3.0)


def test_trajectory_statistics_returns_all_keys():
    traj = np.ones((4, 2, 6))
    stats = trajectory_statistics(traj)
    assert "first_moment" in stats
    assert "enstrophy" in stats
    assert "kurtosis" in stats
    assert stats["enstrophy"].shape == (4, 2)
    assert stats["first_moment"].shape == (4, 2)


# ---------------------------------------------------------------------------
# first_moment
# ---------------------------------------------------------------------------


def test_first_moment_output_shape():
    traj = np.ones((5, 3, 4, 4))
    out = first_moment(traj)
    assert out.shape == (5, 3)


def test_first_moment_constant_field():
    # field = c  →  mean(c) = c
    traj = np.full((2, 3, 8, 8), 3.0)
    np.testing.assert_allclose(first_moment(traj), 3.0)


def test_first_moment_zero_mean_field():
    # antisymmetric field has mean 0
    traj = np.array([1.0, -1.0, 1.0, -1.0]).reshape(1, 1, 4)
    np.testing.assert_allclose(first_moment(traj), 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# energy_spectra
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def field_traj():
    """(n_time=3, n_rollout=2, H=32, W=32) field trajectory."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 2, 32, 32))


@pytest.fixture(scope="module")
def field_traj_chan():
    """(n_time=3, n_rollout=2, H=32, W=32, C=2) channels-last trajectory."""
    rng = np.random.default_rng(43)
    return rng.standard_normal((3, 2, 32, 32, 2))


def test_energy_spectra_output_shapes(field_traj):
    bin_centers, spectra = energy_spectra(field_traj, n_bins=16)
    assert bin_centers.shape == (16,)
    assert spectra.shape == (3, 2, 16)


def test_energy_spectra_log_bins_output_shapes(field_traj):
    bin_centers, spectra = energy_spectra(field_traj, n_bins=16, log_bins=True)
    assert bin_centers.shape == (16,)
    assert spectra.shape == (3, 2, 16)


def test_energy_spectra_bin_centers_positive(field_traj):
    bin_centers, _ = energy_spectra(field_traj, n_bins=16)
    assert np.all(bin_centers > 0)


def test_energy_spectra_log_bins_centers_monotone(field_traj):
    bin_centers, _ = energy_spectra(field_traj, n_bins=16, log_bins=True)
    assert np.all(np.diff(bin_centers) > 0)


def test_energy_spectra_log_bins_centers_log_spaced(field_traj):
    # Ratios of consecutive centres should be approximately constant
    bin_centers, _ = energy_spectra(field_traj, n_bins=16, log_bins=True)
    ratios = bin_centers[1:] / bin_centers[:-1]
    np.testing.assert_allclose(ratios, ratios[0], rtol=0.05)


def test_energy_spectra_spectra_non_negative(field_traj):
    _, spectra = energy_spectra(field_traj, n_bins=16)
    assert np.all(spectra >= 0)


def test_energy_spectra_total_power_parseval(field_traj):
    # Sum of spectral bins ≈ mean(field²) / N² * N² = mean(field²)  up to binning edge effects
    # More precisely: sum(|F̂|²) / N² = mean(field²) by Parseval
    _, spectra = energy_spectra(field_traj, n_bins=256)
    for ti in range(3):
        for ri in range(2):
            field = field_traj[ti, ri]
            expected = np.mean(field**2)
            got = np.sum(spectra[ti, ri])
            np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_energy_spectra_channel_axis_selects_correct_channel(field_traj_chan):
    # Extracting channel 0 via channel_axis=-1 should match computing directly on chan 0
    _, spectra_ch0 = energy_spectra(
        field_traj_chan, n_bins=16, channel_axis=-1, channel_idx=0
    )
    _, spectra_direct = energy_spectra(
        field_traj_chan[..., 0], n_bins=16, channel_axis=None
    )
    np.testing.assert_allclose(spectra_ch0, spectra_direct)


def test_energy_spectra_channel_axis_negative_index(field_traj_chan):
    # channel_axis=-1 and channel_axis=2 (last of 3 state dims H,W,C) are equivalent
    _, s1 = energy_spectra(field_traj_chan, n_bins=8, channel_axis=-1, channel_idx=1)
    _, s2 = energy_spectra(field_traj_chan, n_bins=8, channel_axis=2, channel_idx=1)
    np.testing.assert_array_equal(s1, s2)


def test_energy_spectra_raises_without_channel_axis_for_3d_state():
    traj = np.ones((2, 2, 8, 8, 2))  # 5-D, needs channel selection
    with pytest.raises(ValueError, match="channel_axis"):
        energy_spectra(traj, channel_axis=None)


# ---------------------------------------------------------------------------
# FFT helpers: _dy and _laplacian
# ---------------------------------------------------------------------------


def test_dy_of_sine_is_cosine():
    # f(y) = sin(2π y)  →  ∂_y f = 2π cos(2π y)
    # y = rows (axis 0), so the field must vary along rows.
    N = 64
    y = np.linspace(0, 1, N, endpoint=False)
    f2d = np.tile(np.sin(2 * np.pi * y)[:, None], (1, N))  # (N, N) varies along rows
    df_expected = np.tile((2 * np.pi * np.cos(2 * np.pi * y))[:, None], (1, N))
    df = _dy(f2d)
    np.testing.assert_allclose(df, df_expected, atol=1e-10)


def test_laplacian_of_cosine():
    # f(y) = cos(2π y)  →  ∇²f = -(2π)² cos(2π y)  (x-independent)
    N = 64
    y = np.linspace(0, 1, N, endpoint=False)
    f2d = np.tile(np.cos(2 * np.pi * y)[:, None], (1, N))  # (N, N)
    lap_expected = -((2 * np.pi) ** 2) * f2d
    lap = _laplacian(f2d)
    np.testing.assert_allclose(lap, lap_expected, atol=1e-8)


def test_dy_zero_for_x_only_field():
    # f(x) = sin(2π x) has no y-dependence → ∂_y f = 0
    N = 32
    x = np.linspace(0, 1, N, endpoint=False)
    f2d = np.tile(np.sin(2 * np.pi * x)[None, :], (N, 1))  # (N, N)
    np.testing.assert_allclose(_dy(f2d), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# hw2d physics statistics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hw2d_fields():
    """A tiny pair of random n, phi fields."""
    rng = np.random.default_rng(7)
    N = 32
    n = rng.standard_normal((N, N))
    phi = rng.standard_normal((N, N))
    return n, phi


def test_hw2d_gamma_n_output_shape(hw2d_fields):
    n, phi = hw2d_fields
    result = hw2d_gamma_n(n, phi)
    assert result.shape == ()  # scalar for 2-D input


def test_hw2d_gamma_n_zero_for_x_only_phi():
    # ∂_y φ = 0 when φ depends only on x → Γ_n = 0
    N = 32
    x = np.linspace(0, 1, N, endpoint=False)
    phi_x = np.tile(np.sin(2 * np.pi * x)[None, :], (N, 1))
    n = np.random.default_rng(1).standard_normal((N, N))
    np.testing.assert_allclose(hw2d_gamma_n(n, phi_x), 0.0, atol=1e-12)


def test_hw2d_gamma_c_non_negative(hw2d_fields):
    n, phi = hw2d_fields
    assert float(hw2d_gamma_c(n, phi)) >= 0.0


def test_hw2d_gamma_c_zero_when_n_equals_phi(hw2d_fields):
    n, _ = hw2d_fields
    np.testing.assert_allclose(hw2d_gamma_c(n, n), 0.0, atol=1e-14)


def test_hw2d_gamma_c_scales_with_c1(hw2d_fields):
    n, phi = hw2d_fields
    val1 = hw2d_gamma_c(n, phi, c1=1.0)
    val2 = hw2d_gamma_c(n, phi, c1=3.0)
    np.testing.assert_allclose(val2, 3.0 * val1, rtol=1e-12)


def test_hw2d_energy_non_negative(hw2d_fields):
    n, phi = hw2d_fields
    assert float(hw2d_energy(n, phi)) >= 0.0


def test_hw2d_energy_zero_for_zero_fields():
    N = 16
    np.testing.assert_allclose(
        hw2d_energy(np.zeros((N, N)), np.zeros((N, N))), 0.0, atol=1e-14
    )


def test_hw2d_energy_thermal_only():
    # phi = 0 → E = ½ <n²>
    N = 32
    n = np.random.default_rng(3).standard_normal((N, N))
    expected = 0.5 * np.mean(n**2)
    np.testing.assert_allclose(hw2d_energy(n, np.zeros_like(n)), expected, rtol=1e-12)


def test_hw2d_enstrophy_non_negative(hw2d_fields):
    n, phi = hw2d_fields
    assert float(hw2d_enstrophy(n, phi)) >= 0.0


def test_hw2d_enstrophy_zero_when_n_equals_laplacian_phi():
    # U = ½ <(n - ∇²φ)²> = 0 iff n = ∇²φ
    N = 32
    # construct phi, then set n = ∇²φ exactly
    rng = np.random.default_rng(5)
    phi = rng.standard_normal((N, N))
    n = _laplacian(phi)
    np.testing.assert_allclose(hw2d_enstrophy(n, phi), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# hw2d_statistics trajectory wrapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def traj_5ch():
    """(n_time=3, n_rollout=2, H=16, W=16, C=3) channels-last trajectory array."""
    rng = np.random.default_rng(11)
    return rng.standard_normal((3, 2, 16, 16, 3))


def test_hw2d_statistics_output_shapes(traj_5ch):
    stats = hw2d_statistics(traj_5ch, n_channel=0, phi_channel=1)
    for name, arr in stats.items():
        assert arr.shape == (3, 2), f"{name}: expected (3, 2), got {arr.shape}"


def test_hw2d_statistics_keys(traj_5ch):
    stats = hw2d_statistics(traj_5ch)
    assert set(stats.keys()) == {"gamma_n", "gamma_c", "energy", "hw2d_enstrophy"}


def test_hw2d_statistics_gamma_c_non_negative(traj_5ch):
    stats = hw2d_statistics(traj_5ch)
    assert np.all(stats["gamma_c"] >= 0.0)


def test_hw2d_statistics_energy_non_negative(traj_5ch):
    stats = hw2d_statistics(traj_5ch)
    assert np.all(stats["energy"] >= 0.0)


def test_hw2d_statistics_vectorised_matches_per_frame(traj_5ch):
    """Batch computation must equal looping over frames."""
    stats = hw2d_statistics(traj_5ch, n_channel=0, phi_channel=1)
    n_time, n_rollout = 3, 2
    for ti in range(n_time):
        for ri in range(n_rollout):
            n = traj_5ch[ti, ri, :, :, 0]
            phi = traj_5ch[ti, ri, :, :, 1]
            np.testing.assert_allclose(
                stats["gamma_n"][ti, ri], hw2d_gamma_n(n, phi), rtol=1e-12
            )
            np.testing.assert_allclose(
                stats["energy"][ti, ri], hw2d_energy(n, phi), rtol=1e-12
            )
