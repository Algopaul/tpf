"""Tests for tpflow.processing.

Each test covers one distinct behaviour of a function. Inputs are kept tiny
so the expected output can be worked out by hand before writing the assertion.
"""

import numpy as np
import pytest
import zarr

from tpflow.processing import auto_block_sizes, extract_regression_pairs, open_zarr_array


# ---------------------------------------------------------------------------
# open_zarr_array
# ---------------------------------------------------------------------------

def test_open_zarr_array_returns_array_when_present():
    g = zarr.group()
    g.create_array("data", shape=(4,), dtype="f4")
    assert isinstance(open_zarr_array(g, "data"), zarr.Array)


def test_open_zarr_array_fallback_ones_when_absent_and_size_given():
    g = zarr.group()
    result = open_zarr_array(g, "param", size=5)
    np.testing.assert_array_equal(result, np.ones(5))


def test_open_zarr_array_raises_key_error_when_absent_and_no_size():
    g = zarr.group()
    with pytest.raises(KeyError, match="param"):
        open_zarr_array(g, "param")


def test_open_zarr_array_raises_type_error_when_name_is_a_subgroup():
    g = zarr.group()
    g.create_group("sub")
    with pytest.raises(TypeError, match="sub"):
        open_zarr_array(g, "sub")


# ---------------------------------------------------------------------------
# auto_block_sizes
# ---------------------------------------------------------------------------

def test_auto_block_sizes_returns_positive_ints():
    bs, tbs = auto_block_sizes(state_shape=(8,), n_time=10)
    assert isinstance(bs, int) and bs > 0
    assert isinstance(tbs, int) and tbs > 0


def test_auto_block_sizes_larger_state_gives_smaller_block_size():
    bs_small, _ = auto_block_sizes(state_shape=(8,), n_time=10)
    bs_large, _ = auto_block_sizes(state_shape=(1024,), n_time=10)
    assert bs_large < bs_small


def test_auto_block_sizes_larger_target_mb_gives_larger_block_size():
    bs_1mb, _ = auto_block_sizes(state_shape=(64,), n_time=10, target_mb=1.0)
    bs_4mb, _ = auto_block_sizes(state_shape=(64,), n_time=10, target_mb=4.0)
    assert bs_4mb > bs_1mb


def test_auto_block_sizes_scalar_state_shape_does_not_crash():
    bs, tbs = auto_block_sizes(state_shape=(), n_time=10)
    assert bs > 0 and tbs > 0


# ---------------------------------------------------------------------------
# extract_regression_pairs
# ---------------------------------------------------------------------------

# 3 trajectories, 4 timesteps, state shape (2,) → n_samples = 3 * 3 = 9.

@pytest.fixture
def pairs():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4, 2))
    time = np.array([0.0, 0.25, 0.5, 0.75])
    param = np.array([1.0, 2.0, 3.0])
    cur, nxt, time_flat, param_flat = extract_regression_pairs(data, time, param)
    return dict(data=data, time=time, param=param,
                cur=cur, nxt=nxt, time_flat=time_flat, param_flat=param_flat)


def test_extract_regression_pairs_output_shapes(pairs):
    assert pairs["cur"].shape == (9, 2)
    assert pairs["nxt"].shape == (9, 2)
    assert pairs["time_flat"].shape == (9,)
    assert pairs["param_flat"].shape == (9,)


def test_extract_regression_pairs_cur_and_nxt_are_adjacent_timesteps(pairs):
    data = pairs["data"]
    cur, nxt = pairs["cur"], pairs["nxt"]
    n_traj, n_time = 3, 4
    n_steps = n_time - 1
    for traj in range(n_traj):
        for step in range(n_steps):
            i = traj * n_steps + step
            np.testing.assert_array_equal(cur[i], data[traj, step])
            np.testing.assert_array_equal(nxt[i], data[traj, step + 1])


def test_extract_regression_pairs_time_flat_tiles_time_vector(pairs):
    expected = np.tile(pairs["time"][:-1], 3)
    np.testing.assert_array_equal(pairs["time_flat"], expected)


def test_extract_regression_pairs_param_flat_repeats_per_trajectory(pairs):
    expected = np.repeat(pairs["param"], 3)
    np.testing.assert_array_equal(pairs["param_flat"], expected)
