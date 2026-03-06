"""Tests for tpflow.processing.

Each test covers one distinct behaviour of a function. Inputs are kept tiny
so the expected output can be worked out by hand before writing the assertion.
"""

import numpy as np
import pytest
import zarr

from tpflow.processing import (
    auto_block_sizes,
    extract_regression_pairs,
    load_trajectory_zarr,
    open_zarr_array,
)

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
    return dict(
        data=data,
        time=time,
        param=param,
        cur=cur,
        nxt=nxt,
        time_flat=time_flat,
        param_flat=param_flat,
    )


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


# ---------------------------------------------------------------------------
# load_trajectory_zarr
# ---------------------------------------------------------------------------


def _make_trajectory_zarr(
    tmp_path, n_traj=5, n_time=10, state_shape=(3,), with_param=True, with_time=True
):
    path = str(tmp_path / "traj.zarr")
    g = zarr.open_group(path, mode="w")
    rng = np.random.default_rng(1)
    g.create_array(
        "data", data=rng.standard_normal((n_traj, n_time, *state_shape)).astype("f4")
    )
    if with_param:
        g.create_array("param", data=rng.standard_normal((n_traj,)).astype("f4"))
    if with_time:
        g.create_array("time", data=np.linspace(0, 1, n_time))
    return path, g


def test_load_trajectory_zarr_shapes(tmp_path):
    path, g = _make_trajectory_zarr(tmp_path)
    data, param, time_vector = load_trajectory_zarr(path)
    assert data.shape == (5, 10, 3)
    assert param.shape == (5,)
    assert time_vector.shape == (10,)


def test_load_trajectory_zarr_n_limits_rows(tmp_path):
    path, _ = _make_trajectory_zarr(tmp_path)
    data, param, _ = load_trajectory_zarr(path, n=3)
    assert data.shape[0] == 3
    assert param.shape[0] == 3


def test_load_trajectory_zarr_fallback_param_is_ones(tmp_path):
    path, _ = _make_trajectory_zarr(tmp_path, with_param=False)
    _, param, _ = load_trajectory_zarr(path)
    np.testing.assert_array_equal(param, np.ones(5))


def test_load_trajectory_zarr_fallback_time_is_linspace(tmp_path):
    path, _ = _make_trajectory_zarr(tmp_path, with_time=False)
    _, _, time_vector = load_trajectory_zarr(path)
    assert time_vector.shape == (10,)
    assert float(time_vector[0]) == pytest.approx(0.0)
    assert float(time_vector[-1]) == pytest.approx(1.0)


def test_load_trajectory_zarr_values_match_zarr(tmp_path):
    path, g = _make_trajectory_zarr(tmp_path)
    data, param, time_vector = load_trajectory_zarr(path)
    np.testing.assert_array_equal(data, np.array(g["data"]))
    np.testing.assert_array_equal(param, np.array(g["param"]))
    np.testing.assert_array_equal(time_vector, np.array(g["time"]))
