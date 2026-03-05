"""Test for 04_process_regression_data.

Run with:
    python tests/test_process_regression_data.py
"""

import importlib.util
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import zarr

# Load the module (name starts with a digit so normal import won't work)
_spec = importlib.util.spec_from_file_location(
    'process_regression_data',
    Path(__file__).parent.parent / 'tpflow' / 'apps' / '04_process_regression_data.py',
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[arg-type]
_process = _mod._process


def _arr(group: zarr.Group, name: str) -> zarr.Array:
  return cast(zarr.Array, group[name])


def make_input_zarr(path: str, n_traj: int, n_time: int, state_shape: tuple,
                    with_time: bool = True, with_param: bool = True
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Create a synthetic input zarr and return (data, param, time)."""
  rng = np.random.default_rng(42)
  data = rng.standard_normal((n_traj, n_time, *state_shape)).astype(np.float32)
  param = rng.standard_normal((n_traj,))
  time = np.linspace(0.0, 1.0, n_time)

  g = zarr.open_group(path, mode='w')
  g.create_array('data', data=data)
  if with_param:
    g.create_array('param', data=param)
  if with_time:
    g.create_array('time', data=time)
  return data, param, time


def test_output_shapes():
  n_traj, n_time = 10, 7
  state_shape = (4, 4)
  n_steps = n_time - 1
  n_samples = n_traj * n_steps

  with tempfile.TemporaryDirectory() as tmpdir:
    in_path = str(Path(tmpdir) / 'input.zarr')
    out_path = str(Path(tmpdir) / 'output.zarr')

    make_input_zarr(in_path, n_traj, n_time, state_shape)
    _process(in_path, out_path, block_size=100, trajectory_block_size=4)

    out = zarr.open_group(out_path, mode='r')
    assert _arr(out, 'data').shape == (n_samples, *state_shape)
    assert _arr(out, 'next').shape == (n_samples, *state_shape)
    assert _arr(out, 'time').shape == (n_samples,)
    assert _arr(out, 'param').shape == (n_samples,)
    print(f"  shapes OK: data/next {_arr(out, 'data').shape}, time/param {_arr(out, 'time').shape}")


def test_consecutive_states():
  """data[i] and next[i] must be consecutive states in the same trajectory."""
  n_traj, n_time = 5, 6
  state_shape = (3,)
  n_steps = n_time - 1

  with tempfile.TemporaryDirectory() as tmpdir:
    in_path = str(Path(tmpdir) / 'input.zarr')
    out_path = str(Path(tmpdir) / 'output.zarr')

    data, param, time = make_input_zarr(in_path, n_traj, n_time, state_shape)
    _process(in_path, out_path, block_size=100, trajectory_block_size=2)

    out = zarr.open_group(out_path, mode='r')
    out_data = np.array(_arr(out, 'data'))
    out_next = np.array(_arr(out, 'next'))
    out_time = np.array(_arr(out, 'time'))
    out_param = np.array(_arr(out, 'param'))

    for traj_i in range(n_traj):
      for step_j in range(n_steps):
        idx = traj_i * n_steps + step_j
        np.testing.assert_allclose(out_data[idx], data[traj_i, step_j],
                                   err_msg=f"data traj={traj_i} step={step_j}")
        np.testing.assert_allclose(out_next[idx], data[traj_i, step_j + 1],
                                   err_msg=f"next traj={traj_i} step={step_j}")
        np.testing.assert_allclose(out_time[idx], time[step_j],
                                   err_msg=f"time traj={traj_i} step={step_j}")
        np.testing.assert_allclose(out_param[idx], param[traj_i],
                                   err_msg=f"param traj={traj_i} step={step_j}")
    print("  data/next/time/param values OK")


def test_no_time_array():
  """Without a time array in input, time should default to integer indices."""
  n_traj, n_time = 4, 5
  state_shape = (2,)

  with tempfile.TemporaryDirectory() as tmpdir:
    in_path = str(Path(tmpdir) / 'input.zarr')
    out_path = str(Path(tmpdir) / 'output.zarr')

    make_input_zarr(in_path, n_traj, n_time, state_shape, with_time=False)
    _process(in_path, out_path, block_size=100, trajectory_block_size=4)

    out = zarr.open_group(out_path, mode='r')
    out_time = np.array(_arr(out, 'time'))
    expected = np.tile(np.arange(n_time - 1, dtype=np.float64), n_traj)
    np.testing.assert_array_equal(out_time, expected)
    print("  default integer time OK")


def test_missing_param():
  """Without a param array, param should default to ones."""
  n_traj, n_time = 3, 4
  state_shape = (2,)

  with tempfile.TemporaryDirectory() as tmpdir:
    in_path = str(Path(tmpdir) / 'input.zarr')
    out_path = str(Path(tmpdir) / 'output.zarr')

    make_input_zarr(in_path, n_traj, n_time, state_shape, with_param=False)
    _process(in_path, out_path, block_size=100, trajectory_block_size=4)

    out = zarr.open_group(out_path, mode='r')
    np.testing.assert_array_equal(
        np.array(_arr(out, 'param')),
        np.ones(n_traj * (n_time - 1)),
    )
    print("  missing param defaults to ones OK")


def test_block_boundary():
  """Results must be identical regardless of trajectory_block_size."""
  n_traj, n_time = 9, 5
  state_shape = (2,)

  with tempfile.TemporaryDirectory() as tmpdir:
    in_path = str(Path(tmpdir) / 'input.zarr')

    make_input_zarr(in_path, n_traj, n_time, state_shape)

    ref_path = str(Path(tmpdir) / 'ref.zarr')
    _process(in_path, ref_path, block_size=100, trajectory_block_size=9)  # one block
    ref = zarr.open_group(ref_path, mode='r')

    for bs in [1, 3, 4]:
      out_path = str(Path(tmpdir) / f'out_{bs}.zarr')
      _process(in_path, out_path, block_size=100, trajectory_block_size=bs)
      out = zarr.open_group(out_path, mode='r')
      for key in ('data', 'next', 'time', 'param'):
        np.testing.assert_allclose(
            np.array(_arr(out, key)), np.array(_arr(ref, key)),
            err_msg=f"mismatch in '{key}' with block_size={bs}")
    print("  block boundary consistency OK")


if __name__ == '__main__':
  tests = [
      test_output_shapes,
      test_consecutive_states,
      test_no_time_array,
      test_missing_param,
      test_block_boundary,
  ]
  passed = 0
  for test in tests:
    print(f"Running {test.__name__}...")
    try:
      test()
      print("  PASSED")
      passed += 1
    except Exception as e:
      import traceback
      traceback.print_exc()
      print(f"  FAILED: {e}")
  print(f"\n{passed}/{len(tests)} tests passed")
