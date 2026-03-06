# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Commands

```bash
just test        # run the test suite
just lint        # ruff check tpflow/
just fmt         # ruff format tpflow/
just install     # pip install -e .
```

All pipeline recipes are in the `justfile`. Run `just --list` to see them.

## Architecture

**tpflow** is a two-stage ML pipeline: Conditional Flow Matching (CFM) learns a
parameter-conditioned distribution from data; a regression model then learns to
step along the resulting trajectories one time-step at a time.

### Pipeline (numbered apps in `tpflow/apps/`)

| Step | Script | Purpose |
|------|--------|---------|
| 01   | `01_process_trajectories.py`   | Flatten raw trajectory zarrs into CFM training data (normalise, tile time/param) |
| 01b  | `01b_convert_to_wds.py`        | Convert CFM training data to WebDataset shards for faster I/O |
| 02   | `02_train_cfm.py`              | Train CFM model: noise → state conditioned on (time, param) |
| 03   | `03_gen_cond_trajectories.py`  | Run trained CFM, sweep conditioning param → trajectory zarrs |
| 04   | `04_process_regression_data.py`| Build (state, next_state, time, param) pairs; compute `diff_scale` |
| 05   | `05_train_regression.py`       | Train one-step regression model; eval logs stats plots to wandb |

### Justfile recipes

Dataset-specific data-generation and CFM recipes keep their own names
(`gaurot-data`, `kolflow-cfm`, `hw2d-cfm`, …). The later pipeline steps for
field datasets share generic parameterised recipes:

```
just field-cfm-trajectories   <ds> <checkpoint> <modelname> <env>
just field-cfm-trajectories-processed <ds> <env>
just field-regression         <ds> <env>
```

where `<ds>` is e.g. `kolflow` or `hw2d`.

### Configuration (`tpflow/config/__init__.py`)

All apps use **Hydra** with dataclass configs in the `ConfigStore`. Config names
map to `config_name=` arguments (`cfm`, `regression`, `cond_traj`,
`regression_data`, `wds_convert`). Every config has a `WandbConfig` sub-field
(`cfg.wandb.mode`, `.jobname`, `.group`, `.tag`).

## Module reference

### `tpflow/model.py`

- `CFMDec` — wraps an MLP/UNet; provides `euler_steps` and `rk4_steps` ODE integrators.
- `RegressionDec` / `RegressionUNetDec` — one-step predictors for `step` or `difference` mode.
- `make_flow_fn(model, n_steps)` — returns a **persistent** JIT-compiled function that uses `jax.lax.map` to sweep all conditioning values in one call. **Call this with the current model after each `nnx.merge`** — it closes over the model at call time. Reuse the returned function across batches.
- `flow_inference(model, source_batch, cslist, n_steps)` — Python-loop equivalent of `make_flow_fn`; recompiles on every call. Only used in tests to verify numerical equivalence.
- `regression_rollout(model, x0, time_vector, param, mode, diff_scale=1.0)` — rolls out a regression model step by step; in `difference` mode: `x_{t+1} = x_t + diff_scale * pred`.
- `_save_checkpoint(model, cfg, epoch, sample_shape, info, output_dir)` — pure file-writing (no Hydra dependency); called by `store_model` / `store_regression_model` which resolve the Hydra output dir first.

### `tpflow/processing.py`

Pure-numpy helpers, no Hydra/zarr side-effects — all testable in isolation.

- `open_zarr_array(group, name, size=None)` — returns the named array, or `np.ones(size)` fallback when absent.
- `auto_block_sizes(state_shape, n_time, target_mb=2.0)` — returns `(block_size, trajectory_block_size)` targeting ~2 MB zarr chunks (f4 dtype).
- `extract_regression_pairs(data_block, time_vector, param_block)` — slices `[:, :-1]` / `[:, 1:]` to produce `(cur, nxt, time_flat, param_flat)`, each `(n_traj*(n_time-1), …)`.
- `load_trajectory_zarr(path, n=None)` — loads a raw trajectory zarr fully into memory; returns `(data, param, time_vector)` with `ones` / `linspace(0,1)` fallbacks when arrays are absent.

### `tpflow/statistics.py`

Pure-numpy trajectory statistics. All functions take
`trajectories: (n_time, n_rollout, *state_shape)` and return
`(n_time, n_rollout)` so callers can plot mean ± std over the ensemble.

- `enstrophy(trajectories)` — mean(x²) over state dims per sample.
- `kurtosis(trajectories)` — excess kurtosis of state values per sample.
- `trajectory_statistics(trajectories)` — returns `{"enstrophy": …, "kurtosis": …}`.

### `tpflow/data.py`

- `ZarrData` / `WDSData` — block-shuffled dataloaders for CFM training (zarr and WebDataset backends).
- `RegressionZarrData` — block-shuffled loader for regression data (`data`, `next`, `time`, `param`); exposes `diff_scale: float` from zarr attrs.
- `get_regression_val_data(path, batch_size)` — loads full val set as a list of batch dicts.
- `device_prefetch(it, size=2)` — wraps any iterator to prefetch batches onto the JAX device.

### `tpflow/visualization.py`

- `trace_video(data)` — renders `(n_time, n_particles, 2)` particle data into an RGB video (trail decay effect).
- `angle_color_coded(data, source_data)` — colours particles by their source angle (HSV colormap).

### `tpflow/util.py`

- `init_wandb(cfg, job_type, data_name=None)` — used by every app as a context manager; handles SLURM IDs, job naming, and `cfg.wandb.*` settings.
- `log_duration()` — decorator that logs wall-clock time of the wrapped function.

### `tpflow/tools/list_checkpoints.py`

Typer CLI: lists checkpoints from the `multirun/` output directory.

## Data formats

**Raw trajectories** (input to step 01 and 04):
```
data:  (n_traj, n_time, *state_shape)   float
param: (n_traj,)                        float  — may be absent (treated as ones)
time:  (n_time,)                        float  — may be absent (treated as linspace(0,1))
```

**CFM training data** (output of step 01):
```
data:  (n_samples, *state_shape)   normalised, flattened across traj×time
time:  (n_samples,)
param: (n_samples,)
```

**Regression training data** (output of step 04):
```
data:  (n_samples, *state_shape)   current state
next:  (n_samples, *state_shape)   next state
time:  (n_samples,)
param: (n_samples,)
attrs: diff_scale                  std(x_next - x) over training set
```

**Conditioning trajectories** (output of step 03):
```
data:        (n_samples, n_cond_steps, *state_shape)
source:      (n_samples, *sample_shape)
conditioning:(n_cond_steps,)
```

Hydra outputs → `outputs/` (single runs) or `multirun/` (sweeps).
Checkpoints → `{run_dir}/{epoch}/state/` + `config.yaml` + `checkpoint_info.json`.

## Key concepts

### Regression `difference` mode

Model predicts a normalised increment: target = `(x_next - x) / diff_scale`.
`diff_scale` is computed as `std(x_next - x)` over the full training set
(one-pass accumulation in step 04) and stored as a zarr group attribute.
Rollout: `x_{t+1} = x_t + diff_scale * model(x_t, t, p)`.

### Data types: `hist` vs `field`

`cfg.data_type` controls visualisation and state interpretation:
- `hist` — particle data, `state_shape = (n_particles, 2)`. Visualised with `trace_video`.
- `field` — spatial field data, `state_shape = (H, W[, C])`. Visualised with `frame_rgb` from hdfv.

### `make_flow_fn` usage pattern

```python
# After nnx.merge updates the model, compile fresh for this eval step:
run_fn = make_flow_fn(model, n_steps=cfg.inference.n_ode_steps)
cslist = jnp.linspace(cfg.cond_start, cfg.cond_end, cfg.n_cond_steps)
out = run_fn(source_batch, cslist)  # (n_cond_steps, batch, *state_shape)
```

Do **not** compile once outside the training loop and reuse across epochs —
the closed-over model will be stale after `nnx.merge`.

## Tests

```
tests/
  test_processing.py           open_zarr_array, auto_block_sizes,
                               extract_regression_pairs, load_trajectory_zarr
  test_process_regression_data.py  integration tests for app 04
  test_model.py                make_flow_fn vs flow_inference equivalence,
                               _save_checkpoint / load roundtrip
```

Run with `just test`. All tests use in-memory zarr or `tmp_path`; no real data needed.

## Third-party packages (private repos)

- **`flanch`** — `EmbMLP`, `UNet`, `get_optimizer`, `get_train_step`, `Recorder`
- **`hdfx`** — `zarrshuffle`, `ds_statistics`, `flatten_trajectories`
- **`hdfv`** — `frame_rgb`, `grid_shape`, `histogram_frames` for visualisation
