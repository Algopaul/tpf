# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Lint
ruff check tpflow/

# Format
ruff format tpflow/

# Install (editable)
pip install -e .
```

There are no automated tests. The apps are run directly with Python and Hydra.

## Architecture Overview

**tpflow** is a two-stage ML pipeline for learning parameter-conditioned distributions using Conditional Flow Matching (CFM) followed by regression.

### Pipeline (numbered apps in `tpflow/apps/`)

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `01_process_trajectories.py` | Flatten raw trajectory zarrs into CFM training data (normalises, tiles time/param) |
| 01b | `01b_convert_to_wds.py` | Convert CFM training data to WebDataset shards for faster I/O |
| 02 | `02_train_cfm.py` | Train a CFM model to map noise → state conditioned on (time, param) |
| 03 | `03_gen_cond_trajectories.py` | Run trained CFM model, sweeping conditioning param to produce trajectory zarrs |
| 04 | `04_process_regression_data.py` | Build (state, next_state, time, param) pairs from trajectories; computes `diff_scale` |
| 05 | `05_train_regression.py` | Train a one-step regression model on those pairs |

### Configuration (`tpflow/config/__init__.py`)

All apps use **Hydra** with dataclass configs registered in the `ConfigStore`. Each config dataclass contains a `WandbConfig` field (`cfg.wandb.mode`, `.jobname`, `.group`, `.tag`). Config names map to Hydra `config_name=` arguments (e.g. `cfm`, `regression`, `cond_traj`, `regression_data`, `wds_convert`).

### Core modules

**`tpflow/model.py`**
- `CFMDec` — wraps an MLP/UNet, adds `euler_steps` and `rk4_steps` ODE integrators
- `RegressionDec` / `RegressionUNetDec` — one-step predictors for `step` or `difference` mode
- `make_flow_fn(model, n_steps)` — returns a **persistent** JIT-compiled function; call once before a batch loop and reuse. Uses `jax.lax.map` internally to sweep over conditioning values.
- `flow_inference(model, source_batch, cslist, n_steps)` — convenience wrapper; recompiles on every call, use only for one-off inference
- `regression_rollout(model, x0, time_vector, param, mode, diff_scale=1.0)` — rolls out a regression model; in `difference` mode applies `x += diff_scale * pred`

**`tpflow/data.py`**
- `ZarrData` / `WDSData` — block-shuffled dataloaders for CFM training
- `RegressionZarrData` — block-shuffled loader for regression data; fields: `data`, `next`, `time`, `param`; exposes `diff_scale: float` read from zarr attrs
- `get_regression_val_data` — loads full val set as a list of batches
- `device_prefetch` — wraps an iterator to prefetch batches onto the JAX device

**`tpflow/util.py`**
- `init_wandb(cfg, job_type, data_name=None)` — standard wandb init used by all apps as a context manager; reads `cfg.wandb.*` and optionally `cfg.data.name`
- `log_duration()` — decorator that logs wall-clock time of the decorated function

**`tpflow/tools/list_checkpoints.py`** — CLI tool (Typer) to list checkpoints from the `multirun/` output directory.

### Third-party packages (private repos)

- **`flanch`** — provides `EmbMLP`, `UNet`, `get_optimizer`, `get_train_step`, `Recorder`
- **`hdfx`** — provides `zarrshuffle`, `ds_statistics`, `flatten_trajectories`
- **`hdfv`** — provides `frame_rgb`, `grid_shape` for field visualisation

### Data format

All datasets are **zarr** groups. Raw trajectories: `data (n_traj, n_time, *state_shape)`, `param (n_traj,)`, `time (n_time,)`. Hydra outputs go to `outputs/` (single runs) or `multirun/` (sweeps); checkpoints are saved as `{epoch}/state/` + `config.yaml` + `checkpoint_info.json`.

### Regression `difference` mode

The model predicts a normalised increment. `04_process_regression_data.py` computes `diff_scale = std(x_next - x)` over the full training set (one-pass accumulation) and stores it as a zarr group attribute. Training target: `(x_next - x) / diff_scale`. Rollout: `x_{t+1} = x_t + diff_scale * model(x_t, t, p)`. `RegressionZarrData` exposes `diff_scale` as a float attribute; `05_train_regression.py` reads it once, closes over it in the loss, and passes it to `regression_rollout`.

### Chunk size auto-computation (`04_process_regression_data.py`)

`block_size` and `trajectory_block_size` default to `0` in `RegressionDataConfig`, meaning auto. `_auto_block_sizes(state_shape, n_time, target_mb=2.0)` computes both targeting ~2 MB chunks based on f4 bytes-per-sample. Non-zero config values override the auto values.
