# tpflow

A two-stage ML pipeline for learning parameter-conditioned distributions from trajectory data.

**Stage 1 — Conditional Flow Matching (CFM):** learns a parameter-conditioned distribution from data.
**Stage 2 — Regression:** learns to step along the resulting trajectories one time-step at a time.

## Pipeline

| Step | Script | Purpose |
|------|--------|---------|
| 01   | `01_process_trajectories.py`    | Flatten raw trajectory zarrs into CFM training data (normalise, tile time/param) |
| 01b  | `01b_convert_to_wds.py`         | Convert CFM training data to WebDataset shards for faster I/O |
| 02   | `02_train_cfm.py`               | Train CFM model: noise → state conditioned on (time, param) |
| 03   | `03_gen_cond_trajectories.py`   | Run trained CFM, sweep conditioning param → trajectory zarrs |
| 04   | `04_process_regression_data.py` | Build (state, next_state, time, param) pairs; compute `diff_scale` |
| 05   | `05_train_regression.py`        | Train one-step regression model; eval logs stats/plots to wandb |

## Commands

```bash
just test        # run the test suite
just lint        # ruff check tpflow/
just fmt         # ruff format tpflow/
just install     # pip install -e .
```

All pipeline recipes are in the `justfile`. Run `just --list` to see them.

### Dataset-specific recipes

Dataset-specific data-generation and CFM recipes keep their own names (`gaurot-data`, `kolflow-cfm`, `hw2d-cfm`, …). Later pipeline steps share generic parameterised recipes:

```bash
just field-cfm-trajectories             <ds> <checkpoint> <modelname> <env>
just field-cfm-trajectories-processed   <ds> <env>
just field-regression                   <ds> <env>
```

where `<ds>` is e.g. `kolflow` or `hw2d`.

## Dataset structure

```
data/datasets/{name}/
├── raw_trajectories/
│   ├── train.zarr (data: (n_trajectories, n_time, *state_shape), time: (n_time,), param: (n_trajectories,))
│   └── test.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│
├── cfm_train_data/ # here n_samples = n_trajectories*n_time
│   ├── train.zarr (data: (n_samples, *state_shape), time: (n_samples,), param: (n_samples,))
│   └── test.zarr (data: (n_samples, *state_shape), time: (n_samples,), param: (n_samples,))
│
├── cfm_trajectories/
│   ├── model_1.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│   └── model_2.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│
├── reg_train_data/
│   ├── physics.zarr (data: (n_samples, *state_shape), next: (n_samples, *state_shape), time: (n_samples) param: (n_trajectories,))
│   ├── model1.zarr (data: (n_samples, *state_shape), next: (n_samples, *state_shape), time: (n_samples) param: (n_trajectories,))
│   └── model2.zarr (data: (n_samples, *state_shape), next: (n_samples, *state_shape), time: (n_samples) param: (n_trajectories,))
│
└── stats/
    ├── train/ (n_time, stat_shape)
    │   ├── e.g. enstrophy (mean: n_time, var: n_time)
    │   └── e.g. energy (mean: n_time, var: n_time)
    └── test/
```

## Configuration

All apps use **Hydra** with dataclass configs in the `ConfigStore` (`tpflow/config/__init__.py`). Config names map to `config_name=` arguments: `cfm`, `regression`, `cond_traj`, `regression_data`, `wds_convert`. Every config has a `WandbConfig` sub-field (`cfg.wandb.mode`, `.jobname`, `.group`, `.tag`).

Hydra outputs → `outputs/` (single runs) or `multirun/` (sweeps).
Checkpoints → `{run_dir}/{epoch}/state/` + `config.yaml` + `checkpoint_info.json`.

## Key concepts

### Regression `difference` mode

Model predicts a normalised increment: target = `(x_next - x) / diff_scale`. `diff_scale` is computed as `std(x_next - x)` over the full training set and stored as a zarr group attribute. Rollout: `x_{t+1} = x_t + diff_scale * model(x_t, t, p)`.

### Data types: `hist` vs `field`

`cfg.data_type` controls visualisation and state interpretation:
- `hist` — particle data, `state_shape = (n_particles, 2)`. Visualised with `trace_video`.
- `field` — spatial field data, `state_shape = (H, W[, C])`. Visualised with `frame_rgb` from hdfv.

## Tests

```
tests/
  test_processing.py                open_zarr_array, auto_block_sizes,
                                    extract_regression_pairs, load_trajectory_zarr
  test_process_regression_data.py   integration tests for app 04
  test_model.py                     make_flow_fn vs flow_inference equivalence,
                                    _save_checkpoint / load roundtrip
```

Run with `just test`. All tests use in-memory zarr or `tmp_path`; no real data needed.

## Dependencies (private repos)

- **`flanch`** — `EmbMLP`, `UNet`, `get_optimizer`, `get_train_step`, `Recorder`
- **`hdfx`** — `zarrshuffle`, `ds_statistics`, `flatten_trajectories`
- **`hdfv`** — `frame_rgb`, `grid_shape`, `histogram_frames` for visualisation
