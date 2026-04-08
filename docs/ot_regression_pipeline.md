# OT Regression Pipeline

This document describes the OT (Optimal Transport) path of the **tpflow** pipeline — the alternative to CFM-based trajectory generation for training the one-step regression model.

## Goal

Train a one-step regression model `f(x_t, t, p) ≈ x_{t+1}` without running a CFM model.
Instead of using CFM-generated trajectories, the training pairs `(x_t, x_{t+1})` are
constructed by coupling independent snapshots of the true particle distribution at
consecutive time steps via optimal transport.

## Why OT instead of CFM trajectories?

CFM trajectories are synthetic — they are produced by a generative model and may not
faithfully reproduce the physics. OT trajectories are built directly from the measured
data by finding the minimum-cost permutation of particles between time steps. This gives
a training signal that is grounded in the true distribution dynamics.

## Input format

Raw trajectories zarr (produced outside this pipeline):

```
data:  (n_traj, n_time, *state_shape)   float32  — particle states
param: (n_traj,)                        float32  — per-trajectory parameter (1.0 if unused)
time:  (n_time,)                        float32  — physical time axis
```

For the bump dataset: `n_traj=12500`, `n_time=128`, `state_shape=(2,)`.

## Steps (just recipes)

### Step 3b — `bump-ot-trajectories`

**Script:** `tpflow/apps/03b_gen_ot_trajectories.py`

Reads the raw trajectories and produces OT-coupled trajectories.

What it does:
1. Optionally subsamples to `n_traj` trajectories.
2. Subsamples the time axis by `time_stride` (e.g. stride=4: 128 → 32 frames).
3. Normalises particle states using mean/std from `norm_stats_path` (the CFM training zarr).
   The output zarr is therefore in **normalised space**.
4. For each consecutive time-step pair `(t, t+1)`, solves a linear assignment problem
   to find the permutation of particles that minimises total squared displacement
   (Hungarian algorithm, exact; or Sinkhorn for large n).
5. Chains the per-step assignments to build continuous coupled trajectories:
   particle `i` at step `t+1` is the particle assigned to `i` at step `t`.
6. Saves output zarr with the same layout as the raw trajectories but in normalised space
   and with the (possibly subsampled) time axis.

Output zarr layout:
```
data:  (n_traj, n_time_sub, *state_shape)   float32  — normalised, OT-coupled
param: (n_traj,)
time:  (n_time_sub,)                        physical time values (e.g. [0,4,8,...,124])
```

Key config options:
| Option | Default | Meaning |
|---|---|---|
| `solver` | `hungarian` | `hungarian` (exact, O(n³)) or `sinkhorn` (approximate, GPU-friendly) |
| `n_traj` | 0 (all) | Subsample trajectories to keep Hungarian feasible for large n |
| `time_stride` | 1 | Skip every k frames; larger stride = more informative OT couplings |
| `epsilon` | 0.05 | Sinkhorn regularisation (only used when `solver=sinkhorn`) |
| `norm_stats_path` | `""` | Path to CFM training zarr for normalisation stats |

### Step 4b — `bump-ot-trajectories-processed`

**Script:** `tpflow/apps/04_process_regression_data.py`

Converts OT trajectories into `(x_t, x_{t+1}, t, p)` regression pairs.

What it does:
1. Reads the OT trajectory zarr (already normalised — **do not pass `norm_stats_path`**).
2. Normalises the time axis to `[0, 1]` by dividing by `time[-1]`.
   This ensures the model conditioning time always matches the rollout which uses
   `linspace(0, 1, n_time)`.
3. Extracts all consecutive `(cur, nxt)` state pairs across all trajectories.
4. Computes `diff_scale = std(x_next - x)` over the full training set (per state
   channel). Stored as a zarr attribute; used in training to normalise regression
   targets so `var(target) ≈ 1`.

Output zarr layout:
```
data:  (n_samples, *state_shape)   current state x_t
next:  (n_samples, *state_shape)   next state x_{t+1}
time:  (n_samples,)                normalised time in [0, 1]
param: (n_samples,)
attrs: diff_scale                  std(x_next - x), shape (*state_shape)
```

where `n_samples = n_traj * (n_time_sub - 1)`.

### Step 5b — `bump-ot-regression`

**Script:** `tpflow/apps/05_train_regression.py`

Trains the one-step regression model on OT pairs.

Key settings vs CFM regression:
- `train_data` / `val_data`: the OT regression zarr (no `norm_stats_path` — already normalised)
- `rollout_data`: the OT test trajectories zarr (also already normalised)
- `mode=difference`: model predicts `(x_next - x) / diff_scale`; rollout: `x_{t+1} = x_t + diff_scale * model(x_t, t, p)`
- Larger model recommended (OT targets are harder than physics): `mlp.features_inner=256`, `mlp.default_emb_dim=32`

## Important invariants

1. **Time conditioning must be in `[0, 1]`** during both training and rollout.
   Step 4b normalises the stored time. The rollout uses `linspace(0, 1, n_time)`.
   If you skip step 4b and use the raw OT zarr directly, times will be physical
   (e.g. 0, 4, 8, …, 124) and the model will fail at rollout.

2. **OT data is already normalised** — do not pass `norm_stats_path` to step 4b or
   to the regression training recipe. The normalisation happens once in step 3b.

3. **`diff_scale` ties training and rollout together** — it is computed from the OT
   training set and stored as a zarr attribute. The training script reads it automatically
   from `train_data`. It is also saved alongside each checkpoint and loaded at eval time.

## Evaluation

After training, export the eval zarr and run the W2 comparison:

```bash
just export-eval multirun/<run-dir>/<epoch>
python scripts/w2_comparison_bump.py
```

The exported `rollout.zarr` contains:
- `rollout`: autoregressive rollout from `x_0`
- `one_step_ahead`: teacher-forced single-step predictions (useful to diagnose whether errors are per-step or compounding)
- `reference`: true trajectory from `rollout_data`
- `time`: physical time axis
