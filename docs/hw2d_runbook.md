# hw2d Experiment Runbook

This walks through the full pipeline for the Hasegawa–Wakatani 2D (hw2d) dataset — from raw trajectory generation to regression model training. Each step corresponds to a numbered app in `tpflow/apps/`.

**Current state of the repo** (as of April 2026): steps 0–3 are complete. You need to run steps 4 and 5.

---

## Background

The pipeline has two stages:

1. **CFM (Conditional Flow Matching)** learns a distribution over plasma field states conditioned on time and the `c1` parameter. It maps noise → state but does not model dynamics.
2. **Regression** learns to step one time unit forward, trained on CFM-generated trajectories. The two-stage design lets the regression model generalise across `c1` values by training on diverse CFM samples rather than a limited set of real trajectories.

The hw2d state is `(128, 128, 2)` — density `n` and potential `φ` at each grid point, downsampled from the raw 256×256 simulation.

---

## Step 0 — Generate raw trajectories (already done)

```bash
just hw2d-gen-train "+env=torchcpu"
just hw2d-gen-test  "+env=torchcpu"
```

Runs 2000 hw2d simulations (seeds 0–1999) via SLURM, each producing one trajectory at a randomly sampled `c1`. Results land in `data/datasets/hw2d/raw_trajectories/{train,test}.zarr` with shape `(n_seeds, 801, 128, 128, 2)`.

The `time_stride=10` in `hw2d.py` keeps every 10th snapshot, giving 801 time steps per trajectory (out of 8001 simulated).

---

## Step 1 — Process trajectories into CFM training data (already done)

```bash
just hw2d-data "+env=torchcpu"
```

Flattens `(n_traj, n_time, H, W, 2)` into individual `(state, time, param)` samples and normalises. Uses `normalize_per_time=true` because hw2d field amplitude grows significantly over the conditioning axis — a per-timestep mean/std is more stable than a global one. Output: `data/datasets/hw2d/cfm_train_data/{train,test}.zarr`, ~190 GB.

---

## Step 2 — Train CFM model (already done)

```bash
just hw2d-cfm "+env=greenegpu"
```

Trains a UNet to predict the flow field for CFM. Sweeps `unet.base_ch=32,64` (two runs in parallel). Uses the `imgrot` config as a base (`-cn imgrot`), which sets `batch_size=128` appropriate for 128×128 field data.

Checkpoints land in `multirun/<date>/<time>/<run>/`. Use `just list-checkpoints` to find them.

---

## Step 3 — Generate CFM conditioning trajectories (already done)

```bash
just hw2d-cfm-trajectories <checkpoint_path> model1 "+env=greenegpu"
```

Runs the trained CFM model to sweep `c1` from 0→1 in 801 steps (matching the physical time axis), producing 1000 synthetic trajectories. These are what the regression model trains on. Output: `data/datasets/hw2d/cfm_trajectories/model1.zarr`.

---

## Step 4 — Process regression training data (TODO)

```bash
just field-cfm-trajectories-processed hw2d "+env=torchcpu"
```

Builds `(state, next_state, time, param)` pairs from the CFM trajectories. Also processes the real physics trajectories under the same normalisation so both live in the same space. Computes `diff_scale = std(x_next - x)`, stored as a zarr attribute, which the regression model uses to normalise its prediction target.

Output:
- `data/datasets/hw2d/reg_train_data/model1.zarr` — CFM-generated pairs (training)
- `data/datasets/hw2d/reg_train_data/physics.zarr` — real physics pairs (for comparison)

---

## Step 5 — Train regression model (TODO)

```bash
just hw2d-regression "+env=greenegpu"
```

Trains a UNet to predict `(x_next - x) / diff_scale` given `(x, t, c1)`. The `difference` mode means the model learns normalised increments rather than absolute next states, which is easier to learn and avoids accumulating drift.

Evaluation runs every 50 epochs: rolls out the model from `c1=0` to `c1=1` and compares trajectory statistics (γ_n, γ_c, energy, enstrophy) against the real physics. Results are logged to W&B.

---

## Cluster notes

| Env flag | Use for |
|---|---|
| `+env=torchcpu` | CPU-only SLURM jobs (data gen, preprocessing) |
| `+env=greenegpu` | GPU SLURM jobs (CFM training, regression training) |

All `just` recipes pass `--multi` which submits via Hydra's submitit launcher. Outputs go to `multirun/<date>/<time>/`.

To check available checkpoints:
```bash
just list-checkpoints
# or directly:
.venv/bin/python tpflow/tools/list_checkpoints.py
```
