# Broad Goal
I need to plot the population statistics from hw2d datasets and compare the statistics between the physics (raw_trajectories) and the cfm_generated trajectories. The statistics should be the ones computed during the regression training for hw2d

# Output
1. Make .zarr files that contain the statistics so I can plot them later (I think we have done that previously somewhere)
2. Make preliminalry matplotlib plots

# Potential caviats and resources
1. Data is normalized in a strange (per-time-step) way; you need to take care of this (I want the "unnormalized" statistics)
2. The data format is explained in the README in the section "## Dataset structure"
3. See how we store the statistics in the regression training app. Something like this should be fine

# Logging

## Script written

`scripts/extract_hw2d_statistics.py` — processes all three sources in batches:
- **physics train**: `raw_trajectories/train.zarr` (no unnormalization needed)
- **CFM model1**: `cfm_trajectories/model1.zarr` (unnormalized using `per_time_mean`/`per_time_std` from `cfm_train_data/train.zarr`)
- **physics test**: `raw_trajectories/test.zarr`

Outputs:
- `data/datasets/hw2d/stats/train/physics_statistics.zarr`
- `data/datasets/hw2d/stats/train/cfm_statistics.zarr`
- `data/datasets/hw2d/stats/test/physics_statistics.zarr`
- `data/datasets/hw2d/stats/plots/hw2d_statistics_comparison.png`
- `data/datasets/hw2d/stats/plots/hw2d_test_statistics.png`

Each zarr stores arrays of shape `(n_time, n_traj)` for each statistic:
`gamma_n`, `gamma_c`, `energy`, `hw2d_enstrophy`.

## What to run

The script needs a compute node (~32 GB RAM, no GPU required). Request one and run:

```bash
srun --mem=64G --cpus-per-task=4 --time=2:00:00 \
  .venv/bin/python scripts/extract_hw2d_statistics.py \
  --batch-size 10
```

Or with all defaults (uses all 1000 train trajectories):

```bash
.venv/bin/python scripts/extract_hw2d_statistics.py --batch-size 10
```

Optional flags:
- `--n-traj N`: limit to N trajectories per source (useful for a quick test, e.g. `--n-traj 50`)
- `--batch-size B`: trajectories per batch (default 50; lower if memory is tight)
- `--dataset-root PATH`: override dataset root (default `data/datasets/hw2d`)
