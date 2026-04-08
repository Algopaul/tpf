# ── bump ───────────────────────────────────────────────────────────────────────
# 2-D particle dataset, state_shape=(2,), 25000 particles × 128 time steps.
# No param variation (param=1.0 throughout).
# Source: data/datasets/bump/trajectory.h5 (n_time=128, n_particles=25000, 2)
# mlp.features_in=4: 2 (state) + 1 (time) + 1 (param).
#
# OT note: n=25000 makes Hungarian slow; subsample n_traj=10000 for OT steps.

# Step 0: split raw_trajectories/train.zarr 50/50 into train (12500) / test (12500).
bump-split:
  {{py}} scripts/datagen/bump_split.py

# Step 1: process raw trajectories into CFM training data.
bump-processed extras="":
  {{process}} data.name=bump data.block_size=2048 {{extras}}

# Step 2: train CFM model (2-D hist, MLP — same arch as vlasov/langevin).
bump-cfm extras="":
  {{train_cfm}} --multi \
    data.name=bump \
    data.block_size=1024 \
    data.batch_size=8192 \
    mlp.features_in=4 \
    mlp.features_inner=128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=12500 \
    {{extras}}

# Step 3: generate CFM conditioning trajectories from a trained checkpoint.
# checkpoint: path to {run_dir}/{epoch}/
# modelname:  label for the output zarr (e.g. model1)
bump-cfm-trajectories checkpoint modelname extras="":
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    dataset=bump \
    n_samples=12500 \
    output=data/datasets/bump/cfm_trajectories/{{modelname}}.zarr \
    {{extras}}

# Step 4: process CFM trajectories + raw physics into regression training data.
bump-cfm-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/bump/cfm_trajectories/model1.zarr \
    output=data/datasets/bump/reg_train_data/model1.zarr \
    dataset=bump \
    {{reg_process_defaults}} \
    {{extras}}
  {{process_reg}} --multi \
    input=data/datasets/bump/raw_trajectories/train.zarr \
    output=data/datasets/bump/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/bump/cfm_train_data/train.zarr \
    dataset=bump \
    {{reg_process_defaults}} \
    {{extras}}

# Step 3b: compute OT-coupled trajectories.
# time_stride=4: 128 → 32 frames, 31 OT steps.  n_traj=10000 keeps Hungarian feasible.
bump-ot-trajectories extras="":
  just ot-trajectories bump train "solver=hungarian n_traj=10000 time_stride=4 {{extras}}"
  just ot-trajectories bump test  "solver=hungarian n_traj=10000 time_stride=4 {{extras}}"

# Step 4b: process OT trajectories into regression training data.
# Input is already normalised so norm_stats_path is intentionally omitted.
bump-ot-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/bump/ot_trajectories/train.zarr \
    output=data/datasets/bump/reg_train_data/ot.zarr \
    dataset=bump \
    {{reg_process_defaults}} \
    {{extras}}

# Step 5: train regression on CFM-generated trajectories.
bump-regression extras="":
  {{train_reg}} --multi \
    dataset=bump \
    {{hist_reg_defaults}} \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    train_data=./data/datasets/bump/reg_train_data/model1.zarr \
    val_data=./data/datasets/bump/reg_train_data/model1.zarr \
    rollout_data=./data/datasets/bump/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/bump/cfm_train_data/train.zarr \
    {{extras}}

# Step 5b: train regression on OT-coupled trajectories.
bump-ot-regression extras="":
  {{train_reg}} --multi \
    dataset=bump \
    {{hist_reg_defaults}} \
    mlp.features_inner=256 \
    mlp.default_emb_dim=32 \
    n_rollout=12500 \
    train_data=./data/datasets/bump/reg_train_data/ot.zarr \
    val_data=./data/datasets/bump/reg_train_data/ot.zarr \
    rollout_data=./data/datasets/bump/ot_trajectories/test.zarr \
    {{extras}}

# Step 5c: train regression on raw physics trajectories.
bump-physics-regression extras="":
  {{train_reg}} --multi \
    dataset=bump \
    {{hist_reg_defaults}} \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    train_data=./data/datasets/bump/reg_train_data/physics.zarr \
    val_data=./data/datasets/bump/reg_train_data/physics.zarr \
    rollout_data=./data/datasets/bump/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/bump/cfm_train_data/train.zarr \
    {{extras}}
