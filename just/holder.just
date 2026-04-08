# ── holder ─────────────────────────────────────────────────────────────────────
# 5-particle 2-D system, state_shape=(10,), 5000 traj × 6 time steps, no param variation.
# mlp.features_in=12: 10 (state) + 1 (time) + 1 (param)

# Step 0: copy existing raw zarrs into the pipeline layout.
holder-data:
  {{py}} ./scripts/datagen/holder_setup.py

# Step 1: process raw trajectories into CFM training data.
holder-processed extras="":
  {{process}} data.name=holder data.block_size=2048 {{extras}}

# Step 2: train CFM model (10-D state, MLP).
holder-cfm extras="":
  {{train_cfm}} --multi \
    data.name=holder \
    data.block_size=1000 \
    data.batch_size=4000 \
    mlp.features_in=12 \
    mlp.features_out=10 \
    mlp.features_inner=128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=5000 \
    {{extras}}

# Step 3: generate conditioning trajectories from a trained CFM checkpoint.
# checkpoint: path to {run_dir}/{epoch}/
# modelname:  label for the output zarr (e.g. model1)
holder-cfm-trajectories checkpoint modelname extras="":
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    dataset=holder \
    n_samples=5000 \
    output=data/datasets/holder/cfm_trajectories/{{modelname}}.zarr \
    {{extras}}

# Step 4: process CFM trajectories + raw physics into regression training data.
holder-cfm-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/holder/cfm_trajectories/model1.zarr \
    output=data/datasets/holder/reg_train_data/model1.zarr \
    dataset=holder \
    {{reg_process_defaults}} \
    {{extras}}
  {{process_reg}} --multi \
    input=data/datasets/holder/raw_trajectories/train.zarr \
    output=data/datasets/holder/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/holder/cfm_train_data/train.zarr \
    dataset=holder \
    {{reg_process_defaults}} \
    {{extras}}

# Step 3b: compute OT-coupled trajectories (exact Hungarian assignment, per-step).
# n=5000, 5 time steps → feasible with the exact solver (~1-2 min total).
holder-ot-trajectories extras="":
  just ot-trajectories holder train "solver=hungarian {{extras}}"
  just ot-trajectories holder test "solver=hungarian {{extras}}"

# Step 4b: process OT trajectories into regression training data.
holder-ot-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/holder/ot_trajectories/train.zarr \
    output=data/datasets/holder/reg_train_data/ot.zarr \
    dataset=holder \
    {{reg_process_defaults}} \
    {{extras}}

# Step 5: train regression on CFM-generated trajectories.
holder-regression extras="":
  {{train_reg}} --multi \
    dataset=holder \
    {{hist_reg_defaults}} \
    mlp.features_in=12 \
    mlp.features_out=10 \
    mlp.features_inner=64,128 \
    mlp.default_emb_dim=8,16 \
    n_rollout=5000 \
    train_data=./data/datasets/holder/reg_train_data/model1.zarr \
    val_data=./data/datasets/holder/reg_train_data/model1.zarr \
    rollout_data=./data/datasets/holder/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/holder/cfm_train_data/train.zarr \
    {{extras}}

# Step 5b: train regression on OT-coupled trajectories.
holder-ot-regression extras="":
  {{train_reg}} --multi \
    dataset=holder \
    {{hist_reg_defaults}} \
    mlp.features_in=12 \
    mlp.features_out=10 \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    n_rollout=5000 \
    train_data=./data/datasets/holder/reg_train_data/ot.zarr \
    val_data=./data/datasets/holder/reg_train_data/ot.zarr \
    rollout_data=./data/datasets/holder/ot_trajectories/test.zarr \
    {{extras}}

# Step 5c: train regression on raw physics trajectories.
holder-physics-regression extras="":
  {{train_reg}} --multi \
    dataset=holder \
    {{hist_reg_defaults}} \
    mlp.features_in=12 \
    mlp.features_out=10 \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    n_rollout=5000 \
    train_data=./data/datasets/holder/reg_train_data/physics.zarr \
    val_data=./data/datasets/holder/reg_train_data/physics.zarr \
    rollout_data=./data/datasets/holder/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/holder/cfm_train_data/train.zarr \
    {{extras}}
