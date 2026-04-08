# ── vlasov ─────────────────────────────────────────────────────────────────────
# 1-particle 2-D phase-space system, state_shape=(2,), 12500 traj × 128 time steps.
# No param variation (param=0 throughout).
# mlp.features_in=4: 2 (state) + 1 (time) + 1 (param) — identical to Langevin,
# so hist_reg_defaults needs no override.
#
# OT note: n=12500 makes Hungarian infeasible (cost matrix ~625 MB, hours/step).
# Sinkhorn with epsilon=0.01 is used instead for sharp couplings.

# Step 1: process raw trajectories into CFM training data.
vlasov-processed extras="":
  {{process}} data.name=vlasov data.block_size=2048 {{extras}}

# Step 2: train CFM model (2-D hist, MLP — same arch as Langevin).
vlasov-cfm extras="":
  {{train_cfm}} --multi \
    data.name=vlasov \
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

# Step 3: generate conditioning trajectories from a trained CFM checkpoint.
# checkpoint: path to {run_dir}/{epoch}/
# modelname:  label for the output zarr (e.g. model1)
vlasov-cfm-trajectories checkpoint modelname extras="":
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    dataset=vlasov \
    n_samples=12500 \
    output=data/datasets/vlasov/cfm_trajectories/{{modelname}}.zarr \
    {{extras}}

# Step 4: process CFM trajectories + raw physics into regression training data.
vlasov-cfm-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/vlasov/cfm_trajectories/model1.zarr \
    output=data/datasets/vlasov/reg_train_data/model1.zarr \
    dataset=vlasov \
    {{reg_process_defaults}} \
    {{extras}}
  {{process_reg}} --multi \
    input=data/datasets/vlasov/raw_trajectories/train.zarr \
    output=data/datasets/vlasov/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/vlasov/cfm_train_data/train.zarr \
    dataset=vlasov \
    {{reg_process_defaults}} \
    {{extras}}

# Step 3b: compute OT-coupled trajectories.
# n_traj=5000, time_stride=8: 128 → 16 frames, 15 OT steps.
# Larger dt gives more informative couplings (dt=1 consecutive frames are nearly trivial).
vlasov-ot-trajectories extras="":
  just ot-trajectories vlasov train "solver=hungarian n_traj=10000 time_stride=4 {{extras}}"
  just ot-trajectories vlasov test "solver=hungarian n_traj=10000 time_stride=4 {{extras}}"

# Step 4b: process OT trajectories into regression training data.
vlasov-ot-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/vlasov/ot_trajectories/train.zarr \
    output=data/datasets/vlasov/reg_train_data/ot.zarr \
    dataset=vlasov \
    {{reg_process_defaults}} \
    {{extras}}

# Step 5: train regression on CFM-generated trajectories (sweep over arch sizes).
vlasov-regression extras="":
  {{train_reg}} --multi \
    dataset=vlasov \
    {{hist_reg_defaults}} \
    mlp.features_inner=64,128 \
    mlp.default_emb_dim=8,16 \
    train_data=./data/datasets/vlasov/reg_train_data/model1.zarr \
    val_data=./data/datasets/vlasov/reg_train_data/model1.zarr \
    rollout_data=./data/datasets/vlasov/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/vlasov/cfm_train_data/train.zarr \
    {{extras}}

# Step 5b: train regression on OT-coupled trajectories.
# rollout_data uses the subsampled OT test trajectories (2000 particles).
vlasov-ot-regression extras="":
  {{train_reg}} --multi \
    dataset=vlasov \
    {{hist_reg_defaults}} \
    mlp.features_inner=256 \
    mlp.default_emb_dim=32 \
    n_rollout=10000 \
    train_data=./data/datasets/vlasov/reg_train_data/ot.zarr \
    val_data=./data/datasets/vlasov/reg_train_data/ot.zarr \
    rollout_data=./data/datasets/vlasov/ot_trajectories/test.zarr \
    {{extras}}

# Step 5c: train regression on raw physics trajectories.
vlasov-physics-regression extras="":
  {{train_reg}} --multi \
    dataset=vlasov \
    {{hist_reg_defaults}} \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    train_data=./data/datasets/vlasov/reg_train_data/physics.zarr \
    val_data=./data/datasets/vlasov/reg_train_data/physics.zarr \
    rollout_data=./data/datasets/vlasov/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/vlasov/cfm_train_data/train.zarr \
    {{extras}}
