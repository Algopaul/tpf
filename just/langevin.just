# ── langevin_potential ─────────────────────────────────────────────────────────
# Step 0: generate raw trajectories (N=2048, d=2, dt=1/512).
langevin-data:
  {{py}} ./scripts/datagen/langevin_potential.py

# Step 1: process raw trajectories into CFM training data.
langevin-processed extras="":
  {{process}} data.name=langevin_potential data.block_size=2048 {{extras}}

# Step 2: train CFM model (2-D hist, MLP).
langevin-cfm extras="":
  {{train_cfm}} --multi \
    data.name=langevin_potential \
    data.block_size=1024 \
    data.batch_size=8192 \
    mlp.features_in=4 \
    mlp.features_inner=128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=2048 \
    {{extras}}

# Step 3: generate conditioning trajectories from a trained CFM checkpoint.
# checkpoint: path to {run_dir}/{epoch}/
# modelname:  label for the output zarr (e.g. model1)
langevin-cfm-trajectories checkpoint modelname extras="":
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    dataset=langevin_potential \
    n_samples=2048 \
    output=data/datasets/langevin_potential/cfm_trajectories/{{modelname}}.zarr \
    {{extras}}

# Step 4: process CFM trajectories + raw physics into regression training data.
langevin-cfm-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/langevin_potential/cfm_trajectories/model1.zarr \
    output=data/datasets/langevin_potential/reg_train_data/model1.zarr \
    dataset=langevin_potential \
    {{reg_process_defaults}} \
    {{extras}}
  {{process_reg}} --multi \
    input=data/datasets/langevin_potential/raw_trajectories/train.zarr \
    output=data/datasets/langevin_potential/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/langevin_potential/cfm_train_data/train.zarr \
    dataset=langevin_potential \
    {{reg_process_defaults}} \
    {{extras}}

# Step 3b: compute OT-coupled trajectories (exact Hungarian assignment, per-step).
# n=2048, 511 time steps — slow but gives correct optimal couplings.
langevin-ot-trajectories extras="":
  just ot-trajectories langevin_potential train "solver=hungarian {{extras}}"
  just ot-trajectories langevin_potential test "solver=hungarian {{extras}}"

# Step 4b: process OT trajectories into regression training data.
# Input is already normalised so norm_stats_path is intentionally omitted.
langevin-ot-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/langevin_potential/ot_trajectories/train.zarr \
    output=data/datasets/langevin_potential/reg_train_data/ot.zarr \
    dataset=langevin_potential \
    {{reg_process_defaults}} \
    {{extras}}

# Step 5: train regression on CFM-generated trajectories (sweep over arch sizes).
langevin-regression extras="":
  {{train_reg}} --multi \
    dataset=langevin_potential \
    {{hist_reg_defaults}} \
    mlp.features_inner=64,128 \
    mlp.default_emb_dim=8,16 \
    train_data=./data/datasets/langevin_potential/reg_train_data/model1.zarr \
    val_data=./data/datasets/langevin_potential/reg_train_data/model1.zarr \
    rollout_data=./data/datasets/langevin_potential/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/langevin_potential/cfm_train_data/train.zarr \
    {{extras}}

# Step 5b: train regression on OT-coupled trajectories.
langevin-ot-regression extras="":
  {{train_reg}} --multi \
    dataset=langevin_potential \
    {{hist_reg_defaults}} \
    mlp.features_inner=128 \
    mlp.default_emb_dim=16 \
    train_data=./data/datasets/langevin_potential/reg_train_data/ot.zarr \
    val_data=./data/datasets/langevin_potential/reg_train_data/ot.zarr \
    rollout_data=./data/datasets/langevin_potential/ot_trajectories/test.zarr \
    {{extras}}
