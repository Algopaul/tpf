# ── shared OT-coupling recipe ──────────────────────────────────────────────────
# Compute OT-coupled trajectories for any dataset.
# ds:    dataset name (e.g. langevin_potential, kolflow)
# split: train or test
# extras: optional Hydra overrides (e.g. epsilon=0.01 n_traj=1000)
ot-trajectories ds split="train" extras="":
  {{gen_ot}} --multi \
    input=data/datasets/{{ds}}/raw_trajectories/{{split}}.zarr \
    output=data/datasets/{{ds}}/ot_trajectories/{{split}}.zarr \
    norm_stats_path=data/datasets/{{ds}}/cfm_train_data/train.zarr \
    dataset={{ds}} \
    {{extras}}

# ── shared field-dataset recipes ───────────────────────────────────────────────
# Parameterised by dataset name (ds); reused by kolflow and hw2d below.

field-cfm-trajectories ds checkpoint modelname env:
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    dataset={{ds}} \
    n_samples=1000 \
    output=data/datasets/{{ds}}/cfm_trajectories/{{modelname}}.zarr \
    {{env}}

field-cfm-trajectories-processed ds env:
  {{process_reg}} --multi \
    input=data/datasets/{{ds}}/cfm_trajectories/model1.zarr \
    output=data/datasets/{{ds}}/reg_train_data/model1.zarr \
    dataset={{ds}} \
    {{reg_process_defaults}} \
    {{env}}
  {{process_reg}} --multi \
    input=data/datasets/{{ds}}/raw_trajectories/train.zarr \
    output=data/datasets/{{ds}}/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/{{ds}}/cfm_train_data/train.zarr \
    dataset={{ds}} \
    {{reg_process_defaults}} \
    {{env}}

field-regression ds env:
  {{train_reg}} --multi \
    model_type=unet \
    +unet=mid \
    unet.base_ch=32 \
    dataset={{ds}} \
    train_data=./data/datasets/{{ds}}/reg_train_data/model1.zarr \
    val_data=./data/datasets/{{ds}}/reg_train_data/model1.zarr \
    rollout_data=./data/datasets/{{ds}}/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/{{ds}}/cfm_train_data/train.zarr \
    batch_size=512 \
    block_size=32 \
    mode=difference \
    data_type=field \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    {{env}}
