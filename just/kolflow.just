# ── kolflow ────────────────────────────────────────────────────────────────────
kolflow-train-data extras:
  {{py}} ./scripts/datagen/kolmogorov_flow.py --multi \
    seed=range\(0,1000\) \
    n_seeds=10000 \
    {{extras}}

kolflow-test-data extras:
  {{py}} ./scripts/datagen/kolmogorov_flow.py --multi \
    seed=range\(0,36\) \
    split=test \
    n_seeds=36 \
    {{extras}}

kolflow-processed extras:
  {{process}} {{process_defaults}} data.name=kolflow {{extras}}

kolflow-cfm extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=kolflow unet.base_ch=32 {{extras}}

# state_shape=(H,W,1) → channel_axis=2; compare enstrophy + first_moment + energy spectra
kolflow-cfm-trajectories checkpoint modelname extras:
  just field-cfm-trajectories kolflow {{checkpoint}} {{modelname}} "{{extras}}"

kolflow-cfm-trajectories-processed extras:
  just field-cfm-trajectories-processed kolflow "{{extras}}"

kolflow-regression extras:
  just field-regression kolflow \
    "log_energy_spectra=true \
     energy_spectra.channel_axis=2 \
     'stats=[enstrophy,first_moment,kurtosis]' \
     zero_mean_rollout=true \
     {{extras}}"

# Process raw kolflow physics trajectories (with normalisation) into regression pairs.
kolflow-physics-regression-data extras="":
  {{process_reg}} --multi \
    input=data/datasets/kolflow/raw_trajectories/train.zarr \
    output=data/datasets/kolflow/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/kolflow/cfm_train_data/train.zarr \
    dataset=kolflow \
    {{reg_process_defaults}} \
    {{extras}}

kolflow-physics-regression extras="":
  {{train_reg}} --multi \
    model_type=unet \
    +unet=mid \
    unet.base_ch=32 \
    dataset=kolflow \
    train_data=./data/datasets/kolflow/reg_train_data/physics.zarr \
    val_data=./data/datasets/kolflow/reg_train_data/physics.zarr \
    val_n_samples=50000 \
    rollout_data=./data/datasets/kolflow/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/kolflow/cfm_train_data/train.zarr \
    batch_size=512 \
    block_size=32 \
    mode=difference \
    data_type=field \
    log_energy_spectra=true \
    energy_spectra.channel_axis=2 \
    'stats=[enstrophy,first_moment,kurtosis]' \
    zero_mean_rollout=true \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    {{extras}}

# OT-coupled trajectory recipes for kolflow.
# n_traj=2048 keeps memory manageable (dataset is O(10M) trajectories).
kolflow-ot-trajectories n_traj="2048" extras="":
  just ot-trajectories kolflow train "n_traj={{n_traj}} {{extras}}"
  just ot-trajectories kolflow test "n_traj={{n_traj}} {{extras}}"

kolflow-ot-trajectories-processed extras="":
  {{process_reg}} --multi \
    input=data/datasets/kolflow/ot_trajectories/train.zarr \
    output=data/datasets/kolflow/reg_train_data/ot.zarr \
    dataset=kolflow \
    {{reg_process_defaults}} \
    {{extras}}

kolflow-ot-regression extras:
  just field-regression kolflow \
    "train_data=./data/datasets/kolflow/reg_train_data/ot.zarr \
     val_data=./data/datasets/kolflow/reg_train_data/ot.zarr \
     rollout_data=./data/datasets/kolflow/ot_trajectories/test.zarr \
     log_energy_spectra=true \
     energy_spectra.channel_axis=2 \
     'stats=[enstrophy,first_moment,kurtosis]' \
     zero_mean_rollout=true \
     {{extras}}"
