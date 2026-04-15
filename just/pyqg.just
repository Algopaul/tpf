# ── pyqg ───────────────────────────────────────────────────────────────────────
# Step 0: organize raw h5 files into train/test zarr stores.
# Reads from data/datasets/pyqg_unorganized/trajectories/*.h5
# Writes to  data/datasets/pyqg/raw_trajectories/{train,test}.zarr
pyqg-organize extras="":
  {{py}} ./scripts/datagen/pyqg_organize.py {{extras}}

# Step 1: process raw trajectory zarr into CFM training data.
pyqg-processed extras="":
  {{process}} {{process_defaults}} data.name=pyqg {{extras}}

# Step 2: train CFM model.
pyqg-cfm extras="":
  {{train_cfm}} {{field_cfm_defaults}} data.name=pyqg unet.base_ch=32 data.batch_size=512 data.block_size=32 {{extras}}

# Step 3: generate CFM-conditioned trajectories.
pyqg-cfm-trajectories checkpoint modelname extras="":
  just field-cfm-trajectories pyqg {{checkpoint}} {{modelname}} "{{extras}}"

# Step 3b: process CFM trajectories into regression training data.
pyqg-cfm-trajectories-processed extras="":
  just field-cfm-trajectories-processed pyqg "{{extras}}"

# Step 5: train regression model on CFM trajectories (default).
pyqg-regression extras="":
  just field-regression pyqg \
    "log_energy_spectra=true \
     energy_spectra.channel_axis=2 \
     'stats=[enstrophy,first_moment,kurtosis]' \
     zero_mean_rollout=true \
     {{extras}}"

# Process raw pyqg physics trajectories into regression pairs (no CFM needed).
pyqg-physics-regression-data extras="":
  {{process_reg}} --multi \
    input=data/datasets/pyqg/raw_trajectories/train.zarr \
    output=data/datasets/pyqg/reg_train_data/physics.zarr \
    norm_stats_path=data/datasets/pyqg/cfm_train_data/train.zarr \
    dataset=pyqg \
    {{reg_process_defaults}} \
    {{extras}}

# Train regression model directly on physics trajectories.
pyqg-physics-regression extras="":
  {{train_reg}} --multi \
    model_type=unet \
    +unet=mid \
    unet.base_ch=32 \
    dataset=pyqg \
    train_data=./data/datasets/pyqg/reg_train_data/physics.zarr \
    val_data=./data/datasets/pyqg/reg_train_data/physics.zarr \
    val_n_samples=50000 \
    rollout_data=./data/datasets/pyqg/raw_trajectories/test.zarr \
    norm_stats_path=./data/datasets/pyqg/cfm_train_data/train.zarr \
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

# Resume recipes
resume-pyqg-cfm run_dir extras="":
  {{train_cfm}} {{field_cfm_defaults}} data.name=pyqg unet.base_ch=32 data.batch_size=512 data.block_size=32 restart_from={{run_dir}} {{extras}}

resume-pyqg-regression run_dir extras="":
  just pyqg-regression "restart_from={{run_dir}} {{extras}}"
