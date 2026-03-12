# ── interpreters / app shortcuts ──────────────────────────────────────────────
py           := ".venv/bin/python"
process      := py + " ./tpflow/apps/01_process_trajectories.py"
train_cfm    := py + " ./tpflow/apps/02_train_cfm.py"
gen_traj     := py + " ./tpflow/apps/03_gen_cond_trajectories.py"
process_reg  := py + " ./tpflow/apps/04_process_regression_data.py"
train_reg    := py + " ./tpflow/apps/05_train_regression.py"

# ── shared defaults ────────────────────────────────────────────────────────────
# data.block_size sets the inner zarr chunk size (~2 MB target via auto_block_sizes).
# Shard size is chosen automatically to target ~1 GB per shard file.
process_defaults   := "--multi data.trajectory_block_size=8 data.block_size=32"
field_cfm_defaults := "-cn imgrot --multi inference.n_samples=36"

# block_size=32 matches the training block_size in field-regression so that
# shard_size (auto × 32) remains divisible by the training block_size.
reg_process_defaults := "block_size=32"

# ── pipeline status ────────────────────────────────────────────────────────────
# Show progress and print the next command for a field dataset.
# extras: optional args appended to the suggested command (e.g. "+env=slurm")
status ds extras="":
  {{py}} tpflow/tools/pipeline_status.py {{ds}} --extras "{{extras}}"

# Safe on login nodes: prints shard/chunk layout without reading any data.
bench-inspect ds extras="":
  {{py}} tpflow/tools/benchmark_dataloader.py dataset={{ds}} inspect=true {{extras}}

# Full dataloader benchmark — submit to a CPU node via +env=torchcpu.
bench ds extras="":
  {{py}} tpflow/tools/benchmark_dataloader.py --multi dataset={{ds}} {{extras}}

# ── dev ────────────────────────────────────────────────────────────────────────
test:
  .venv/bin/pytest tests/ -v

lint:
  .venv/bin/ruff check tpflow/

fmt:
  .venv/bin/ruff format tpflow/

install:
  pip install -e .

# ── gaurot ─────────────────────────────────────────────────────────────────────
gaurot-data:
  {{py}} ./scripts/datagen/rotating_gaussians.py
  {{process}} data.block_size=10_000

gaurot-cfm-quick extras:
  {{train_cfm}} --multi \
    mlp.features_in=4 \
    mlp.features_inner=64 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \
    {{extras}}

gaurot-cfm extras:
  {{train_cfm}} --multi \
    mlp.features_in=4 \
    mlp.features_inner=64,128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8,16 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \
    {{extras}}

# ── imgrot ─────────────────────────────────────────────────────────────────────
imgrot-raw extras:
  {{py}} ./scripts/datagen/rotation_image.py --multi sharpness=1.0 grid_dim=64,128 speed_schedule=const,acc {{extras}}

imgrot-processed extras:
  {{process}} {{process_defaults}} data.name=imgrot-64-acc,imgrot-128-acc,imgrot-64-const,imgrot-128-const {{extras}}

imgrot-cfm-small extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=imgrot-64-acc,imgrot-64-const unet.base_ch=32,64 {{extras}}

imgrot-cfm-reg extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=imgrot-64-acc unet.base_ch=32 \
    conditioning_reg=1e-6,1e-4,1e-2,1.0 opt.clip_grad_norm=1.0 {{extras}}

# ── shared field-dataset recipes ───────────────────────────────────────────────
# Parameterised by dataset name (ds); reused by kolflow and hw2d below.

field-cfm-trajectories ds checkpoint modelname env:
  {{gen_traj}} --multi \
    checkpoint={{checkpoint}} \
    n_samples=1000 \
    n_cond_steps=128 \
    output=data/datasets/{{ds}}/cfm_trajectories/{{modelname}}.zarr \
    {{env}}

field-cfm-trajectories-processed ds env:
  {{process_reg}} --multi \
    input=data/datasets/{{ds}}/cfm_trajectories/model1.zarr \
    output=data/datasets/{{ds}}/reg_train_data/model1.zarr \
    {{reg_process_defaults}} \
    {{env}}
  {{process_reg}} --multi \
    input=data/datasets/{{ds}}/raw_trajectories/train.zarr \
    output=data/datasets/{{ds}}/reg_train_data/physics.zarr \
    {{reg_process_defaults}} \
    {{env}}

field-regression ds env:
  {{train_reg}} --multi \
    model_type=unet \
    +unet=mid \
    unet.base_ch=32 \
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
     {{extras}}"

hw2d-regression extras:
  just field-regression hw2d \
    "data_type=hw2d \
     unet.channels_inout=2 \
     'stats=[gamma_n,gamma_c,energy,hw2d_enstrophy]' \
     {{extras}}"

# ── hw2d ───────────────────────────────────────────────────────────────────────
# Step 0: generate raw trajectories (density + phi) on the cluster.
# Each seed is one SLURM job via Hydra submitit.  Use +env=torchcpu for CPU nodes.
hw2d-gen-train extras:
  {{py}} ./scripts/datagen/hw2d.py --multi seed=range\(0,1000\) split=train {{extras}}

hw2d-gen-test extras:
  {{py}} ./scripts/datagen/hw2d.py --multi seed=range\(0,100\) split=test n_seeds=100 {{extras}}

# Step 1: process the raw trajectory zarr into CFM training data.
hw2d-data extras:
  {{process}} {{process_defaults}} data.name=hw2d {{extras}}

hw2d-cfm extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=hw2d unet.base_ch=32,64 unet.channels_inout=2 {{extras}}
