py := ".venv/bin/python"
process := py + " ./tpflow/apps/01_process_trajectories.py"
train := py + " ./tpflow/apps/02_train_cfm.py"
process_defaults := "--multi data.trajectory_block_size=8 data.block_size=32"
field_cfm_defaults := "-cn imgrot --multi inference.n_samples=36"

gaurot-data:
  {{py}} ./scripts/datagen/rotating_gaussians.py
  {{process}} data.block_size=10_000

gaurot-cfm-quick extras:
  {{train}} --multi \
    mlp.features_in=4 \
    mlp.features_inner=64 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \
    {{extras}}

gaurot-cfm:
  {{train}} --multi +env=torch \
    mlp.features_in=4 \
    mlp.features_inner=64,128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8,16 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000

imgrot-raw extras:
  {{py}} ./scripts/datagen/rotation_image.py --multi sharpness=1.0 grid_dim=64,128 speed_schedule=const,acc {{extras}}

imgrot-processed extras:
  {{process}} {{process_defaults}} data.name=imgrot-64-acc,imgrot-128-acc,imgrot-64-const,imgrot-128-const {{extras}}

imgrot-cfm-small extras:
  {{train}} {{field_cfm_defaults}} data.name=imgrot-64-acc,imgrot-64-const unet.base_ch=32,64 {{extras}}

imgrot-cfm-reg extras:
  {{train}} {{field_cfm_defaults}} data.name=imgrot-64-acc unet.base_ch=32 \
    conditioning_reg=1e-6,1e-4,1e-2,1.0 opt.clip_grad_norm=1.0 {{extras}}

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
  {{train}} {{field_cfm_defaults}} data.name=kolflow unet.base_ch=32 {{extras}}

kolflow-cfm-trajectories checkpoint modelname env:
  {{py}} ./tpflow/apps/03_gen_cond_trajectories.py --multi checkpoint={{checkpoint}} n_samples=1000 n_cond_steps=128 output=data/datasets/kolflow/cfm_trajectories/{{modelname}}.zarr {{env}}

kolflow-cfm-trajectories-processed env:
  {{py}} ./tpflow/apps/04_process_regression_data.py --multi input=data/datasets/kolflow/cfm_trajectories/model1.zarr output=data/datasets/kolflow/reg_train_data/model1.zarr {{env}}
  {{py}} ./tpflow/apps/04_process_regression_data.py --multi input=data/datasets/kolflow/raw_trajectories/train.zarr output=data/datasets/kolflow/reg_train_data/physics.zarr {{env}}

kolflow-regression env:
  {{py}} ./tpflow/apps/05_train_regression.py --multi \
  input=data/datasets/kolflow/cfm_trajectories/model1.zarr \
  output=data/datasets/kolflow/reg_train_data/model1.zarr \
  model_type=unet \
  unet.base_ch=32,64 \
  train_data=./data/datasets/kolflow/reg_train_data/model1.zarr \
  val_data=./data/datasets/kolflow/reg_train_data/model1.zarr \
  rollout_data=./data/datasets/kolflow/cfm_trajectories/model1.zarr \
  batch_size=512 \
  block_size=32 \
  mode=difference \
  data_type=field \
  {{env}}

hw2d-data extras:
  {{process}} {{process_defaults}} data.name=hw2d {{extras}}

hw2d-cfm extras:
  {{train}} {{field_cfm_defaults}} data.name=hw2d unet.base_ch=32,64 {{extras}}

hw2d-cfm-trajectories checkpoint modelname env:
  {{py}} ./tpflow/apps/03_gen_cond_trajectories.py --multi checkpoint={{checkpoint}} n_samples=1000 n_cond_steps=128 output=data/datasets/hw2d/cfm_trajectories/{{modelname}}.zarr {{env}}

hw2d-cfm-trajectories-processed env:
  {{py}} ./tpflow/apps/04_process_regression_data.py --multi input=data/datasets/hw2d/cfm_trajectories/model1.zarr output=data/datasets/hw2d/reg_train_data/model1.zarr {{env}}
  {{py}} ./tpflow/apps/04_process_regression_data.py --multi input=data/datasets/hw2d/raw_trajectories/train.zarr output=data/datasets/hw2d/reg_train_data/physics.zarr {{env}}

hw2d-regression env:
  {{py}} ./tpflow/apps/05_train_regression.py --multi \
  input=data/datasets/hw2d/cfm_trajectories/model1.zarr \
  output=data/datasets/hw2d/reg_train_data/model1.zarr \
  model_type=unet \
  unet.base_ch=32,64 \
  train_data=./data/datasets/hw2d/reg_train_data/model1.zarr \
  val_data=./data/datasets/hw2d/reg_train_data/model1.zarr \
  rollout_data=./data/datasets/hw2d/cfm_trajectories/model1.zarr \
  batch_size=512 \
  block_size=32 \
  mode=difference \
  data_type=field \
  {{env}}
