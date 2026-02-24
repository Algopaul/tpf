py := ".venv/bin/python"

gaurot-data:
  {{py}} ./scripts/datagen/rotating_gaussians.py
  {{py}} ./tpflow/apps/01_process_trajectories.py data.block_size=10_000

gaurot-cfm-quick extras:
  {{py}} ./tpflow/apps/02_train_cfm.py --multi {{extras}} \
    mlp.features_in=4 \
    mlp.features_inner=64 \
    mlp.layers=4 \
    mlp.default_emb_dim=8 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \

gaurot-cfm:
  {{py}} ./tpflow/apps/02_train_cfm.py --multi +env=torch \
    mlp.features_in=4 \
    mlp.features_inner=64,128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8,16 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \

imgrot-raw extras:
  {{py}} ./scripts/datagen/rotation_image.py --multi sharpness=1.0 grid_dim=64,128 speed_schedule=const,acc {{extras}}

imgrot-processed extras:
  {{py}} ./tpflow/apps/01_process_trajectories.py --multi block_size=8 data.name=imgrot-64-acc,imgrot-128-acc,imgrot-64-const,imgrot-128-const {{extras}}


imgrot-cfm-small extras:
  {{py}} ./tpflow/apps/02_train_cfm.py -cn imgrot --multi data.name=imgrot-64-acc,imgrot-64-const unet.base_ch=32,64 inference.n_samples=36 {{extras}}
