py := ".venv/bin/python"

gaurot-data:
  {{py}} ./scripts/datagen/rotating_gaussians.py
  {{py}} ./tpflow/apps/01_process_trajectories.py block_size=10_000

gaurot-cfm:
  {{py}} ./tpflow/apps/02_train_cfm.py --multi +env=torch \
    mlp.features_in=4 \
    mlp.features_inner=64,128 \
    mlp.layers=4 \
    mlp.default_emb_dim=8,16 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4 \
    inference.n_samples=20_000 \

imgrot-data:
  {{py}} ./scripts/datagen/rotation_image.py
  {{py}} ./tpflow/apps/01_process_trajectories.py block_size=8 data.name=imgrot


imgrot-cfm:
  {{py}} ./tpflow/apps/02_train_cfm.py -cn imgrot
