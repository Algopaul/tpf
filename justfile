py := ".venv/bin/python"

gaurot-data:
  {{py}} ./scripts/datagen/rotating_gaussians.py
  {{py}} ./tpflow/apps/01_process_trajectories.py block_size=10_000

gaurot-cfm:
  {{py}} ./tpflow/apps/02_train_cfm.py --multi +env=torch \
    mlp.features_in=4 \
    mlp.features_inner=32,64,128,256 \
    mlp.layers=4,6 \
    mlp.default_emb_dim=8,16 \
    opt.epochs=2_000 \
    opt.learning_rate=1e-4

