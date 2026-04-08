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
