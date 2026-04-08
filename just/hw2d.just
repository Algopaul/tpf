# ── hw2d ───────────────────────────────────────────────────────────────────────
# Step 0: generate raw trajectories (density + phi) on the cluster.
# Each seed is one SLURM job via Hydra submitit.  Use +env=torchcpu for CPU nodes.
hw2d-gen-train extras:
  {{py}} ./scripts/datagen/hw2d.py --multi seed=range\(0,1000\) split=train {{extras}}

hw2d-gen-test extras:
  {{py}} ./scripts/datagen/hw2d.py --multi seed=range\(0,100\) split=test n_seeds=100 {{extras}}

# Step 1: process the raw trajectory zarr into CFM training data.
hw2d-data extras:
  {{process}} {{process_defaults}} data.name=hw2d normalize_per_time=true {{extras}}

hw2d-cfm extras:
  {{train_cfm}} {{field_cfm_defaults}} data.name=hw2d unet.base_ch=32,64 unet.channels_inout=2 {{extras}}

hw2d-regression extras:
  just field-regression hw2d \
    "unet.channels_inout=2 \
     val_n_samples=5000 \
     'stats=[gamma_n,gamma_c,energy,hw2d_enstrophy]' \
     {{extras}}"
