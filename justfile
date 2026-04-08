# ── interpreters / app shortcuts ──────────────────────────────────────────────
py           := ".venv/bin/python"
process      := py + " ./tpflow/apps/01_process_trajectories.py"
train_cfm    := py + " ./tpflow/apps/02_train_cfm.py"
gen_traj     := py + " ./tpflow/apps/03_gen_cond_trajectories.py"
gen_ot       := py + " ./tpflow/apps/03b_gen_ot_trajectories.py"
process_reg  := py + " ./tpflow/apps/04_process_regression_data.py"
train_reg    := py + " ./tpflow/apps/05_train_regression.py"
export_eval  := py + " ./tpflow/apps/06_export_eval.py"

# ── shared defaults ────────────────────────────────────────────────────────────
# data.block_size sets the inner zarr chunk size (~2 MB target via auto_block_sizes).
# Shard size is chosen automatically to target ~1 GB per shard file.
process_defaults   := "--multi data.trajectory_block_size=8 data.block_size=32"
field_cfm_defaults := "-cn imgrot --multi inference.n_samples=36"

# block_size=0 uses auto_block_sizes: targets ~2 MB zarr chunks based on state_shape.
# For 2-D particle data this gives ~262K samples/chunk; for large fields it still
# gives ≥32, so training block_size=32 (a divisor of any auto size) always works.
reg_process_defaults := ""

# Shared regression training defaults for 2-D particle (hist) datasets.
# batch_size/block_size tuned for Langevin (n_samples ~1M, state_shape=(2,)).
hist_reg_defaults := "model_type=mlp mlp.features_in=4 mlp.layers=4 \
    batch_size=4096 block_size=2048 \
    mode=difference data_type=hist \
    zero_mean_rollout=false n_rollout=512 \
    opt.epochs=2_000 opt.learning_rate=1e-4"

# ── imports ────────────────────────────────────────────────────────────────────
import 'just/shared.just'
import 'just/gaurot.just'
import 'just/imgrot.just'
import 'just/kolflow.just'
import 'just/hw2d.just'
import 'just/langevin.just'
import 'just/holder.just'
import 'just/vlasov.just'
import 'just/bump.just'

# ── dev ────────────────────────────────────────────────────────────────────────
test:
  .venv/bin/pytest tests/ -v

lint:
  .venv/bin/ruff check tpflow/

fmt:
  .venv/bin/ruff format tpflow/

install:
  pip install -e .

# ── utilities ──────────────────────────────────────────────────────────────────
# Re-export zarr eval data for an existing checkpoint (auto-detects CFM vs regression).
# checkpoint: path to {run_dir}/{epoch}/ checkpoint directory.
# extras: optional Hydra overrides, e.g. "+env=slurm" or "rollout_data=/new/path"
export-eval checkpoint extras="":
  {{export_eval}} --multi checkpoint={{checkpoint}} {{extras}}

# Show pipeline progress and print the next command for a field dataset.
status ds extras="":
  {{py}} tpflow/tools/pipeline_status.py {{ds}} --extras "{{extras}}"

# Dry-run: show what would be deleted to restart from step N (1-5).
# Pass yes=true to actually delete: just restart-from kolflow 3 yes=true
restart-from ds step yes="":
  {{py}} tpflow/tools/pipeline_restart.py {{ds}} --from {{step}} {{if yes == "true" { "--yes" } else { "" } }}

# Safe on login nodes: prints shard/chunk layout without reading any data.
bench-inspect ds extras="":
  {{py}} tpflow/tools/benchmark_dataloader.py dataset={{ds}} inspect=true {{extras}}

# Full dataloader benchmark — submit to a CPU node via +env=torchcpu.
bench ds extras="":
  {{py}} tpflow/tools/benchmark_dataloader.py --multi dataset={{ds}} {{extras}}

# ── resume recipes ─────────────────────────────────────────────────────────────
resume-gaurot-cfm run_dir extras="":
  {{train_cfm}} --multi restart_from={{run_dir}} {{extras}}

resume-kolflow-cfm run_dir extras="":
  {{train_cfm}} {{field_cfm_defaults}} data.name=kolflow unet.base_ch=32 restart_from={{run_dir}} {{extras}}

resume-hw2d-cfm run_dir extras="":
  {{train_cfm}} {{field_cfm_defaults}} data.name=hw2d unet.base_ch=32 unet.channels_inout=2 restart_from={{run_dir}} {{extras}}

resume-kolflow-regression run_dir extras="":
  just kolflow-regression "restart_from={{run_dir}} {{extras}}"

resume-hw2d-regression run_dir extras="":
  just hw2d-regression "restart_from={{run_dir}} {{extras}}"
