"""Re-export zarr eval data for an existing checkpoint.

Loads the training config from ``{run_dir}/.hydra/config.yaml`` alongside the
checkpoint, then re-runs the eval pass and writes zarr files to
``{checkpoint}/eval/`` (or ``cfg.eval_dir`` if set).

Checkpoint type (CFM vs regression) is auto-detected from checkpoint_info.json.

Usage::

    python tpflow/apps/06_export_eval.py --multirun checkpoint=/run_dir/50
    python tpflow/apps/06_export_eval.py --multirun checkpoint=/run_dir/50 +env=slurm
    python tpflow/apps/06_export_eval.py --multirun checkpoint=/run_dir/50 \\
        rollout_data=/new/path/to/test.zarr

Or via justfile::

    just export-eval /run_dir/50
    just export-eval /run_dir/50 "+env=slurm"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from tpflow.config import ExportEvalConfig
from tpflow.util import log_duration


@hydra.main(version_base=None, config_name="export_eval", config_path="../../conf")
@log_duration()
def main(cfg: ExportEvalConfig) -> None:
    checkpoint_dir = Path(to_absolute_path(cfg.checkpoint))
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if not (checkpoint_dir / "checkpoint_info.json").exists():
        from tpflow.model import _find_latest_checkpoint
        checkpoint_dir = _find_latest_checkpoint(checkpoint_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoints found under {cfg.checkpoint}")
        logging.info("Auto-selected latest checkpoint: %s", checkpoint_dir)

    with open(checkpoint_dir / "checkpoint_info.json") as f:
        info = json.load(f)

    hydra_cfg_path = checkpoint_dir.parent / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(
            f"No Hydra config found at {hydra_cfg_path}. "
            "The checkpoint must live inside a Hydra output directory."
        )

    train_cfg = OmegaConf.load(hydra_cfg_path)
    if cfg.rollout_data:
        train_cfg = OmegaConf.merge(train_cfg, {"rollout_data": cfg.rollout_data})

    eval_dir = Path(to_absolute_path(cfg.eval_dir)) if cfg.eval_dir else checkpoint_dir / "eval"

    is_regression = "time_conditioned" in info
    if is_regression:
        _export_regression(checkpoint_dir, train_cfg, eval_dir, info)
    else:
        _export_cfm(checkpoint_dir, train_cfg, eval_dir, info)


def _export_cfm(checkpoint_dir: Path, train_cfg, eval_dir: Path, info: dict) -> None:
    from tpflow.eval import export_cfm_eval, run_cfm_eval
    from tpflow.model import load_model

    logging.info("Loading CFM model from %s", checkpoint_dir)
    model = load_model(checkpoint_dir)
    sample_shape = tuple(info["sample_shape"])

    logging.info("Running CFM eval (sample_shape=%s)", sample_shape)
    result = run_cfm_eval(model, train_cfg, sample_shape)

    logging.info("Writing zarr to %s", eval_dir)
    export_cfm_eval(result, eval_dir)


def _export_regression(
    checkpoint_dir: Path,
    train_cfg,
    eval_dir: Path,
    info: dict,
) -> None:
    from tpflow.eval import export_regression_eval, run_regression_eval
    from tpflow.model import load_regression_model

    # Prefer diff_scale stored in checkpoint info; fall back to loading training data.
    if "diff_scale" in info:
        diff_scale = np.array(info["diff_scale"], dtype=np.float32)
        logging.info("Loaded diff_scale from checkpoint info (shape=%s)", diff_scale.shape)
    else:
        from tpflow.data import RegressionZarrData
        logging.info("Loading diff_scale from training data at %s", train_cfg.train_data)
        train_data = RegressionZarrData(train_cfg.train_data, train_cfg.batch_size, train_cfg.block_size)
        diff_scale = train_data.diff_scale

    logging.info("Loading regression model from %s", checkpoint_dir)
    model = load_regression_model(checkpoint_dir)

    logging.info("Running regression eval")
    result = run_regression_eval(model, train_cfg, diff_scale)

    logging.info("Writing zarr to %s", eval_dir)
    export_regression_eval(result, eval_dir)


if __name__ == "__main__":
    main()
