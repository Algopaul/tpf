import functools
import logging
import os
import time

import humanize
from omegaconf import OmegaConf

import wandb


def log_duration(
    label: str | None = None,
    *,
    minimum_unit: str = "microseconds",
):
    """
    Decorator that logs how long a function takes.

    Args:
      label: Optional name to show in the log; defaults to func.__name__.
      minimum_unit: humanize.naturaldelta's minimum_unit.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                name = label or func.__name__
                elapsed = humanize.naturaldelta(
                    time.time() - start, minimum_unit=minimum_unit
                )
                logging.info("%s complete. Took %s", name, elapsed)

        return wrapper

    return decorator


def init_wandb(cfg, job_type, data_name=None, resume_run_id: str | None = None):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    sid = os.environ.get("SLURM_JOB_ID")
    if sid is None:
        sid = ""
    else:
        sid = "_" + str(sid)
    config_dict["slurm_id"] = sid  # pyright: ignore
    if resume_run_id:
        return wandb.init(
            id=resume_run_id,
            resume="must",
            project="two-parameter-flow",
            config=config_dict,  # pyright: ignore
            mode=cfg.wandb.mode,
        )
    if len(cfg.wandb.tag) > 0:
        tag = f"_{cfg.wandb.tag}"
    else:
        tag = ""
    if data_name is None:
        data_name = getattr(getattr(cfg, "data", None), "name", "")
    name_suffix = f"_{data_name}" if data_name else ""
    jobname = cfg.wandb.jobname or job_type + name_suffix + tag + sid
    group = cfg.wandb.group or data_name or job_type
    return wandb.init(
        name=jobname,
        project="two-parameter-flow",
        group=group,
        job_type=job_type,
        config=config_dict,  # pyright: ignore
        mode=cfg.wandb.mode,
    )
