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


def init_wandb(cfg, job_type):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    sid = os.environ.get("SLURM_JOB_ID")
    if sid is None:
        sid = ""
    else:
        sid = "_" + str(sid)
    config_dict["slurm_id"] = sid  # pyright: ignore
    if len(cfg.wandb.tag) > 0:
        tag = f"_{cfg.wandb.tag}"
    else:
        tag = ""
    jobname = cfg.wandb.jobname or job_type + "_" + cfg.data.name + tag + sid
    group = cfg.wandb.group or cfg.data.name
    return wandb.init(
        name=jobname,
        project="two-parameter-flow",
        group=group,
        job_type=job_type,
        config=config_dict,  # pyright: ignore
        mode=cfg.wandb.mode,
    )
