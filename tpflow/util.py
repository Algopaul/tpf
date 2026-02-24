import functools
import logging
import os
import time

import humanize
import matplotlib
import numpy as np
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
            time.time() - start, minimum_unit=minimum_unit)
        logging.info("%s complete. Took %s", name, elapsed)

    return wrapper

  return decorator


def init_wandb(cfg, job_type):
  config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
  sid = os.environ.get("SLURM_JOB_ID")
  if sid is None:
    sid = ''
  else:
    sid = '_' + str(sid)
  config_dict["slurm_id"] = sid  # pyright: ignore
  if len(cfg.wandb.tag) > 0:
    tag = f'_{cfg.wandb.tag}'
  else:
    tag = ''
  jobname = cfg.wandb.jobname or job_type + '_' + cfg.data.name + tag + sid
  group = cfg.wandb.group or cfg.data.name
  return wandb.init(
      name=jobname,
      project='two-parameter-flow',
      group=group,
      job_type=job_type,
      config=config_dict,  # pyright: ignore
      mode=cfg.wandb.mode,
  )


def to_pixel(x, y, xlim, ylim, resolution):
  xmin, xmax = xlim
  ymin, ymax = ylim
  px = ((x - xmin) / (xmax - xmin) * (resolution - 1)).astype(int)
  py = ((y - ymin) / (ymax - ymin) * (resolution - 1)).astype(int)
  return px, py


def trace_video(
    data,
    *,
    resolution=512,
    xlim=(-1, 1),
    ylim=(-1, 1),
    trail_decay=0.92,  # closer to 1 = longer trails
    dot_intensity=1.0,  # brightness of current position
):
  """
    data: (n_time, n_particles, 2)
    """

  n_time, _, _ = data.shape

  # Frame buffer (float for accumulation)
  frame = np.zeros((resolution, resolution, 3), dtype=np.float32)

  out_data = []

  for t in range(n_time):

    # Fade previous frame (trail effect)
    frame *= trail_decay

    x = data[t, :, 0]
    y = data[t, :, 1]

    px, py = to_pixel(x, y, xlim, ylim, resolution)

    # Clip valid pixels
    mask = ((px >= 0) & (px < resolution) & (py >= 0) & (py < resolution))
    px = px[mask]
    py = py[mask]

    # Draw bright current points (cyan-ish)
    frame[py, px, 0] += 0.1 * dot_intensity
    frame[py, px, 1] += 0.8 * dot_intensity
    frame[py, px, 2] += 1.0 * dot_intensity

    # Clip for display
    img = np.clip(frame, 0, 1)
    rgb = (255 * img).astype(np.uint8)
    out_data.append(rgb)

  return out_data


def angle_color_coded(
    data,
    source_data,
    *,
    resolution=512,
    xlim=(-1, 1),
    ylim=(-1, 1),
):
  n_time, _, _ = data.shape
  cmap = matplotlib.colormaps['hsv']  # cyclic colormap
  angles = np.arctan2(source_data[:, 1], source_data[:, 0])
  angles = (np.pi + angles) / (2 * np.pi)
  colors = cmap(angles)[:, :3]
  out_data = []

  for t in range(n_time):
    frame = np.zeros((resolution, resolution, 3), dtype=np.float32)
    x = data[t, :, 0]
    y = data[t, :, 1]
    px, py = to_pixel(x, y, xlim, ylim, resolution)
    mask = ((px >= 0) & (px < resolution) & (py >= 0) & (py < resolution))
    px = px[mask]
    py = py[mask]
    frame[py, px] = colors[mask]
    img = np.clip(frame, 0, 1)
    rgb = (255 * img).astype(np.uint8)
    out_data.append(rgb)

  return out_data
