import logging
from os.path import join

import h5py
import hydra
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import zarr
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hdfv.histogram_videos import histogram_frames
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from tpflow.config import CFMTraining
from tpflow.data import get_data
from tpflow.model import flow_inference, get_model
from tpflow.util import init_wandb, log_duration


def velo_err_pure(model, batch):
  x_target, conditioning, key = batch
  k0, k1 = jrd.split(key, 2)
  t = jrd.uniform(k0, x_target.shape[0])
  x_source = jrd.normal(k1, x_target.shape)
  tt = t[:, *((None,) * len(x_source.shape[1:]))]
  x = (1 - tt) * x_source + tt * x_target

  if conditioning.ndim == 1:
    conditioning = conditioning[:, None]
  if t.ndim == 1:
    t = t[:, None]

  pred = model(
      x.astype(jnp.float32),
      t.astype(jnp.float32),
      conditioning.astype(jnp.float32),
  ).astype(jnp.float32)
  given = (x_target - x_source).astype(jnp.float32)
  return jnp.mean((pred - given)**2)


def batch_prep(batch):
  batch, key = batch
  return batch['data'], batch['time'], key


@hydra.main(version_base=None, config_name='cfm', config_path='../../conf')
@log_duration()
def main(cfg: CFMTraining) -> None:
  with init_wandb(cfg, 'cfm-train') as run:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))
    rngs = nnx.Rngs(0)
    model = get_model(cfg, rngs=rngs)
    data = get_data(cfg.data)
    opt = get_optimizer(model, cfg.opt, len(data))
    metrics = nnx.metrics.Average()
    r = Recorder()
    ts, graphdef, state = get_train_step(
        model,
        opt,
        metrics,
        velo_err_pure,
        batch_prep=batch_prep,
    )
    for epoch in range(cfg.opt.epochs):
      model.train()
      keys = jrd.split(rngs.param(), len(data))
      pbar = tqdm(enumerate(data), total=len(data))
      for i, batch in pbar:
        b = (jax.device_put(batch), keys[i])
        loss_val, state = ts(state, b)
        met = r({'loss_val': loss_val})
        pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

      model, opt, avg_metric = nnx.merge(graphdef, state)
      logging.info('Epoch %i: Avg. loss %.4e', epoch + 1, avg_metric.compute())
      run.log({"train/avg_loss": avg_metric.compute()}, step=epoch + 1)
      avg_metric.reset()

      if (epoch + 1) % cfg.eval_interval == 0:
        model.eval()
        out = flow_inference(
            model,
            jrd.normal(jrd.key(0), (20_000, 2)),
            jnp.linspace(0, 1, 32),
        )
        if cfg.data.type == 'hist':
          frames = histogram_frames(out)
          video = np.transpose(frames, (0, 3, 1, 2))
          video = wandb.Video(video, fps=30, format='mp4')
          run.log({"train/histogram_video": video}, step=epoch + 1)


if __name__ == "__main__":
  main()
