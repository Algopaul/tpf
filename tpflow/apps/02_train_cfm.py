import logging

import hydra
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hdfv.histogram_videos import histogram_frames
from hdfv.images import frame_rgb, grid_shape
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from tpflow.config import CFMTraining
from tpflow.data import ZarrData, get_data
from tpflow.model import flow_inference, get_model
from tpflow.util import init_wandb, log_duration, trajectory_video_numpy


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
    jax.block_until_ready(model)
    logging.info('Model loaded')
    data = ZarrData(cfg.data, 'train_shuffled')
    val_data = get_data(cfg.data, 'test')
    jax.block_until_ready(data)
    logging.info('Data prepared')
    opt = get_optimizer(model, cfg.opt, len(data))
    jax.block_until_ready(opt)
    logging.info('Optimizer initialized')
    train_err = nnx.metrics.Average()
    val_err = nnx.metrics.Average()
    r = Recorder()
    ts, graphdef, state, loss_fn = get_train_step(
        model,
        opt,
        train_err,
        velo_err_pure,
        batch_prep=batch_prep,
    )
    for epoch in range(cfg.opt.epochs):
      model.train()
      keys = jrd.split(rngs.param(), len(data))
      pbar = tqdm(enumerate(data.iter_batches(epoch)), total=len(data))
      for i, batch in pbar:
        b = (jax.device_put(batch), keys[i])
        loss_val, state = ts(state, b)
        met = r({'loss_val': loss_val})
        pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

      model, opt, avg_metric = nnx.merge(graphdef, state)
      logging.info('Epoch %i: Avg. loss %.4e', epoch + 1, avg_metric.compute())
      run.log({"train/avg_loss": avg_metric.compute()}, step=epoch + 1)
      avg_metric.reset()

      model.eval()
      pbar = tqdm(enumerate(val_data), total=len(val_data))
      keys = jrd.split(rngs.param(), len(val_data))
      for i, batch in pbar:
        b = (batch['data'], batch['time'], keys[i])
        loss_val = loss_fn(state, b)
        met = r({'loss_val': loss_val})
        val_err.update(values=loss_val)
        pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

      logging.info('Epoch %i: Val. loss %.4e', epoch + 1, val_err.compute())
      run.log({"val/avg_loss": val_err.compute()}, step=epoch + 1)
      val_err.reset()

      if (epoch + 1) % cfg.eval_interval == 0:
        sample_shape = batch['data'].shape[1:]
        out = flow_inference(
            model,
            jrd.normal(jrd.key(0), (cfg.inference.n_samples, *sample_shape)),
            jnp.linspace(0, 1, cfg.inference.n_param_steps),
            n_steps=cfg.inference.n_param_steps,
        )
        if cfg.data.type == 'hist':
          frames = histogram_frames(out)
          video = np.transpose(frames, (0, 3, 1, 2))
          video = wandb.Video(video, fps=30, format='mp4')
          run.log({"train/cfm_trajectories": video}, step=epoch + 1)
          frames = trajectory_video_numpy(out[:, :200, :])
          video = np.transpose(frames, (0, 3, 1, 2))
          video = wandb.Video(video, fps=20, format='mp4')
          run.log({"train/traces": video}, step=epoch + 1)
        elif cfg.data.type == 'field':
          nrows, ncols = grid_shape(cfg.inference.n_samples)
          frames = [
              frame_rgb(o, grid=True, nrows=nrows, ncols=ncols) for o in out
          ]
          video = np.transpose(frames, (0, 3, 1, 2))
          video = wandb.Video(video, fps=30, format='mp4')
          run.log({"train/cfm_trajectories": video}, step=epoch + 1)


if __name__ == "__main__":
  main()
