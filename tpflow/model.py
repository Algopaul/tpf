import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flanch.model import EmbMLP, UNet
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import CFMTraining


def flow_inference(model, source_batch, cslist, n_steps=128):

  @jax.jit
  def run(source_batch, cond):
    return model.rk4_steps(source_batch, cond, 0.0, 1.0, n_steps)

  outs = np.zeros((len(cslist), *source_batch.shape))
  t_base = jnp.ones((source_batch.shape[0], 1))

  pbar = tqdm(enumerate(cslist), total=len(cslist), desc='Generating samples')
  for i, cs in pbar:
    cond = cs * t_base
    out = run(source_batch, cond)
    jax.block_until_ready(out)
    outs[i] = out

  return outs


class CFMDec(nnx.Module):

  def __init__(self, model):
    self.model = model

  def __call__(self, x, tf, tp):
    x = jnp.concatenate((x, tf, tp), axis=-1)
    return self.model(x)

  def euler_steps(self, x0, conditioning, t0, t1, n_steps):
    ts = jnp.linspace(t0, t1, n_steps, endpoint=False)
    dt = ts[1] - ts[0]

    def body_fn(i, x):
      return x + dt[:, None, None, None] * self(x, ts[i], conditioning)

    return jax.lax.fori_loop(0, n_steps, body_fn, x0)

  def rk4_steps(self, x0, conditioning, t0, t1, n_steps=32):
    ts = jnp.linspace(t0, t1, n_steps + 1, endpoint=False)
    dt = ts[1] - ts[0]
    dtf = dt
    ones = jnp.ones((x0.shape[0], 1))

    def body(i, x):
      t = ts[i] * ones
      t2 = (ts[i] + 0.5 * dtf) * ones
      t1n = ts[i + 1] * ones

      k1 = self(x, t, conditioning)
      k2 = self(x + 0.5 * dtf * k1, t2, conditioning)
      k3 = self(x + 0.5 * dtf * k2, t2, conditioning)
      k4 = self(x + dtf * k3, t1n, conditioning)
      return x + (dtf / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.lax.fori_loop(0, n_steps, body, x0)


def get_model(cfg: CFMTraining, rngs=None):
  if rngs is None:
    rngs = nnx.Rngs(0)
  match cfg.model_type:
    case 'mlp':
      return CFMDec(EmbMLP.from_config(cfg.mlp, rngs=rngs))
    case 'unet':
      return UNet.from_config(cfg.unet, rngs=rngs)
    case _:
      raise ValueError('model_type not supported')


def store_model(model, cfg, epoch):
  run_dir = HydraConfig.get().runtime.output_dir
  logging.info('Storing model at %s', run_dir)
  f = Path(run_dir) / f'{epoch}' / 'final'
  f = f.absolute()
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(model)
  checkpointer.save(f.parent / 'state', state)
  checkpointer.wait_until_finished()
  with open(f.parent / 'config.yaml', 'w') as f:
    match cfg.model_type:
      case 'mlp':
        f.writelines(OmegaConf.to_yaml(cfg.mlp))
      case 'unet':
        f.writelines(OmegaConf.to_yaml(cfg.unet))
      case _:
        raise ValueError('model_type not supported')
