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

from tpflow.config import CFMTraining, RegressionTraining


def make_flow_fn(model, n_steps=128):
    """Return a JIT-compiled inference function. Call once and reuse across batches.

    Args:
        model: CFM model with a ``rk4_steps`` method.
        n_steps: Number of ODE integration steps.

    Returns:
        ``run(source_batch, cslist) -> array of shape (len(cslist), *batch_shape)``
    """

    @jax.jit
    def run(source_batch, cslist):
        t_base = jnp.ones((source_batch.shape[0], 1))

        def run_one(cs):
            return model.rk4_steps(source_batch, cs * t_base, 0.0, 1.0, n_steps)

        return jax.lax.map(run_one, cslist)

    return run


def flow_inference(model, source_batch, cslist, n_steps=128):

    @jax.jit
    def run(source_batch, cond):
        return model.rk4_steps(source_batch, cond, 0.0, 1.0, n_steps)

    outs = np.zeros((len(cslist), *source_batch.shape))
    t_base = jnp.ones((source_batch.shape[0], 1))

    pbar = tqdm(enumerate(cslist), total=len(cslist), desc="Generating samples")
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


class RegressionDec(nnx.Module):
    """One-step prediction model wrapping an MLP.

    With time_conditioned=True  input is concat(x, time, param).
    With time_conditioned=False input is concat(x, param) — time is dropped
    entirely, giving a genuinely smaller architecture.
    """

    def __init__(self, model, time_conditioned: bool):
        self.model = model
        self.time_conditioned = time_conditioned

    def __call__(self, x, time, param):
        if self.time_conditioned:
            inp = jnp.concatenate((x, time, param), axis=-1)
        else:
            inp = jnp.concatenate((x, param), axis=-1)
        return self.model(inp)


class RegressionUNetDec(nnx.Module):
    """One-step prediction model wrapping a UNet.

    With time_conditioned=True  time is passed through unchanged.
    With time_conditioned=False time is zeroed out before the UNet call.
    """

    def __init__(self, model, time_conditioned: bool):
        self.model = model
        self.time_conditioned = time_conditioned

    def __call__(self, x, time, param):
        t = time if self.time_conditioned else jnp.zeros_like(time)
        return self.model(x, t, param)


def get_regression_model(cfg: RegressionTraining, rngs=None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    match cfg.model_type:
        case "mlp":
            return RegressionDec(
                EmbMLP.from_config(cfg.mlp, rngs=rngs), cfg.time_conditioned
            )
        case "unet":
            return RegressionUNetDec(
                UNet.from_config(cfg.unet, rngs=rngs), cfg.time_conditioned
            )
        case _:
            raise ValueError(f"model_type {cfg.model_type!r} not supported")


def regression_rollout(model, x0, time_vector, param, mode: str):
    """Roll out a one-step regression model from initial conditions.

    Args:
      model:       RegressionDec (or compatible)
      x0:          (n_rollout, *state_shape) — initial states
      time_vector: (n_time,) — conditioning-time values
      param:       (n_rollout, 1) — per-trajectory params
      mode:        'step' or 'difference'

    Returns:
      numpy array of shape (n_time, n_rollout, *state_shape)
    """

    @jax.jit
    def step(x, t_val, dt):
        t = jnp.full((x.shape[0], 1), t_val, dtype=jnp.float32)
        pred = model(x, t, param.astype(jnp.float32)).astype(jnp.float32)
        if mode == "difference":
            return x + dt * pred
        return pred

    n_time = len(time_vector)
    outs = np.zeros((n_time, *x0.shape))
    x = x0.astype(jnp.float32)
    outs[0] = np.array(x)
    for t_idx in tqdm(range(n_time - 1), desc="Rollout"):
        dt = float(time_vector[t_idx + 1] - time_vector[t_idx])
        x = step(x, float(time_vector[t_idx]), dt)
        jax.block_until_ready(x)
        outs[t_idx + 1] = np.array(x)
    return outs


def store_regression_model(
    model, cfg: RegressionTraining, epoch: int, sample_shape: tuple
):
    import json

    run_dir = HydraConfig.get().runtime.output_dir
    logging.info("Storing regression model at %s", run_dir)
    checkpoint_dir = Path(run_dir) / f"{epoch}"
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    checkpointer.save(checkpoint_dir / "state", state)
    checkpointer.wait_until_finished()
    model_cfg = getattr(cfg, cfg.model_type, None)
    if model_cfg is None:
        raise ValueError(f'model_type "{cfg.model_type}" not supported')
    with open(checkpoint_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(model_cfg))
    info = {
        "model_type": cfg.model_type,
        "time_conditioned": cfg.time_conditioned,
        "mode": cfg.mode,
        "sample_shape": list(sample_shape),
        "epoch": epoch,
    }
    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)


def load_regression_model(checkpoint_dir: str | Path):
    import json

    from flanch.config import MLPConfig, UNetConfig

    checkpoint_dir = Path(checkpoint_dir)
    with open(checkpoint_dir / "checkpoint_info.json") as f:
        info = json.load(f)
    model_type = info["model_type"]
    time_conditioned = info["time_conditioned"]
    raw_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    rngs = nnx.Rngs(0)
    match model_type:
        case "mlp":
            model_cfg = OmegaConf.merge(OmegaConf.structured(MLPConfig()), raw_cfg)
            model = RegressionDec(
                EmbMLP.from_config(model_cfg, rngs=rngs), time_conditioned
            )
        case "unet":
            model_cfg = OmegaConf.merge(OmegaConf.structured(UNetConfig()), raw_cfg)
            model = RegressionUNetDec(
                UNet.from_config(model_cfg, rngs=rngs), time_conditioned
            )
        case _:
            raise ValueError(f"Unknown model_type: {model_type}")
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(checkpoint_dir / "state", abstract_state)
    return nnx.merge(graphdef, state)


def get_model(cfg: CFMTraining, rngs=None):
    if rngs is None:
        rngs = nnx.Rngs(0)
    match cfg.model_type:
        case "mlp":
            return CFMDec(EmbMLP.from_config(cfg.mlp, rngs=rngs))
        case "unet":
            return UNet.from_config(cfg.unet, rngs=rngs)
        case _:
            raise ValueError("model_type not supported")


def load_checkpoint_info(checkpoint_dir: str | Path) -> dict:
    import json

    with open(Path(checkpoint_dir) / "checkpoint_info.json") as f:
        return json.load(f)


def load_model(checkpoint_dir: str | Path):
    from flanch.config import MLPConfig, UNetConfig

    checkpoint_dir = Path(checkpoint_dir)
    info = load_checkpoint_info(checkpoint_dir)
    model_type = info["model_type"]
    raw_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    rngs = nnx.Rngs(0)
    match model_type:
        case "mlp":
            model_cfg = OmegaConf.merge(OmegaConf.structured(MLPConfig()), raw_cfg)
            model = CFMDec(EmbMLP.from_config(model_cfg, rngs=rngs))
        case "unet":
            model_cfg = OmegaConf.merge(OmegaConf.structured(UNetConfig()), raw_cfg)
            model = UNet.from_config(model_cfg, rngs=rngs)
        case _:
            raise ValueError(f"Unknown model_type: {model_type}")
    graphdef, abstract_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(checkpoint_dir / "state", abstract_state)
    return nnx.merge(graphdef, state)


def store_model(model, cfg, epoch, sample_shape: tuple):
    import json

    run_dir = HydraConfig.get().runtime.output_dir
    logging.info("Storing model at %s", run_dir)
    checkpoint_dir = Path(run_dir) / f"{epoch}"
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    checkpointer.save(checkpoint_dir / "state", state)
    checkpointer.wait_until_finished()
    model_cfg = getattr(cfg, cfg.model_type, None)
    if model_cfg is None:
        raise ValueError(f'model_type "{cfg.model_type}" not supported')
    with open(checkpoint_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(model_cfg))
    info = {
        "model_type": cfg.model_type,
        "data_name": cfg.data.name,
        "sample_shape": list(sample_shape),
        "epoch": epoch,
    }
    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)
