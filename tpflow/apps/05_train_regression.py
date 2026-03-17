"""Train a one-step regression model on conditioning trajectories.

Learns to predict the next state given (current_state, conditioning_time, param).
The 'conditioning_time' here is the axis along which the trajectory was swept in
03_gen_cond_trajectories (i.e. the cond_values), not the flow-matching time used
in 02_train_cfm.

Two prediction modes:
  step       -- minimises  ||model(x, t, p) - x_next||
  difference -- minimises  ||model(x, t, p) - (x_next - x) / diff_scale||
               rollout: x_{t+1} = x_t + diff_scale * model(x_t, t, p)
               diff_scale = std(x_next - x) over training set (zarr attribute)

Two conditioning modes (set via time_conditioned):
  True  -- model receives (x, t, p) as input   (architecture input size n+2)
  False -- model receives (x, p)    as input   (architecture input size n+1)

Usage:
    python tpflow/apps/05_train_regression.py \\
        train_data=data/.../regression_train_data/train_shuffled.zarr \\
        val_data=data/.../regression_train_data/test.zarr \\
        rollout_data=data/.../raw_trajectories/test.zarr \\
        mode=step \\
        time_conditioned=true
"""

import logging
import time
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import RegressionTraining
from tpflow.data import RegressionZarrData, device_prefetch, get_regression_val_data
from tpflow.eval import export_regression_eval, log_regression_eval, run_regression_eval
from tpflow.model import (
    advance_opt_steps,
    get_regression_model,
    load_regression_model,
    restart_state,
    store_regression_model,
)
from tpflow.util import init_wandb, log_duration


def get_regression_loss(mode: str, diff_scale: np.ndarray | float = 1.0):

    def regression_loss(model, batch):
        x, x_next, time, param = batch
        if time.ndim == 1:
            time = time[:, None]
        if param.ndim == 1:
            param = param[:, None]
        x = x.astype(jnp.float32)
        time = time.astype(jnp.float32)
        param = param.astype(jnp.float32)
        pred = model(x, time, param).astype(jnp.float32)
        if mode == "difference":
            target = ((x_next - x) / diff_scale).astype(jnp.float32)
        else:
            target = x_next.astype(jnp.float32)
        return jnp.mean((pred - target) ** 2)

    return regression_loss


def batch_prep(batch):
    batch_dict, _ = batch
    return (
        batch_dict["data"],
        batch_dict["next"],
        batch_dict["time"],
        batch_dict["param"],
    )


@hydra.main(version_base=None, config_name="regression", config_path="../../conf")
@log_duration()
def main(cfg: RegressionTraining) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    start_epoch = 0
    restart_path = None
    if cfg.restart_from:
        start_epoch, restart_path = restart_state(cfg.restart_from)
        if start_epoch >= cfg.opt.epochs:
            logging.warning(
                "start_epoch %d >= total epochs %d — nothing to train",
                start_epoch, cfg.opt.epochs,
            )

    with init_wandb(cfg, "regression-train", data_name=cfg.dataset or None) as run:
        logging.info("\n%s", OmegaConf.to_yaml(cfg))

        rngs = nnx.Rngs(0)
        if restart_path is not None:
            model = load_regression_model(restart_path)
        else:
            model = get_regression_model(cfg, rngs=rngs)
        jax.block_until_ready(model)
        logging.info("Model loaded")

        train_data = RegressionZarrData(cfg.train_data, cfg.batch_size, cfg.block_size)
        diff_scale = train_data.diff_scale
        sample_shape: tuple[int, ...] = train_data._arrays["data"].shape[1:]
        val_data = get_regression_val_data(cfg.val_data, cfg.batch_size, cfg.val_n_samples)
        logging.info(
            "Data prepared: %d train batches, %d val batches",
            len(train_data),
            len(val_data),
        )
        logging.info("diff_scale shape=%s mean=%.6g min=%.6g max=%.6g",
                     diff_scale.shape, float(diff_scale.mean()),
                     float(diff_scale.min()), float(diff_scale.max()))

        opt = get_optimizer(model, cfg.opt, len(train_data))
        jax.block_until_ready(opt)
        logging.info("Optimizer initialized")

        loss_fn_inner = get_regression_loss(cfg.mode, diff_scale)
        train_err = nnx.metrics.Average()
        val_err = nnx.metrics.Average()
        r = Recorder()

        ts, graphdef, state, loss_fn = get_train_step(
            model,
            opt,
            train_err,
            loss_fn_inner,
            batch_prep=batch_prep,
        )
        if start_epoch > 0:
            state = advance_opt_steps(state, start_epoch * len(train_data))

        for epoch in range(start_epoch, cfg.opt.epochs):
            model.train()
            keys = jrd.split(rngs.param(), len(train_data))
            ep_data = device_prefetch(train_data.iter_batches(epoch))
            pbar = tqdm(enumerate(ep_data), total=len(train_data))
            load_times: list[float] = []
            dispatch_times: list[float] = []
            t_loop = time.perf_counter()
            for i, batch in pbar:
                t_got = time.perf_counter()
                load_times.append(t_got - t_loop)
                b = (jax.device_put(batch), keys[i])
                loss_val, state = ts(state, b)
                t_loop = time.perf_counter()
                dispatch_times.append(t_loop - t_got)
                met = r({"loss_val": loss_val})
                pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

            model, opt, avg_metric = nnx.merge(graphdef, state)
            logging.info("Epoch %d: Avg. loss %.4e", epoch + 1, avg_metric.compute())
            lt = np.array(load_times[1:]) * 1000
            dt = np.array(dispatch_times[1:]) * 1000
            run.log({
                "train/avg_loss": avg_metric.compute(),
                "perf/load_ms_mean": float(np.mean(lt)),
                "perf/load_ms_p95": float(np.percentile(lt, 95)),
                "perf/dispatch_ms_mean": float(np.mean(dt)),
            }, step=epoch + 1)
            avg_metric.reset()

            model.eval()
            pbar = tqdm(enumerate(device_prefetch(val_data)), total=len(val_data))
            for i, batch in pbar:
                b = jax.device_put(
                    (batch["data"], batch["next"], batch["time"], batch["param"])
                )
                loss_val = loss_fn(state, b)
                met = r({"loss_val": loss_val})
                val_err.update(values=loss_val)
                pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

            logging.info("Epoch %d: Val. loss %.4e", epoch + 1, val_err.compute())
            run.log({"val/avg_loss": val_err.compute()}, step=epoch + 1)
            val_err.reset()

            if (epoch + 1) % cfg.eval_interval == 0:
                store_regression_model(model, cfg, epoch + 1, sample_shape, run_dir, diff_scale)
                eval_result = run_regression_eval(model, cfg, diff_scale)
                export_regression_eval(eval_result, run_dir / f"{epoch + 1}" / "eval")
                log_regression_eval(eval_result, run, cfg, epoch + 1)


if __name__ == "__main__":
    main()
