import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax.random as jrd
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import CFMTraining
from tpflow.data import ZarrData, device_prefetch, get_data
from tpflow.eval import export_cfm_eval, log_cfm_eval, run_cfm_eval
from tpflow.model import (
    advance_opt_steps,
    get_model,
    load_model,
    restart_state,
    store_model,
)
from tpflow.util import init_wandb, log_duration


def get_velo_err(cfg: CFMTraining):

    def velo_err(model, batch):
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

        x = x.astype(jnp.float32)
        t = t.astype(jnp.float32)
        conditioning = conditioning.astype(jnp.float32)

        pred = model(x, t, conditioning).astype(jnp.float32)

        given = (x_target - x_source).astype(jnp.float32)
        pred_err = jnp.mean((pred - given) ** 2)

        if cfg.conditioning_reg > 0:
            predh = model(x, t, conditioning + cfg.conditioning_stepsize)
            reg = (predh.astype(jnp.float32) - pred) / cfg.conditioning_stepsize
            reg = jnp.mean(reg**2)
            return pred_err + reg * cfg.conditioning_reg
        else:
            return pred_err

    return velo_err


def batch_prep(batch):
    batch, key = batch
    return batch["data"], batch["time"], key


@hydra.main(version_base=None, config_name="cfm", config_path="../../conf")
@log_duration()
def main(cfg: CFMTraining) -> None:
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

    with init_wandb(cfg, "cfm-train") as run:
        logging.info("\n%s", OmegaConf.to_yaml(cfg))
        rngs = nnx.Rngs(0)
        if restart_path is not None:
            model = load_model(restart_path)
        else:
            model = get_model(cfg, rngs=rngs)
        jax.block_until_ready(model)
        logging.info("Model loaded")
        data = ZarrData(cfg.data, "train")
        val_data = get_data(cfg.data, "test")
        logging.info("Data prepared")
        opt = get_optimizer(model, cfg.opt, len(data))
        jax.block_until_ready(opt)
        logging.info("Optimizer initialized")
        train_err = nnx.metrics.Average()
        val_err = nnx.metrics.Average()
        r = Recorder()
        velo_err = get_velo_err(cfg)
        ts, graphdef, state, loss_fn = get_train_step(
            model,
            opt,
            train_err,
            velo_err,
            batch_prep=batch_prep,
        )
        if start_epoch > 0:
            state = advance_opt_steps(state, start_epoch * len(data))

        for epoch in range(start_epoch, cfg.opt.epochs):
            model.train()
            keys = jrd.split(rngs.param(), len(data))
            ep_data = device_prefetch(data.iter_batches(epoch))
            pbar = tqdm(enumerate(ep_data), total=len(data))
            for i, batch in pbar:
                b = (jax.device_put(batch), keys[i])
                loss_val, state = ts(state, b)
                met = r({"loss_val": loss_val})
                pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

            model, opt, avg_metric = nnx.merge(graphdef, state)
            logging.info("Epoch %i: Avg. loss %.4e", epoch + 1, avg_metric.compute())
            run.log({"train/avg_loss": avg_metric.compute()}, step=epoch + 1)
            avg_metric.reset()

            model.eval()
            pbar = tqdm(enumerate(device_prefetch(val_data)), total=len(val_data))
            keys = jrd.split(rngs.param(), len(val_data))
            for i, batch in pbar:
                b = jax.device_put((batch["data"], batch["time"], keys[i]))
                loss_val = loss_fn(state, b)
                met = r({"loss_val": loss_val})
                val_err.update(values=loss_val)
                pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

            logging.info("Epoch %i: Val. loss %.4e", epoch + 1, val_err.compute())
            run.log({"val/avg_loss": val_err.compute()}, step=epoch + 1)
            val_err.reset()

            if (epoch + 1) % cfg.eval_interval == 0:
                sample_shape = batch["data"].shape[1:]
                store_model(model, cfg, epoch + 1, sample_shape, run_dir)
                eval_result = run_cfm_eval(model, cfg, sample_shape)
                export_cfm_eval(eval_result, run_dir / f"{epoch + 1}" / "eval")
                log_cfm_eval(eval_result, run, cfg, epoch + 1)


if __name__ == "__main__":
    main()
