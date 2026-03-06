"""Train a one-step regression model on conditioning trajectories.

Learns to predict the next state given (current_state, conditioning_time, param).
The 'conditioning_time' here is the axis along which the trajectory was swept in
03_gen_cond_trajectories (i.e. the cond_values), not the flow-matching time used
in 02_train_cfm.

Two prediction modes:
  step       -- minimises  ||model(x, t, p) - x_next||
  difference -- minimises  ||model(x, t, p) - (x_next - x) / diff_scale||
               rollout is a forward-Euler step: x_{t+1} = x_t + diff_scale * model(x_t, t, p)
               diff_scale = std(x_next - x) over the training set (stored as zarr attribute)

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

import hydra
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hdfv.images import frame_rgb, grid_shape
from omegaconf import OmegaConf
from tqdm import tqdm

import matplotlib.pyplot as plt
import wandb
from tpflow.config import RegressionTraining
from tpflow.data import RegressionZarrData, device_prefetch, get_regression_val_data
from tpflow.model import (
    get_regression_model,
    regression_rollout,
    store_regression_model,
)
from tpflow.processing import load_trajectory_zarr
from tpflow.statistics import trajectory_statistics
from tpflow.util import init_wandb, log_duration
from tpflow.visualization import trace_video


def get_regression_loss(mode: str, diff_scale: float = 1.0):

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
    batch_dict, _key = batch
    return (
        batch_dict["data"],
        batch_dict["next"],
        batch_dict["time"],
        batch_dict["param"],
    )


@hydra.main(version_base=None, config_name="regression", config_path="../../conf")
@log_duration()
def main(cfg: RegressionTraining) -> None:
    with init_wandb(cfg, "regression-train") as run:
        logging.info("\n%s", OmegaConf.to_yaml(cfg))

        rngs = nnx.Rngs(0)
        model = get_regression_model(cfg, rngs=rngs)
        jax.block_until_ready(model)
        logging.info("Model loaded")

        train_data = RegressionZarrData(cfg.train_data, cfg.batch_size, cfg.block_size)
        diff_scale = train_data.diff_scale
        val_data = get_regression_val_data(cfg.val_data, cfg.batch_size)
        logging.info(
            "Data prepared: %d train batches, %d val batches",
            len(train_data),
            len(val_data),
        )
        logging.info("diff_scale = %.6g", diff_scale)

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

        for epoch in range(cfg.opt.epochs):
            model.train()
            keys = jrd.split(rngs.param(), len(train_data))
            ep_data = device_prefetch(train_data.iter_batches(epoch))
            pbar = tqdm(enumerate(ep_data), total=len(train_data))
            for i, batch in pbar:
                b = (jax.device_put(batch), keys[i])
                loss_val, state = ts(state, b)
                met = r({"loss_val": loss_val})
                pbar.set_postfix({"loss": f"{met['loss_val']:.2e}"})

            model, opt, avg_metric = nnx.merge(graphdef, state)
            logging.info("Epoch %d: Avg. loss %.4e", epoch + 1, avg_metric.compute())
            run.log({"train/avg_loss": avg_metric.compute()}, step=epoch + 1)
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
                sample_shape = batch["data"].shape[1:]
                store_regression_model(model, cfg, epoch + 1, sample_shape)
                _log_rollout_eval(model, cfg, run, epoch + 1, diff_scale)


def _log_rollout_eval(
    model, cfg: RegressionTraining, run, step: int, diff_scale: float = 1.0
):
    traj_data, traj_param, time_vector = load_trajectory_zarr(cfg.rollout_data, n=cfg.n_rollout)

    x0 = jnp.array(traj_data[:, 0])  # (n_rollout, *state_shape)
    param = jnp.array(traj_param[:, None])  # (n_rollout, 1)

    model.eval()
    out = regression_rollout(model, x0, time_vector, param, cfg.mode, diff_scale)
    # out: (n_time, n_rollout, *state_shape)

    if cfg.data_type == "hist":
        frames = trace_video(out)  # expects (n_time, n_particles, 2)
        video = np.array(np.transpose(frames, (0, 3, 1, 2)))
        run.log({"eval/rollout": wandb.Video(video, fps=20, format="mp4")}, step=step)
    elif cfg.data_type == "field":
        nrows, ncols = grid_shape(cfg.n_rollout)
        frames = [
            frame_rgb(o, grid=True, nrows=nrows, ncols=ncols, channel=0) for o in out
        ]
        video = np.array(np.transpose(frames, (0, 3, 1, 2)))
        run.log({"eval/rollout": wandb.Video(video, fps=30, format="mp4")}, step=step)

    # reference: (n_rollout, n_time, *state) → (n_time, n_rollout, *state)
    ref = np.moveaxis(traj_data, 0, 1)
    rollout_stats = trajectory_statistics(out)
    ref_stats = trajectory_statistics(ref)
    for stat_name in rollout_stats:
        fig = _stats_figure(stat_name, rollout_stats[stat_name], ref_stats[stat_name], time_vector)
        run.log({f"eval/{stat_name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


def _stats_figure(
    stat_name: str,
    rollout_vals: np.ndarray,
    ref_vals: np.ndarray,
    time_vector: np.ndarray,
):
    """Create a mean ± std plot comparing rollout and reference ensembles.

    Args:
        stat_name:    name used for the y-axis label and title
        rollout_vals: ``(n_time, n_rollout)`` from the model
        ref_vals:     ``(n_time, n_rollout)`` from the reference data
        time_vector:  ``(n_time,)`` x-axis values

    Returns:
        Matplotlib figure (caller is responsible for closing it).
    """
    t = time_vector

    rm = np.mean(rollout_vals, axis=1)
    rs = np.std(rollout_vals, axis=1)
    fm = np.mean(ref_vals, axis=1)
    fs = np.std(ref_vals, axis=1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(t, rm - rs, rm + rs, alpha=0.25, color="tab:blue")
    ax.plot(t, rm, color="tab:blue", label="rollout")
    ax.fill_between(t, fm - fs, fm + fs, alpha=0.25, color="tab:orange")
    ax.plot(t, fm, color="tab:orange", linestyle="--", label="reference")
    ax.set_xlabel("time")
    ax.set_ylabel(stat_name)
    ax.set_title(stat_name)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
