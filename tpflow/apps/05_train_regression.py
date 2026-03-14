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
import zarr
import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
from flanch import Recorder, get_optimizer
from flanch.optimizer import get_train_step
from flax import nnx
from hdfv.images import frame_rgb, grid_shape
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from tpflow.config import RegressionTraining
from tpflow.data import RegressionZarrData, device_prefetch, get_regression_val_data
from tpflow.model import (
    _find_latest_checkpoint,
    get_regression_model,
    load_checkpoint_info,
    load_regression_model,
    regression_rollout,
    store_regression_model,
)
from tpflow.processing import load_trajectory_zarr
from tpflow.statistics import energy_spectra, hw2d_statistics, trajectory_statistics
from tpflow.util import init_wandb, log_duration
from tpflow.visualization import trace_video


def get_regression_loss(mode: str, diff_scale=1.0):

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
    start_epoch = 0
    resume_run_id = None
    restart_path = None
    if cfg.restart_from:
        restart_path = Path(cfg.restart_from).resolve()
        if not (restart_path / "checkpoint_info.json").exists():
            restart_path = _find_latest_checkpoint(restart_path)
            if restart_path is None:
                raise FileNotFoundError(f"No checkpoints found under {cfg.restart_from}")
        info = load_checkpoint_info(restart_path)
        start_epoch = info["epoch"]
        resume_run_id = info.get("wandb_run_id")
        logging.info("Restarting from epoch %d at %s", start_epoch, restart_path)
        if start_epoch >= cfg.opt.epochs:
            logging.warning(
                "start_epoch %d >= total epochs %d — nothing to train",
                start_epoch, cfg.opt.epochs,
            )

    with init_wandb(cfg, "regression-train", data_name=cfg.dataset or None, resume_run_id=resume_run_id) as run:
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
        val_data = get_regression_val_data(cfg.val_data, cfg.batch_size)
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
            # Advance optimizer step counters so the LR schedule resumes at the
            # correct position rather than restarting warmup from step 0.
            steps_done = start_epoch * len(train_data)
            state = jax.tree_util.tree_map(
                lambda x: jnp.array(steps_done, dtype=x.dtype)
                if (x.shape == () and jnp.issubdtype(x.dtype, jnp.integer))
                else x,
                state,
            )

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
            # Skip first batch (cold start) before logging timing stats
            lt = np.array(load_times[1:]) * 1000  # ms
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
                store_regression_model(model, cfg, epoch + 1, sample_shape)
                _log_rollout_eval(model, cfg, run, epoch + 1, diff_scale)


def _spectra_figure(
    bin_centers: np.ndarray,
    rollout_spectra: np.ndarray,
    ref_spectra: np.ndarray,
):
    """Mean ± std energy spectrum plot comparing rollout and reference.

    Averages over the time axis before computing ensemble statistics, giving
    a single stationary spectrum per trajectory.

    Args:
        bin_centers:     ``(n_bins,)`` wavenumber bin centres.
        rollout_spectra: ``(n_time, n_rollout, n_bins)`` from the model.
        ref_spectra:     ``(n_time, n_rollout, n_bins)`` from reference data.

    Returns:
        Matplotlib figure (caller is responsible for closing it).
    """
    # Average over time → (n_rollout, n_bins), then mean/std over ensemble
    r = np.mean(rollout_spectra, axis=0)   # (n_rollout, n_bins)
    f = np.mean(ref_spectra, axis=0)       # (n_rollout, n_bins)
    rm, rs = np.mean(r, axis=0), np.std(r, axis=0)
    fm, fs = np.mean(f, axis=0), np.std(f, axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(bin_centers, rm - rs, rm + rs, alpha=0.25, color="tab:blue")
    ax.plot(bin_centers, rm, color="tab:blue", label="rollout")
    ax.fill_between(bin_centers, fm - fs, fm + fs, alpha=0.25, color="tab:orange")
    ax.plot(bin_centers, fm, color="tab:orange", linestyle="--", label="reference")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber")
    ax.set_ylabel("power")
    ax.set_title("energy spectra")
    ax.legend()
    fig.tight_layout()
    return fig


def _log_rollout_eval(
    model, cfg: RegressionTraining, run, step: int, diff_scale=1.0
):
    traj_data, traj_param, _ = load_trajectory_zarr(
        cfg.rollout_data, n=cfg.n_rollout
    )
    n_time = traj_data.shape[1]
    time_vector = np.linspace(cfg.cond_start, cfg.cond_end, n_time, dtype=np.float32)

    if cfg.norm_stats_path:
        stats = zarr.open(cfg.norm_stats_path, mode="r")
        if "per_time_mean" in stats.attrs:
            # Per-time normalization: stats are (n_time, C); broadcast over rollout+spatial
            per_time_mean = np.asarray(stats.attrs["per_time_mean"])  # (n_time, C)
            per_time_std = np.asarray(stats.attrs["per_time_std"])
            state_shape = traj_data.shape[2:]
            n_spatial = len(state_shape) - 1
            bc_shape = (1, n_time) + (1,) * n_spatial + (state_shape[-1],)
            traj_data = (traj_data - per_time_mean.reshape(bc_shape)) / per_time_std.reshape(bc_shape)
        else:
            data_mean = np.asarray(stats.attrs["data_mean"])
            data_std = np.asarray(stats.attrs["data_std"])
            traj_data = (traj_data - data_mean) / data_std

    x0 = jnp.array(traj_data[:, 0])  # (n_rollout, *state_shape)
    param = jnp.array(traj_param[:, None])  # (n_rollout, 1)

    model.eval()
    out = regression_rollout(model, x0, time_vector, param, cfg.mode, diff_scale,
                             zero_mean=cfg.zero_mean_rollout)
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

    if cfg.stats:
        if cfg.dataset == "hw2d":
            rollout_stats = hw2d_statistics(out)
            ref_stats = hw2d_statistics(ref)
        else:
            rollout_stats = trajectory_statistics(out)
            ref_stats = trajectory_statistics(ref)
        for stat_name in cfg.stats:
            if stat_name not in rollout_stats:
                logging.warning("Unknown stat %r — skipping", stat_name)
                continue
            fig = _stats_figure(
                stat_name, rollout_stats[stat_name], ref_stats[stat_name], time_vector
            )
            run.log({f"eval/{stat_name}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    if cfg.log_energy_spectra:
        ch_axis = cfg.energy_spectra.channel_axis
        channel_axis = ch_axis if ch_axis >= 0 else None
        rollout_bins, rollout_spectra = energy_spectra(
            out,
            n_bins=cfg.energy_spectra.n_bins,
            log_bins=cfg.energy_spectra.log_bins,
            channel_axis=channel_axis,
            channel_idx=cfg.energy_spectra.channel_idx,
        )
        _, ref_spectra = energy_spectra(
            ref,
            n_bins=cfg.energy_spectra.n_bins,
            log_bins=cfg.energy_spectra.log_bins,
            channel_axis=channel_axis,
            channel_idx=cfg.energy_spectra.channel_idx,
        )
        fig = _spectra_figure(rollout_bins, rollout_spectra, ref_spectra)
        run.log({"eval/energy_spectra": wandb.Image(fig)}, step=step)
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
