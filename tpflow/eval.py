"""Evaluation helpers shared between training scripts and the standalone export script.

Each model type has three functions:
  run_*_eval    -- pure computation; returns a result dataclass
  export_*_eval -- writes zarr files to eval_dir; no wandb dependency
  log_*_eval    -- logs videos / figures to wandb
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrd
import matplotlib.pyplot as plt
import numpy as np
import zarr

import wandb
from hdfv.histogram_videos import histogram_frames
from hdfv.images import frame_rgb, grid_shape

from tpflow.model import make_flow_fn, regression_one_step_ahead, regression_rollout
from tpflow.processing import load_trajectory_zarr
from tpflow.statistics import energy_spectra, hw2d_statistics, trajectory_statistics
from tpflow.visualization import angle_color_coded, trace_video


# ── CFM ─────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class CFMEvalResult:
    data: np.ndarray               # (n_cond_steps, n_samples, *state_shape)  f32
    conditioning: np.ndarray       # (n_cond_steps,)  f32
    source: np.ndarray             # (n_samples, *state_shape)  f32
    reference: np.ndarray | None   # (n_cond_steps, n_samples, *state_shape)  f32, or None


def _load_reference(cfg, n_samples: int, conditioning: np.ndarray) -> np.ndarray | None:
    """Load normalised reference trajectories sampled at the conditioning steps.

    Returns (n_cond_steps, n_samples, *state_shape) float32, or None when the
    raw trajectory zarr or normalisation stats are not available.
    """
    basedir = Path(f"data/datasets/{cfg.data.name}")
    test_path = basedir / "raw_trajectories" / "test.zarr"
    stats_path = basedir / "cfm_train_data" / "train.zarr"
    if not test_path.exists() or not stats_path.exists():
        return None

    test_zarr = zarr.open(str(test_path), mode="r")
    n = min(n_samples, test_zarr["data"].shape[0])
    ref = np.array(test_zarr["data"][:n], dtype=np.float32)  # (n, n_time, *state)

    stats = zarr.open(str(stats_path), mode="r")
    data_mean = np.asarray(stats.attrs["data_mean"], dtype=np.float32)
    data_std = np.asarray(stats.attrs["data_std"], dtype=np.float32)
    ref = (ref - data_mean) / data_std

    # Map each conditioning value to the nearest time index in the trajectory
    n_time = ref.shape[1]
    t_idx = np.round(conditioning * (n_time - 1)).astype(int).clip(0, n_time - 1)
    return np.stack([ref[:, ti] for ti in t_idx], axis=0)  # (n_cond_steps, n, *state)


def run_cfm_eval(model, cfg, sample_shape: tuple) -> CFMEvalResult:
    """Run CFM inference sweep over conditioning values."""
    source_batch = jrd.normal(jrd.key(0), (cfg.inference.n_samples, *sample_shape))
    cslist = jnp.linspace(0, 1, cfg.inference.n_param_steps)
    run_fn = make_flow_fn(model, n_steps=cfg.inference.n_param_steps)
    data = np.array(run_fn(source_batch, cslist))
    reference = _load_reference(cfg, cfg.inference.n_samples, np.array(cslist))
    return CFMEvalResult(
        data=data.astype(np.float32),
        conditioning=np.array(cslist, dtype=np.float32),
        source=np.array(source_batch, dtype=np.float32),
        reference=reference,
    )


def export_cfm_eval(result: CFMEvalResult, eval_dir: Path) -> None:
    """Write CFM eval arrays to ``eval_dir/trajectories.zarr``."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    store = zarr.open_group(str(eval_dir / "trajectories.zarr"), mode="w")
    store.create_array("data", data=result.data, chunks=(1, *result.data.shape[1:]))
    store.create_array("conditioning", data=result.conditioning)


def log_cfm_eval(result: CFMEvalResult, run, cfg, step: int) -> None:
    """Log CFM eval videos to wandb."""
    out = result.data
    if cfg.data.type == "hist":
        # Compute limits once from the full trajectory so all frames share the
        # same fixed range (2nd/98th percentile → symmetric around zero).
        absmax = float(np.percentile(np.abs(out), 98))
        lim = (-absmax, absmax)
        frames = np.array([f.data for f in histogram_frames(out, xlim=lim, ylim=lim)])
        run.log({"train/cfm_trajectories": wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=30, format="mp4")}, step=step)
        frames = trace_video(out[:, :200, :], xlim=lim, ylim=lim)
        run.log({"train/traces": wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=20, format="mp4")}, step=step)
        frames = angle_color_coded(out, result.source, xlim=lim, ylim=lim)
        run.log({"train/colorcoded": wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=20, format="mp4")}, step=step)
        if result.reference is not None:
            ref_frames = np.array([f.data for f in histogram_frames(result.reference, xlim=lim, ylim=lim)])
            run.log({"train/reference_particles": wandb.Video(np.transpose(ref_frames, (0, 3, 1, 2)), fps=30, format="mp4")}, step=step)
    elif cfg.data.type == "field":
        nrows, ncols = grid_shape(cfg.inference.n_samples)
        frames = [frame_rgb(o, grid=True, nrows=nrows, ncols=ncols, channel=0) for o in out]
        run.log({"train/cfm_trajectories": wandb.Video(np.array(np.transpose(frames, (0, 3, 1, 2))), fps=30, format="mp4")}, step=step)


# ── Regression ──────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class RegressionEvalResult:
    rollout: np.ndarray           # (n_time, n_rollout, *state_shape)  f32
    one_step_ahead: np.ndarray    # (n_time, n_rollout, *state_shape)  f32
    reference: np.ndarray         # (n_time, n_rollout, *state_shape)  f32
    time: np.ndarray              # (n_time,)  f32
    param: np.ndarray             # (n_rollout,)  f32
    rollout_stats: dict[str, np.ndarray] | None  # each (n_time, n_rollout)
    ref_stats: dict[str, np.ndarray] | None
    bin_centers: np.ndarray | None      # (n_bins,)
    rollout_spectra: np.ndarray | None  # (n_time, n_rollout, n_bins)
    ref_spectra: np.ndarray | None


def run_regression_eval(
    model,
    cfg,
    diff_scale: np.ndarray | float,
) -> RegressionEvalResult:
    """Run regression rollout and compute all configured statistics."""
    traj_data, traj_param, traj_time = load_trajectory_zarr(cfg.rollout_data, n=cfg.n_rollout)
    n_time = traj_data.shape[1]
    # time_vector: model conditioning values in [cond_start, cond_end] — must match training data
    # traj_time:   physical time from the rollout dataset — stored in the output zarr for plotting
    time_vector = np.linspace(cfg.cond_start, cfg.cond_end, n_time, dtype=np.float32)

    norm_per_time_mean: np.ndarray | None = None
    norm_per_time_std: np.ndarray | None = None
    if cfg.norm_stats_path:
        stats = zarr.open(cfg.norm_stats_path, mode="r")
        if "per_time_mean" in stats.attrs:
            norm_per_time_mean = np.asarray(stats.attrs["per_time_mean"])
            norm_per_time_std = np.asarray(stats.attrs["per_time_std"])
            state_shape = traj_data.shape[2:]
            n_spatial = len(state_shape) - 1
            bc_shape = (1, n_time) + (1,) * n_spatial + (state_shape[-1],)
            traj_data = (traj_data - norm_per_time_mean.reshape(bc_shape)) / norm_per_time_std.reshape(bc_shape)
        else:
            data_mean = np.asarray(stats.attrs["data_mean"])
            data_std = np.asarray(stats.attrs["data_std"])
            traj_data = (traj_data - data_mean) / data_std

    x0 = jnp.array(traj_data[:, 0])
    param = jnp.array(traj_param[:, None])

    model.eval()
    out = regression_rollout(
        model, x0, time_vector, param, cfg.mode, diff_scale,
        zero_mean=cfg.zero_mean_rollout,
    )
    ref = np.moveaxis(traj_data, 0, 1)  # (n_time, n_rollout, *state)
    osa = regression_one_step_ahead(model, ref, time_vector, param, cfg.mode, diff_scale)

    rollout_stats = ref_stats = None
    if cfg.stats:
        if cfg.dataset == "hw2d":
            # hw2d statistics require physical-unit fields — unnormalise before computing
            if norm_per_time_mean is not None:
                state_shape = out.shape[2:]
                n_spatial = len(state_shape) - 1
                bc_shape = (n_time, 1) + (1,) * n_spatial + (state_shape[-1],)
                out_phys = out * norm_per_time_std.reshape(bc_shape) + norm_per_time_mean.reshape(bc_shape)
                ref_phys = ref * norm_per_time_std.reshape(bc_shape) + norm_per_time_mean.reshape(bc_shape)
            else:
                out_phys, ref_phys = out, ref
            all_rollout = hw2d_statistics(out_phys)
            all_ref = hw2d_statistics(ref_phys)
        else:
            all_rollout = trajectory_statistics(out)
            all_ref = trajectory_statistics(ref)
        rollout_stats = {k: v for k, v in all_rollout.items() if k in cfg.stats}
        ref_stats = {k: v for k, v in all_ref.items() if k in cfg.stats}

    bin_centers = rollout_spectra = ref_spectra = None
    if cfg.log_energy_spectra:
        ch_axis = cfg.energy_spectra.channel_axis
        channel_axis = ch_axis if ch_axis >= 0 else None
        spectra_kwargs = dict(
            n_bins=cfg.energy_spectra.n_bins,
            log_bins=cfg.energy_spectra.log_bins,
            channel_axis=channel_axis,
            channel_idx=cfg.energy_spectra.channel_idx,
        )
        bin_centers, rollout_spectra = energy_spectra(out, **spectra_kwargs)
        _, ref_spectra = energy_spectra(ref, **spectra_kwargs)

    return RegressionEvalResult(
        rollout=out,
        one_step_ahead=osa,
        reference=ref,
        time=traj_time.astype(np.float32),
        param=np.array(traj_param, dtype=np.float32),
        rollout_stats=rollout_stats,
        ref_stats=ref_stats,
        bin_centers=bin_centers,
        rollout_spectra=rollout_spectra,
        ref_spectra=ref_spectra,
    )


def export_regression_eval(result: RegressionEvalResult, eval_dir: Path) -> None:
    """Write regression eval arrays to zarr files in eval_dir."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.open_group(str(eval_dir / "rollout.zarr"), mode="w")
    store.create_array("rollout", data=result.rollout.astype(np.float32), chunks=(1, *result.rollout.shape[1:]))
    store.create_array("one_step_ahead", data=result.one_step_ahead.astype(np.float32), chunks=(1, *result.one_step_ahead.shape[1:]))
    store.create_array("reference", data=result.reference.astype(np.float32), chunks=(1, *result.reference.shape[1:]))
    store.create_array("time", data=result.time)
    store.create_array("param", data=result.param)

    if result.rollout_stats is not None or result.bin_centers is not None:
        stats_store = zarr.open_group(str(eval_dir / "statistics.zarr"), mode="w")
        if result.rollout_stats is not None:
            for name, vals in result.rollout_stats.items():
                stats_store.create_array(f"rollout_{name}", data=vals.astype(np.float32))
                stats_store.create_array(f"ref_{name}", data=result.ref_stats[name].astype(np.float32))
        if result.bin_centers is not None:
            stats_store.create_array("bin_centers", data=result.bin_centers.astype(np.float32))
            stats_store.create_array("rollout_spectra", data=result.rollout_spectra.astype(np.float32))
            stats_store.create_array("ref_spectra", data=result.ref_spectra.astype(np.float32))


def log_regression_eval(result: RegressionEvalResult, run, cfg, step: int) -> None:
    """Log regression eval videos and figures to wandb."""
    if cfg.data_type == "hist":
        absmax = float(np.nanpercentile(np.abs(np.concatenate([result.rollout, result.reference], axis=1)), 98))
        absmax = absmax if np.isfinite(absmax) and absmax > 0 else 3.0
        lim = (-absmax, absmax)
        frames = np.array([f.data for f in histogram_frames(result.rollout, xlim=lim, ylim=lim)])
        run.log({"eval/histogram": wandb.Video(np.transpose(frames, (0, 3, 1, 2)), fps=30, format="mp4")}, step=step)
        frames = trace_video(result.rollout, xlim=lim, ylim=lim)
        run.log({"eval/rollout": wandb.Video(np.array(np.transpose(frames, (0, 3, 1, 2))), fps=20, format="mp4")}, step=step)
        frames = angle_color_coded(result.rollout, result.rollout[0], xlim=lim, ylim=lim)
        run.log({"eval/colorcoded": wandb.Video(np.transpose(np.array(frames), (0, 3, 1, 2)), fps=20, format="mp4")}, step=step)
        osa_frames = np.array([f.data for f in histogram_frames(result.one_step_ahead, xlim=lim, ylim=lim)])
        run.log({"eval/one_step_ahead_histogram": wandb.Video(np.transpose(osa_frames, (0, 3, 1, 2)), fps=30, format="mp4")}, step=step)
        ref_frames = np.array([f.data for f in histogram_frames(result.reference, xlim=lim, ylim=lim)])
        run.log({"eval/reference_histogram": wandb.Video(np.transpose(ref_frames, (0, 3, 1, 2)), fps=30, format="mp4")}, step=step)
    elif cfg.data_type == "field":
        nrows, ncols = grid_shape(cfg.n_rollout)
        frames = [frame_rgb(o, grid=True, nrows=nrows, ncols=ncols, channel=0) for o in result.rollout]
        run.log({"eval/rollout": wandb.Video(np.array(np.transpose(frames, (0, 3, 1, 2))), fps=30, format="mp4")}, step=step)
        osa_frames = [frame_rgb(o, grid=True, nrows=nrows, ncols=ncols, channel=0) for o in result.one_step_ahead]
        run.log({"eval/one_step_ahead": wandb.Video(np.array(np.transpose(osa_frames, (0, 3, 1, 2))), fps=30, format="mp4")}, step=step)

    if result.rollout_stats is not None:
        for stat_name in cfg.stats:
            if stat_name not in result.rollout_stats:
                logging.warning("Unknown stat %r — skipping", stat_name)
                continue
            fig = _stats_figure(stat_name, result.rollout_stats[stat_name], result.ref_stats[stat_name], result.time)
            run.log({f"eval/{stat_name}": wandb.Image(fig)}, step=step)
            plt.close(fig)

    if result.bin_centers is not None:
        fig = _spectra_figure(result.bin_centers, result.rollout_spectra, result.ref_spectra)
        run.log({"eval/energy_spectra": wandb.Image(fig)}, step=step)
        plt.close(fig)


def _stats_figure(
    stat_name: str,
    rollout_vals: np.ndarray,
    ref_vals: np.ndarray,
    time_vector: np.ndarray,
):
    """Mean ± std comparison plot for a scalar statistic."""
    rm, rs = np.mean(rollout_vals, axis=1), np.std(rollout_vals, axis=1)
    fm, fs = np.mean(ref_vals, axis=1), np.std(ref_vals, axis=1)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(time_vector, rm - rs, rm + rs, alpha=0.25, color="tab:blue")
    ax.plot(time_vector, rm, color="tab:blue", label="rollout")
    ax.fill_between(time_vector, fm - fs, fm + fs, alpha=0.25, color="tab:orange")
    ax.plot(time_vector, fm, color="tab:orange", linestyle="--", label="reference")
    ax.set_xlabel("time")
    ax.set_ylabel(stat_name)
    ax.set_title(stat_name)
    ax.legend()
    fig.tight_layout()
    return fig


def _spectra_figure(
    bin_centers: np.ndarray,
    rollout_spectra: np.ndarray,
    ref_spectra: np.ndarray,
    n_chunks: int = 3,
    mid_freq_frac: tuple[float, float] = (0.2, 0.7),
):
    """Log-log energy spectrum comparison, split into time chunks.

    ``n_chunks`` equal slices of the time axis are each shown as a separate
    colour.  Rollout curves are solid; reference curves are dashed.  A
    spectral slope *α* is estimated by least-squares fit of
    ``log E ~ α log k`` over the mid-frequency band and appended to each
    legend label.  The fitting band is shaded in gray.

    Args:
        bin_centers: ``(n_bins,)`` wavenumber bin centres.
        rollout_spectra: ``(n_time, n_rollout, n_bins)``.
        ref_spectra: ``(n_time, n_rollout, n_bins)``.
        n_chunks: number of equal time intervals to display.
        mid_freq_frac: ``(lo, hi)`` fractions of the log-wavenumber range
            used for slope fitting.
    """
    n_time = rollout_spectra.shape[0]

    # ── time-chunk boundaries ─────────────────────────────────────────────────
    starts = [i * n_time // n_chunks for i in range(n_chunks)]
    ends = [(i + 1) * n_time // n_chunks for i in range(n_chunks)]
    pct_labels = [
        f"{100 * s // n_time}–{100 * e // n_time}%"
        for s, e in zip(starts, ends)
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_chunks))

    # ── mid-frequency mask for slope fitting ──────────────────────────────────
    valid = bin_centers > 0
    log_k_all = np.where(valid, np.log10(np.where(valid, bin_centers, 1.0)), np.nan)
    log_k_valid = log_k_all[valid]
    if log_k_valid.size >= 4:
        lk_lo = log_k_valid.min() + mid_freq_frac[0] * (log_k_valid.max() - log_k_valid.min())
        lk_hi = log_k_valid.min() + mid_freq_frac[1] * (log_k_valid.max() - log_k_valid.min())
        slope_mask = valid & (log_k_all >= lk_lo) & (log_k_all <= lk_hi)
    else:
        slope_mask = valid

    def _fit_slope(k: np.ndarray, e: np.ndarray, mask: np.ndarray) -> float | None:
        km, em = k[mask], e[mask]
        pos = em > 0
        km, em = km[pos], em[pos]
        if km.size < 3:
            return None
        return float(np.polyfit(np.log10(km), np.log10(em), 1)[0])

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))

    for s, e, pct, color in zip(starts, ends, pct_labels, colors):
        # mean over time-chunk steps, then over rollouts → (n_bins,)
        rm = np.mean(np.mean(rollout_spectra[s:e], axis=0), axis=0)
        fm = np.mean(np.mean(ref_spectra[s:e], axis=0), axis=0)

        r_slope = _fit_slope(bin_centers, rm, slope_mask)
        f_slope = _fit_slope(bin_centers, fm, slope_mask)

        r_suffix = f" (α={r_slope:.2f})" if r_slope is not None else ""
        f_suffix = f" (α={f_slope:.2f})" if f_slope is not None else ""

        ax.plot(bin_centers, rm, color=color, label=f"rollout {pct}{r_suffix}")
        ax.plot(bin_centers, fm, color=color, linestyle="--", label=f"ref {pct}{f_suffix}")

    # shade slope-fitting band
    k_fit = bin_centers[slope_mask]
    if k_fit.size >= 2:
        ax.axvspan(k_fit[0], k_fit[-1], alpha=0.08, color="gray", label="fit range")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("power E(k)")
    ax.set_title("energy spectra")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig
