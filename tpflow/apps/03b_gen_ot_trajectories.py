"""Generate OT-coupled trajectories from normalised input trajectories.

For each consecutive timestep pair t → t+1, finds a hard assignment
(permutation) between the two particle clouds and chains the per-step
assignments to produce coupled trajectories.

Two solvers are available (``solver`` config key):

* ``"hungarian"`` (default) — exact optimal assignment via
  ``scipy.optimize.linear_sum_assignment`` (LAPJV).  Always yields a valid
  permutation and minimises total squared displacement with no random
  tie-breaking.  Scales as O(n²·n) worst-case but is much faster in
  practice; recommended for n ≤ 10 000 with few time steps.

* ``"sinkhorn"`` — regularised OT via ott-jax Sinkhorn followed by argmax
  with random conflict resolution.  Faster for large n or many time steps
  (GPU-friendly); tune ``epsilon`` to control sharpness.

Works for both particle data (state_shape = (d,)) and field data
(state_shape = (H, W[, C])) — the state is flattened to a vector for cost
computation only; the stored output keeps the original state shape.

Typical usage (holder example):

    python tpflow/apps/03b_gen_ot_trajectories.py \\
        input=data/datasets/holder/raw_trajectories/train.zarr \\
        output=data/datasets/holder/ot_trajectories/train.zarr \\
        norm_stats_path=data/datasets/holder/cfm_train_data/train.zarr \\
        solver=hungarian
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import zarr
from hydra.core.config_store import ConfigStore
from hdfv.histogram_videos import histogram_frames
from hdfv.images import frame_rgb, grid_shape
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from tpflow.config import WandbConfig
from tpflow.util import init_wandb, log_duration
from tpflow.visualization import angle_color_coded, trace_video


@dataclass
class OTTrajConfig:
    input: str = "MISSING"      # path to input trajectory zarr (n_traj, n_time, *state)
    output: str = "MISSING"     # path to output trajectory zarr
    dataset: str = ""           # dataset name for wandb run title
    # When set, states are normalised with stored mean/std before OT computation
    # and the output zarr contains normalised states.
    norm_stats_path: str = ""
    solver: str = "hungarian"   # "hungarian" (exact, recommended) or "sinkhorn" (faster for large n)
    epsilon: float = 0.05       # Sinkhorn regularisation — only used when solver="sinkhorn"
    n_traj: int = 0             # 0 = use all trajectories in the zarr
    time_stride: int = 1        # subsample time axis before OT: keep every k-th frame.
                                # stride=4 on 128 frames → 32 frames, 31 OT steps instead of 127.
                                # Larger steps give more informative couplings.
    n_vis: int = 2048           # trajectories used for wandb visualisation
    wandb: WandbConfig = field(default_factory=WandbConfig)


cs = ConfigStore.instance()
cs.store(name="ot_traj", node=OTTrajConfig)


# ── OT solvers ───────────────────────────────────────────────────────────────

def _hungarian_assignment(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact optimal assignment via scipy's LAPJV (linear_sum_assignment).

    Minimises total squared Euclidean distance.  Always returns a valid
    permutation with no conflicts and no random tie-breaking.
    """
    cost = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)   # (n, n) float32
    _, col_ind = linear_sum_assignment(cost)
    return col_ind.astype(np.int32)


@jax.jit
def _sinkhorn_matrix(x: jnp.ndarray, y: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    """Return the (n, n) Sinkhorn coupling matrix for point clouds x and y."""
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    return sinkhorn.Sinkhorn()(prob).matrix


def _argmax_to_permutation(assignment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert a (possibly non-injective) argmax assignment into a valid permutation.

    For each target claimed by multiple sources, the source with the lowest index
    keeps the assignment; the remaining "losers" are randomly shuffled to the
    unoccupied targets.
    """
    n = len(assignment)
    perm = assignment.copy()
    order = np.argsort(perm, kind="stable")
    sorted_targets = perm[order]
    is_first = np.concatenate([[True], sorted_targets[1:] != sorted_targets[:-1]])
    winner_sources = order[is_first]
    loser_mask = np.ones(n, dtype=bool)
    loser_mask[winner_sources] = False
    losers = np.where(loser_mask)[0]
    target_taken = np.zeros(n, dtype=bool)
    target_taken[perm[winner_sources]] = True
    unoccupied = np.where(~target_taken)[0]
    perm[losers] = rng.permutation(unoccupied)
    return perm.astype(np.int32)


def _sinkhorn_assignment(x: np.ndarray, y: np.ndarray, epsilon: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Sinkhorn OT + argmax + conflict resolution (fallback for large n)."""
    matrix = np.array(_sinkhorn_matrix(jnp.array(x), jnp.array(y), epsilon))
    return _argmax_to_permutation(matrix.argmax(axis=1).astype(np.int32), rng)


def _ot_assignment(x: np.ndarray, y: np.ndarray, solver: str,
                   epsilon: float, rng: np.random.Generator) -> np.ndarray:
    if solver == "hungarian":
        return _hungarian_assignment(x, y)
    return _sinkhorn_assignment(x, y, epsilon, rng)


# ── normalisation ─────────────────────────────────────────────────────────────

def _normalise(data: np.ndarray, stats_path: str) -> np.ndarray:
    """Normalise (n_traj, n_time, *state) using stored mean/std attrs.

    Mirrors the logic in 01_process_trajectories.py:
    - If ``per_time_mean``/``per_time_std`` are present (written when
      ``normalize_per_time=True``), use those — shape (n_time, *state).
    - Otherwise fall back to the global ``data_mean``/``data_std``.
    Both broadcast correctly against (n_traj, n_time, *state).
    """
    stats = zarr.open(stats_path, mode="r")
    if "per_time_mean" in stats.attrs:
        mean = np.asarray(stats.attrs["per_time_mean"], dtype=np.float32)
        std = np.asarray(stats.attrs["per_time_std"], dtype=np.float32)
        logging.info("Using per_time normalisation stats, shape %s", mean.shape)
    else:
        mean = np.asarray(stats.attrs["data_mean"], dtype=np.float32)
        std = np.asarray(stats.attrs["data_std"], dtype=np.float32)
        logging.info("Using global normalisation stats, shape %s", mean.shape)
    return (data - mean) / std


# ── visualisation ─────────────────────────────────────────────────────────────

def _log_vis(run, coupled: np.ndarray, n_vis: int) -> None:
    """Log trajectory videos to wandb.

    Detects data type from state_shape:
    - 1-D state  → 2-D particle scatter (histogram + trace videos)
    - ≥2-D state → field grid video (subset of trajectories)
    """
    state_shape = coupled.shape[2:]
    n_traj, n_time = coupled.shape[:2]
    n_vis = min(n_vis, n_traj)

    # coupled: (n_traj, n_time, *state) → vis subset: (n_vis, n_time, *state)
    vis = coupled[:n_vis]

    if len(state_shape) == 1:
        # ── particle data ────────────────────────────────────────────────────
        # histogram_frames / trace_video expect (n_time, n_particles, 2)
        traj = np.transpose(vis, (1, 0, 2))   # (n_time, n_vis, d)
        absmax = float(np.percentile(np.abs(traj), 98))
        lim = (-absmax, absmax)

        frames = np.array([f.data for f in histogram_frames(traj, xlim=lim, ylim=lim)])
        run.log({"ot/histogram": wandb.Video(
            np.transpose(frames, (0, 3, 1, 2)), fps=30, format="mp4")})

        frames = trace_video(traj, xlim=lim, ylim=lim)
        run.log({"ot/traces": wandb.Video(
            np.transpose(np.array(frames), (0, 3, 1, 2)), fps=20, format="mp4")})

        frames = angle_color_coded(traj, traj[0], xlim=lim, ylim=lim)
        run.log({"ot/colorcoded": wandb.Video(
            np.transpose(np.array(frames), (0, 3, 1, 2)), fps=20, format="mp4")})

    else:
        # ── field data ───────────────────────────────────────────────────────
        # Show a grid of n_vis fields evolving over time.
        # vis: (n_vis, n_time, H, W[, C])
        n_grid = min(n_vis, 16)
        nrows, ncols = grid_shape(n_grid)
        frames = [
            frame_rgb(vis[:n_grid, t], grid=True, nrows=nrows, ncols=ncols, channel=0)
            for t in range(n_time)
        ]
        frames = np.array(frames)   # (n_time, H_grid, W_grid, 3)
        run.log({"ot/trajectories": wandb.Video(
            np.transpose(frames, (0, 3, 1, 2)), fps=30, format="mp4")})


# ── main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_name="ot_traj", config_path="../../conf")
@log_duration()
def main(cfg: OTTrajConfig) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    with init_wandb(cfg, "ot-trajectories") as run:
        # Load input trajectories
        in_group = zarr.open(cfg.input, mode="r")
        data = np.array(in_group["data"], dtype=np.float32)   # (n_traj, n_time, *state)
        time_vec = np.array(in_group["time"], dtype=np.float32)
        param = np.array(in_group["param"], dtype=np.float32)

        if cfg.n_traj > 0:
            data = data[:cfg.n_traj]
            param = param[:cfg.n_traj]

        if cfg.time_stride > 1:
            data = data[:, ::cfg.time_stride]
            time_vec = time_vec[::cfg.time_stride]
            logging.info("Time subsampled by stride=%d: %d → %d frames",
                         cfg.time_stride, len(time_vec) * cfg.time_stride, len(time_vec))

        n_traj, n_time = data.shape[:2]
        state_shape = data.shape[2:]
        logging.info("Input: n_traj=%d  n_time=%d  state_shape=%s", n_traj, n_time, state_shape)

        # Normalise for OT if stats provided; output will be in normalised space
        if cfg.norm_stats_path:
            logging.info("Normalising with stats from %s", cfg.norm_stats_path)
            data = _normalise(data, cfg.norm_stats_path)

        # Flatten spatial dims for cost computation: (n_traj, n_time, d)
        flat = data.reshape(n_traj, n_time, -1)

        rng = np.random.default_rng(0)
        logging.info("Solver: %s", cfg.solver)

        if cfg.solver == "sinkhorn":
            # Warm up JIT on the first pair before the tqdm loop
            logging.info("Warming up Sinkhorn JIT…")
            _ = _ot_assignment(flat[:, 0], flat[:, 1], cfg.solver, cfg.epsilon, rng)

        # Compute per-step OT assignments (each is a valid permutation)
        assignments = []   # list of n_time-1 arrays of shape (n_traj,)
        for t in tqdm(range(n_time - 1), desc="OT assignments"):
            assignments.append(
                _ot_assignment(flat[:, t], flat[:, t + 1], cfg.solver, cfg.epsilon, rng)
            )

        # Chain assignments to build coupled trajectories.
        # chain[i] = which original trajectory index provides the state at time t
        chain = np.arange(n_traj, dtype=np.int32)
        coupled = np.empty_like(data)
        coupled[:, 0] = data[:, 0]
        for t, perm in enumerate(tqdm(assignments, desc="Chaining")):
            chain = perm[chain]
            coupled[:, t + 1] = data[chain, t + 1]

        # Save output zarr (trajectory format: n_traj, n_time, *state_shape)
        out_path = Path(cfg.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_group = zarr.create_group(str(out_path), overwrite=True)
        chunk_traj = min(n_traj, 256)
        arr = out_group.create_array(
            "data",
            shape=(n_traj, n_time, *state_shape),
            chunks=(chunk_traj, n_time, *state_shape),
            dtype="f4",
        )
        arr[:] = coupled
        arr.attrs["dims"] = ["trajectory", "time", *[f"s{i}" for i in range(len(state_shape))]]
        out_group.create_array("time", data=time_vec)
        out_group.create_array("param", data=param)
        logging.info("Saved OT trajectories → %s", cfg.output)

        # Visualise and log to wandb
        logging.info("Logging visualisation to wandb…")
        _log_vis(run, coupled, cfg.n_vis)


if __name__ == "__main__":
    main()
