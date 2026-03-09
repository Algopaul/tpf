"""Generate one hw2d trajectory and append (density, phi) to a shared zarr.

Follows the same pattern as kolmogorov_flow.py: one invocation per seed,
zarr opened in append mode so many seeds can be written in parallel.

Single seed (local):
    python scripts/datagen/hw2d.py seed=42 split=train

Sweep on the cluster (submitit_slurm):
    python scripts/datagen/hw2d.py --multi \\
        seed=range(0,1000) split=train +env=torchcpu

    python scripts/datagen/hw2d.py --multi \\
        seed=range(0,100) split=test n_seeds=100 +env=torchcpu

Output zarr layout (channels-last, matching the tpflow field convention):
    data  : (n_seeds, n_time, H, W, 2)  — channel 0 = density (n),
                                           channel 1 = phi
    param : (n_seeds,)                  — c1 value per trajectory
    time  : (n_time,)                   — normalised to [0, 1]
"""

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import hydra
import numpy as np
import zarr
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd


@dataclass
class HW2DConfig:
    # ── hw2d physics parameters ────────────────────────────────────────────
    step_size: float = 0.025
    end_time: float = 200.0
    grid_pts: int = 256  # simulation grid; downsampled before storing
    c1: float = 1.0
    k0: float = 0.15
    N: int = 3
    nu: float = 5.0e-8
    buffer_length: int = 100  # HDF5 write buffer (not snapshot interval)
    snaps: int = 1  # 1 = save every simulation step
    downsample_factor: int = 2  # spatial: grid_pts → grid_pts // factor

    # ── dataset parameters ─────────────────────────────────────────────────
    seed: int = 0
    split: str = "train"
    n_seeds: int = 1000  # total seeds; used to pre-allocate zarr shape
    time_stride: int = 20  # keep every N-th snapshot (401 out of 8001)

    # ── output ─────────────────────────────────────────────────────────────
    output_dir: str = "data/datasets/hw2d/raw_trajectories"


cs = ConfigStore.instance()
cs.store(name="hw2d", node=HW2DConfig)


def _open_or_create(
    zarr_path: Path, n_seeds: int, n_time: int, H: int, W: int
) -> zarr.Group:
    """Open existing zarr in append mode, creating arrays if absent."""
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    store: zarr.Group = zarr.open_group(str(zarr_path), mode="a")
    if "data" not in store:
        store.require_array(
            "data",
            shape=(n_seeds, n_time, H, W, 2),
            dtype=np.float32,
            chunks=(1, n_time, H, W, 2),  # one chunk per trajectory
        )
    if "param" not in store:
        store.require_array("param", shape=(n_seeds,), dtype=np.float32)
    if "time" not in store:
        t: zarr.Array = store.require_array("time", shape=(n_time,), dtype=np.float32)
        t[:] = np.linspace(0.0, 1.0, n_time, dtype=np.float32)
    return store


@hydra.main(version_base=None, config_name="hw2d", config_path="../../conf")
def main(cfg: HW2DConfig) -> None:
    orig_cwd = Path(get_original_cwd())
    zarr_path = orig_cwd / cfg.output_dir / f"{cfg.split}.zarr"

    H_out = W_out = cfg.grid_pts // cfg.downsample_factor

    # Number of saved snapshots: step_size * n_steps = end_time,
    # snaps=1 → every step is written → n_steps+1 total rows in the h5.
    n_steps = round(cfg.end_time / cfg.step_size)  # 8000
    time_indices = list(range(0, n_steps + 1, cfg.time_stride))  # 401 values
    n_time = len(time_indices)

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / f"hw2d_{cfg.seed}.h5"

        # Run hw2d using the same Python interpreter (respects .venv).
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hw2d",
                f"--step_size={cfg.step_size}",
                f"--end_time={cfg.end_time}",
                f"--grid_pts={cfg.grid_pts}",
                f"--c1={cfg.c1}",
                f"--k0={cfg.k0}",
                f"--N={cfg.N}",
                f"--nu={cfg.nu}",
                f"--output_path={h5_path}",
                f"--buffer_length={cfg.buffer_length}",
                f"--snaps={cfg.snaps}",
                f"--downsample_factor={cfg.downsample_factor}",
                "--movie=0",
                "--debug=0",
                f"--seed={cfg.seed}",
            ],
            check=True,
            cwd=str(orig_cwd),
        )

        with h5py.File(h5_path, "r") as f:
            # hw2d stores density (n) and phi; subsampled along the time axis.
            # If your hw2d version uses different key names, update here.
            # shape: (n_time, H, W)
            density = np.asarray(f["density"])[time_indices].astype(np.float32)
            phi = np.asarray(f["phi"])[time_indices].astype(np.float32)

    # Stack channels last: (n_time, H, W, 2)
    data = np.stack([density, phi], axis=-1)

    store = _open_or_create(zarr_path, cfg.n_seeds, n_time, H_out, W_out)
    data_arr: zarr.Array = store["data"]  # type: ignore[assignment]
    param_arr: zarr.Array = store["param"]  # type: ignore[assignment]
    data_arr[cfg.seed] = data
    param_arr[cfg.seed] = np.float32(cfg.c1)


if __name__ == "__main__":
    main()
