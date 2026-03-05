from dataclasses import dataclass
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import numpy as np
import zarr
from hydra.core.config_store import ConfigStore


@dataclass
class KolmogorovFlow:
    viscosity: float = 1e-3
    max_velocity: float = 7.0
    n_dim: int = 256
    n_out_dim: int = 128
    rng_seed: int = 0
    split: str = "train"
    n_seeds: int = 1_000
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="kolflow", node=KolmogorovFlow)


def open_or_create(cfg: KolmogorovFlow, result_shape, n_seeds):

    STORE_PATH = f"data/datasets/kolflow/raw_trajectories/{cfg.split}.zarr"
    Path(STORE_PATH).parent.mkdir(parents=True, exist_ok=True)

    store = zarr.open(STORE_PATH, mode="a")  # append mode

    if "data" not in store:
        store.require_array(
            "data",
            shape=(n_seeds, *result_shape),
            dtype=np.float32,
            chunks=(1, *result_shape),  # chunk per seed
        )

    if "time" not in store:
        store.require_array(
            "time",
            shape=(result_shape[0],),
            dtype=np.float32,
        )
        store["time"] = np.linspace(0, 1, result_shape[0])

    return store["data"]


@hydra.main(version_base=None, config_name="kolflow", config_path="../../conf")
def main(cfg: KolmogorovFlow) -> None:
    viscosity = cfg.viscosity
    max_velocity = cfg.max_velocity
    n_dim = cfg.n_dim
    grid = grids.Grid((n_dim, n_dim), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    dt = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, grid)

    # setup step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing

    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt
    )

    final_time = 25.0
    outer_steps = 128
    inner_steps = (final_time // dt) // outer_steps

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
    )

    seed = cfg.seed if cfg.split == "train" else cfg.seed + 10_000
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid, max_velocity, 4
    )
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    # transform the trajectory into real-space and wrap in xarray for plotting
    spatial_coord = (
        jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
    )  # same for x and y
    coords = {
        "time": dt * jnp.arange(outer_steps) * inner_steps,
        "x": spatial_coord,
        "y": spatial_coord,
    }
    out = jnp.fft.irfftn(trajectory, axes=(1, 2))
    ts = out.shape
    out = jax.image.resize(out, (ts[0], cfg.n_out_dim, cfg.n_out_dim), "bilinear")
    arr = open_or_create(
        cfg,
        (outer_steps, cfg.n_out_dim, cfg.n_out_dim, 1),
        cfg.n_seeds,
    )
    arr[cfg.seed] = np.expand_dims(out, -1)


if __name__ == "__main__":
    main()
