"""Generate conditioning trajectories from a trained CFM model.

A conditioning trajectory fixes a set of source noise vectors and sweeps the
conditioning parameter from `cond_start` to `cond_end` across `n_cond_steps`
values. For each conditioning value, the ODE is integrated from the same
source noise to produce output samples.

Output zarr layout:
    output.zarr/
        data:         (n_samples, n_cond_steps, *sample_shape)  generated samples
        source:       (n_samples, *sample_shape)                source noise
        conditioning: (n_cond_steps,)                          conditioning values

Usage:
    python tpflow/apps/03_gen_cond_trajectories.py \\
        checkpoint=/path/to/outputs/date/time/epoch \\
        n_samples=1024 \\
        n_cond_steps=64 \\
        cond_start=0.0 \\
        cond_end=1.0 \\
        n_ode_steps=128 \\
        batch_size=256 \\
        output=cond_traj.zarr
"""

import logging
from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm import tqdm

from tpflow.config import CondTrajConfig
from tpflow.model import load_checkpoint_info, load_model, make_flow_fn
from tpflow.util import log_duration


def make_output_zarr(
    path: str,
    n_cond_steps: int,
    n_samples: int,
    sample_shape: tuple,
    cond_values: np.ndarray,
):
    out = zarr.open_group(path, mode="w")
    out.create_array(
        "data",
        shape=(n_samples, n_cond_steps, *sample_shape),
        chunks=(min(n_samples, 1024), 1, *sample_shape),
        dtype="f4",
    )
    out.create_array("source", shape=(n_samples, *sample_shape), dtype="f4")
    cond_arr = out.create_array("conditioning", shape=(n_cond_steps,), dtype="f8")
    cond_arr[:] = cond_values
    return out


@hydra.main(version_base=None, config_name="cond_traj", config_path="../../conf")
@log_duration()
def main(cfg: CondTrajConfig) -> None:
    logging.info("\n%s", OmegaConf.to_yaml(cfg))

    checkpoint = str(Path(cfg.checkpoint).resolve())
    logging.info("Loading model from %s", checkpoint)
    model = load_model(checkpoint)
    model.eval()
    logging.info("Model loaded")

    info = load_checkpoint_info(checkpoint)
    sample_shape = tuple(info["sample_shape"])
    cond_values = np.linspace(cfg.cond_start, cfg.cond_end, cfg.n_cond_steps)

    out = make_output_zarr(
        cfg.output, cfg.n_cond_steps, cfg.n_samples, sample_shape, cond_values
    )

    run_fn = make_flow_fn(model, cfg.n_ode_steps)
    cslist = jnp.array(cond_values)

    key = jrd.key(cfg.seed)
    n_batches = int(np.ceil(cfg.n_samples / cfg.batch_size))
    pbar = tqdm(range(n_batches), desc="Batches")
    for b in pbar:
        key, subkey = jrd.split(key)
        start = b * cfg.batch_size
        end = min(start + cfg.batch_size, cfg.n_samples)
        source_batch = jrd.normal(subkey, (end - start, *sample_shape))
        out["source"][start:end] = np.array(source_batch)

        # run_fn returns (n_cond_steps, batch, *sample_shape)
        batch_out = np.array(run_fn(source_batch, cslist))
        out["data"][start:end] = np.moveaxis(batch_out.astype(np.float32), 0, 1)

    logging.info("Saved conditioning trajectories to %s", cfg.output)


if __name__ == "__main__":
    main()
