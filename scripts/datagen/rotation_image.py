"""Generates raw trajectories for dataset imgrot"""
from dataclasses import dataclass

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import zarr
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


def gaussian_bump_2d(X, Y, bump_pos, sharpness):
  Xs = X - bump_pos[0]
  Ys = Y - bump_pos[1]
  return jnp.exp(-sharpness * (Xs**2 + Ys**2))


def double_bump(angle, radius, sharpness, grid):
  X, Y = grid
  r = radius / 2
  a = angle
  p1 = jnp.stack([r * jnp.cos(a), r * jnp.sin(a)])
  p2 = jnp.stack([-r * jnp.cos(a), -r * jnp.sin(a)])
  b1 = gaussian_bump_2d(X, Y, p1, sharpness)
  b2 = gaussian_bump_2d(X, Y, p2, sharpness)
  b = b1 + b2
  return b - jnp.mean(b)


def get_grid(dim: int = 128):
  a, b = -2 * jnp.pi, 2 * jnp.pi
  x = jnp.linspace(a, b, dim, endpoint=False)
  X, Y = jnp.meshgrid(x, x, indexing='ij')
  return (X, Y)


def trajectory(
    timepoints,
    rotation_speed_fn,
    merging_schedule,
    *,
    initial_distance=1.2,
    initial_angle=0.0,
    sharpness=10.0,
    grid=None,
):
  if grid is None:
    grid = get_grid()

  dbb = lambda angle, distance: double_bump(angle, distance, sharpness, grid)
  rot_speeds = jax.vmap(rotation_speed_fn)(timepoints)
  distances = initial_distance * jax.vmap(merging_schedule)(timepoints)
  angles = jnp.cumsum(rot_speeds) + initial_angle
  return jax.vmap(dbb)(angles, distances)


@dataclass
class ImgRot:
  grid_dim: int = 128
  speed_schedule: str = 'acc'
  n_train_samples: int = 3_000
  n_test_samples: int = 128
  sharpness: float = 10.0


cs = ConfigStore.instance()
cs.store(name='imgrot', node=ImgRot)


@hydra.main(version_base=None, config_name='imgrot', config_path='../../conf')
def main(cfg: ImgRot) -> None:
  n_timepoints = 100
  time = jnp.linspace(0, 1, n_timepoints)
  if cfg.speed_schedule == 'acc':
    rotation_speed_fn = lambda t: t / 4 + 0.2
  elif cfg.speed_schedule == 'const':
    rotation_speed_fn = lambda _: 0.2
  elif cfg.speed_schedule == 'zero':
    rotation_speed_fn = lambda _: 0.0
  else:
    raise ValueError('Unknown speed schedule')
  merging_schedule = lambda t: 5.0 - 5 * t
  n_samples = {'train': cfg.n_train_samples, 'test': cfg.n_test_samples}
  grid = get_grid(cfg.grid_dim)
  for k, v in n_samples.items():
    name = f'imgrot-{cfg.grid_dim}-{cfg.speed_schedule}'
    root = zarr.create_group(
        f'data/datasets/{name}/raw_trajectories/{k}.zarr',
        overwrite=True,
    )
    initial_angles = np.random.rand(v) * 2 * jnp.pi
    data = root.create_array(
        name='data',
        shape=(v, n_timepoints, cfg.grid_dim, cfg.grid_dim, 1),
        chunks=(1, 100, cfg.grid_dim, cfg.grid_dim, 1),
        dtype='f8',
    )
    data.attrs.update({
        "dims": ["trajectory", "time", "height", "width", "channel"],
    })
    root.create_array(name='time', data=np.array(time))
    root.create_array(name='param', data=1.0 * np.ones(v))
    for i in tqdm(range(v), desc=f'Samples for {k} data'):
      data[i, ...] = np.array(
          trajectory(
              time,
              rotation_speed_fn,
              merging_schedule,
              initial_angle=initial_angles[i],
              sharpness=cfg.sharpness,
              grid=grid,
          ))[..., None]


if __name__ == "__main__":
  main()
