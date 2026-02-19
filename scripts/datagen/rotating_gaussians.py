"""Generate raw trajectories for dataset gaurot of rotating Gaussians"""
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import zarr


def gaussians(
    key,
    n_samples,
    *,
    n_bumps=8,
    radius=0.7,
    scale=0.07,
    angle=0.0,
):
  key_comp, key_gauss = jrd.split(key)
  angles = jnp.linspace(0, 2 * jnp.pi, n_bumps, endpoint=False) + angle
  means_x = radius * jnp.cos(angles)
  means_y = radius * jnp.sin(angles)
  means = jnp.stack((means_x, means_y)).T
  component_ids = jrd.randint(key_comp, n_samples, 0, n_bumps)
  z = jrd.normal(key_gauss, shape=(n_samples, 2))
  m = means[component_ids]
  return m + z * scale


def trajectory(seed, n_samples, n_timepoints):
  out = []
  for angle in jnp.linspace(0, 2 * jnp.pi / 8, n_timepoints):
    i, seed = jrd.split(seed)
    out.append(gaussians(i, n_samples, angle=angle))
  return jnp.transpose(jnp.stack(out), [1, 0, 2])


def main():
  n_timepoints = 30
  n_samples = {'train': 200_000, 'test': 20_000}
  for i, (k, v) in enumerate(n_samples.items()):
    root = zarr.create_group(
        f'data/datasets/gaurot/raw_trajectories/{k}.zarr',
        overwrite=True,
    )
    data = root.create_array('data', shape=(v, n_timepoints, 2), dtype='f8')
    data.attrs.update({"dims": ["trajectory", "time", "dimension"]})
    root.create_array("time", data=np.linspace(0, 1, n_timepoints))
    root.create_array(name='param', data=1.0 * np.ones(v))
    traj = trajectory(jrd.key(i), v, n_timepoints)
    data[:] = np.array(traj)


if __name__ == "__main__":
  main()
