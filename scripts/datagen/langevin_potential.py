"""Generate particle trajectories under a time-dependent potential.

    V(t, x) = sin(π/2 · t)² · V₁(x) + cos(π/2 · t)² · V₀(x)

with separable per-dimension potentials (eq. 76):

    V₀(x) = Σᵢ  sin(xᵢ) + cos(xᵢ) + xᵢ² + xᵢ
    V₁(x) = Σᵢ  xᵢ⁴ - 16·xᵢ² + 5·xᵢ

Particles are propagated with overdamped Langevin dynamics (Euler–Maruyama)
at unit temperature:

    dx = -∇V(t, x) dt + √2 dW
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrd
import zarr
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Potentials (eq. 76)
# ---------------------------------------------------------------------------

def v0(x):
    """V₀(x) = Σᵢ (sin(xᵢ) + cos(xᵢ) + xᵢ² + xᵢ)"""
    return jnp.sum(jnp.sin(x) + jnp.cos(x) + x ** 2 + x)


def v1(x):
    """V₁(x) = Σᵢ (xᵢ⁴ - 16·xᵢ² + 5·xᵢ)"""
    return jnp.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x)


def time_potential(x, t):
    """V(t, x) = sin(π/2·t)² V₁(x) + cos(π/2·t)² V₀(x)"""
    s2 = jnp.sin(jnp.pi / 2.0 * t) ** 2
    c2 = jnp.cos(jnp.pi / 2.0 * t) ** 2
    return c2 * v0(x) + s2 * v1(x)


# Gradient w.r.t. x
_grad_potential = jax.grad(time_potential)


# ---------------------------------------------------------------------------
# Langevin stepper
# ---------------------------------------------------------------------------

def make_step_fn(dt):
    """Return a JIT-compiled Euler–Maruyama step for a batch of particles.

    Returns:
        step(x, t, key) → x_new  where x has shape (n_particles, d)
    """
    sqrt2dt = jnp.sqrt(2.0 * dt)

    def step(x, t, key):
        grad = jax.vmap(_grad_potential, in_axes=(0, None))(x, t)
        noise = jrd.normal(key, x.shape)
        return x - grad * dt + sqrt2dt * noise

    return jax.jit(step)


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectories(key, n_particles, d, time_vector, dt):
    """Simulate Langevin trajectories.

    Particles are initialised i.i.d. from ρ₀ = N(0, I) and propagated with
    one Euler–Maruyama step per saved frame (dt = 1/512).

    Returns:
        data: (n_particles, n_time, d)
    """
    step_fn = make_step_fn(dt)

    key, k0 = jrd.split(key)
    x = jrd.normal(k0, (n_particles, d))  # ρ₀ = N(0, I)

    frames = []
    for t_val in tqdm(time_vector, desc="  time steps"):
        frames.append(np.array(x))
        key, k = jrd.split(key)
        x = step_fn(x, jnp.float32(t_val), k)

    return np.stack(frames, axis=1)  # (n_particles, n_time, d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    d = 2
    dt = 1.0 / 512
    n_timepoints = 512
    time_vector = np.linspace(0.0, 1.0, n_timepoints, endpoint=False)

    n_samples = {"train": 2048, "test": 2048}

    for split_idx, (split, n) in enumerate(n_samples.items()):
        print(f"\nGenerating {split} split (N={n}, d={d}, dt=1/512)…")
        key = jrd.key(split_idx)

        data = generate_trajectories(key, n, d, time_vector, dt)

        out_path = f"data/datasets/langevin_potential/raw_trajectories/{split}.zarr"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        root = zarr.create_group(out_path, overwrite=True)

        arr = root.create_array(
            "data",
            shape=(n, n_timepoints, d),
            chunks=(n, n_timepoints, d),
            dtype="f4",
        )
        arr.attrs.update({"dims": ["trajectory", "time", "dimension"]})
        root.create_array("time", data=time_vector.astype("f4"))
        root.create_array("param", data=np.ones(n, dtype="f4"))

        arr[:] = data.astype("f4")
        print(f"  saved → {out_path}")


if __name__ == "__main__":
    main()
