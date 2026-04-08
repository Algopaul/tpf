"""Convert data/dice_*.npy files to rollout.zarr format for W2 comparison.

Input shape:  (n_particles, n_time, 2)  float32
Output zarr:  rollout  (n_time, n_particles, 2)
              reference (n_time, n_particles, 2)
              time      (n_time,)

Usage:
    python dice_npy_to_zarr.py
"""

import numpy as np
import zarr
from pathlib import Path

GEN_PATH  = Path("data/dice_gen_analytic.npy")
DATA_PATH = Path("data/dice_data_analytic.npy")
OUT_PATH  = Path("data/dice_rollout.zarr")

gen  = np.load(GEN_PATH)   # (n_particles, n_time, 2)
data = np.load(DATA_PATH)  # (n_particles, n_time, 2)

# Transpose to (n_time, n_particles, 2)
rollout   = np.transpose(gen,  (1, 0, 2)).astype(np.float32)
reference = np.transpose(data, (1, 0, 2)).astype(np.float32)
n_time    = rollout.shape[0]
time      = np.linspace(0.0, 1.0, n_time, dtype=np.float32)

print(f"rollout:   {rollout.shape}")
print(f"reference: {reference.shape}")
print(f"time:      {time.shape}  [{time[0]:.4f} … {time[-1]:.4f}]")

store = zarr.open(str(OUT_PATH), mode="w")
store["rollout"]   = rollout
store["reference"] = reference
store["time"]      = time
print(f"Written to {OUT_PATH}")
