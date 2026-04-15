"""Organize pyqg_unorganized h5 trajectory files into train/test zarr stores.

Each h5 file contains one trajectory:
  data:  (n_time=100, H=128, W=128)  float32

Output zarr layout (mirrors kolflow raw_trajectories):
  data:  (n_traj, n_time, H, W, 1)   float32   chunked (1, n_time, H, W, 1)
  time:  (n_time,)                   float32   linspace(0, 1)

Usage:
  .venv/bin/python scripts/datagen/pyqg_organize.py
  .venv/bin/python scripts/datagen/pyqg_organize.py --train 5000 --test 999 --seed 42
"""

from tqdm import tqdm
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import zarr


SRC = Path("data/datasets/pyqg_unorganized/trajectories")
DST = Path("data/datasets/pyqg/raw_trajectories")

N_TIME = 100
H = 128
W = 128
STATE_SHAPE = (N_TIME, H, W, 1)


def write_split(files: list[Path], out_path: Path) -> None:
    n = len(files)
    out_path.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_path), mode="w")

    data_arr = store.require_array(
        "data",
        shape=(n, *STATE_SHAPE),
        dtype=np.float32,
        chunks=(1, *STATE_SHAPE),
    )
    time_arr = store.require_array(
        "time",
        shape=(N_TIME,),
        dtype=np.float32,
        chunks=(N_TIME,),
    )
    time_arr[:] = np.linspace(0.0, 1.0, N_TIME, dtype=np.float32)

    for i, fp in enumerate(tqdm(files)):
        with h5py.File(fp, "r") as f:
            traj = f["data"][:]          # (100, 128, 128)
        data_arr[i] = traj[:, :, :, np.newaxis]   # (100, 128, 128, 1)

    print(f"Wrote {n} trajectories → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=5000, help="Number of train trajectories")
    parser.add_argument("--test", type=int, default=999, help="Number of test trajectories")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffle")
    args = parser.parse_args()

    all_files = sorted(SRC.glob("*.h5"))
    total = len(all_files)
    print(f"Found {total} h5 files in {SRC}")

    need = args.train + args.test
    if total < need:
        raise ValueError(f"Not enough files: need {need}, found {total}")

    rng = random.Random(args.seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    train_files = shuffled[: args.train]
    test_files = shuffled[args.train : args.train + args.test]

    print(f"Writing train split ({args.train} trajectories)...")
    write_split(train_files, DST / "train.zarr")

    print(f"Writing test split ({args.test} trajectories)...")
    write_split(test_files, DST / "test.zarr")


if __name__ == "__main__":
    main()
