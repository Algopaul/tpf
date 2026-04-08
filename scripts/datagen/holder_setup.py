"""Copy holder raw zarrs into the pipeline raw_trajectories layout.

The source data lives at data/datasets/holder/{train,test}.zarr.
This script copies them to raw_trajectories/ so the standard pipeline
steps (01_process_trajectories.py etc.) can find them.
"""

import shutil
from pathlib import Path

BASE = Path("data/datasets/holder")
RAW  = BASE / "raw_trajectories"
RAW.mkdir(exist_ok=True)

for split in ["train", "test"]:
    src = BASE / f"{split}.zarr"
    dst = RAW  / f"{split}.zarr"
    if dst.exists():
        print(f"  {dst} already exists, skipping")
    else:
        shutil.copytree(src, dst)
        print(f"  Copied {src} -> {dst}")
