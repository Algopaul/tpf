"""Split data/datasets/bump/raw_trajectories/train.zarr 50/50 into train/test."""
import numpy as np
import zarr

src = zarr.open("data/datasets/bump/raw_trajectories/train.zarr", mode="r")
data  = np.array(src["data"])    # (25000, 128, 2)
param = np.array(src["param"])
time  = np.array(src["time"])

n = len(data) // 2  # 12500 each
for name, sl in [("train", slice(None, n)), ("test", slice(n, None))]:
    g = zarr.open_group(f"data/datasets/bump/raw_trajectories/{name}.zarr", mode="w")
    g.create_array("data",  data=data[sl],  chunks=(256, data.shape[1], 2))
    g.create_array("param", data=param[sl], chunks=(256,))
    g.create_array("time",  data=time)
    print(f"{name}: {data[sl].shape}")
