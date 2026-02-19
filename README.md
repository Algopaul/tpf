## Dataset storage
## Dataset  structure

```
data/datasets/{name}/
├── raw_trajectories/
│   ├── train.zarr (data: (n_trajectories, n_time, *state_shape), time: (n_time,), param: (n_trajectories,))
│   └── test.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│
├── cfm_train_data/ # here n_samples = n_trajectories*n_time
│   ├── train.zarr (data: (n_samples, *state_shape), time: (n_samples,), param: (n_samples,))
│   └── test.zarr (data: (n_samples, *state_shape), time: (n_samples,), param: (n_samples,))
│
├── cfm_trajectories/
│   ├── model_1.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│   └── model_2.zarr (data: (n_trajectories, n_time, *state_shape), param: (n_trajectories,))
│
└── stats/
    ├── train/ (n_time, stat_shape)
    │   ├── e.g. enstrophy (mean: n_time, var: n_time)
    │   └── e.g. energy (mean: n_time, var: n_time)
    └── test/
```
