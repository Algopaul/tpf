from tpflow.w2 import compare_w2, load_rollout, plot_w2_comparison

results = compare_w2(
    {
        "OT regression": "multirun/2026-03-26/10-00-00/0/50/eval/rollout.zarr",
        "CFM regression": "multirun/2026-03-26/11-00-00/0/50/eval/rollout.zarr",
        "colleague baseline": "data/external/colleague_rollout.zarr",  # needs same zarr layout
    },
    stride=10,
    epsilon=1e-3,
)
