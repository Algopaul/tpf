from dataclasses import dataclass, field

from flanch.config import MLPConfig, OptimizerConfig, UNetConfig
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
    name: str = "gaurot"
    type: str = "hist"
    batch_size: int = 100_000
    block_size: int = 10_000
    trajectory_block_size: int = 8
    fields: tuple = ("data", "time")


@dataclass
class WandbConfig:
    mode: str = "online"
    jobname: str = ""
    group: str = ""
    tag: str = ""


@dataclass
class WDSConvertConfig:
    data: DataConfig = field(default_factory=DataConfig)
    splits: tuple = ("train", "test")
    blocks_per_shard: int = 500
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class TrajectoryProcessing:
    tag: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    dryrun: bool = False
    # Normalize each time step independently: one mean/std per (time, channel),
    # averaging over spatial dims.  Useful when field amplitude grows strongly over
    # the conditioning axis (e.g. hw2d).  Stored in train.zarr attrs as
    # per_time_mean / per_time_std of shape (n_time, C).
    normalize_per_time: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class InferenceConfig:
    n_samples: int = 256
    n_sampling_steps: int = 128
    n_param_steps: int = 64


@dataclass
class EnergySpectraConfig:
    n_bins: int = 32
    log_bins: bool = False
    # -1 means the state is already a plain 2-D (H, W) field with no channel axis.
    # Set to 0 (or any valid axis index) when state_shape is (H, W, C) or (C, H, W).
    channel_axis: int = -1
    channel_idx: int = 0


@dataclass
class RegressionTraining:
    model_type: str = "mlp"
    mlp: MLPConfig = field(default_factory=MLPConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataset: str = ""  # dataset name included in wandb run title
    train_data: str = "MISSING"  # path to training regression zarr
    val_data: str = "MISSING"  # path to validation regression zarr
    rollout_data: str = "MISSING"  # path to trajectory zarr for rollout eval
    # Path to cfm_train_data/train.zarr; when set, raw rollout_data trajectories
    # are normalised with the stored data_mean/data_std before evaluation.
    norm_stats_path: str = ""
    batch_size: int = 100_000
    block_size: int = 10_000
    mode: str = "step"  # 'step' or 'difference'
    time_conditioned: bool = True
    cond_start: float = 0.0  # conditioning range used in step 03; rollout sweeps this range
    cond_end: float = 1.0
    eval_interval: int = 50
    n_rollout: int = 16
    zero_mean_rollout: bool = False  # subtract spatial mean after each step (e.g. vorticity)
    data_type: str = "hist"  # 'hist' or 'field'
    # Scalar statistics to compare in rollout eval.  Any subset of:
    # "first_moment", "enstrophy", "kurtosis"
    stats: tuple = ("first_moment", "enstrophy", "kurtosis")
    # Set True to also compare radially-averaged energy spectra (field data only).
    log_energy_spectra: bool = False
    energy_spectra: EnergySpectraConfig = field(default_factory=EnergySpectraConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    restart_from: str = ""  # path to a {run_dir}/{epoch}/ checkpoint dir; "" = fresh start


@dataclass
class BenchmarkConfig:
    dataset: str = "MISSING"
    loader: str = "cfm"        # "cfm" or "regression"
    model: str = "model1"      # model name for regression data path
    n_batches: int = 80
    batch_size: int = 512
    block_size: int = 32
    prefetch: int = 2
    compute_ms: float = 0.0    # simulate GPU step of this many ms
    compare_prefetch: bool = False  # sweep prefetch=2,4,8
    inspect: bool = False      # metadata only, no data loaded
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class RegressionDataConfig:
    input: str = "MISSING"  # path to input .zarr
    output: str = "regression_data.zarr"
    dataset: str = ""  # dataset name included in wandb run title
    block_size: int = 0  # 0 = auto: targets ~2 MB zarr chunks based on state_shape
    trajectory_block_size: int = 0  # 0 = auto: targets ~2 MB input buffer
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class CondTrajConfig:
    checkpoint: str = "MISSING"  # path to {run_dir}/{epoch}/
    dataset: str = ""  # dataset name included in wandb run title
    n_samples: int = 256  # total source noise vectors
    batch_size: int = 256  # samples per forward pass (memory limit)
    seed: int = 0
    n_cond_steps: int = 32  # number of conditioning values
    cond_start: float = 0.0
    cond_end: float = 1.0
    n_ode_steps: int = 128  # RK4 integration steps per sample
    output: str = "cond_trajectories.zarr"
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class ExportEvalConfig:
    checkpoint: str = "MISSING"  # path to {run_dir}/{epoch}/ checkpoint directory
    rollout_data: str = ""        # override rollout_data path (regression only)
    eval_dir: str = ""            # override output dir; default: {checkpoint}/eval/


@dataclass
class CFMTraining:
    model_type: str = "mlp"
    mlp: MLPConfig = field(default_factory=MLPConfig)
    unet: UNetConfig = field(default_factory=UNetConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    conditioning_reg: float = 0.0
    conditioning_stepsize: float = 1e-4
    eval_interval: int = 50
    restart_from: str = ""  # path to a {run_dir}/{epoch}/ checkpoint dir; "" = fresh start


unet_deep = UNetConfig(
    channels_inout=1,
    base_ch=64,
    use_attn=tuple(6 * [False]),
    head_dim_multipliers=tuple(6 * [1]),
    head_multipliers=tuple(6 * [1]),
    channel_multipliers=(1, 2, 2, 4, 4, 8),
    strides=(1, 1, 1, 2, 2, 2),
)

unet_mid = UNetConfig(
    channels_inout=1,
    base_ch=64,
    use_attn=tuple(5 * [False]),
    head_dim_multipliers=tuple(5 * [1]),
    head_multipliers=tuple(5 * [1]),
    channel_multipliers=(1, 2, 4, 4, 8),
    strides=(1, 1, 1, 2, 2),
)

unet_xs = UNetConfig(
    channels_inout=1,
    base_ch=1,
    use_attn=tuple(2 * [False]),
    head_dim_multipliers=tuple(2 * [1]),
    head_multipliers=tuple(2 * [1]),
    channel_multipliers=(1, 2),
    strides=(8, 4),
)

cs = ConfigStore.instance()
cs.store(name="export_eval", node=ExportEvalConfig)
cs.store(name="benchmark", node=BenchmarkConfig)
cs.store(name="wds_convert", node=WDSConvertConfig)
cs.store(name="cond_traj", node=CondTrajConfig)
cs.store(name="config", node=TrajectoryProcessing)
cs.store(name="cfm", node=CFMTraining)
cs.store(name="regression_data", node=RegressionDataConfig)
cs.store(name="regression", node=RegressionTraining)
cs.store(
    name="imgrot",
    node=CFMTraining(
        model_type="unet",
        unet=UNetConfig(channels_inout=1, base_ch=64, use_attn=tuple(4 * [False])),
        opt=OptimizerConfig(learning_rate=1e-4, epochs=1_000),
        data=DataConfig(
            "imgrot",
            type="field",
            batch_size=128,
            block_size=8,
        ),
    ),
)

cs.store(group="unet", name="deep", node=unet_deep)
cs.store(group="unet", name="mid", node=unet_mid)
cs.store(group="unet", name="xs", node=unet_xs)
