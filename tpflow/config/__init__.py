from dataclasses import dataclass, field

from flanch.config import MLPConfig, OptimizerConfig, UNetConfig
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
  name: str = 'gaurot'
  type: str = 'hist'
  batch_size: int = 100_000
  block_size: int = 10_000
  fields: tuple = ('data', 'time')


@dataclass
class WandbConfig:
  mode: str = 'online'
  jobname: str = ''
  group: str = ''
  tag: str = ''


@dataclass
class TrajectoryProcessing:
  tag: str = 'default'
  data: DataConfig = field(default_factory=DataConfig)
  dryrun: bool = False
  wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class InferenceConfig:
  n_samples: int = 256
  n_sampling_steps: int = 128
  n_param_steps: int = 64


@dataclass
class CFMTraining:
  model_type: str = 'mlp'
  mlp: MLPConfig = field(default_factory=MLPConfig)
  unet: UNetConfig = field(default_factory=UNetConfig)
  opt: OptimizerConfig = field(default_factory=OptimizerConfig)
  data: DataConfig = field(default_factory=DataConfig)
  wandb: WandbConfig = field(default_factory=WandbConfig)
  inference: InferenceConfig = field(default_factory=InferenceConfig)
  eval_interval: int = 50


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
cs.store(name='config', node=TrajectoryProcessing)
cs.store(name='cfm', node=CFMTraining)
cs.store(
    name='imgrot',
    node=CFMTraining(
        model_type='unet',
        unet=UNetConfig(
            channels_inout=1, base_ch=64, use_attn=tuple(4 * [False])),
        opt=OptimizerConfig(learning_rate=1e-4, epochs=1_000),
        data=DataConfig(
            'imgrot',
            type='field',
            batch_size=128,
            block_size=8,
        ),
    ),
)

cs.store(group='unet', name='deep', node=unet_deep)
cs.store(group='unet', name='mid', node=unet_mid)
cs.store(group='unet', name='xs', node=unet_xs)
