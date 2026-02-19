from dataclasses import dataclass, field

from flanch.config import MLPConfig, OptimizerConfig
from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
  name: str = 'gaurot'
  type: str = 'hist'
  batch_size: int = 100_000
  fields: tuple = ('data', 'time')
  gen_size: int = 100_000


@dataclass
class WandbConfig:
  mode: str = 'online'
  jobname: str = ''
  group: str = ''


@dataclass
class TrajectoryProcessing:
  tag: str = 'default'
  data: DataConfig = field(default_factory=DataConfig)
  dryrun: bool = False
  block_size: int = 64
  wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class CFMTraining:
  model_type: str = 'mlp'
  mlp: MLPConfig = field(default_factory=MLPConfig)
  opt: OptimizerConfig = field(default_factory=OptimizerConfig)
  data: DataConfig = field(default_factory=DataConfig)
  wandb: WandbConfig = field(default_factory=WandbConfig)
  eval_interval: int = 50


cs = ConfigStore.instance()
cs.store(name='config', node=TrajectoryProcessing)
cs.store(name='cfm', node=CFMTraining)
