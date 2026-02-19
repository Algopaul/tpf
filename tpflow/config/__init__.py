from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class DataConfig:
  name: str = 'gaurot'
  blocksize: int = 64


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


cs = ConfigStore.instance()
cs.store(name='config', node=TrajectoryProcessing)
