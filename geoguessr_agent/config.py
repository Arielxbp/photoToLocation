import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    pretrained_weights: str = "IMAGENET1K_V2"
    image_size: tuple[int, int] = (320, 180)
    embed_dim: int = 1536
    dropout: float = 0.3
    num_countries: int = 95
    num_regions: int = 1500
    num_continents: int = 7
    region_type: str = "s2_l6"


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.2
    loss_country_weight: float = 0.6
    loss_region_weight: float = 0.3
    loss_continent_weight: float = 0.1
    haversine_temperature: float = 1.0
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 42
    mixed_precision: bool = True
    use_tta: bool = True


@dataclass
class DPOConfig:
    beta: float = 0.02
    learning_rate: float = 1e-5
    buffer_size: int = 10_000
    batch_size: int = 32
    dpo_epochs: int = 1
    fine_tune_every_n_rounds: int = 500
    collect_rounds_per_session: int = 50
    exploration_epsilon: float = 0.1
    validation_enabled: bool = True
    max_degradation_pct: float = 5.0
    dpo_loss_weight: float = 0.1
    sft_loss_weight: float = 1.0


@dataclass
class GameConfig:
    url: str = "https://openguessr.com/"
    headless: bool = False
    stealth: bool = True
    screenshot_dir: str = "data/screenshots"
    viewport_width: int = 1280
    viewport_height: int = 720
    round_timeout_ms: int = 15_000
    guess_delay_ms: tuple[int, int] = (200, 500)
    max_daily_rounds: int = 500
    crop_panorama: bool = True


@dataclass
class DataConfig:
    kaggle_dir: str = "data/kaggle"
    plonkit_cache_dir: str = "data/plonkit_cache"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_mapped_subset: bool = False
    min_images_per_country: int = 50
    filter_duplicate_coords: bool = True
    filter_low_variance: bool = True
    variance_threshold: float = 15.0
    laplacian_threshold: float = 50.0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "geoguessr-agent"
    entity: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    game: GameConfig = field(default_factory=GameConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls()
        if data:
            for section, values in data.items():
                if hasattr(config, section):
                    dc = getattr(config, section)
                    if isinstance(dc, object) and not isinstance(dc, (str, int)):
                        for k, v in values.items():
                            if hasattr(dc, k):
                                setattr(dc, k, v)
        return config

    def to_yaml(self, path: str | Path) -> None:
        data = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if hasattr(value, "__dataclass_fields__"):
                data[field_name] = {
                    k: v for k, v in value.__dict__.items() if not k.startswith("_")
                }
            else:
                data[field_name] = value
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def ensure_dirs(self) -> None:
        for p in [
            self.data.kaggle_dir,
            self.data.plonkit_cache_dir,
            self.data.checkpoint_dir,
            self.data.log_dir,
            self.game.screenshot_dir,
        ]:
            Path(p).mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(path: Optional[str] = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    if os.path.exists(path):
        return Config.from_yaml(path)
    config = Config()
    config.to_yaml(path)
    return config
