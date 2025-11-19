from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class PipelineConfig:
    data_dir: str
    output_dir: str
    image_size: Tuple[int, int] = (256, 256)
    train_ratio: float = 0.7
    seed: int = 42
    batch_size: int = 32
    epochs: int = 75
    learning_rate: float = 1e-4
    resnet_variant: str = "resnet50v2"
    freeze_backbone: bool = True
    early_stopping_patience: int = 5
    model_dir: str = "models"
    plots_dir: str = "plots"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    use_dataset_cache: bool = True
    cache_dir: str = "processed/cache"
    use_mixed_precision: bool = True
    framework: str = "tf"
    num_workers: int = 4
    amp_dtype: str = "bf16"
    min_images_harmful: int = 175


class ConfigLoader:
    def __init__(self, config_path: Path):
        self.config_path = config_path

    def load(self) -> PipelineConfig:
        import yaml
        if not self.config_path.exists():
            raise FileNotFoundError(str(self.config_path))
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return PipelineConfig(**cfg)