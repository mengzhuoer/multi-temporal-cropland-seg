"""配置文件解析与校验"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    data_dir: str = 'data'
    save_dir: str = 'results/checkpoints'
    log_dir: str = 'results/logs'
    model_name: str = 'unet'
    num_classes: int = 5
    num_bands: int = 6
    num_temporal: int = 6
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epochs: int = 100
    num_workers: int = 4
    seed: int = 42
    classes: dict = field(default_factory=lambda: {
        'background': 0,
        'cropland': 1,
        'forest': 2,
        'water': 3,
        'built_up': 4
    })

    @classmethod
    def from_json(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_variant_config(base: TrainConfig, *, num_bands=None,
                          num_temporal=None, model_name=None) -> TrainConfig:
    copy = TrainConfig(**base.__dict__)
    if num_bands is not None:
        copy.num_bands = num_bands
    if num_temporal is not None:
        copy.num_temporal = num_temporal
    if model_name is not None:
        copy.model_name = model_name
    return copy


def output_dir(base_dir: str | Path, model_name: str, num_temporal: int,
               num_bands: int, seed: int) -> Path:
    run_name = f"{model_name}_t{num_temporal}_b{num_bands}_seed{seed}"
    path = Path(base_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path
