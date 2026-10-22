"""YAML config loading for the per-model training configs (configs/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelCfg:
    name: str
    pretrained: str
    native_resolution: int
    sampler: str = "ddim"


@dataclass
class LoraCfg:
    rank: int = 8
    alpha: int = 8
    dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])


@dataclass
class TrainCfg:
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 10000
    seed: int = 0
    mixed_precision: str = "fp16"
    conditioning_dropout: float = 0.1
    gradient_checkpointing: bool = False
    use_8bit_adam: bool = False
    # Equalise per-batch class frequency (LungHist700 is ~1.08:1 ADC:SCC; kept on for
    # parity with the original 9.6:1 protocol). Set False to fall back to uniform shuffling (for ablation).
    balanced_sampling: bool = True


@dataclass
class DataCfg:
    cache_latents: bool = True
    cache_text_embeddings: bool = True


@dataclass
class Config:
    model: ModelCfg
    lora: LoraCfg
    training: TrainCfg
    data: DataCfg
    fits_local_8gb: bool = True


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        model=ModelCfg(**raw["model"]),
        lora=LoraCfg(**raw.get("lora", {})),
        training=TrainCfg(**raw.get("training", {})),
        data=DataCfg(**raw.get("data", {})),
        fits_local_8gb=raw.get("hardware", {}).get("fits_local_8gb", True),
    )
