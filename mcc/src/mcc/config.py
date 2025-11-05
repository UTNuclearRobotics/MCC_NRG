from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import yaml
import os

@dataclass
class ModelParams:
    """Define parameters and inference settings for MCC"""
    input_size: int = 224
    occupancy_weight: float = 1.0
    rgb_weight: float = 0.01
    drop_path: float = 0.1
    regress_color: bool = False
    granularity: float = 0.05
    score_thresholds: List[float] = field(default_factory=lambda: [0.3])
    temperature: float = 0.1
    shrink_threshold: float = 10.0
    query_volume: float = 3.0

@dataclass
class MCCConfig:
    model: ModelParams

def load_config(path: str) -> MCCConfig:
    """Load configuration from a YAML file."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as file:
        raw = yaml.safe_load(file)

    model_params = ModelParams(**raw["model"])
    return MCCConfig(model=model_params)