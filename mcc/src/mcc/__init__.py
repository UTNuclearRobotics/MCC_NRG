# src/mcc/__init__.py
from typing import Any

__all__ = ["load_model", "MCCPredictor"]  # add "MCCModel" if desired

def __getattr__(name: str) -> Any:
    if name == "load_model":
        from mcc.loader import load_model as _load_model
        return _load_model
    if name == "MCCPredictor":
        from mcc.predictor import MCCPredictor as _MCCPredictor
        return _MCCPredictor
    raise AttributeError(f"module 'mcc' has no attribute {name}")
