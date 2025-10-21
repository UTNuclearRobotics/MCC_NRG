__all__ = ["load_model"]

def load_model(*args, **kwargs):
    """Lazy import wrapper for the real model loader."""
    from mcc.loader import load_model as _load_model
    return _load_model(*args, **kwargs)