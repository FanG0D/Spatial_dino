"""
Utility functions for Spatial-SVG
"""

import importlib
from typing import Dict, Any


def instantiate_from_config(config: Dict[str, Any]):
    """Instantiate a class from a config dict with 'target' and 'params'."""
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string: str, reload: bool = False):
    """Get object from string (module.name)."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def count_params(model):
    """Count trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
