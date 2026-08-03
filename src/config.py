import argparse
from pathlib import Path

import yaml

BASE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def load_config(experiment_config_path=None):
    """Load configs/base.yaml, then apply overrides from an experiment yaml file.

    Returns an argparse.Namespace so downstream code (training.py) can keep
    using `args.foo` attribute access, same as the original notebook.
    """
    with open(BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    if experiment_config_path is not None:
        with open(experiment_config_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg.update(overrides)

    return argparse.Namespace(**cfg)
