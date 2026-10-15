"""Experiment configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML file and validate its required top-level fields."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Configuration must define an 'experiment' mapping.")

    seed = experiment.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("experiment.seed must be a non-negative integer.")

    return config

