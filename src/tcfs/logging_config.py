"""Logging setup driven by a YAML dictionary configuration."""

import logging.config
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def configure_logging(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Configure logging and resolve relative file handlers from project root."""
    path = Path(config_path).resolve()
    root = Path(project_root).resolve() if project_root else path.parent.parent

    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("Logging configuration must contain a YAML mapping.")

    config = deepcopy(loaded)
    handlers = config.get("handlers", {})
    if not isinstance(handlers, dict):
        raise ValueError("Logging 'handlers' must be a mapping.")

    for handler in handlers.values():
        if not isinstance(handler, dict) or "filename" not in handler:
            continue
        filename = Path(handler["filename"])
        if not filename.is_absolute():
            filename = root / filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        handler["filename"] = str(filename)

    logging.config.dictConfig(config)
    return config

