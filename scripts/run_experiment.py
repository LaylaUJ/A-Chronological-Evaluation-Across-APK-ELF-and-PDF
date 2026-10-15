"""Initialize a configured TCFS experiment without acquiring data."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tcfs import configure_logging, load_config, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.yaml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--logging-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "logging.yaml",
        help="Path to the logging YAML configuration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and initialize reproducibility controls."""
    args = parse_args(argv)
    configure_logging(args.logging_config, project_root=PROJECT_ROOT)
    config = load_config(args.config)

    experiment = config["experiment"]
    seed_everything(experiment["seed"])

    logger = logging.getLogger(__name__)
    logger.info(
        "Initialized experiment '%s' with seed %d; no data was downloaded.",
        experiment["name"],
        experiment["seed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
