"""Create TCFS temporal assignments; optionally authorize final test access."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs import configure_logging
from tcfs.temporal_split import (
    PROTOCOL_VERSION,
    create_temporal_assignments,
    unseal_official_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse temporal-protocol arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ember2024_metadata.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "temporal_assignments.parquet",
    )
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="Authorize official weeks 53-64 for final evaluation and audit-log it.",
    )
    parser.add_argument(
        "--unseal-log",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "test_unseal_log.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write protocol assignments while keeping test rows sealed by default."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml",
        project_root=PROJECT_ROOT,
    )
    logger = logging.getLogger(__name__)
    metadata = pd.read_parquet(args.metadata)
    assignments = create_temporal_assignments(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(args.output, index=False)
    logger.info("Applied temporal protocol %s", PROTOCOL_VERSION)

    if args.unseal_test:
        indices, authorization = unseal_official_test(
            assignments,
            unseal_test=True,
            audit_log=args.unseal_log,
        )
        logger.warning(
            "UNSEALED %d official test rows under %s at %s",
            len(indices),
            authorization.protocol_version,
            authorization.unsealed_at_utc,
        )
    else:
        logger.info(
            "Official test weeks 53-64 remain sealed; use --unseal-test only "
            "for final evaluation"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

