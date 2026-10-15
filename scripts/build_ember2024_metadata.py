"""Build row-aligned Parquet metadata for downloaded EMBER2024 features."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tcfs import configure_logging
from tcfs.metadata import build_metadata_parquet, write_alignment_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse metadata builder arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ember2024" / "extracted",
        help="Directory containing train/ and test/ JSONL shard trees.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ember2024_metadata.parquet",
        help="Destination Parquet file.",
    )
    parser.add_argument("--x-train", type=Path, help="Optional train X .dat/.npy/.npz.")
    parser.add_argument("--x-test", type=Path, help="Optional test X .dat/.npy/.npz.")
    parser.add_argument(
        "--feature-dim",
        type=int,
        help="Feature columns; required when checking thrember .dat matrices.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create Parquet metadata and its label-free alignment report."""
    args = parse_args(argv)
    if (args.x_train is None) != (args.x_test is None):
        raise ValueError("--x-train and --x-test must be supplied together")

    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml",
        project_root=PROJECT_ROOT,
    )
    vectorized = None
    if args.x_train is not None:
        vectorized = {"train": args.x_train, "test": args.x_test}

    report = build_metadata_parquet(
        args.raw_dir,
        args.output,
        vectorized_features=vectorized,
        feature_dim=args.feature_dim,
    )
    report_path = args.output.with_suffix(".alignment.json")
    write_alignment_report(report, report_path)
    logging.getLogger(__name__).info(
        "Wrote %d row-aligned records to %s",
        sum(report.metadata_rows.values()),
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

