"""Run non-destructive quality control on EMBER2024 metadata and features."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tcfs import configure_logging
from tcfs.data_quality import run_data_quality

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse quality-control command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ember2024_metadata.parquet",
        help="Row-aligned metadata Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "data_quality",
        help="Directory for CSV, JSON, and Markdown reports.",
    )
    parser.add_argument("--x-train", type=Path, help="Optional train feature matrix.")
    parser.add_argument("--x-test", type=Path, help="Optional test feature matrix.")
    parser.add_argument(
        "--feature-dim",
        type=int,
        help="Number of columns for raw thrember .dat feature matrices.",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        help="Optional text file containing one feature name per line.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and report its output directory."""
    args = parse_args(argv)
    if (args.x_train is None) != (args.x_test is None):
        raise ValueError("--x-train and --x-test must be supplied together")
    features = None
    if args.x_train is not None:
        features = {"train": args.x_train, "test": args.x_test}
    names = None
    if args.feature_names is not None:
        names = args.feature_names.read_text(encoding="utf-8").splitlines()

    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml",
        project_root=PROJECT_ROOT,
    )
    result = run_data_quality(
        args.metadata,
        args.output_dir,
        feature_paths=features,
        feature_dim=args.feature_dim,
        feature_names=names,
    )
    logging.getLogger(__name__).info(
        "Audited %d records; wrote reports to %s; deleted 0 records",
        result.input_rows,
        result.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

