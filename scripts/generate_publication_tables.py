"""Generate the two manuscript tables from saved QC and metric artifacts."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tcfs.reporting import generate_table_1, generate_table_2

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse publication-table input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-counts", type=Path, required=True)
    parser.add_argument("--check-results", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--noninferiority", type=Path, required=True)
    parser.add_argument("--secondary-comparisons", type=Path, required=True)
    parser.add_argument("--tcfs-method", default="tcfs")
    parser.add_argument("--all-features-method", default="all_features")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "publication",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate both manuscript tables from immutable result files."""
    args = parse_args(argv)
    generate_table_1(args.record_counts, args.check_results, args.output_dir)
    generate_table_2(
        args.metrics,
        args.noninferiority,
        args.secondary_comparisons,
        args.output_dir,
        tcfs_method=args.tcfs_method,
        all_features_method=args.all_features_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
