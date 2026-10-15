"""Generate the four manuscript figures from saved analysis artifacts."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tcfs.reporting import (
    generate_figure_1,
    generate_figure_2,
    generate_figure_3,
    generate_figure_4,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse publication-figure input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--jaccard-stability", type=Path, required=True)
    parser.add_argument("--spearman-stability", type=Path, required=True)
    parser.add_argument("--budget-metrics", type=Path, required=True)
    parser.add_argument("--ablation-sensitivity", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "publication",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all manuscript figures without refitting or data access."""
    args = parse_args(argv)
    generate_figure_1(args.assignments, args.output_dir)
    generate_figure_2(
        args.jaccard_stability, args.spearman_stability, args.output_dir
    )
    generate_figure_3(args.budget_metrics, args.output_dir)
    generate_figure_4(args.ablation_sensitivity, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
