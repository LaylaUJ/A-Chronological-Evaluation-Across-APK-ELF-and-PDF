"""Combine three genuine validation-only TCFS ablation prediction files."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs.validation_outputs import validate_ablation_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ablation prediction paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "validation_ablation_sensitivity_predictions.parquet",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, combine, and save row-aligned Figure 4 predictions."""
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    combined = pd.concat(
        [pd.read_parquet(path) for path in args.predictions], ignore_index=True
    )
    result = validate_ablation_predictions(combined)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
