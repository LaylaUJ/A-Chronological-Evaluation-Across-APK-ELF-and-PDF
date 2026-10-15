"""Combine and reduce genuine validation predictions for publication analyses."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs.validation_outputs import choose_budget_predictions


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse validation prediction inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--budget-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write one row-aligned prediction set per method and feature budget."""
    args = parse_args(argv)
    if args.budget_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.budget_output}")
    combined = pd.concat(
        [pd.read_parquet(path) for path in args.predictions], ignore_index=True
    )
    budget = choose_budget_predictions(combined)
    args.budget_output.parent.mkdir(parents=True, exist_ok=True)
    budget.to_parquet(args.budget_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
