"""Build Figure 4 data from genuine row-level validation predictions."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs.publication_aggregation import (
    aggregate_ablation_sensitivity,
    save_publication_aggregate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse validation-prediction aggregation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "outputs" / "reports" / "ablation_sensitivity.csv"
        ),
    )
    parser.add_argument("--ablation-baseline", default="full_tcfs")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate validation-only ablation and rho-sensitivity predictions."""
    args = parse_args(argv)
    predictions = _read_table(args.predictions)
    result = aggregate_ablation_sensitivity(
        predictions,
        ablation_baseline=args.ablation_baseline,
        n_bootstrap=args.bootstrap_samples,
        random_state=args.random_state,
    )
    save_publication_aggregate(result, args.output)
    return 0


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
