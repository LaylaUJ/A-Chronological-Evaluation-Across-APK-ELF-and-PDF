"""Evaluate aligned APK, ELF, and PDF predictions with paired bootstrap CIs."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs import configure_logging
from tcfs.evaluation import evaluate_predictions, save_evaluation_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--experiment-metadata", type=Path, required=True)
    parser.add_argument("--feature-results", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tcfs-method", default="tcfs")
    parser.add_argument("--all-features-method", default="all_features")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "evaluation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the per-format evaluation and write machine-readable tables."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)
    predictions = _load_frame(args.predictions)
    metadata = _load_frame(args.experiment_metadata)
    feature_results = (
        None if args.feature_results is None else _load_frame(args.feature_results)
    )
    results = evaluate_predictions(
        predictions,
        metadata,
        n_bootstrap=args.bootstrap_samples,
        random_state=args.random_state,
        tcfs_method=args.tcfs_method,
        all_features_method=args.all_features_method,
        feature_results=feature_results,
    )
    save_evaluation_results(results, args.output_dir)
    logger.info(
        "Saved per-format evaluation with %d paired stratified bootstrap samples "
        "to %s",
        args.bootstrap_samples,
        args.output_dir,
    )
    return 0


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
