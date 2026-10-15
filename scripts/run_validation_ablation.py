"""Generate genuine validation predictions for TCFS ablations and rho sensitivity."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.run_model_selection import (
    _load_aligned_development_data,
    _load_feature_names,
    _load_frame,
)

from tcfs.ablation import build_validation_ablation_predictions
from tcfs.final_evaluation import verify_final_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse frozen validation-ablation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--consensus-scores", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the TCFS freeze and fit discovery-only ablation models."""
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"ablation predictions already exist: {args.output}")
    readiness = verify_final_readiness(args.experiment_dir, args.output)
    configuration = readiness.configuration
    if configuration.get("method") != "tcfs":
        raise ValueError("ablation requires a frozen TCFS experiment")
    if configuration.get("classifier") != "lightgbm":
        raise ValueError("ablation requires the frozen primary LightGBM experiment")

    feature_names = _load_feature_names(args.feature_names)
    assignments = _load_frame(args.assignments)
    metadata = _load_frame(args.metadata)
    X, labels, assignments = _load_aligned_development_data(
        args.features,
        args.labels,
        assignments,
        feature_names,
        metadata=metadata,
        exclusions=args.exclusions,
        file_type=args.file_type,
    )
    scores = _load_frame(args.consensus_scores)
    predictions = build_validation_ablation_predictions(
        X,
        labels,
        assignments,
        feature_names,
        scores,
        readiness.frozen_model,
        file_type=args.file_type,
        feature_budget=float(configuration["feature_budget"]),
        selected_rho=float(configuration["rho"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output, index=False)
    manifest = {
        "file_type": args.file_type,
        "rows": len(predictions),
        "source_experiment": str(args.experiment_dir),
        "feature_budget": float(configuration["feature_budget"]),
        "selected_rho": float(configuration["rho"]),
        "protocol_version": configuration["protocol_version"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
