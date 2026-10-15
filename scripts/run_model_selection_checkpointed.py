"""Run long validation searches with one durable checkpoint per method."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from scripts.run_model_selection import (
    PROJECT_ROOT,
    _load_aligned_development_data,
    _load_configuration,
    _load_feature_names,
    _load_frame,
)

from tcfs import configure_logging
from tcfs.model_checkpoints import (
    merge_method_checkpoints,
    method_checkpoint_complete,
)
from tcfs.modeling import run_development_experiments

METHOD_ORDER = (
    "all_features",
    "mutual_information",
    "l1_logistic",
    "lightgbm_permutation",
    "random_matched",
    "tcfs",
    "shared_tcfs",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse checkpointed model-selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"), required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_selection.yaml",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Resume completed methods, run missing methods, then merge artifacts."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)
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
    rankings = _load_frame(args.rankings)
    rankings = rankings.loc[rankings["file_type"].eq(args.file_type)].copy()
    available_methods = set(rankings["method"])
    methods = tuple(
        method for method in METHOD_ORDER if method in available_methods
    )
    if set(methods) != set(METHOD_ORDER):
        raise ValueError("rankings do not contain every required method")
    configuration = _load_configuration(args.config)
    model_selection = configuration["model_selection"]
    classifier_grids = {
        name: classifier["parameter_grid"]
        for name, classifier in configuration["classifiers"].items()
    }
    for method in methods:
        checkpoint = args.checkpoint_root / method
        if method_checkpoint_complete(checkpoint, method):
            logger.info("Reusing complete %s checkpoint", method)
            continue
        if checkpoint.exists():
            raise RuntimeError(
                f"Incomplete checkpoint requires inspection before retry: {checkpoint}"
            )
        logger.info("Starting durable checkpoint for %s/%s", args.file_type, method)
        results = run_development_experiments(
            X,
            labels,
            assignments,
            feature_names,
            rankings.loc[rankings["method"].eq(method)].copy(),
            classifier_grids=classifier_grids,
            feature_budgets=model_selection["feature_budgets"],
            thresholds=model_selection["thresholds"],
            seeds=model_selection["seeds"],
            output_dir=checkpoint,
            file_type=args.file_type,
        )
        results.selected_configurations.to_csv(
            checkpoint / "selected_configurations.csv", index=False
        )
        if not method_checkpoint_complete(checkpoint, method):
            raise RuntimeError(f"Failed to validate completed checkpoint: {checkpoint}")
        logger.info("Completed durable checkpoint for %s/%s", args.file_type, method)
    merge_method_checkpoints(args.checkpoint_root, args.output_dir, methods)
    logger.info("Merged all %s checkpoints into %s", args.file_type, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
