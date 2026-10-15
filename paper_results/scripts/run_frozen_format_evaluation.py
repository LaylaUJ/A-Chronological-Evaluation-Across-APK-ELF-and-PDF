"""Evaluate every frozen experiment for one format with one sealed-data load."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from scripts.run_final_evaluation import (
    PROJECT_ROOT,
    _load_aligned_final_data,
    _load_feature_names,
    _load_frame,
)

from tcfs import configure_logging
from tcfs.final_evaluation import run_final_evaluation, verify_final_readiness


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one-format frozen evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"), required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--test-labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unseal-test", action="store_true")
    parser.add_argument(
        "--unseal-log",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "test_unseal_log.jsonl",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify all freezes, load sealed rows once, and evaluate each pipeline."""
    args = parse_args(argv)
    if not args.unseal_test:
        raise PermissionError("pass --unseal-test for frozen final evaluation")
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)
    selected_path = args.experiment_dir / "selected_configurations.csv"
    selected = _load_frame(selected_path)
    if "experiment_id" not in selected or selected.empty:
        raise ValueError("selected configurations do not identify experiments")
    experiment_ids = sorted(selected["experiment_id"].astype(str).unique())
    if len(experiment_ids) != len(selected):
        raise ValueError("selected configurations contain duplicate experiment IDs")

    pending = []
    readiness = {}
    for experiment_id in experiment_ids:
        output = args.output_dir / experiment_id
        if output.exists():
            manifest_path = output / "final_evaluation_manifest.json"
            if not manifest_path.is_file():
                raise FileExistsError(
                    f"incomplete prior final-test output blocks rerun: {output}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise FileExistsError(
                    f"failed prior final-test output blocks rerun: {output}"
                )
            logger.info("Keeping completed one-shot result for %s", experiment_id)
            continue
        pending.append(experiment_id)
        readiness[experiment_id] = verify_final_readiness(
            args.experiment_dir / experiment_id, output
        )
    if not pending:
        logger.info("All %s final evaluations are already complete", args.file_type)
        return 0

    assignments = _load_frame(args.assignments)
    feature_names = _load_feature_names(args.feature_names)
    X, labels, assignments = _load_aligned_final_data(
        args.train_features,
        args.train_labels,
        args.test_features,
        args.test_labels,
        _load_frame(args.metadata),
        assignments,
        feature_names,
        args.exclusions,
        args.file_type,
    )
    for experiment_id in pending:
        logger.warning("Starting one-shot final evaluation for %s", experiment_id)
        run_final_evaluation(
            X,
            labels,
            assignments,
            feature_names,
            readiness[experiment_id],
            file_type=args.file_type,
            unseal_test=True,
            unseal_log=args.unseal_log,
            n_bootstrap=args.bootstrap_samples,
            random_state=args.random_state,
        )
        logger.warning("Completed one-shot final evaluation for %s", experiment_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
