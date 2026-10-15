"""Tune development classifiers without accessing official test rows."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tcfs import configure_logging
from tcfs.modeling import run_development_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse model-selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"))
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument(
        "--feature-names",
        type=Path,
        help="JSON list required when --features is a NumPy matrix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "model_selection.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "experiments",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation-only model selection and write experiment artifacts."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)
    feature_names = _load_feature_names(args.feature_names)
    assignments = _load_frame(args.assignments)
    metadata = _load_frame(args.metadata) if args.metadata is not None else None
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
    if "file_type" in rankings:
        if args.file_type is None:
            raise ValueError("--file-type is required for multi-format rankings")
        rankings = rankings.loc[rankings["file_type"].eq(args.file_type)].copy()
    configuration = _load_configuration(args.config)

    model_selection = configuration["model_selection"]
    classifier_grids = {
        name: classifier["parameter_grid"]
        for name, classifier in configuration["classifiers"].items()
    }
    logger.info(
        "Starting validation-only tuning on weeks 41-52; official test data "
        "must not be supplied"
    )
    results = run_development_experiments(
        X,
        labels,
        assignments,
        feature_names,
        rankings,
        classifier_grids=classifier_grids,
        feature_budgets=model_selection["feature_budgets"],
        thresholds=model_selection["thresholds"],
        seeds=model_selection["seeds"],
        output_dir=args.output_dir,
        force=args.force,
        file_type=args.file_type,
    )
    results.selected_configurations.to_csv(
        args.output_dir / "selected_configurations.csv", index=False
    )
    logger.info(
        "Saved %d finalized experiments under %s",
        len(results.selected_configurations),
        args.output_dir,
    )
    return 0


def _load_aligned_development_data(
    feature_path: Path,
    label_path: Path,
    assignments: pd.DataFrame,
    feature_names: list[str],
    *,
    metadata: pd.DataFrame | None,
    exclusions: Path | None,
    file_type: str | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load one format, apply approved DQ003 rows, and omit sealed-test rows."""
    if feature_path.suffix.lower() == ".dat":
        if metadata is None or file_type is None:
            raise ValueError(".dat inputs require --metadata and --file-type")
        format_mask = metadata["file_type"].eq(file_type) & metadata[
            "official_split"
        ].eq("train")
        format_metadata = metadata.loc[format_mask].reset_index(drop=True)
        format_assignments = assignments.loc[format_mask].reset_index(drop=True)
        rows = len(format_metadata)
        dimensions = len(feature_names)
        expected_feature_bytes = rows * dimensions * np.dtype(np.float32).itemsize
        expected_label_bytes = rows * np.dtype(np.int32).itemsize
        if feature_path.stat().st_size != expected_feature_bytes:
            raise ValueError("feature file size disagrees with metadata")
        if label_path.stat().st_size != expected_label_bytes:
            raise ValueError("label file size disagrees with metadata")
        features = np.memmap(
            feature_path, dtype=np.float32, mode="r", shape=(rows, dimensions)
        )
        labels = np.memmap(label_path, dtype=np.int32, mode="r", shape=rows)
        if not np.array_equal(
            labels, format_metadata["label"].to_numpy(dtype=np.int32)
        ):
            raise ValueError("labels are not row-aligned with metadata")
        keep = np.ones(rows, dtype=bool)
        if exclusions is not None:
            ledger = pd.read_csv(exclusions)
            required = {"rule_id", "decision", "official_split", "row_id"}
            if missing := required - set(ledger):
                raise ValueError(
                    f"approved exclusion ledger is missing columns: {sorted(missing)}"
                )
            if set(ledger["rule_id"].astype(str)) != {"DQ003"} or set(
                ledger["decision"].astype(str)
            ) != {"exclude"}:
                raise ValueError("only an approved DQ003 exclusion ledger is accepted")
            excluded_ids = set(
                ledger.loc[
                    ledger["official_split"].eq("train"), "row_id"
                ].astype(int)
            )
            keep &= ~format_metadata["row_id"].isin(excluded_ids).to_numpy()
        development = ~format_assignments["protocol_role"].eq("sealed_test").to_numpy()
        keep &= development
        return (
            np.asarray(features[keep]),
            np.asarray(labels[keep]),
            format_assignments.loc[keep].reset_index(drop=True),
        )

    if feature_path.suffix.lower() == ".npy":
        features = np.load(feature_path, allow_pickle=False)
    else:
        features = _load_frame(feature_path).to_numpy()
    if label_path.suffix.lower() == ".npy":
        labels = np.load(label_path, allow_pickle=False)
    else:
        label_frame = _load_frame(label_path)
        if "label" not in label_frame.columns:
            raise ValueError("tabular labels must contain a 'label' column")
        labels = label_frame["label"].to_numpy()
    return features, labels, assignments


def _load_frame(path: Path | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("tabular path is required")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular file extension: {path.suffix}")


def _load_feature_names(path: Path | None) -> list[str]:
    if path is None:
        raise ValueError("--feature-names is required for NumPy feature matrices")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("feature-names JSON must contain a list of strings")
    return values


def _load_configuration(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    if not isinstance(configuration, dict):
        raise ValueError("model-selection configuration must be a mapping")
    if not isinstance(configuration.get("model_selection"), dict):
        raise ValueError("configuration is missing model_selection")
    if not isinstance(configuration.get("classifiers"), dict):
        raise ValueError("configuration is missing classifiers")
    return configuration


if __name__ == "__main__":
    raise SystemExit(main())
