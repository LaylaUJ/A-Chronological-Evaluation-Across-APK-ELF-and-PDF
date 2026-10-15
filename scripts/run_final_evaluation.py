"""Run one audited evaluation of a frozen pipeline on official test weeks."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tcfs import configure_logging
from tcfs.final_evaluation import run_final_evaluation, verify_final_readiness

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately explicit final-evaluation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="Required acknowledgement for the one official-test evaluation.",
    )
    parser.add_argument(
        "--unseal-log",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "test_unseal_log.jsonl",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the freeze before loading test data, then execute exactly once."""
    args = parse_args(argv)
    if not args.unseal_test:
        raise PermissionError(
            "Final test data remain sealed; pass --unseal-test for the single "
            "frozen evaluation."
        )
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)

    # This preflight intentionally occurs before any feature or label file is read.
    readiness = verify_final_readiness(args.experiment_dir, args.output_dir)
    logger.warning(
        "Freeze verified for %s; loading official-test inputs for one evaluation",
        args.experiment_dir,
    )
    assignments = _load_frame(args.assignments)
    if args.features.suffix.lower() == ".dat":
        feature_names = _load_feature_names(args.feature_names)
        if any(
            value is None
            for value in (
                args.test_features,
                args.test_labels,
                args.metadata,
                args.exclusions,
            )
        ):
            raise ValueError(
                ".dat inputs require --test-features, --test-labels, "
                "--metadata, and --exclusions"
            )
        X, labels, assignments = _load_aligned_final_data(
            args.features,
            args.labels,
            args.test_features,
            args.test_labels,
            _load_frame(args.metadata),
            assignments,
            feature_names,
            args.exclusions,
            args.file_type,
        )
    else:
        X, inferred_names = _load_features(args.features)
        feature_names = inferred_names or _load_feature_names(args.feature_names)
        labels = _load_labels(args.labels)
    output = run_final_evaluation(
        X,
        labels,
        assignments,
        feature_names,
        readiness,
        file_type=args.file_type,
        unseal_test=True,
        unseal_log=args.unseal_log,
        n_bootstrap=args.bootstrap_samples,
        random_state=args.random_state,
    )
    logger.warning("Final evaluation complete; immutable output written to %s", output)
    return 0


def _load_features(path: Path) -> tuple[np.ndarray, list[str] | None]:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False), None
    frame = _load_frame(path)
    return frame.to_numpy(), [str(column) for column in frame.columns]


def _load_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    frame = _load_frame(path)
    if "label" not in frame.columns:
        raise ValueError("tabular labels must contain a 'label' column")
    return frame["label"].to_numpy()


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def _load_feature_names(path: Path | None) -> list[str]:
    if path is None:
        raise ValueError("--feature-names is required for NumPy feature matrices")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise ValueError("feature-names JSON must contain a list of strings")
    return values


def _load_aligned_final_data(
    train_feature_path: Path,
    train_label_path: Path,
    test_feature_path: Path | None,
    test_label_path: Path | None,
    metadata: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_names: list[str],
    exclusions_path: Path | None,
    file_type: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load pre-vectorized splits after freeze verification and apply DQ003."""
    if test_feature_path is None or test_label_path is None or exclusions_path is None:
        raise ValueError("final .dat inputs and exclusions must be supplied")
    required_metadata = {"row_id", "file_type", "official_split", "label"}
    if missing := required_metadata - set(metadata):
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if len(metadata) != len(assignments) or not np.array_equal(
        metadata["row_id"].to_numpy(), assignments["row_id"].to_numpy()
    ):
        raise ValueError("metadata and assignments are not row-aligned")
    ledger = pd.read_csv(exclusions_path)
    required_ledger = {"rule_id", "decision", "official_split", "row_id"}
    if missing := required_ledger - set(ledger):
        raise ValueError(f"exclusion ledger is missing columns: {sorted(missing)}")
    if set(ledger["rule_id"].astype(str)) != {"DQ003"} or set(
        ledger["decision"].astype(str)
    ) != {"exclude"}:
        raise ValueError("only the approved DQ003 exclusion ledger is accepted")

    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    assignment_frames: list[pd.DataFrame] = []
    dimensions = len(feature_names)
    paths = {
        "train": (train_feature_path, train_label_path),
        "test": (test_feature_path, test_label_path),
    }
    for split in ("train", "test"):
        mask = metadata["file_type"].eq(file_type) & metadata[
            "official_split"
        ].eq(split)
        split_metadata = metadata.loc[mask].reset_index(drop=True)
        split_assignments = assignments.loc[mask].reset_index(drop=True)
        rows = len(split_metadata)
        feature_path, label_path = paths[split]
        expected_feature_bytes = rows * dimensions * np.dtype(np.float32).itemsize
        expected_label_bytes = rows * np.dtype(np.int32).itemsize
        if feature_path.stat().st_size != expected_feature_bytes:
            raise ValueError(f"{split} feature file size disagrees with metadata")
        if label_path.stat().st_size != expected_label_bytes:
            raise ValueError(f"{split} label file size disagrees with metadata")
        matrix = np.memmap(
            feature_path, dtype=np.float32, mode="r", shape=(rows, dimensions)
        )
        split_labels = np.memmap(
            label_path, dtype=np.int32, mode="r", shape=rows
        )
        if not np.array_equal(
            split_labels, split_metadata["label"].to_numpy(dtype=np.int32)
        ):
            raise ValueError(f"{split} labels are not row-aligned with metadata")
        excluded_ids = set(
            ledger.loc[ledger["official_split"].eq(split), "row_id"].astype(int)
        )
        keep = ~split_metadata["row_id"].isin(excluded_ids).to_numpy()
        matrices.append(np.asarray(matrix[keep]))
        labels.append(np.asarray(split_labels[keep]))
        assignment_frames.append(split_assignments.loc[keep].reset_index(drop=True))
    return (
        np.concatenate(matrices),
        np.concatenate(labels),
        pd.concat(assignment_frames, ignore_index=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
