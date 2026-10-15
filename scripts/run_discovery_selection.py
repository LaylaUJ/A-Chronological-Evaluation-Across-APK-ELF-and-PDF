"""Fit component selectors in four discovery windows without validation access."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from tcfs import configure_logging
from tcfs.feature_selection import (
    L1LogisticSelector,
    LightGBMPermutationSelector,
    MutualInformationSelector,
    RandomMatchedSelector,
)
from tcfs.temporal_split import PROTOCOL_VERSION, TemporalLeakageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("mutual_information", "l1_logistic", "lightgbm_permutation")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse discovery-selection inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--file-type", choices=("APK", "ELF", "PDF"), required=True)
    parser.add_argument("--feature-budget", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create complete component and control rankings from discovery rows."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml", project_root=PROJECT_ROOT
    )
    logger = logging.getLogger(__name__)
    names = json.loads(args.feature_names.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("feature names must be a JSON list of strings")

    metadata = pd.read_parquet(args.metadata)
    assignments = pd.read_parquet(args.assignments)
    if len(metadata) != len(assignments):
        raise ValueError("metadata and assignments are not aligned")
    versions = set(assignments["protocol_version"].astype(str))
    if versions != {PROTOCOL_VERSION}:
        raise ValueError("assignments do not use the frozen protocol version")
    format_mask = metadata["file_type"].eq(args.file_type) & metadata[
        "official_split"
    ].eq("train")
    format_metadata = metadata.loc[format_mask].reset_index(drop=True)
    format_assignments = assignments.loc[format_mask].reset_index(drop=True)

    dimension = len(names)
    rows = len(format_metadata)
    expected_x_bytes = rows * dimension * np.dtype(np.float32).itemsize
    expected_y_bytes = rows * np.dtype(np.int32).itemsize
    if args.features.stat().st_size != expected_x_bytes:
        raise ValueError("feature file size disagrees with metadata and names")
    if args.labels.stat().st_size != expected_y_bytes:
        raise ValueError("label file size disagrees with metadata")
    X = np.memmap(args.features, dtype=np.float32, mode="r", shape=(rows, dimension))
    y = np.memmap(args.labels, dtype=np.int32, mode="r", shape=rows)
    if not np.array_equal(y, format_metadata["label"].to_numpy(dtype=np.int32)):
        raise ValueError("vectorized labels are not row-aligned with metadata")

    exclusions = pd.read_csv(args.exclusions)
    required_exclusion_columns = {
        "rule_id",
        "official_split",
        "row_id",
        "decision",
        "retain_rule",
        "approved_at_utc",
    }
    if missing := required_exclusion_columns - set(exclusions):
        raise ValueError(
            "exclusions must be an approved ledger; missing columns: "
            f"{sorted(missing)}"
        )
    if set(exclusions["rule_id"].astype(str)) != {"DQ003"}:
        raise ValueError("approved exclusion ledger must contain only DQ003")
    if set(exclusions["decision"].astype(str)) != {"exclude"}:
        raise ValueError("approved exclusion ledger contains a non-exclude decision")
    if set(exclusions["retain_rule"].astype(str)) != {
        "lowest_row_id_within_official_split_and_sha256"
    }:
        raise ValueError("approved exclusion ledger uses an unexpected retain rule")
    excluded_ids = set(
        exclusions.loc[
            exclusions["rule_id"].eq("DQ003")
            & exclusions["official_split"].eq("train"),
            "row_id",
        ].astype(int)
    )
    keep = ~format_metadata["row_id"].isin(excluded_ids).to_numpy()
    if not keep.any():
        raise ValueError("duplicate exclusion rules removed every row")

    score_frames: list[pd.DataFrame] = []
    random_frames: list[pd.DataFrame] = []
    for window in (1, 2, 3, 4):
        window_mask = (
            keep
            & format_assignments["protocol_role"].eq("discovery").to_numpy()
            & format_assignments["discovery_window"].eq(window).to_numpy()
        )
        indices = np.flatnonzero(window_mask)
        if len(indices) == 0:
            raise ValueError(f"discovery window {window} is empty")
        if not set(format_assignments.iloc[indices]["protocol_week"]).isdisjoint(
            range(41, 65)
        ):
            raise TemporalLeakageError("discovery selection reached week 41 or later")
        matrix = np.asarray(X[indices])
        labels = np.asarray(y[indices])
        scaled = StandardScaler().fit_transform(matrix).astype(np.float32, copy=False)
        seed = args.random_state + window
        selectors = {
            "mutual_information": MutualInformationSelector(
                args.feature_budget, random_state=seed
            ),
            "l1_logistic": L1LogisticSelector(
                args.feature_budget, random_state=seed
            ),
            "lightgbm_permutation": LightGBMPermutationSelector(
                args.feature_budget, random_state=seed
            ),
        }
        for selector_name, selector in selectors.items():
            logger.info(
                "Fitting %s for %s discovery window %d on %d rows",
                selector_name,
                args.file_type,
                window,
                len(indices),
            )
            selector_matrix = scaled if selector_name == "l1_logistic" else matrix
            scores = selector.fit(
                selector_matrix, labels, feature_names=names
            ).get_feature_scores()
            scores.insert(0, "selector", selector_name)
            scores.insert(0, "discovery_window", window)
            scores.insert(0, "file_type", args.file_type)
            score_frames.append(scores)

        random_scores = RandomMatchedSelector(
            args.feature_budget, random_state=seed
        ).fit(matrix, labels, feature_names=names).get_feature_scores()
        random_scores.insert(0, "selector", "random_matched")
        random_scores.insert(0, "discovery_window", window)
        random_scores.insert(0, "file_type", args.file_type)
        random_frames.append(random_scores)

    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    scores_path = destination / "selector_scores.parquet"
    controls_path = destination / "control_scores.parquet"
    manifest_path = destination / "discovery_manifest.json"
    if any(path.exists() for path in (scores_path, controls_path, manifest_path)):
        raise FileExistsError("discovery outputs already exist")
    pd.concat(score_frames, ignore_index=True).to_parquet(scores_path, index=False)
    pd.concat(random_frames, ignore_index=True).to_parquet(controls_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "file_type": args.file_type,
                "protocol_version": PROTOCOL_VERSION,
                "feature_budget": args.feature_budget,
                "random_state": args.random_state,
                "component_selectors": list(COMPONENTS),
                "duplicate_exclusion_rule": "DQ003",
                "retained_train_rows": int(keep.sum()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
