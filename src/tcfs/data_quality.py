"""Non-destructive data-quality checks for row-aligned EMBER2024 data."""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcfs.metadata import ALLOWED_FORMATS, OFFICIAL_SPLITS

DATASET_START = pd.Timestamp("2023-09-24", tz="UTC")
DATASET_END = pd.Timestamp("2024-12-14 23:59:59", tz="UTC")

RULES = {
    "DQ001": "record counts",
    "DQ002": "duplicate row IDs",
    "DQ003": "exact SHA-256 duplicates within a split",
    "DQ004": "cross-split duplicate SHA-256 hashes",
    "DQ005": "missing feature values",
    "DQ006": "infinite feature values",
    "DQ007": "constant features",
    "DQ008": "inconsistent file types",
    "DQ009": "impossible timestamps",
    "DQ010": "supplied/derived week mismatch",
    "DQ011": "class prevalence over time",
}

EXCLUSION_COLUMNS = [
    "rule_id",
    "official_split",
    "row_id",
    "sha256",
    "reason",
    "proposed_action",
]
FEATURE_QUALITY_COLUMNS = [
    "feature_index",
    "feature_name",
    "missing_count",
    "infinite_count",
    "finite_min",
    "finite_max",
    "constant",
    "rule_ids",
    "proposed_action",
]


@dataclass(frozen=True)
class CheckResult:
    """Machine-readable status for one quality-control rule."""

    rule_id: str
    check: str
    status: str
    findings: int | None
    note: str


@dataclass(frozen=True)
class QualityControlResult:
    """Paths and aggregate counts from a completed audit."""

    input_rows: int
    proposed_row_exclusions: int
    proposed_feature_exclusions: int
    checks: list[CheckResult]
    output_dir: str


def run_data_quality(
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    feature_paths: Mapping[str, str | Path] | None = None,
    feature_dim: int | None = None,
    feature_names: list[str] | None = None,
    chunk_rows: int = 10_000,
) -> QualityControlResult:
    """Run all checks and write reports without modifying source records."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    metadata_file = Path(metadata_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(metadata_file)
    _validate_metadata_schema(metadata)
    input_rows = len(metadata)

    record_counts = _record_counts(metadata)
    prevalence = _class_prevalence(metadata)
    duplicate_row_ids = metadata.loc[
        metadata.duplicated(["official_split", "row_id"], keep=False)
    ].sort_values(["official_split", "row_id"])
    duplicate_hashes = _duplicate_hash_summary(metadata)
    cross_split_hashes = duplicate_hashes.loc[duplicate_hashes["split_count"] > 1]

    exclusion_parts = [
        _duplicate_row_id_exclusions(duplicate_row_ids),
        _duplicate_hash_exclusions(metadata),
        _cross_split_exclusions(metadata, cross_split_hashes),
    ]

    inconsistent_mask = _inconsistent_file_type_mask(metadata)
    exclusion_parts.append(
        _row_exclusions(
            metadata.loc[inconsistent_mask],
            "DQ008",
            "Invalid file_type or one SHA-256 is assigned multiple file types",
        )
    )
    timestamp_mask, timestamp_reasons = _impossible_timestamp_issues(metadata)
    exclusion_parts.append(
        _row_exclusions(
            metadata.loc[timestamp_mask],
            "DQ009",
            timestamp_reasons.loc[timestamp_mask],
        )
    )
    week_mask = (
        metadata["week_id"].notna()
        & metadata["derived_week_id"].notna()
        & (metadata["week_id"] != metadata["derived_week_id"])
    )
    exclusion_parts.append(
        _row_exclusions(
            metadata.loc[week_mask],
            "DQ010",
            "Supplied week_id differs from derived_week_id; review temporal assignment",
        )
    )

    feature_quality = pd.DataFrame(columns=FEATURE_QUALITY_COLUMNS)
    feature_checks = _not_run_feature_checks()
    if feature_paths is not None:
        missing_rows, infinite_rows, feature_quality = _scan_features(
            metadata,
            feature_paths,
            feature_dim=feature_dim,
            feature_names=feature_names,
            chunk_rows=chunk_rows,
        )
        exclusion_parts.extend(
            [
                _feature_row_exclusions(metadata, missing_rows, "DQ005"),
                _feature_row_exclusions(metadata, infinite_rows, "DQ006"),
            ]
        )
        feature_checks = [
            _check(
                "DQ005",
                len(missing_rows),
                "Rows containing one or more NaN values",
            ),
            _check(
                "DQ006",
                len(infinite_rows),
                "Rows containing one or more positive/negative infinity values",
            ),
            _check(
                "DQ007",
                int(feature_quality["constant"].sum()),
                "Features constant across all supplied splits",
            ),
        ]

    exclusions = pd.concat(exclusion_parts, ignore_index=True)
    if exclusions.empty:
        exclusions = pd.DataFrame(columns=EXCLUSION_COLUMNS)
    else:
        exclusions = exclusions[EXCLUSION_COLUMNS].sort_values(
            ["rule_id", "official_split", "row_id"],
            kind="stable",
        )

    checks = [
        CheckResult(
            "DQ001",
            RULES["DQ001"],
            "PASS",
            len(record_counts),
            f"{input_rows} records summarized; no records removed",
        ),
        _check("DQ002", len(duplicate_row_ids), "Rows with non-unique split/row_id"),
        _check(
            "DQ003",
            len(duplicate_hashes),
            "SHA-256 groups occurring more than once",
        ),
        _check(
            "DQ004",
            len(cross_split_hashes),
            "SHA-256 groups present in both train and test",
        ),
        *feature_checks,
        _check(
            "DQ008",
            int(inconsistent_mask.sum()),
            "Rows with invalid or hash-inconsistent file types",
        ),
        _check("DQ009", int(timestamp_mask.sum()), "Rows with impossible timestamps"),
        _check(
            "DQ010",
            int(week_mask.sum()),
            "Rows where supplied and derived week IDs differ",
        ),
        CheckResult(
            "DQ011",
            RULES["DQ011"],
            "PASS",
            len(prevalence),
            "Prevalence groups written without modifying records",
        ),
    ]

    _write_csv(record_counts, destination / "record_counts.csv")
    _write_csv(prevalence, destination / "class_prevalence_over_time.csv")
    _write_csv(duplicate_row_ids, destination / "duplicate_row_ids.csv")
    _write_csv(duplicate_hashes, destination / "duplicate_hashes.csv")
    _write_csv(cross_split_hashes, destination / "cross_split_hashes.csv")
    _write_csv(feature_quality, destination / "feature_quality.csv")
    _write_csv(exclusions, destination / "proposed_exclusions.csv")
    _write_csv(
        pd.DataFrame([asdict(check) for check in checks]),
        destination / "check_results.csv",
    )

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "metadata_path": str(metadata_file.resolve()),
        "input_rows": input_rows,
        "output_rows": input_rows,
        "records_deleted": 0,
        "proposed_row_exclusion_entries": len(exclusions),
        "proposed_feature_exclusions": int(
            (feature_quality["proposed_action"] == "exclude_feature").sum()
        ),
        "checks": [asdict(check) for check in checks],
        "rules": RULES,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (destination / "quality_control_report.md").write_text(
        _markdown_report(summary, checks),
        encoding="utf-8",
    )

    return QualityControlResult(
        input_rows=input_rows,
        proposed_row_exclusions=len(exclusions),
        proposed_feature_exclusions=summary["proposed_feature_exclusions"],
        checks=checks,
        output_dir=str(destination),
    )


def _validate_metadata_schema(metadata: pd.DataFrame) -> None:
    required = {
        "row_id",
        "sha256",
        "file_type",
        "label",
        "first_submission_date",
        "last_analysis_date",
        "official_split",
        "week_id",
        "derived_week_id",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    invalid_splits = set(metadata["official_split"].dropna().unique()) - set(
        OFFICIAL_SPLITS
    )
    if invalid_splits:
        raise ValueError(f"Unexpected official splits: {sorted(invalid_splits)}")


def _record_counts(metadata: pd.DataFrame) -> pd.DataFrame:
    return (
        metadata.groupby(
            ["official_split", "file_type", "label", "week_id"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("record_count")
        .reset_index()
        .sort_values(["official_split", "week_id", "file_type", "label"])
    )


def _class_prevalence(metadata: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metadata.groupby(
            ["official_split", "file_type", "derived_week_id", "label"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("class_count")
        .reset_index()
    )
    totals = grouped.groupby(
        ["official_split", "file_type", "derived_week_id"],
        observed=True,
    )["class_count"].transform("sum")
    grouped["total_count"] = totals
    grouped["prevalence"] = grouped["class_count"] / totals
    return grouped.sort_values(
        ["official_split", "derived_week_id", "file_type", "label"]
    )


def _duplicate_hash_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    duplicated = metadata.loc[metadata.duplicated("sha256", keep=False)]
    if duplicated.empty:
        return pd.DataFrame(
            columns=[
                "sha256",
                "occurrence_count",
                "split_count",
                "splits",
                "file_types",
            ]
        )
    return (
        duplicated.groupby("sha256", sort=True)
        .agg(
            occurrence_count=("sha256", "size"),
            split_count=("official_split", "nunique"),
            splits=("official_split", lambda values: "|".join(sorted(set(values)))),
            file_types=("file_type", lambda values: "|".join(sorted(set(values)))),
        )
        .reset_index()
    )


def _duplicate_row_id_exclusions(rows: pd.DataFrame) -> pd.DataFrame:
    return _row_exclusions(
        rows,
        "DQ002",
        "row_id is duplicated within the official split",
    )


def _duplicate_hash_exclusions(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = metadata.loc[
        metadata.duplicated(["official_split", "sha256"], keep="first")
    ]
    return _row_exclusions(
        rows,
        "DQ003",
        "Exact SHA-256 duplicate within split; retain the lowest row_id if approved",
    )


def _cross_split_exclusions(
    metadata: pd.DataFrame,
    cross_split_hashes: pd.DataFrame,
) -> pd.DataFrame:
    hashes = set(cross_split_hashes["sha256"])
    rows = metadata.loc[
        metadata["sha256"].isin(hashes) & metadata["official_split"].eq("test")
    ]
    return _row_exclusions(
        rows,
        "DQ004",
        "SHA-256 also occurs in train; proposed test exclusion to prevent leakage",
    )


def _inconsistent_file_type_mask(metadata: pd.DataFrame) -> pd.Series:
    invalid = ~metadata["file_type"].isin(ALLOWED_FORMATS)
    type_counts = metadata.groupby("sha256", observed=True)["file_type"].transform(
        "nunique"
    )
    return invalid | type_counts.gt(1)


def _impossible_timestamp_issues(
    metadata: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    first = pd.to_datetime(metadata["first_submission_date"], utc=True, errors="coerce")
    last = pd.to_datetime(metadata["last_analysis_date"], utc=True, errors="coerce")
    missing_first = first.isna()
    outside_dataset = first.lt(DATASET_START) | first.gt(DATASET_END)
    last_before_first = last.notna() & first.notna() & last.lt(first)
    train_wrong_period = metadata["official_split"].eq("train") & ~metadata[
        "derived_week_id"
    ].between(0, 51)
    test_wrong_period = metadata["official_split"].eq("test") & ~metadata[
        "derived_week_id"
    ].between(52, 63)
    mask = (
        missing_first
        | outside_dataset
        | last_before_first
        | train_wrong_period
        | test_wrong_period
    )
    reasons = pd.Series("", index=metadata.index, dtype="string")
    reason_masks = [
        (missing_first, "missing/invalid first_submission_date"),
        (outside_dataset, "first_submission_date outside EMBER2024 collection range"),
        (last_before_first, "last_analysis_date precedes first_submission_date"),
        (train_wrong_period, "train row has derived week outside 0..51"),
        (test_wrong_period, "test row has derived week outside 52..63"),
    ]
    for reason_mask, text in reason_masks:
        reasons.loc[reason_mask] = reasons.loc[reason_mask].map(
            lambda current, reason=text: f"{current}; {reason}".lstrip("; ")
        )
    return mask, reasons


def _row_exclusions(
    rows: pd.DataFrame,
    rule_id: str,
    reason: str | pd.Series,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=EXCLUSION_COLUMNS)
    if isinstance(reason, pd.Series):
        reason_values = reason.reindex(rows.index).astype(str).to_numpy()
    else:
        reason_values = np.repeat(reason, len(rows))
    return pd.DataFrame(
        {
            "rule_id": rule_id,
            "official_split": rows["official_split"].to_numpy(),
            "row_id": rows["row_id"].to_numpy(),
            "sha256": rows["sha256"].to_numpy(),
            "reason": reason_values,
            "proposed_action": "exclude_row_after_review",
        }
    )


def _feature_row_exclusions(
    metadata: pd.DataFrame,
    rows_by_split: Mapping[str, np.ndarray],
    rule_id: str,
) -> pd.DataFrame:
    parts = []
    text = (
        "Feature row contains NaN"
        if rule_id == "DQ005"
        else "Feature row contains infinity"
    )
    for split, row_ids in rows_by_split.items():
        if len(row_ids) == 0:
            continue
        selected = metadata.loc[
            metadata["official_split"].eq(split)
            & metadata["row_id"].isin(row_ids)
        ]
        parts.append(_row_exclusions(selected, rule_id, text))
    if not parts:
        return pd.DataFrame(columns=EXCLUSION_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _scan_features(
    metadata: pd.DataFrame,
    feature_paths: Mapping[str, str | Path],
    *,
    feature_dim: int | None,
    feature_names: list[str] | None,
    chunk_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    if set(feature_paths) != set(OFFICIAL_SPLITS):
        raise ValueError(f"feature_paths must contain exactly {OFFICIAL_SPLITS}")

    missing_rows: dict[str, np.ndarray] = {}
    infinite_rows: dict[str, np.ndarray] = {}
    missing_counts: np.ndarray | None = None
    infinite_counts: np.ndarray | None = None
    finite_min: np.ndarray | None = None
    finite_max: np.ndarray | None = None

    for split in OFFICIAL_SPLITS:
        expected_rows = int(metadata["official_split"].eq(split).sum())
        with _open_feature_matrix(feature_paths[split], feature_dim) as matrix:
            if matrix.ndim != 2:
                raise ValueError(f"Feature matrix for {split} must be two-dimensional")
            if matrix.shape[0] != expected_rows:
                raise ValueError(
                    f"Metadata and feature row counts differ for {split}: "
                    f"{expected_rows} != {matrix.shape[0]}"
                )
            n_features = matrix.shape[1]
            if missing_counts is None:
                missing_counts = np.zeros(n_features, dtype=np.int64)
                infinite_counts = np.zeros(n_features, dtype=np.int64)
                finite_min = np.full(n_features, np.inf, dtype=np.float64)
                finite_max = np.full(n_features, -np.inf, dtype=np.float64)
            elif len(missing_counts) != n_features:
                raise ValueError("Train and test feature dimensions differ")

            split_missing: list[np.ndarray] = []
            split_infinite: list[np.ndarray] = []
            for start in range(0, len(matrix), chunk_rows):
                values = np.asarray(matrix[start : start + chunk_rows])
                missing = np.isnan(values)
                infinite = np.isinf(values)
                finite = np.isfinite(values)
                missing_counts += missing.sum(axis=0)
                infinite_counts += infinite.sum(axis=0)
                split_missing.append(np.flatnonzero(missing.any(axis=1)) + start)
                split_infinite.append(np.flatnonzero(infinite.any(axis=1)) + start)
                chunk_min = np.min(values, axis=0, where=finite, initial=np.inf)
                chunk_max = np.max(values, axis=0, where=finite, initial=-np.inf)
                finite_min = np.minimum(finite_min, chunk_min)
                finite_max = np.maximum(finite_max, chunk_max)
            missing_rows[split] = np.concatenate(split_missing)
            infinite_rows[split] = np.concatenate(split_infinite)

    assert missing_counts is not None
    assert infinite_counts is not None
    assert finite_min is not None
    assert finite_max is not None
    n_features = len(missing_counts)
    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries; expected {n_features}"
        )
    names = feature_names or [f"feature_{index}" for index in range(n_features)]
    constants = (
        (missing_counts == 0)
        & (infinite_counts == 0)
        & np.isfinite(finite_min)
        & (finite_min == finite_max)
    )
    rule_ids = []
    proposed_actions = []
    for index in range(n_features):
        rules = []
        if missing_counts[index]:
            rules.append("DQ005")
        if infinite_counts[index]:
            rules.append("DQ006")
        if constants[index]:
            rules.append("DQ007")
        rule_ids.append("|".join(rules))
        proposed_actions.append("exclude_feature" if rules else "retain")
    quality = pd.DataFrame(
        {
            "feature_index": np.arange(n_features),
            "feature_name": names,
            "missing_count": missing_counts,
            "infinite_count": infinite_counts,
            "finite_min": np.where(np.isfinite(finite_min), finite_min, np.nan),
            "finite_max": np.where(np.isfinite(finite_max), finite_max, np.nan),
            "constant": constants,
            "rule_ids": rule_ids,
            "proposed_action": proposed_actions,
        }
    )
    return missing_rows, infinite_rows, quality


@contextmanager
def _open_feature_matrix(path: str | Path, feature_dim: int | None):
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()
    archive = None
    if suffix == ".npy":
        matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(feature_path, allow_pickle=False)
        if "X" not in archive:
            archive.close()
            raise ValueError(f"NPZ feature archive has no X array: {feature_path}")
        matrix = archive["X"]
    elif suffix == ".dat":
        if feature_dim is None or feature_dim <= 0:
            raise ValueError("feature_dim is required for raw .dat matrices")
        row_bytes = np.dtype(np.float32).itemsize * feature_dim
        rows, remainder = divmod(feature_path.stat().st_size, row_bytes)
        if remainder:
            raise ValueError(f"Invalid raw feature matrix size: {feature_path}")
        matrix = np.memmap(
            feature_path,
            dtype=np.float32,
            mode="r",
            shape=(rows, feature_dim),
        )
    else:
        raise ValueError(f"Unsupported feature matrix: {feature_path}")
    try:
        yield matrix
    finally:
        if archive is not None:
            archive.close()


def _not_run_feature_checks() -> list[CheckResult]:
    return [
        CheckResult(
            rule_id,
            RULES[rule_id],
            "NOT_RUN",
            None,
            "No feature matrix supplied",
        )
        for rule_id in ("DQ005", "DQ006", "DQ007")
    ]


def _check(rule_id: str, findings: int, note: str) -> CheckResult:
    return CheckResult(
        rule_id,
        RULES[rule_id],
        "PASS" if findings == 0 else "WARN",
        findings,
        note,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _markdown_report(
    summary: dict[str, Any],
    checks: list[CheckResult],
) -> str:
    lines = [
        "# EMBER2024 quality-control report",
        "",
        f"Generated: {summary['generated_at_utc']}",
        "",
        f"- Input records: {summary['input_rows']:,}",
        f"- Output records: {summary['output_rows']:,}",
        "- Records deleted: 0",
        (
            "- Proposed row-exclusion entries: "
            f"{summary['proposed_row_exclusion_entries']:,}"
        ),
        (
            "- Proposed feature exclusions: "
            f"{summary['proposed_feature_exclusions']:,}"
        ),
        "",
        "| Rule | Check | Status | Findings |",
        "| --- | --- | --- | ---: |",
    ]
    for check in checks:
        findings = "—" if check.findings is None else f"{check.findings:,}"
        lines.append(
            f"| {check.rule_id} | {check.check} | {check.status} | {findings} |"
        )
    lines.extend(
        [
            "",
            "No records were changed or deleted. `proposed_exclusions.csv` records ",
            "every row-level proposal with its rule ID and reason. Feature-level ",
            "proposals are recorded in `feature_quality.csv`.",
            "",
        ]
    )
    return "\n".join(lines)
