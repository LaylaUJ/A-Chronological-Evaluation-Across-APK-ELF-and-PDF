"""Build row-aligned EMBER2024 metadata from official JSONL feature shards."""

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ALLOWED_FORMATS = ("APK", "ELF", "PDF")
OFFICIAL_SPLITS = ("train", "test")
DATASET_EPOCH = datetime(2023, 9, 24, tzinfo=UTC)
SHARD_PATTERN = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2})_"
    r"(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?P<file_type>APK|ELF|PDF)_"
    r"(?P<split>train|test)\.jsonl$"
)

METADATA_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("file_type", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=True),
        pa.field(
            "first_submission_date",
            pa.timestamp("ms", tz="UTC"),
            nullable=False,
        ),
        pa.field(
            "last_analysis_date",
            pa.timestamp("ms", tz="UTC"),
            nullable=True,
        ),
        pa.field("official_split", pa.string(), nullable=False),
        pa.field("week_id", pa.int32(), nullable=True),
        pa.field("derived_week_id", pa.int32(), nullable=False),
        pa.field("family", pa.string(), nullable=True),
    ]
)


@dataclass(frozen=True)
class AlignmentReport:
    """Counts proving that source, metadata, and optional vector rows agree."""

    source_rows: dict[str, int]
    metadata_rows: dict[str, int]
    vectorized_feature_rows: dict[str, int] | None
    week_id_mismatches: dict[str, int]
    shard_order: dict[str, list[str]]


def discover_shards(raw_dir: str | Path, split: str) -> list[Path]:
    """Discover shards in the same filename order used by thrember."""
    if split not in OFFICIAL_SPLITS:
        raise ValueError(f"split must be one of {OFFICIAL_SPLITS}")
    split_dir = Path(raw_dir) / split
    paths = sorted(split_dir.rglob("*.jsonl"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No JSONL shards found under {split_dir}")
    if len({path.name for path in paths}) != len(paths):
        raise ValueError(f"Duplicate shard filenames found under {split_dir}")

    for path in paths:
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None or match.group("split") != split:
            raise ValueError(f"Unexpected EMBER2024 shard name: {path.name}")
    return paths


def iter_metadata_rows(
    paths: Iterable[Path],
    split: str,
) -> Iterator[dict[str, Any]]:
    """Yield metadata in exact shard/line order with split-local row IDs."""
    row_id = 0
    for path in paths:
        shard = _parse_shard_name(path)
        if shard["split"] != split:
            raise ValueError(f"Shard split does not match {split}: {path}")
        with path.open(encoding="utf-8") as stream:
            for source_line, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{source_line}") from error
                yield _metadata_record(
                    record,
                    row_id=row_id,
                    split=split,
                    shard_file_type=shard["file_type"],
                    source=f"{path}:{source_line}",
                )
                row_id += 1


def build_metadata_parquet(
    raw_dir: str | Path,
    output_path: str | Path,
    *,
    vectorized_features: Mapping[str, str | Path] | None = None,
    feature_dim: int | None = None,
    batch_size: int = 10_000,
) -> AlignmentReport:
    """Stream all metadata to Parquet and stop on any alignment mismatch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    shards = {split: discover_shards(raw_dir, split) for split in OFFICIAL_SPLITS}
    source_counts = {split: 0 for split in OFFICIAL_SPLITS}
    week_mismatches: dict[str, int]
    writer: pq.ParquetWriter | None = None

    try:
        writer = pq.ParquetWriter(
            temporary,
            METADATA_SCHEMA,
            compression="zstd",
            use_dictionary=["file_type", "official_split", "family"],
        )
        for split in OFFICIAL_SPLITS:
            batch: list[dict[str, Any]] = []
            for record in iter_metadata_rows(shards[split], split):
                batch.append(record)
                source_counts[split] += 1
                if len(batch) == batch_size:
                    _write_batch(writer, batch)
                    batch.clear()
            if batch:
                _write_batch(writer, batch)
        writer.close()
        writer = None

        metadata_counts = _parquet_split_counts(temporary)
        _assert_equal_counts(source_counts, metadata_counts, "source", "metadata")
        week_mismatches = _parquet_week_mismatches(temporary)

        vector_counts = None
        if vectorized_features is not None:
            vector_counts = {
                split: feature_row_count(vectorized_features[split], feature_dim)
                for split in OFFICIAL_SPLITS
            }
            _assert_equal_counts(
                metadata_counts,
                vector_counts,
                "metadata",
                "vectorized features",
            )

        temporary.replace(output)
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    return AlignmentReport(
        source_rows=source_counts,
        metadata_rows=metadata_counts,
        vectorized_feature_rows=vector_counts,
        week_id_mismatches=week_mismatches,
        shard_order={
            split: [path.name for path in split_paths]
            for split, split_paths in shards.items()
        },
    )


def feature_row_count(path: str | Path, feature_dim: int | None = None) -> int:
    """Read only a feature matrix shape; never inspect labels."""
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        return _matrix_rows(matrix.shape, feature_path)
    if suffix == ".npz":
        with np.load(feature_path, allow_pickle=False) as archive:
            if "X" not in archive:
                raise ValueError(f"NPZ feature archive has no X array: {feature_path}")
            return _matrix_rows(archive["X"].shape, feature_path)
    if suffix == ".dat":
        if feature_dim is None or feature_dim <= 0:
            raise ValueError(
                "feature_dim is required and must be positive for .dat files"
            )
        row_bytes = np.dtype(np.float32).itemsize * feature_dim
        size = feature_path.stat().st_size
        rows, remainder = divmod(size, row_bytes)
        if remainder:
            raise ValueError(
                f"Feature file size is not divisible by one row ({row_bytes} bytes): "
                f"{feature_path}"
            )
        return rows
    raise ValueError(f"Unsupported feature matrix format: {feature_path}")


def write_alignment_report(report: AlignmentReport, path: str | Path) -> None:
    """Write a JSON sidecar without labels or feature values."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _metadata_record(
    record: dict[str, Any],
    *,
    row_id: int,
    split: str,
    shard_file_type: str,
    source: str,
) -> dict[str, Any]:
    required = {"sha256", "file_type", "label", "first_submission_date"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Missing fields {sorted(missing)} at {source}")

    sha256 = record["sha256"]
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
        raise ValueError(f"Invalid sha256 at {source}")
    file_type = record["file_type"]
    if file_type != shard_file_type or file_type not in ALLOWED_FORMATS:
        raise ValueError(f"file_type does not match shard at {source}")

    first_submission = _utc_timestamp(
        record["first_submission_date"],
        field="first_submission_date",
        source=source,
        required=True,
    )
    derived_week = (first_submission.date() - DATASET_EPOCH.date()).days // 7
    if derived_week < 0:
        raise ValueError(f"first_submission_date predates dataset epoch at {source}")

    dataset_week = record.get("week_id")
    if dataset_week is not None:
        if not isinstance(dataset_week, int) or isinstance(dataset_week, bool):
            raise ValueError(f"week_id is not an integer at {source}")

    label = record["label"]
    if label is not None and (not isinstance(label, int) or isinstance(label, bool)):
        raise ValueError(f"label is not an integer or null at {source}")
    family = record.get("family")
    if family is not None and not isinstance(family, str):
        raise ValueError(f"family is not a string or null at {source}")

    return {
        "row_id": row_id,
        "sha256": sha256.lower(),
        "file_type": file_type,
        "label": label,
        "first_submission_date": first_submission,
        "last_analysis_date": _utc_timestamp(
            record.get("last_analysis_date"),
            field="last_analysis_date",
            source=source,
            required=False,
        ),
        "official_split": split,
        "week_id": dataset_week,
        "derived_week_id": derived_week,
        "family": family,
    }


def _utc_timestamp(
    value: object,
    *,
    field: str,
    source: str,
    required: bool,
) -> datetime | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is null at {source}")
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} is not a Unix integer timestamp at {source}")
    try:
        return datetime.fromtimestamp(value, UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError(f"Invalid {field} at {source}") from error


def _parse_shard_name(path: Path) -> dict[str, str]:
    match = SHARD_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected EMBER2024 shard name: {path.name}")
    return match.groupdict()


def _write_batch(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=METADATA_SCHEMA)
    writer.write_table(table)


def _parquet_split_counts(path: Path) -> dict[str, int]:
    table = pq.read_table(path, columns=["official_split"])
    return {
        split: int(pc.sum(pc.equal(table["official_split"], split)).as_py())
        for split in OFFICIAL_SPLITS
    }


def _parquet_week_mismatches(path: Path) -> dict[str, int]:
    table = pq.read_table(
        path,
        columns=["official_split", "week_id", "derived_week_id"],
    )
    comparable = pc.is_valid(table["week_id"])
    differs = pc.not_equal(table["week_id"], table["derived_week_id"])
    mismatch = pc.and_(comparable, pc.fill_null(differs, False))
    return {
        split: int(
            pc.sum(
                pc.and_(pc.equal(table["official_split"], split), mismatch)
            ).as_py()
        )
        for split in OFFICIAL_SPLITS
    }


def _assert_equal_counts(
    expected: Mapping[str, int],
    actual: Mapping[str, int],
    expected_name: str,
    actual_name: str,
) -> None:
    for split in OFFICIAL_SPLITS:
        if expected.get(split) != actual.get(split):
            raise ValueError(
                f"{expected_name} and {actual_name} row counts differ for {split}: "
                f"{expected.get(split)} != {actual.get(split)}"
            )


def _matrix_rows(shape: tuple[int, ...], path: Path) -> int:
    if len(shape) != 2:
        raise ValueError(f"Feature matrix must be two-dimensional: {path} has {shape}")
    return int(shape[0])
