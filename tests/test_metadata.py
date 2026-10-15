import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from tcfs.metadata import (
    build_metadata_parquet,
    discover_shards,
    feature_row_count,
)


def _write_shard(
    root: Path,
    split: str,
    file_type: str,
    records: list[dict[str, object]],
) -> Path:
    directory = root / split / file_type
    directory.mkdir(parents=True, exist_ok=True)
    start = "2023-09-24" if split == "train" else "2024-09-22"
    end = "2023-09-30" if split == "train" else "2024-09-28"
    path = directory / f"{start}_{end}_{file_type}_{split}.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _record(file_type: str, week: int, suffix: str) -> dict[str, object]:
    timestamp = int(DATASET_START.timestamp()) + week * 7 * 24 * 60 * 60
    return {
        "sha256": (suffix * 64)[:64],
        "file_type": file_type,
        "label": 1,
        "first_submission_date": timestamp,
        "last_analysis_date": timestamp + 60,
        "week_id": week,
        "family": None,
    }


DATASET_START = datetime(2023, 9, 24, tzinfo=UTC)


def test_builds_row_aligned_parquet_in_thrember_order(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_shard(raw, "train", "PDF", [_record("PDF", 0, "a")])
    _write_shard(raw, "train", "APK", [_record("APK", 0, "b")])
    _write_shard(raw, "test", "PDF", [_record("PDF", 52, "c")])
    _write_shard(raw, "test", "APK", [_record("APK", 52, "d")])
    output = tmp_path / "metadata.parquet"

    report = build_metadata_parquet(raw, output, batch_size=1)
    table = pq.read_table(output)

    assert report.source_rows == {"train": 2, "test": 2}
    assert report.metadata_rows == report.source_rows
    assert table.column_names == [field.name for field in table.schema]
    assert table["row_id"].to_pylist() == [0, 1, 0, 1]
    assert table["file_type"].to_pylist() == ["APK", "PDF", "APK", "PDF"]
    assert table["derived_week_id"].to_pylist() == [0, 0, 52, 52]
    assert table["first_submission_date"].type.tz == "UTC"


def test_feature_count_mismatch_stops_without_output(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for split, week in (("train", 0), ("test", 52)):
        _write_shard(raw, split, "APK", [_record("APK", week, "a")])
    train_x = tmp_path / "X_train.npy"
    test_x = tmp_path / "X_test.npy"
    np.save(train_x, np.zeros((2, 3), dtype=np.float32))
    np.save(test_x, np.zeros((1, 3), dtype=np.float32))
    output = tmp_path / "metadata.parquet"

    with pytest.raises(ValueError, match="row counts differ for train"):
        build_metadata_parquet(
            raw,
            output,
            vectorized_features={"train": train_x, "test": test_x},
        )

    assert not output.exists()


def test_feature_count_match_is_recorded(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for split, week in (("train", 0), ("test", 52)):
        _write_shard(raw, split, "ELF", [_record("ELF", week, "a")])
    paths = {}
    for split in ("train", "test"):
        path = tmp_path / f"X_{split}.npy"
        np.save(path, np.zeros((1, 3), dtype=np.float32))
        paths[split] = path

    report = build_metadata_parquet(
        raw,
        tmp_path / "metadata.parquet",
        vectorized_features=paths,
    )

    assert report.vectorized_feature_rows == {"train": 1, "test": 1}


def test_dataset_and_derived_week_mismatch_is_preserved(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    bad = _record("APK", 0, "a")
    bad["week_id"] = 1
    _write_shard(raw, "train", "APK", [bad])
    _write_shard(raw, "test", "APK", [_record("APK", 52, "b")])

    report = build_metadata_parquet(raw, tmp_path / "metadata.parquet")

    assert report.week_id_mismatches == {"train": 1, "test": 0}


def test_dat_feature_count_requires_exact_row_size(tmp_path: Path) -> None:
    path = tmp_path / "X_train.dat"
    np.zeros((2, 3), dtype=np.float32).tofile(path)

    assert feature_row_count(path, feature_dim=3) == 2
    with pytest.raises(ValueError, match="not divisible"):
        feature_row_count(path, feature_dim=5)


def test_discovery_rejects_unexpected_shard_name(tmp_path: Path) -> None:
    path = tmp_path / "train" / "APK" / "unknown.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected EMBER2024 shard"):
        discover_shards(tmp_path, "train")
