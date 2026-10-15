import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tcfs.ember2024 import download_subsets, selected_subsets


class MockDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, str]] = []

    def __call__(self, target: str, *, file_type: str, split: str) -> None:
        target_path = Path(target)
        self.calls.append((target_path, file_type, split))
        (target_path / "features.jsonl").write_bytes(
            f"{file_type}:{split}\n".encode()
        )


def test_selected_subsets_never_includes_windows_pe() -> None:
    subsets = selected_subsets()

    assert subsets == (
        ("APK", "train"),
        ("APK", "test"),
        ("ELF", "train"),
        ("ELF", "test"),
        ("PDF", "train"),
        ("PDF", "test"),
    )
    assert not {"PE", "Win32", "Win64", "Dot_Net"} & {
        file_format for file_format, _ in subsets
    }


def test_can_select_one_format_or_one_split() -> None:
    assert selected_subsets("elf") == (("ELF", "train"), ("ELF", "test"))
    assert selected_subsets(split="test") == (
        ("APK", "test"),
        ("ELF", "test"),
        ("PDF", "test"),
    )
    assert selected_subsets("pdf", "train") == (("PDF", "train"),)


def test_download_uses_thrember_and_records_manifest(tmp_path: Path) -> None:
    mock = MockDownloader()
    timestamp = datetime(2026, 10, 10, 12, 30, tzinfo=UTC)
    raw_dir = tmp_path / "raw"

    records = download_subsets(
        raw_dir,
        file_format="apk",
        split="train",
        downloader=mock,
        thrember_version="0.1.0",
        now=lambda: timestamp,
    )

    artifact = raw_dir / "ember2024" / "apk" / "train" / "features.jsonl"
    assert mock.calls == [(artifact.parent, "APK", "train")]
    assert len(records) == 1
    assert records[0].sha256 == hashlib.sha256(b"APK:train\n").hexdigest()

    with (tmp_path / "data_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "format": "APK",
            "split": "train",
            "path": "ember2024/apk/train/features.jsonl",
            "source_url": (
                "https://huggingface.co/datasets/joyce8/EMBER2024/resolve/"
                "main/APK_train.zip"
            ),
            "downloaded_at_utc": timestamp.isoformat(),
            "size_bytes": str(len(b"APK:train\n")),
            "version": "EMBER2024-feature-v3; thrember-0.1.0",
            "sha256": hashlib.sha256(b"APK:train\n").hexdigest(),
        }
    ]


def test_existing_subset_requires_force(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    target = raw_dir / "ember2024" / "pdf" / "test"
    target.mkdir(parents=True)
    (target / "existing.jsonl").write_text("old", encoding="utf-8")
    mock = MockDownloader()

    with pytest.raises(FileExistsError, match="--force"):
        download_subsets(
            raw_dir,
            file_format="PDF",
            split="test",
            downloader=mock,
        )

    assert mock.calls == []


def test_force_allows_existing_subset_and_replaces_manifest_rows(
    tmp_path: Path,
) -> None:
    mock = MockDownloader()
    raw_dir = tmp_path / "raw"
    kwargs = {
        "file_format": "ELF",
        "split": "test",
        "downloader": mock,
        "thrember_version": "0.1.0",
    }
    download_subsets(raw_dir, **kwargs)

    download_subsets(raw_dir, force=True, **kwargs)

    with (tmp_path / "data_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert len(mock.calls) == 2


@pytest.mark.parametrize("file_format", ["PE", "Win32", "Win64", "Dot_Net"])
def test_windows_formats_are_rejected(file_format: str) -> None:
    with pytest.raises(ValueError, match="format must be one of"):
        selected_subsets(file_format)
