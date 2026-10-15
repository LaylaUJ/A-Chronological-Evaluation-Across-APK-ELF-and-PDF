"""Safe, auditable access to the non-PE portions of EMBER2024."""

import csv
import hashlib
import importlib
import importlib.metadata
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

ALLOWED_FORMATS = ("APK", "ELF", "PDF")
ALLOWED_SPLITS = ("train", "test")
DATASET_REPOSITORY = "joyce8/EMBER2024"
DATASET_REVISION = "main"
MANIFEST_COLUMNS = (
    "format",
    "split",
    "path",
    "source_url",
    "downloaded_at_utc",
    "size_bytes",
    "version",
    "sha256",
)


@dataclass(frozen=True)
class ManifestRecord:
    """Provenance metadata for one locally extracted dataset artifact."""

    format: str
    split: str
    path: str
    source_url: str
    downloaded_at_utc: str
    size_bytes: int
    version: str
    sha256: str


def selected_subsets(
    file_format: str | None = None,
    split: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return an explicitly allowlisted cross-product of formats and splits."""
    formats = (_normalize_format(file_format),) if file_format else ALLOWED_FORMATS
    splits = (_normalize_split(split),) if split else ALLOWED_SPLITS
    return tuple(
        (current_format, current_split)
        for current_format in formats
        for current_split in splits
    )


def download_subsets(
    data_dir: str | Path,
    *,
    file_format: str | None = None,
    split: str | None = None,
    force: bool = False,
    downloader: Callable[..., object] | None = None,
    thrember_version: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> list[ManifestRecord]:
    """Download allowed EMBER2024 subsets through the official thrember API.

    Files are feature/label artifacts. This function never executes their contents
    and exposes no route to the Windows PE, challenge, or raw-malware subsets.
    """
    root = Path(data_dir).resolve()
    dataset_root = root / "ember2024"
    manifest_path = root.parent / "data_manifest.csv"
    subsets = selected_subsets(file_format, split)

    occupied = []
    for subset in subsets:
        target = _subset_dir(dataset_root, *subset)
        if target.exists() and any(target.iterdir()):
            occupied.append(target)
    if occupied and not force:
        paths = ", ".join(str(path) for path in occupied)
        raise FileExistsError(
            f"Refusing to overwrite non-empty subset directories: {paths}. "
            "Pass --force to permit replacement."
        )

    if downloader is None:
        module = _load_thrember()
        downloader = module.download_dataset
    version = thrember_version or _installed_thrember_version()
    clock = now or (lambda: datetime.now(UTC))

    records: list[ManifestRecord] = []
    for current_format, current_split in subsets:
        target = _subset_dir(dataset_root, current_format, current_split)
        target.mkdir(parents=True, exist_ok=True)
        working_directory = Path.cwd()
        try:
            downloader(
                str(target),
                file_type=current_format,
                split=current_split,
            )
        finally:
            os.chdir(working_directory)

        files = sorted(path for path in target.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(
                f"thrember returned without creating files for "
                f"{current_format}/{current_split}"
            )

        downloaded_at = clock().astimezone(UTC).isoformat()
        source_url = _source_url(current_format, current_split)
        records.extend(
            ManifestRecord(
                format=current_format,
                split=current_split,
                path=path.relative_to(root).as_posix(),
                source_url=source_url,
                downloaded_at_utc=downloaded_at,
                size_bytes=path.stat().st_size,
                version=f"EMBER2024-feature-v3; thrember-{version}",
                sha256=_sha256(path),
            )
            for path in files
        )

    _update_manifest(manifest_path, records, replaced_subsets=set(subsets))
    return records


def _normalize_format(value: str) -> str:
    normalized = value.upper()
    if normalized not in ALLOWED_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(ALLOWED_FORMATS)}")
    return normalized


def _normalize_split(value: str) -> str:
    normalized = value.lower()
    if normalized not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of: {', '.join(ALLOWED_SPLITS)}")
    return normalized


def _subset_dir(root: Path, file_format: str, split: str) -> Path:
    return root / file_format.lower() / split


def _source_url(file_format: str, split: str) -> str:
    filename = f"{file_format}_{split}.zip"
    return (
        f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
        f"{DATASET_REVISION}/{filename}"
    )


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_thrember() -> ModuleType:
    try:
        return importlib.import_module("thrember")
    except ImportError as error:
        raise RuntimeError(
            "The official thrember package is required. Install this project "
            "with `python -m pip install -e .`."
        ) from error


def _installed_thrember_version() -> str:
    try:
        return importlib.metadata.version("thrember")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            raise ValueError(
                f"Unexpected manifest columns in {path}: {reader.fieldnames}"
            )
        return list(reader)


def _update_manifest(
    path: Path,
    records: Iterable[ManifestRecord],
    *,
    replaced_subsets: set[tuple[str, str]],
) -> None:
    existing = [
        row
        for row in _read_manifest(path)
        if (row["format"], row["split"]) not in replaced_subsets
    ]
    rows = existing + [
        {key: str(value) for key, value in asdict(record).items()}
        for record in records
    ]
    rows.sort(key=lambda row: (row["format"], row["split"], row["path"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)
