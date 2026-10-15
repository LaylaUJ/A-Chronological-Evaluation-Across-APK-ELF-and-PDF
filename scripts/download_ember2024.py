"""Download only the APK, ELF, and PDF feature subsets of EMBER2024."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tcfs import configure_logging
from tcfs.ember2024 import ALLOWED_FORMATS, ALLOWED_SPLITS, download_subsets

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse downloader command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download EMBER2024 feature data through the official thrember "
            "package. Windows PE and challenge subsets are never available."
        )
    )
    parser.add_argument(
        "--format",
        type=str.upper,
        choices=ALLOWED_FORMATS,
        help="Download one format; omit to select APK, ELF, and PDF.",
    )
    parser.add_argument(
        "--split",
        type=str.lower,
        choices=ALLOWED_SPLITS,
        help="Download one split; omit to select train and test.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Raw-data root (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow thrember to overwrite files in selected subset directories.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded downloader and record file provenance."""
    args = parse_args(argv)
    configure_logging(
        PROJECT_ROOT / "configs" / "logging.yaml",
        project_root=PROJECT_ROOT,
    )
    logger = logging.getLogger(__name__)
    logger.warning(
        "EMBER2024 feature/label artifacts are for defensive research only. "
        "Do not execute or redistribute malware."
    )

    records = download_subsets(
        args.data_dir,
        file_format=args.format,
        split=args.split,
        force=args.force,
    )
    logger.info(
        "Recorded %d artifacts in %s",
        len(records),
        args.data_dir.parent / "data_manifest.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
