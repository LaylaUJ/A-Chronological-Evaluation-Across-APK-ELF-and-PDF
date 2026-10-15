"""Vectorize one EMBER2024 non-PE format with the official thrember code."""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from thrember.features import PEFeatureExtractor, StringExtractor
from thrember.model import vectorize_subset

from tcfs.final_evaluation import verify_final_readiness
from tcfs.temporal_split import PROTOCOL_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMATS = ("APK", "ELF", "PDF")
SPLITS = ("train", "test")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse vectorization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--file-type", choices=FORMATS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument(
        "--feature-spec",
        type=Path,
        default=PROJECT_ROOT / "configs" / "thrember_nonpe_features.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "experiments",
    )
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="Required before reading official test shards.",
    )
    parser.add_argument(
        "--unseal-log",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "test_vectorization_unseal_log.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Create row-aligned X/y memory maps without overwriting existing data."""
    args = parse_args(argv)
    if args.split == "test":
        _verify_test_vectorization_ready(
            args.file_type,
            args.experiment_root,
            args.unseal_log,
            unseal_test=args.unseal_test,
        )
    source = args.raw_root / "extracted" / args.split / args.file_type
    paths = sorted(source.glob("*.jsonl"), key=lambda path: path.name)
    expected_shards = 52 if args.split == "train" else 12
    if len(paths) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} {args.file_type}/{args.split} shards; "
            f"found {len(paths)}"
        )

    destination = args.output_root / args.file_type
    destination.mkdir(parents=True, exist_ok=True)
    x_path = destination / f"X_{args.split}.dat"
    y_path = destination / f"y_{args.split}.dat"
    names_path = destination / "feature_names.json"
    manifest_path = destination / f"{args.split}_vectorization.json"
    existing = [path for path in (x_path, y_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite vectorized files: {existing}")

    extractor = PEFeatureExtractor(args.feature_spec)
    feature_names = _feature_names(extractor)
    if len(feature_names) != extractor.dim:
        raise RuntimeError("feature-name count differs from thrember dimension")
    if names_path.exists():
        recorded = json.loads(names_path.read_text(encoding="utf-8"))
        if recorded != feature_names:
            raise ValueError("existing feature names disagree with thrember spec")
    else:
        names_path.write_text(
            json.dumps(feature_names, indent=2) + "\n", encoding="utf-8"
        )

    row_count = sum(1 for path in paths for _ in path.open(encoding="utf-8"))
    vectorize_subset(x_path, y_path, paths, extractor, row_count)
    expected_x_bytes = row_count * extractor.dim * np.dtype(np.float32).itemsize
    expected_y_bytes = row_count * np.dtype(np.int32).itemsize
    if x_path.stat().st_size != expected_x_bytes:
        raise RuntimeError("vectorized feature file has an unexpected size")
    if y_path.stat().st_size != expected_y_bytes:
        raise RuntimeError("vectorized label file has an unexpected size")
    manifest_path.write_text(
        json.dumps(
            {
                "file_type": args.file_type,
                "split": args.split,
                "rows": row_count,
                "feature_dimension": extractor.dim,
                "feature_spec": str(args.feature_spec.resolve()),
                "source_shards": [str(path.resolve()) for path in paths],
                "x_path": str(x_path.resolve()),
                "y_path": str(y_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _feature_names(extractor: PEFeatureExtractor) -> list[str]:
    names: list[str] = []
    for feature in extractor.features:
        if feature.name == "general":
            names.extend(
                [
                    "general.size",
                    "general.entropy",
                    "general.is_pe",
                    "general.start_byte_0",
                    "general.start_byte_1",
                    "general.start_byte_2",
                    "general.start_byte_3",
                ]
            )
        elif feature.name in {"histogram", "byteentropy"}:
            names.extend(
                f"{feature.name}.{index:03d}" for index in range(feature.dim)
            )
        elif isinstance(feature, StringExtractor):
            names.extend(
                [
                    "strings.numstrings",
                    "strings.avlength",
                    "strings.printables",
                ]
            )
            names.extend(f"strings.printabledist.{index:02d}" for index in range(96))
            names.append("strings.entropy")
            ordered_regexes = sorted(feature.regex_idxs, key=feature.regex_idxs.get)
            names.extend(f"strings.regex.{value}" for value in ordered_regexes)
        else:
            names.extend(f"{feature.name}.{index:03d}" for index in range(feature.dim))
    return names


def _verify_test_vectorization_ready(
    file_type: str,
    experiment_root: Path,
    unseal_log: Path,
    *,
    unseal_test: bool,
) -> None:
    """Verify all selected pipelines before the first official-test read."""
    if not unseal_test:
        raise PermissionError(
            "Official test shards remain sealed; pass --unseal-test only after "
            "all selected pipelines for this format are frozen."
        )
    format_root = experiment_root / file_type
    selected_path = format_root / "selected_configurations.csv"
    if not selected_path.is_file():
        raise FileNotFoundError(f"selected configurations are missing: {selected_path}")
    selected = pd.read_csv(selected_path)
    if "experiment_id" not in selected or selected.empty:
        raise ValueError("selected configurations do not identify experiments")
    experiment_ids = sorted(selected["experiment_id"].astype(str).unique())
    if len(experiment_ids) != len(selected):
        raise ValueError("selected configurations contain duplicate experiment IDs")
    for experiment_id in experiment_ids:
        verify_final_readiness(
            format_root / experiment_id,
            format_root / f".__vectorization_readiness__{experiment_id}",
        )
    unseal_log.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "official_test_vectorization_unsealed",
        "file_type": file_type,
        "protocol_version": PROTOCOL_VERSION,
        "unsealed_at_utc": datetime.now(UTC).isoformat(),
        "verified_experiments": experiment_ids,
    }
    with unseal_log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
