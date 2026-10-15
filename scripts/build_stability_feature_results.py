"""Build publication stability inputs from discovery scores and frozen choices."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from tcfs.stability_outputs import build_feature_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse feature-score and selected-configuration inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--control-scores", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--selected-configurations", type=Path, nargs="+", required=True
    )
    parser.add_argument("--file-types", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write a complete window-level feature-results Parquet artifact."""
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if len(args.selected_configurations) != len(args.file_types):
        raise ValueError("one file type is required per selected-configuration file")
    selectors = pd.concat(
        [pd.read_parquet(path) for path in args.selector_scores], ignore_index=True
    )
    controls = pd.concat(
        [pd.read_parquet(path) for path in args.control_scores], ignore_index=True
    )
    configurations = pd.concat(
        [
            pd.read_csv(path).assign(file_type=file_type)
            for path, file_type in zip(
                args.selected_configurations, args.file_types, strict=True
            )
        ],
        ignore_index=True,
    )
    results = build_feature_results(selectors, controls, configurations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
