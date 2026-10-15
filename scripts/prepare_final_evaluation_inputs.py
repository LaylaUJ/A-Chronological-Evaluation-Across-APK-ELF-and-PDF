"""Assemble verified per-experiment final outputs for publication evaluation."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tcfs.final_outputs import assemble_final_evaluation_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-output assembly arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "final_evaluation",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "experiments",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "final_test_predictions.parquet",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "reports"
        / "evaluation"
        / "experiment_metadata.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write canonical final prediction and resource-metadata inputs."""
    args = parse_args(argv)
    for path in (args.predictions_output, args.metadata_output):
        if path.exists():
            raise FileExistsError(f"final aggregate already exists: {path}")
    predictions, metadata = assemble_final_evaluation_inputs(
        args.final_root, args.experiment_root
    )
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.predictions_output, index=False)
    metadata.to_csv(args.metadata_output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
