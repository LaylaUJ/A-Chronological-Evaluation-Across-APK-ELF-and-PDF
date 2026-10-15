"""Materialize an approved data-quality exclusion policy without deleting rows."""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RULE_ID = "DQ003"
DECISION = "exclude"
RETAIN_RULE = "lowest_row_id_within_official_split_and_sha256"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse paths for the approved DQ003 policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--proposed-exclusions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args(argv)


def build_approved_dq003_exclusions(
    metadata: pd.DataFrame,
    proposed: pd.DataFrame,
    *,
    approved_at_utc: str,
) -> pd.DataFrame:
    """Validate and return exactly the non-lowest rows in duplicate-hash groups."""
    metadata_required = {"row_id", "sha256", "file_type", "official_split"}
    proposed_required = {"rule_id", "row_id", "sha256", "official_split"}
    if missing := metadata_required - set(metadata):
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if missing := proposed_required - set(proposed):
        raise ValueError(f"proposed exclusions are missing columns: {sorted(missing)}")
    if metadata.duplicated(["official_split", "row_id"]).any():
        raise ValueError("row_id must be unique within each official split")

    expected = metadata.loc[
        metadata.duplicated(["official_split", "sha256"], keep="first"),
        ["official_split", "row_id", "sha256", "file_type"],
    ].copy()
    # Metadata is row-aligned in ascending row_id order within each split. Recompute
    # explicitly so the retained record is independent of source ordering.
    minimum_ids = metadata.groupby(
        ["official_split", "sha256"], sort=False, observed=True
    )["row_id"].transform("min")
    expected = metadata.loc[
        metadata["row_id"].ne(minimum_ids),
        ["official_split", "row_id", "sha256", "file_type"],
    ].copy()

    supplied = proposed.loc[
        proposed["rule_id"].eq(RULE_ID),
        ["official_split", "row_id", "sha256"],
    ].copy()
    key = ["official_split", "row_id", "sha256"]
    if supplied.duplicated(key).any():
        raise ValueError("proposed DQ003 exclusions contain duplicate entries")
    expected_keys = set(expected[key].itertuples(index=False, name=None))
    supplied_keys = set(supplied[key].itertuples(index=False, name=None))
    if supplied_keys != expected_keys:
        raise ValueError(
            "proposed DQ003 exclusions do not exactly match recomputed duplicates"
        )

    approved = expected.sort_values(
        ["official_split", "row_id"], kind="stable", ignore_index=True
    )
    approved.insert(0, "rule_id", RULE_ID)
    approved["decision"] = DECISION
    approved["retain_rule"] = RETAIN_RULE
    approved["reason"] = (
        "Exact SHA-256 duplicate within official split; lowest row_id retained"
    )
    approved["approved_at_utc"] = approved_at_utc
    return approved


def main(argv: Sequence[str] | None = None) -> int:
    """Write the validated exclusion ledger and policy summary."""
    args = parse_args(argv)
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    approved_at = datetime.now(UTC).isoformat()
    metadata = pd.read_parquet(args.metadata)
    proposed = pd.read_csv(args.proposed_exclusions)
    approved = build_approved_dq003_exclusions(
        metadata, proposed, approved_at_utc=approved_at
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(args.output, index=False)
    counts = approved.groupby(
        ["file_type", "official_split"], sort=True, observed=True
    ).size()
    summary = {
        "approved_at_utc": approved_at,
        "rule_id": RULE_ID,
        "decision": DECISION,
        "retain_rule": RETAIN_RULE,
        "excluded_rows": len(approved),
        "excluded_rows_by_format_and_split": {
            f"{file_type}_{split}": int(count)
            for (file_type, split), count in counts.items()
        },
        "records_deleted": 0,
        "ledger": str(args.output.resolve()),
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
