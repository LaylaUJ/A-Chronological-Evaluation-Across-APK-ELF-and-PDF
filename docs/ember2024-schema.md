# Actual EMBER2024 schema and row alignment

## Inspected files

The local official download was inspected from `joyce8/EMBER2024`. It contains
192 JSON Lines shards totaling approximately 3.38 GB:

| Split | Weeks | Formats | JSONL shards | Raw rows |
| --- | ---: | --- | ---: | ---: |
| train | 52 | APK, ELF, PDF | 156 | 572,000 |
| test | 12 | APK, ELF, PDF | 36 | 132,000 |

The 704,000 raw rows are twice the documented 352,000 non-PE records because the
current upstream archives duplicate almost every record. ELF test contains 5,989
unique hashes rather than the documented 6,000. Metadata generation preserves
every raw row so it remains aligned with official thrember vectorization. Any
later de-duplication must apply the identical row mask to both metadata and `X`.

## Observed JSONL record schema

Every inspected format and split used the same 33 top-level fields:

| Fields | Observed JSON type |
| --- | --- |
| `md5`, `sha1`, `sha256`, `tlsh`, `detection_ratio`, `file_type` | string |
| `first_submission_date`, `last_analysis_date`, `label`, `week_id` | integer |
| `family` | string or null |
| `family_confidence` | number or null |
| `behavior`, `file_property`, `packer`, `exploit`, `group` | array |
| `histogram`, `byteentropy` | array |
| `strings`, `general`, `header`, `section`, `imports`, `authenticode` | object |
| `exports`, `datadirectories`, `richheader`, `pefilewarnings` | array |
| `caps`, `ttps`, `mbc` | array |

The date fields are Unix seconds. `week_id` starts at zero for the week beginning
2023-09-24, but it is a dataset assignment rather than a guaranteed derivation
from each record's timestamp. Actual records exist where it differs from
`derived_week_id`; both values are therefore preserved. The inspected data has
zero disagreements in train and 644 in test. APK and PDF use only the common
non-PE feature groups in practice; PE-specific containers remain present but can
be empty.

## Parquet metadata schema

`scripts/build_ember2024_metadata.py` writes:

| Column | Parquet type | Meaning |
| --- | --- | --- |
| `row_id` | int64 | Zero-based row in the official split's feature matrix |
| `sha256` | string | Lowercase sample digest |
| `file_type` | string | APK, ELF, or PDF |
| `label` | nullable int8 | Upstream binary label; not summarized for test |
| `first_submission_date` | timestamp[ms, UTC] | Converted Unix timestamp |
| `last_analysis_date` | nullable timestamp[ms, UTC] | Converted when present |
| `official_split` | string | `train` or `test` from the official shard |
| `week_id` | nullable int32 | Upstream dataset week |
| `derived_week_id` | int32 | Whole weeks since 2023-09-24 UTC |
| `family` | nullable string | Upstream family when assigned |

The composite key `(official_split, row_id)` is unique. `row_id` restarts at zero
for test because thrember creates separate `X_train.dat` and `X_test.dat` files.

## Alignment contract

Thrember sorts shard filenames, then enumerates every line in each shard. The
builder uses the same order: filename sort across all three format directories,
then original line order. It asserts that:

1. path split/type and record split/type agree;
2. enumerated source rows and written Parquet rows agree per split;
3. upstream/derived week disagreements are counted in the alignment report;
4. when `--x-train` and `--x-test` are provided, each matrix row count equals its
   metadata count. Only `X` shapes are inspected; test label values are never
   read for this check.

For raw thrember `.dat` matrices, also pass `--feature-dim`. `.npy` and `.npz`
matrices carry their own shapes. Any mismatch aborts and removes the temporary
Parquet output.
