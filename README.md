# tcfs-malware

A reproducible Python research project for temporal-consistency feature selection
(TCFS) experiments in malware detection. The repository intentionally contains no
dataset; data acquisition and provenance should be documented before experiments
are run.

## Requirements

- Python 3.11 or later
- A virtual environment is strongly recommended

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The exact direct dependency versions used by the project are recorded in
`requirements.txt`. Runtime dependency ranges and tooling configuration live in
`pyproject.toml`.

## Layout

```text
configs/             Experiment and logging configuration
data/raw/            Immutable source data (not tracked)
data/interim/        Intermediate transformations (not tracked)
data/processed/      Model-ready datasets (not tracked)
docs/                Research and reproducibility notes
notebooks/           Exploratory notebooks
outputs/figures/     Generated figures
outputs/models/      Serialized estimators
outputs/predictions/ Model predictions
outputs/reports/     Metrics, logs, and reports
scripts/             Executable research entry points
src/tcfs/            Installable package
tests/               Automated tests
```

## Run an experiment

The starter command validates the YAML configuration, initializes logging, and
sets deterministic random seeds. It does not fetch or process data.

```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

## Download EMBER2024 feature data

The guarded downloader uses the official `thrember` package and permits only
APK, ELF, and PDF train/test feature subsets. Windows PE and challenge subsets
cannot be selected. Existing subset directories are not overwritten unless
`--force` is supplied.

```bash
# One subset
python scripts/download_ember2024.py --format APK --split train

# Both ELF splits
python scripts/download_ember2024.py --format ELF

# All three non-PE test splits
python scripts/download_ember2024.py --split test
```

Each extracted artifact is recorded in `data/data_manifest.csv` with its
official source archive URL, retrieval time, byte size, feature/package version,
and SHA-256 digest. The dataset contains features and labels for defensive
research. This project does not execute or redistribute malware; users must do
the same and comply with the upstream dataset license and access terms.

## Build row-aligned metadata

After extraction, build the Parquet metadata table from the weekly JSONL shards:

```bash
python scripts/build_ember2024_metadata.py \
  --raw-dir data/raw/ember2024/extracted
```

To assert alignment with existing vectorized matrices, provide both feature
paths. For thrember's raw `.dat` matrices, also provide their feature dimension:

```bash
python scripts/build_ember2024_metadata.py \
  --raw-dir data/raw/ember2024/extracted \
  --x-train data/processed/X_train.npy \
  --x-test data/processed/X_test.npy
```

For thrember `.dat` matrices, add `--feature-dim` using the installed extractor's
actual `PEFeatureExtractor().dim` value.

See `docs/ember2024-schema.md` for the observed raw schema, upstream duplication
warning, canonical row order, and Parquet types.

## Run data quality checks

```bash
python scripts/run_data_quality.py
```

The audit writes CSV/JSON machine outputs and a Markdown summary under
`outputs/reports/data_quality/`. It never deletes records. Every proposed row
exclusion includes a rule ID and reason in `proposed_exclusions.csv`; feature
proposals are recorded in `feature_quality.csv`. Supply `--x-train` and
`--x-test` to enable missing, infinite, and constant-feature checks. Without
matrices, those checks are explicitly reported as `NOT_RUN`.

## Temporal protocol

Protocol `tcfs-temporal-v1.0` partitions the 64 dataset weeks into four disjoint
discovery windows (1–10, 11–20, 21–30, and 31–40), validation weeks 41–52, and
sealed official-test weeks 53–64. Create the row assignments with:

```bash
python scripts/run_temporal_split.py
```

Preprocessing and discovery importance helpers fit only the requested discovery
window. Threshold selection and hyperparameter tuning can access discovery plus
validation, never official test. Final test access requires `--unseal-test` and
appends the protocol version and UTC unsealing time to
`outputs/reports/test_unseal_log.jsonl`.

## Feature selectors

`tcfs.feature_selection` provides a common sklearn-compatible interface for
mutual-information, L1 logistic-regression, LightGBM permutation-importance,
seeded random matched-feature, and all-features selectors. Each implements
`fit`, `transform`, `get_support`, and `get_feature_scores`. Budgeted selectors
accept 5%, 10%, 20%, 40%, or 60% and use ceiling rounding. The score table ranks
every input feature and records its name, raw importance, normalized rank, and
selection status. The LightGBM selector calculates permutation importance on a
deterministic internal holdout drawn only from the data passed to `fit`; callers
must pass an eligible discovery window, never validation or test data.

## Temporal consensus

`tcfs.consensus` combines the complete component-selector tables from all four
discovery windows. For each feature it calculates mean normalized rank `M`,
top-k selection frequency `F`, and population rank variance `V`, then defines
`TCFS(rho) = (M * F) / (1 + rho * V)`. Format-specific scores use the 12
observations from four windows and three selectors. Shared scores combine all
observations for an identically named feature across the formats where it is
present. Rho is selected from `{0, 0.25, 0.5, 1}` through a guarded
validation-only evaluator; discovery or sealed-test rows are rejected.

## Classifier model selection

LightGBM is the primary classifier and scaled logistic regression is the
secondary classifier. `scripts/run_model_selection.py` accepts a development
matrix containing discovery weeks 1-40 and validation weeks 41-52; it rejects
official-test rows. The fixed search in `configs/model_selection.yaml` tunes
feature budget, TCFS rho, classifier parameters, threshold, and either one seed
or a mean-probability seed ensemble. Every feature-selection method receives
the same classifier parameter grid.

For each method/classifier pair, the runner chooses the smallest feature budget
whose validation macro-F1 is within 0.01 of the best result. Remaining ties use
macro-F1, MCC, feature count, and deterministic configuration order. Once all
choices are frozen, the final development model is refit through week 52. Each
experiment directory contains its configuration, selected features, fitted
model, held-out validation predictions, complete tuning results, runtime, and
software versions. Official test evaluation remains a separate explicitly
unsealed operation.

## Statistical evaluation

`tcfs.evaluation` reports predictive and resource metrics independently for APK,
ELF, and PDF. Predictive results include macro-F1, MCC, balanced accuracy,
positive-class precision and recall, PR-AUC, ROC-AUC, and TPR at maximum FPRs of
1% and 0.1%. Resource results include selected-feature count, serialized model
size, and median probability-inference latency per sample. Pairwise feature
stability is reported with Jaccard similarity and Spearman rank correlation.

The evaluation runner defaults to 2,000 class-stratified bootstrap resamples.
Within each format, every method uses the same resampled row indices, preserving
the paired design for differences. TCFS versus all features uses a one-sided
macro-F1 non-inferiority margin of -0.01. Confirmatory secondary comparisons use
two-sided bootstrap P values with Holm adjustment within format. Every output
table carries total sample size and positive/negative class counts.

```bash
python scripts/run_evaluation.py \
  --predictions outputs/predictions/evaluation_predictions.parquet \
  --experiment-metadata outputs/reports/evaluation_metadata.csv \
  --feature-results outputs/reports/feature_replicates.parquet
```

## One-shot final evaluation

Every finalized development experiment now includes `freeze_manifest.json`,
which records the frozen protocol, selected features, hyperparameters,
threshold, seed or ensemble rule, and SHA-256 hashes of the configuration,
feature list, and fitted pipeline. Final evaluation refuses modified artifacts
or any destination containing an earlier attempt.

```bash
python scripts/run_final_evaluation.py \
  --experiment-dir outputs/reports/experiments/tcfs__lightgbm \
  --features data/processed/APK_X.npy \
  --labels data/processed/APK_y.npy \
  --feature-names data/processed/APK_feature_names.json \
  --assignments data/processed/APK_temporal_assignments.parquet \
  --file-type APK \
  --output-dir outputs/reports/final/APK_tcfs_lightgbm \
  --unseal-test
```

Preflight verification runs before feature or label files are loaded. The
destination is then reserved before official-test unsealing; even a failed
attempt leaves a blocking execution record. The frozen pipeline is refit on
weeks 1-52 and evaluated once on weeks 53-64 without any test-driven changes.
Outputs contain timestamped predictions, bootstrap confidence intervals, the
refitted model, an audit record, and SHA-256 checksums.

## Publication figures and tables

`scripts/generate_publication_figures.py` creates the chronological design,
feature stability, retained-feature performance, and ablation/sensitivity
figures. `scripts/generate_publication_tables.py` creates dataset/QC and
confirmatory performance/efficiency tables. Both operate exclusively on saved
CSV or Parquet artifacts. Every rendered item is exported as vector PDF, SVG,
and 300-DPI PNG; tables additionally include CSV and LaTeX sources. See
`docs/publication-outputs.md` for commands and required schemas.

Paths in the experiment configuration are relative to the project root. Record
new experiment-specific values in a copied YAML file rather than changing code.

## Quality checks

```bash
python -m ruff check .
python -m pytest
```

See `docs/reproducibility.md` for the experiment checklist and known limits of
determinism.
