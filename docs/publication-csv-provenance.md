# Publication CSV provenance

This document traces every CSV consumed by the publication scripts. Commands
assume the repository root is the working directory and `PYTHONPATH=src`.
Publication builders consume saved results; they do not fit models or unseal
official-test rows.

## Existing quality-control inputs

### `outputs/reports/data_quality/record_counts.csv`

- Producer: `scripts/run_data_quality.py`, through
  `tcfs.data_quality.run_data_quality`.
- Inputs: aligned metadata Parquet and, when available, the feature matrix.
- Canonical columns: `official_split`, `file_type`, `label`, `week_id`,
  `record_count`.
- Command:

  ```bash
  python scripts/run_data_quality.py \
    --metadata data/processed/ember2024_metadata.parquet \
    --output-dir outputs/reports/data_quality
  ```

- Status: present at its canonical name.

### `outputs/reports/data_quality/check_results.csv`

- Producer, inputs, and command: same data-quality run as `record_counts.csv`.
- Canonical columns: `rule_id`, `check`, `status`, `findings`, `note`.
- Status: present at its canonical name.

## Evaluation and feature-stability inputs

The following five missing files share one producer:
`scripts/run_evaluation.py`, through `tcfs.evaluation.evaluate_predictions`
and `tcfs.evaluation.save_evaluation_results`.

Required input artifacts are:

- a long, row-aligned prediction table with `file_type`, `method`, `row_id`,
  `y_true`, `probability`, and `prediction`;
- experiment metadata with `file_type`, `method`, `feature_count`,
  `model_size_bytes`, and `inference_latency_ms_per_sample`;
- for stability outputs, a complete feature-results table with `file_type`,
  `method`, `replicate`, `feature_name`, `selected`, and `normalized_rank`.

Canonical command, after the frozen experiments have produced those inputs:

```bash
python scripts/run_evaluation.py \
  --predictions outputs/predictions/final_test_predictions.parquet \
  --experiment-metadata outputs/reports/evaluation/experiment_metadata.csv \
  --feature-results outputs/reports/feature_selection/feature_results.parquet \
  --bootstrap-samples 2000 \
  --random-state 42 \
  --output-dir outputs/reports/evaluation
```

### `outputs/reports/evaluation/metrics.csv`

- Canonical columns: `file_type`, `method`, `metric`, `estimate`, `ci_lower`,
  `ci_upper`, `confidence_level`, `bootstrap_samples`,
  `bootstrap_random_state`, `n_samples`, `n_positive`, `n_negative`.
- Status: missing. It is not stored under another name. A per-experiment
  `test_metrics.csv` would be an intermediate from final evaluation, not a
  schema-compatible replacement for this cross-method table.

### `outputs/reports/evaluation/noninferiority.csv`

- Canonical columns: `file_type`, `method`, `reference_method`, `metric`,
  `difference`, `ci_lower`, `ci_upper`, `noninferiority_margin`, `p_value`,
  `noninferior_at_0_05`, `bootstrap_samples`, `bootstrap_random_state`,
  `n_samples`, `n_positive`, `n_negative`.
- Status: missing; no differently named equivalent exists.

### `outputs/reports/evaluation/secondary_comparisons.csv`

- Canonical columns: `file_type`, `method`, `reference_method`, `metric`,
  `difference`, `ci_lower`, `ci_upper`, `p_value`, `holm_adjusted_p_value`,
  `significant_at_0_05`, `bootstrap_samples`, `bootstrap_random_state`,
  `n_samples`, `n_positive`, `n_negative`.
- Status: missing; no differently named equivalent exists.

### `outputs/reports/evaluation/jaccard_stability.csv`

- Canonical columns: `file_type`, `method`, `replicate_a`, `replicate_b`,
  `n_samples`, `n_positive`, `n_negative`, `jaccard`.
- Status: missing; no differently named equivalent exists. The discovery-window
  feature-results input is also missing.

### `outputs/reports/evaluation/spearman_stability.csv`

- Canonical columns: `file_type`, `method`, `replicate_a`, `replicate_b`,
  `n_samples`, `n_positive`, `n_negative`, `spearman_rho`.
- Status: missing; no differently named equivalent exists. It shares the same
  missing feature-results dependency as Jaccard stability.

## Validation-only publication aggregations

### `outputs/reports/budget_metrics.csv`

- Producer: `scripts/build_budget_metrics.py`, through
  `tcfs.publication_aggregation.aggregate_budget_metrics`.
- Input artifact: genuine row-level validation predictions at the canonical
  proposed path
  `outputs/predictions/validation_budget_predictions.parquet`.
- Required input columns: `file_type`, `method`, `feature_budget`, `row_id`,
  `y_true`, `prediction`, `protocol_role`, and `protocol_week`; an optional
  `protocol_version` must equal the frozen protocol version.
- Output columns: `file_type`, `method`, `feature_budget`, `metric`, `estimate`,
  `ci_lower`, `ci_upper`, `n_samples`, `n_positive`, `n_negative`,
  `bootstrap_samples`, `bootstrap_random_state`, `protocol_version`.
- Command:

  ```bash
  python scripts/build_budget_metrics.py \
    --predictions outputs/predictions/validation_budget_predictions.parquet \
    --bootstrap-samples 2000 \
    --random-state 42 \
    --output outputs/reports/budget_metrics.csv
  ```

- Status: output missing because its row-level validation-prediction dependency
  is missing. No differently named equivalent was found.

### `outputs/reports/ablation_sensitivity.csv`

- Producer: `scripts/build_ablation_sensitivity.py`, through
  `tcfs.publication_aggregation.aggregate_ablation_sensitivity`.
- Input artifact: genuine row-level validation predictions at the canonical
  proposed path
  `outputs/predictions/validation_ablation_sensitivity_predictions.parquet`.
- Required input columns: `file_type`, `analysis`, `setting`, `x_value`,
  `row_id`, `y_true`, `prediction`, `protocol_role`, and `protocol_week`; an
  optional `protocol_version` must equal the frozen protocol version.
- Output columns: `file_type`, `analysis`, `setting`, `metric`, `estimate`,
  `ci_lower`, `ci_upper`, `x_value`, `n_samples`, `n_positive`, `n_negative`,
  `bootstrap_samples`, `bootstrap_random_state`, `protocol_version`.
- Command:

  ```bash
  python scripts/build_ablation_sensitivity.py \
    --predictions \
      outputs/predictions/validation_ablation_sensitivity_predictions.parquet \
    --ablation-baseline full_tcfs \
    --bootstrap-samples 2000 \
    --random-state 42 \
    --output outputs/reports/ablation_sensitivity.csv
  ```

- Status: output missing because its row-level validation-prediction dependency
  is missing. No differently named equivalent was found.

Both aggregation functions reject any protocol role other than `validation`
and any week outside 41-52. Ablation confidence intervals use paired bootstrap
differences against `full_tcfs`; rho sensitivity reports absolute validation
macro-F1 intervals.

## Final-test gate

The presence of `tcfs-temporal-v1.0` in temporal assignments freezes the split
definition, but does not freeze a fitted pipeline. Official weeks 53-64 may be
opened only after every experiment has a `freeze_manifest.json` covering the
selected features, classifier hyperparameters, threshold, seed or ensemble
rule, model, and protocol version. No such experiment directory or manifest is
currently present, so final-test execution is not yet authorized by the
project's readiness checks.

