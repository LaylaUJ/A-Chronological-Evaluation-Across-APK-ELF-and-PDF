# Publication outputs

The publication scripts are read-only with respect to experimental data. They
consume saved CSV or Parquet artifacts and never fit models, select features, or
open sealed data.

## Figures

Run:

```bash
python scripts/generate_publication_figures.py \
  --assignments data/processed/temporal_assignments.parquet \
  --jaccard-stability outputs/reports/evaluation/jaccard_stability.csv \
  --spearman-stability outputs/reports/evaluation/spearman_stability.csv \
  --budget-metrics outputs/reports/budget_metrics.csv \
  --ablation-sensitivity outputs/reports/ablation_sensitivity.csv
```

`budget_metrics` uses the columns `file_type`, `method`, `feature_budget`,
`metric`, `estimate`, `ci_lower`, and `ci_upper`. It must contain saved future
`macro_f1` results for each retained-feature budget.

`ablation_sensitivity` uses `file_type`, `analysis`, `setting`, `metric`,
`estimate`, `ci_lower`, `ci_upper`, and `x_value`. `analysis=ablation` stores
macro-F1 differences from full TCFS; `analysis=sensitivity` stores validation
macro-F1 with numeric rho in `x_value`.

Build these two inputs from saved row-level validation predictions with:

```bash
python scripts/build_budget_metrics.py \
  --predictions outputs/predictions/validation_budget_predictions.parquet

python scripts/build_ablation_sensitivity.py \
  --predictions \
    outputs/predictions/validation_ablation_sensitivity_predictions.parquet
```

Both builders reject discovery rows and official-test weeks. See
`docs/publication-csv-provenance.md` for complete schemas and dependency status.

## Tables

Run:

```bash
python scripts/generate_publication_tables.py \
  --record-counts outputs/reports/data_quality/record_counts.csv \
  --check-results outputs/reports/data_quality/check_results.csv \
  --metrics outputs/reports/evaluation/metrics.csv \
  --noninferiority outputs/reports/evaluation/noninferiority.csv \
  --secondary-comparisons outputs/reports/evaluation/secondary_comparisons.csv
```

Table source data are also exported as CSV and booktabs-compatible LaTeX.
Figures and rendered tables are written as vector PDF, text-preserving SVG, and
300-DPI PNG. SVG identifiers and export metadata are deterministic.
