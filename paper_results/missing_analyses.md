# Missing analyses

No manuscript-required predictive, stability, non-inferiority, secondary-comparison, or ablation analysis is missing.

The saved data-quality run marks feature-matrix checks DQ005 (missing values), DQ006 (infinite values), and DQ007 (constant features) as NOT_RUN because no feature matrix was supplied to that audit. To complete them without touching the sealed evaluation logic, run `python scripts/run_data_quality.py` with all six processed feature matrices and the 696 feature names, writing to a new QC directory. This package does not invent those findings.
