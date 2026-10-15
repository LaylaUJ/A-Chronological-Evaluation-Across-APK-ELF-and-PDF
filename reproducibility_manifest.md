# Reproducibility manifest

Generated: 2026-10-15T04:43:29.891635+00:00

Protocol: `tcfs-temporal-v1.0`. Dataset: EMBER2024, local processed metadata at `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\data\processed\ember2024_metadata.parquet`.

Code version: No Git commit available; working tree is unversioned.

## Repository inventory

- scripts: 29
- configs: 4
- processed_data_files: 25
- prediction_files: 10
- model_files: 42
- log_files: 3
- existing_figures: 13

## Environment

- python: 3.12.14
- platform: Windows-11-10.0.26300-SP0
- pandas: 3.0.6
- numpy: 2.5.3
- scipy: 1.18.1
- scikit-learn: 1.9.1
- lightgbm: 4.7.0
- pyarrow: 22.0.0
- matplotlib: 3.11.2
- seaborn: 0.13.2
- pyyaml: 6.0.3
- joblib: 1.6.0
- pytest: 9.1.1

Random seeds: model ensemble 17, 42, 73; bootstrap 42; 2,000 resamples.

## Commands

- standardized package: `python scripts/build_paper_results_package.py --output-dir C:/workspace/paper_results --force`
- sealed final evaluation: `python scripts/run_frozen_format_evaluation.py ... --unseal-test (already completed; do not rerun)`
- budget aggregation: `python scripts/build_budget_metrics.py --predictions outputs/predictions/validation_budget_predictions.parquet --output outputs/reports/budget_metrics.csv`
- ablation aggregation: `python scripts/build_ablation_sensitivity.py --predictions outputs/predictions/validation_ablation_sensitivity_predictions.parquet --output outputs/reports/ablation_sensitivity.csv`

## Figure and table mapping

- Figure 2 and table primary: `budget_metrics.csv`, `metrics.csv`.
- Figure 3 and table stability: `jaccard_stability.csv`, `spearman_stability.csv`.
- Figure 4 and table ablation: `ablation_sensitivity.csv`.
- Table non-inferiority: `noninferiority.csv`, `secondary_comparisons.csv`.

## Source SHA-256 hashes

- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\data\processed\ember2024_metadata.parquet`: `ce5864864316e77f3cc6fc0e5f34402f00aa09941869b2ddfe2f3ac15086c8eb`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\data\processed\temporal_assignments.parquet`: `0b908ca74aa934c78b7018421aaf06148af3a200f9d36f338edaefd9998125e5`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\predictions\final_test_predictions.parquet`: `04cbca79dab98c54335d5d3cf66af223a0f47745920bf8cf36525a022c6f4b08`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\predictions\validation_budget_predictions.parquet`: `2dcd1585d7d320250d087def6d9a48c528c38f895669ca2aead8cc7fe7d290f2`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\predictions\validation_ablation_sensitivity_predictions.parquet`: `0cdcd63ee13244b4528bc92a1142fad51c030af36bf8ef8abbe4096a8cfbf35b`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\reports\data_quality\approved_exclusions.csv`: `2386c78e9bc06c8fa10686921eb7b13f56f08b132e961a84e6e41a2dc80dcd2a`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\reports\evaluation\metrics.csv`: `dbf2c8bdb631bc75f209f61d4c5730ea8a42621459a2dcaf4eef26e736e496ae`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\reports\evaluation\noninferiority.csv`: `6c41275adb9198632fe6a287f6a041bdab57c9393be8ec338609bd0ebd39e546`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\reports\evaluation\secondary_comparisons.csv`: `244cf9ae5658c01e4e2630a6112299adf3044468aa7df59b7131f7a678d2df2b`
- `C:\Users\LENOVO\OneDrive - zuj.edu.jo\Documents\ChatGPT\malware detection\tcfs-malware\outputs\reports\feature_selection\feature_results.parquet`: `d9111e32c46605846363f53786a9f43a4d68ac275d19a08bb70a28edc7f73007`
