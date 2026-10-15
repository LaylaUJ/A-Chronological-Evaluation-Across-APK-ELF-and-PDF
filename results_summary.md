# Results summary

## Implemented design

EMBER2024 APK, ELF, and PDF records were assigned chronologically: discovery weeks 1–40 (four non-overlapping 10-week windows), internal validation weeks 41–52, and official future-test weeks 53–64. Component selectors were mutual information, L1 logistic regression, and LightGBM permutation importance. TCFS aggregated their four-window rankings. LightGBM was primary and logistic regression secondary. Budgets were 5%, 10%, 20%, 40%, and 60%; rho values were 0, 0.25, 0.5, and 1. Models used a mean-probability ensemble over seeds 17, 42, and 73.

Only validation data selected budget, rho, hyperparameters, threshold, and ensemble rule. Frozen pipelines were refit on weeks 1–52 and evaluated once on weeks 53–64. Approved DQ003 exclusions removed exact within-split duplicate hashes while retaining the lowest row_id.

## Primary future-test findings

- **APK:** TCFS used 70/696 features (10.1%); future-test macro-F1 was 0.906167 versus 0.920916 for all features (difference -0.014750, 95% CI -0.016542 to -0.012895); **Not non-inferior** under the strict lower-bound > -0.01 rule.
- **ELF:** TCFS used 70/696 features (10.1%); future-test macro-F1 was 0.956921 versus 0.962919 for all features (difference -0.005999, 95% CI -0.010508 to -0.001663); **Not non-inferior** under the strict lower-bound > -0.01 rule.
- **PDF:** TCFS used 35/696 features (5.0%); future-test macro-F1 was 0.959314 versus 0.957973 for all features (difference +0.001341, 95% CI -0.001081 to 0.003842); **Non-inferior** under the strict lower-bound > -0.01 rule.

## Stability

- APK: TCFS mean Jaccard 0.2514; mean Spearman rho 0.6041 across six window pairs.
- ELF: TCFS mean Jaccard 0.1557; mean Spearman rho 0.3137 across six window pairs.
- PDF: TCFS mean Jaccard 0.2359; mean Spearman rho 0.5627 across six window pairs.

## Ablation and sensitivity

- APK: largest observed ablation loss was -0.006791 for `without_frequency`.
- ELF: largest observed ablation loss was -0.006005 for `without_frequency`.
- PDF: largest observed ablation loss was -0.005335 for `mean_rank_only`.

Ablation and rho-sensitivity results use validation weeks 41–52 and are exploratory, not confirmatory future-test analyses. Budget curves are also validation analyses.

## Evidence-supported limitations

- The project directory has no resolvable Git commit, so code provenance is limited to file hashes.
- Feature-level missing, infinite, and constant-value QC was marked NOT_RUN in the saved DQ report.
- The stored ELF non-inferiority boolean used a p-value rule, but the requested manuscript rule is CI-based; the standardized table therefore reports ELF as not non-inferior.
- The 60% candidate budget exceeds the manuscript's stated maximum of 50%; selected primary TCFS models remained at 5–10%, but the 60% validation curve is protocol-deviating exploratory evidence.
