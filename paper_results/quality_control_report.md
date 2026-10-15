# Quality-control report

## Data quality

- DQ001 record counts: PASS (384 findings).
- DQ002 duplicate row IDs: PASS (0 findings).
- DQ003 exact SHA-256 duplicates within a split: WARN (351989 findings).
- DQ004 cross-split duplicate SHA-256 hashes: PASS (0 findings).
- DQ005 missing feature values: NOT_RUN (not available findings).
- DQ006 infinite feature values: NOT_RUN (not available findings).
- DQ007 constant features: NOT_RUN (not available findings).
- DQ008 inconsistent file types: PASS (0 findings).
- DQ009 impossible timestamps: PASS (0 findings).
- DQ010 supplied/derived week mismatch: WARN (644 findings).
- DQ011 class prevalence over time: PASS (384 findings).

Approved exclusions contain only DQ003 exact within-split duplicate hashes and retain the lowest row_id. Saved QC found zero cross-split duplicate hashes, zero duplicate split/row identifiers, zero inconsistent file types, and zero impossible timestamps. There were 644 supplied-versus-derived week discrepancies; the protocol uses derived weeks.

## Leakage audit

- Discovery selectors explicitly reject weeks 41–64 and fit per-window scaling only on their discovery window.
- Model-selection code rejects sealed-test rows and trains validation candidates on weeks 1–40 only.
- Budget, rho, hyperparameters, thresholds, and ensemble choices use validation weeks 41–52.
- Final-evaluation code requires frozen manifests and an explicit logged `--unseal-test`, then refits on weeks 1–52.
- Final prediction rows contain only `sealed_test` weeks 53–64.
- No cross-split duplicate hashes were reported. No evidence of test-label use in model selection was found.

## Result integrity

Recalculated point metrics from sample-level final predictions differed from stored estimates by at most 1.11e-16. All probabilities and reported metrics were checked for finite values and valid bounds. Confidence intervals were checked for lower <= estimate <= upper.

There is 1 stored non-inferiority flag conflict with the requested strict CI rule: ELF. The standardized decision uses the requested rule and preserves the original p-value.

## Traceability

Every numerical claim in `results_summary.md` is derived from one of the standardized CSVs. Exploratory validation analyses are explicitly separated from confirmatory future-test results.

## Final verification

- Required CSV schema and non-empty-row checks: PASS.
- Point-metric cross-check against stored summaries: PASS (maximum absolute difference 1.11e-16).
- Four XLSX files reopened successfully after export; all sheets rendered and checked; no formula errors: PASS.
- Three figures exported as 300-DPI PNG, PDF, and SVG: PASS.
- Test suite: 97 passed.
- Ruff lint: all checks passed.
