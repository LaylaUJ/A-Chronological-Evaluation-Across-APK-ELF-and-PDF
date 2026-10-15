# Reproducibility checklist

Before reporting an experiment:

1. Record the Python version and installed packages (`python -m pip freeze`).
2. Preserve the exact YAML configuration used for the run.
3. Record source-data provenance, checksums, and acquisition date.
4. Keep raw data immutable and write transformations to `data/interim/` or
   `data/processed/`.
5. Use the configured seed for Python, NumPy, scikit-learn, and LightGBM.
6. Store metrics, predictions, fitted models, and figures under `outputs/`.
7. Run `python -m ruff check .` and `python -m pytest` before committing.

Fixed seeds make pseudo-random operations repeatable on the same software and
hardware stack, but bit-for-bit LightGBM results can still vary across platforms,
thread counts, compiler versions, and dependency versions. The baseline config
uses one model thread to reduce this source of variation.

