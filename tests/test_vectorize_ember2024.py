from pathlib import Path

import pytest
from scripts.vectorize_ember2024 import main


def test_test_vectorization_requires_explicit_unseal_before_source_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="remain sealed"):
        main(
            [
                "--raw-root",
                str(tmp_path / "missing_raw"),
                "--file-type",
                "ELF",
                "--split",
                "test",
                "--experiment-root",
                str(tmp_path / "missing_experiments"),
            ]
        )
