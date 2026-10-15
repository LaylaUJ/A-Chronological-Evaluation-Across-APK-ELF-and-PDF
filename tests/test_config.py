from pathlib import Path

import pytest

from tcfs.config import load_config


def test_load_config_reads_experiment_seed(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("experiment:\n  name: smoke\n  seed: 7\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["experiment"]["name"] == "smoke"
    assert config["experiment"]["seed"] == 7


@pytest.mark.parametrize("seed_yaml", ["true", "-1", "'42'", "null"])
def test_load_config_rejects_invalid_seed(tmp_path: Path, seed_yaml: str) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        f"experiment:\n  name: invalid\n  seed: {seed_yaml}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"experiment\.seed"):
        load_config(config_path)
