import logging
from pathlib import Path

from tcfs.logging_config import configure_logging


def test_configure_logging_resolves_and_creates_log_directory(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "logging.yaml"
    config_path.write_text(
        """
version: 1
handlers:
  file:
    class: logging.FileHandler
    filename: outputs/reports/test.log
root:
  level: INFO
  handlers: [file]
""".strip(),
        encoding="utf-8",
    )

    resolved = configure_logging(config_path, project_root=tmp_path)
    logging.getLogger("tcfs.test").info("configured")

    expected = tmp_path / "outputs" / "reports" / "test.log"
    assert resolved["handlers"]["file"]["filename"] == str(expected)
    assert expected.read_text(encoding="utf-8") == "configured\n"

