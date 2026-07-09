"""Utilities for reading YAML files."""

from pathlib import Path
from typing import Any, cast

import yaml


def read_yaml(file_path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Dictionary containing the parsed YAML content.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file is empty or does not contain a mapping.
    """
    path = Path(file_path)

    with path.open(encoding="utf-8") as file:
        loaded_data = yaml.safe_load(file)

    if loaded_data is None:
        msg = f"YAML file is empty: {path}"
        raise ValueError(msg)

    if not isinstance(loaded_data, dict):
        msg = f"YAML file must contain a top-level mapping: {path}"
        raise ValueError(msg)

    return cast(dict[str, Any], loaded_data)
