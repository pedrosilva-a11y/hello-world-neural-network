"""Utilities for reading and writing JSON files."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

JsonObject = dict[str, object]


def save_json(data: Mapping[str, object], file_path: str | Path) -> None:
    """Save a mapping as a JSON file.

    If the parent directory does not exist, it is created automatically.
    If the file already exists, it is overwritten.

    Args:
        data: Mapping to save as JSON.
        file_path: Destination JSON file path.
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode="w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def read_json(file_path: str | Path) -> JsonObject:
    """Read a JSON file into a dictionary.

    Args:
        file_path: Source JSON file path.

    Returns:
        JSON file contents as a dictionary.
    """
    input_path = Path(file_path)

    with input_path.open(mode="r", encoding="utf-8") as json_file:
        return cast("JsonObject", json.load(json_file))
