"""Utilities for saving and loading trained model parameters."""

from pathlib import Path
from typing import Any, cast

import numpy as np


def save_parameters(
    parameters: dict[str, np.ndarray],
    file_path: Path,
) -> None:
    """Save trained model parameters to a compressed NumPy file.

    Args:
        parameters: Dictionary containing trained weights and biases.
        file_path: Destination path for the saved parameter file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        file_path,
        **cast(dict[str, Any], parameters),
    )


def load_parameters(
    file_path: Path,
) -> dict[str, np.ndarray]:
    """Load trained model parameters from a NumPy file.

    Args:
        file_path: Path to the saved parameter file.

    Returns:
        Dictionary containing loaded weights and biases.

    Raises:
        FileNotFoundError: If the parameter file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {file_path}")

    with np.load(file_path) as loaded_parameters:
        return {
            parameter_name: loaded_parameters[parameter_name]
            for parameter_name in loaded_parameters.files
        }
