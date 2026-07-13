"""Weight and bias parameter initialization utilities."""

import math

import numpy as np


def _create_weights_ndarray(
    neurons_before: int,
    neurons_current: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Create the weight matrix for one layer.

    Args:
        neurons_before: Quantity of neurons in the previous layer.
        neurons_current: Quantity of neurons in the current layer.
        random_generator: NumPy random generator used for reproducible
            initialization.

    Returns:
        Weight matrix for the current layer.
    """
    return random_generator.standard_normal(
        size=(neurons_before, neurons_current),
    ) * math.sqrt(2 / neurons_before)


def _create_bias_ndarray(neurons_current: int) -> np.ndarray:
    """Create the bias vector for one layer.

    Args:
        neurons_current: Quantity of neurons in the current layer.

    Returns:
        Bias vector for the current layer.
    """
    return np.zeros((1, neurons_current))


def initialize_weights_and_bias(
    x_train: np.ndarray,
    neurons_profile: list[int],
    random_seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Initialize the weight matrix and bias vector for all neural network layers.

    Args:
        x_train: NumPy array containing the input features.
        neurons_profile: List containing the number of neurons per layer, in order.
        random_seed: Optional seed for reproducible parameter initialization.

    Returns:
        Dictionary containing initialized weight matrices and bias vectors.

    Raises:
        ValueError: If neurons_profile is empty.
    """
    if not neurons_profile:
        raise ValueError("neurons_profile must contain at least one layer.")

    input_size = x_train.shape[1]
    sizes = [input_size, *neurons_profile]
    random_generator = np.random.default_rng(random_seed)

    parameters_mapping: dict[str, np.ndarray] = {}

    for i in range(len(neurons_profile)):
        parameters_mapping[f"W{i + 1}"] = _create_weights_ndarray(
            neurons_before=sizes[i],
            neurons_current=sizes[i + 1],
            random_generator=random_generator,
        )
        parameters_mapping[f"b{i + 1}"] = _create_bias_ndarray(
            neurons_current=neurons_profile[i],
        )

    return parameters_mapping
