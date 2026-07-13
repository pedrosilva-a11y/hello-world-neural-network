"""Gradient descent optimizer utilities."""

import numpy as np


def update_parameters_with_gradient_descent(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, np.ndarray]:
    """Update parameters using vanilla gradient descent.

    Args:
        parameters: Current model parameters.
        gradients: Current gradient dictionary.
        learning_rate: Learning rate.

    Returns:
        Updated parameter dictionary.

    Raises:
        KeyError: If a gradient is missing for a parameter.
    """
    updated_parameters: dict[str, np.ndarray] = {}

    for parameter_name, parameter_value in parameters.items():
        gradient_name = f"d{parameter_name}"

        if gradient_name not in gradients:
            raise KeyError(f"Missing gradient for parameter: {parameter_name}")

        updated_parameters[parameter_name] = (
            parameter_value - learning_rate * gradients[gradient_name]
        )

    return updated_parameters
