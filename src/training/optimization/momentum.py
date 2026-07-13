"""Momentum optimizer utilities."""

import numpy as np


def _validate_beta(beta: float) -> None:
    """Validate the momentum coefficient.

    Args:
        beta: Momentum coefficient.

    Raises:
        ValueError: If beta is outside the supported range.
    """
    if beta < 0.0 or beta >= 1.0:
        raise ValueError("beta must be greater than or equal to 0.0 and less than 1.0.")


def initialize_velocity(
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Initialize momentum velocity arrays for each parameter.

    Args:
        parameters: Current model parameters.

    Returns:
        Dictionary of zero velocity arrays matching each parameter.
    """
    return {
        parameter_name: np.zeros_like(parameter_value)
        for parameter_name, parameter_value in parameters.items()
    }


def update_parameters_with_momentum(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    velocity: dict[str, np.ndarray],
    learning_rate: float,
    beta: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Update parameters using momentum.

    Args:
        parameters: Current model parameters.
        gradients: Current gradient dictionary.
        velocity: Current velocity dictionary.
        learning_rate: Learning rate.
        beta: Momentum coefficient.

    Returns:
        Tuple containing updated parameters and updated velocity.

    Raises:
        ValueError: If beta is outside the supported range.
        KeyError: If a gradient or velocity is missing for a parameter.
    """
    _validate_beta(beta=beta)

    updated_parameters: dict[str, np.ndarray] = {}
    updated_velocity: dict[str, np.ndarray] = {}

    for parameter_name, parameter_value in parameters.items():
        gradient_name = f"d{parameter_name}"

        if gradient_name not in gradients:
            raise KeyError(f"Missing gradient for parameter: {parameter_name}")

        if parameter_name not in velocity:
            raise KeyError(f"Missing velocity for parameter: {parameter_name}")

        updated_velocity[parameter_name] = (
            beta * velocity[parameter_name]
            + gradients[gradient_name]
        )

        updated_parameters[parameter_name] = (
            parameter_value
            - learning_rate * updated_velocity[parameter_name]
        )

    return updated_parameters, updated_velocity
