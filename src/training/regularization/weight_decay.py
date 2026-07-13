"""L2 regularization loss and gradient term computations."""

from typing import cast

import numpy as np


def _get_regularization_sample_count(
    x_train: np.ndarray,
    regularization_sample_count: int | None,
) -> int:
    """Get the sample count used to scale L2 regularization.

    Args:
        x_train: Feature matrix used in the current training step.
        regularization_sample_count: Optional explicit sample count for
            regularization scaling.

    Returns:
        Positive sample count used as the L2 denominator.

    Raises:
        ValueError: If the resolved sample count is less than 1.
    """
    sample_count = (
        x_train.shape[0]
        if regularization_sample_count is None
        else regularization_sample_count
    )

    if sample_count < 1:
        raise ValueError("regularization_sample_count must be at least 1.")

    return int(sample_count)


def weight_decay_loss_term(
    x_train: np.ndarray,
    lambda_coefficient: float,
    parameters: dict[str, np.ndarray],
    regularization_sample_count: int | None = None,
) -> float:
    """Compute the L2 weight decay term added to the training loss.

    Args:
        x_train: NumPy array containing the input features.
        lambda_coefficient: Weight decay coefficient. When equal to 0,
            regularization is turned off.
        parameters: Dictionary containing the weights and bias parameters for
            each layer.
        regularization_sample_count: Optional explicit sample count used to scale
            the L2 term. Defaults to x_train.shape[0].

    Returns:
        L2 weight decay loss term.
    """
    if lambda_coefficient == 0:
        return 0.0

    sample_count = _get_regularization_sample_count(
        x_train=x_train,
        regularization_sample_count=regularization_sample_count,
    )
    weight_squared_sum = 0.0

    for parameter_name, parameter_value in parameters.items():
        if parameter_name.startswith("W"):
            weight_squared_sum += float(np.sum(np.square(parameter_value)))

    return cast(
        float,
        (lambda_coefficient / (2 * sample_count)) * weight_squared_sum,
    )


def weight_decay_gradient_term(
    x_train: np.ndarray,
    lambda_coefficient: float,
    weight_parameter: np.ndarray,
    regularization_sample_count: int | None = None,
) -> np.ndarray:
    """Compute the L2 weight decay term added to a weight gradient.

    Args:
        x_train: NumPy array containing the input features.
        lambda_coefficient: Weight decay coefficient. When equal to 0,
            regularization is turned off.
        weight_parameter: Weight matrix for the current layer.
        regularization_sample_count: Optional explicit sample count used to scale
            the L2 gradient term. Defaults to x_train.shape[0].

    Returns:
        L2 weight decay gradient term for the current weight matrix.
    """
    if lambda_coefficient == 0:
        return cast(np.ndarray, np.zeros_like(weight_parameter))

    sample_count = _get_regularization_sample_count(
        x_train=x_train,
        regularization_sample_count=regularization_sample_count,
    )

    return cast(
        np.ndarray,
        (lambda_coefficient / sample_count) * weight_parameter,
    )


def apply_weight_decay_to_gradients(
    x_train: np.ndarray,
    lambda_coefficient: float,
    gradients: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    layers: int,
    regularization_sample_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Add L2 weight decay terms to weight gradients.

    Args:
        x_train: NumPy array containing the input features.
        lambda_coefficient: Weight decay coefficient.
        gradients: Dictionary containing dW and db gradients.
        parameters: Dictionary containing weights and biases.
        layers: Number of layers in the neural network.
        regularization_sample_count: Optional explicit sample count used to scale
            L2 gradient terms. Defaults to x_train.shape[0].

    Returns:
        Dictionary containing updated gradients.
    """
    if lambda_coefficient == 0:
        return gradients

    for layer in range(1, layers + 1):
        gradient_name = f"dW{layer}"
        weight_name = f"W{layer}"

        gradients[gradient_name] = cast(
            np.ndarray,
            gradients[gradient_name]
            + weight_decay_gradient_term(
                x_train=x_train,
                lambda_coefficient=lambda_coefficient,
                weight_parameter=parameters[weight_name],
                regularization_sample_count=regularization_sample_count,
            ),
        )

    return gradients
