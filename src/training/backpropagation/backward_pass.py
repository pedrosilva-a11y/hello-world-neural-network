"""Backward pass orchestration for the Digit Recognizer model."""

from typing import Any

import numpy as np

from training.backpropagation.gradient.gradients_cross_entropy import (
    gradient_computations_relu,
    gradient_computations_softmax,
)
from training.backpropagation.optimizer.batch_gradient_descent import batch_gradient_descent
from training.error.categorial_cross_entropy import categorical_cross_entropy


def run_backward_pass(
    x_train: np.ndarray,
    forward_pass_results: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
    learning_rate: float = 1e-5,
) -> dict[str, Any]:
    """Run loss, gradient, and parameter-update computations.

    Args:
        x_train: Training feature matrix.
        forward_pass_results: Dictionary containing Z, and A for all layers.
            Also predictions and the one-hot sparse representation of the true labels.
        parameters: Dictionary containing the weights and bias parameters for each layer.
        learning_rate: Step size used to update the parameters.
        neurons_profile: Quantity of neurons per layer, in order.

    Returns:
        Dictionary containing loss, gradients, and updated parameters.
    """
    y_one_hot = forward_pass_results["Y_one_hot"]

    layers = len(neurons_profile)

    loss = categorical_cross_entropy(
        y_one_hot=y_one_hot,
        y_pred=forward_pass_results[f"A{layers}"],
    )

    gradients: dict[str, np.ndarray] = {}

    for i in range(layers - 1, -1, -1):

        if i != (layers - 1):
            gradients = gradient_computations_relu(
                x=x_train,
                gradients=gradients,
                parameters=parameters,
                forward_pass_results=forward_pass_results,
                layer=i + 1,
            )
        else:
            gradients = gradient_computations_softmax(
                x=x_train,
                yhot=y_one_hot,
                forward_pass_results=forward_pass_results,
                layer=i + 1,
            )

    for i in range(layers):
        batch_gradient_descent(
            gradients=gradients,
            parameters=parameters,
            layer=i + 1,
            learning_rate=learning_rate,
        )

    return {
        "loss": loss,
        "gradients": gradients,
        "parameters": parameters,
    }
