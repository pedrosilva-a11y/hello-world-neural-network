"""Backward pass orchestration for the Digit Recognizer model."""

from typing import Any

import numpy as np

from training.backpropagation.gradient.gradients_cross_entropy import (
    gradient_computations_relu,
    gradient_computations_softmax,
)
from training.error.categorial_cross_entropy import categorical_cross_entropy
from training.regularization.weight_decay import (
    apply_weight_decay_to_gradients,
    weight_decay_loss_term,
)


def run_backward_pass(
    x_train: np.ndarray,
    forward_pass_results: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
    lambda_coefficient: float,
    regularization_sample_count: int | None = None,
) -> dict[str, Any]:
    """Run loss and gradient computations.

    Args:
        x_train: Training feature matrix.
        forward_pass_results: Dictionary containing Z and A values for all layers,
            predictions, and the one-hot representation of the true labels.
        parameters: Dictionary containing the weights and bias parameters for each layer.
        neurons_profile: Quantity of neurons per layer, in order.
        lambda_coefficient: Weight decay coefficient.
        regularization_sample_count: Optional explicit sample count used to scale
            L2 regularization. Defaults to x_train.shape[0].

    Returns:
        Dictionary containing loss and gradients.
    """
    y_one_hot = forward_pass_results["Y_one_hot"]
    layers = len(neurons_profile)

    loss = categorical_cross_entropy(
        y_one_hot=y_one_hot,
        y_pred=forward_pass_results[f"A{layers}"],
    ) + weight_decay_loss_term(
        x_train=x_train,
        lambda_coefficient=lambda_coefficient,
        parameters=parameters,
        regularization_sample_count=regularization_sample_count,
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

    gradients = apply_weight_decay_to_gradients(
        x_train=x_train,
        lambda_coefficient=lambda_coefficient,
        gradients=gradients,
        parameters=parameters,
        layers=layers,
        regularization_sample_count=regularization_sample_count,
    )

    return {
        "loss": loss,
        "gradients": gradients,
    }
