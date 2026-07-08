"""Backward pass orchestration for the Digit Recognizer model."""

from typing import TypedDict

import numpy as np

from training.backpropagation.batch_gradient_descent import batch_gradient_descent
from training.backpropagation.gradients import gradient_computations_softmax
from training.error.categorial_cross_entropy import categorical_cross_entropy


class BackwardPassOutput(TypedDict):
    """Output values from the backward pass."""

    loss: float
    dZ: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    updated_W1: np.ndarray
    updated_b1: np.ndarray


def run_backward_pass(
    x_train: np.ndarray,
    y_one_hot: np.ndarray,
    activation: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    learning_rate: float = 1e-5,
) -> BackwardPassOutput:
    """Run loss, gradient, and parameter-update computations.

    Args:
        x_train: Training feature matrix.
        y_one_hot: One-hot representation of the true labels.
        activation: Softmax probability matrix.
        W1: Current weight matrix.
        b1: Current bias vector.
        learning_rate: Step size used to update the parameters.

    Returns:
        Dictionary containing loss, gradients, and updated parameters.
    """
    loss = categorical_cross_entropy(
        y_one_hot=y_one_hot,
        y_pred=activation,
    )

    gradients = gradient_computations_softmax(
        x=x_train,
        yhot=y_one_hot,
        activation=activation,
    )

    updated_parameters = batch_gradient_descent(
        W1=W1,
        b1=b1,
        dW1=gradients["dW"],
        db1=gradients["db"],
        learning_rate=learning_rate,
    )

    return {
        "loss": loss["loss"],
        "dZ": gradients["dZ"],
        "dW": gradients["dW"],
        "db": gradients["db"],
        "updated_W1": updated_parameters["W1"],
        "updated_b1": updated_parameters["b1"],
    }
