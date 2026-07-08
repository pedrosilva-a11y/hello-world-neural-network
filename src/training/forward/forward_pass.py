"""Forward pass orchestration for the Digit Recognizer model."""

from typing import TypedDict

import numpy as np

from training.activation.pre_activation import compute_pre_activation
from training.activation.softmax import softmax
from training.encoding.label_one_hot_representation import label_one_hot_representation
from training.prediction.get_predictions import get_predictions


class ForwardPassOutput(TypedDict):
    """Output values from the forward pass."""

    Z: np.ndarray
    A: np.ndarray
    predictions: np.ndarray
    Y_one_hot: np.ndarray


def run_forward_pass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
) -> ForwardPassOutput:
    """Run the model forward pass.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        W1: Weight matrix.
        b1: Bias vector.

    Returns:
        Dictionary containing Z, A, predictions, and Y_one_hot.
    """
    pre_activation = compute_pre_activation(
        x=x_train,
        w=W1,
        b=b1,
    )

    activation = softmax(logits=pre_activation["Z"])
    predictions = get_predictions(activation=activation["A"])
    y_one_hot = label_one_hot_representation(labels=y_train)

    return {
        "Z": pre_activation["Z"],
        "A": activation["A"],
        "predictions": predictions["predictions"],
        "Y_one_hot": y_one_hot,
    }
