"""Training orchestration utilities for the Digit Recognizer model."""

from typing import TypedDict

import numpy as np

from training.activation.pre_activation import compute_pre_activation
from training.activation.softmax import softmax
from training.error.categorial_cross_entropy import categorical_cross_entropy
from training.parameter.initialize import initialize_weights_and_bias
from training.prediction.get_predictions import get_predictions
from training.prediction.label_one_hot_representation import label_one_hot_representation

OUTPUT_LAYER_NEURONS = 10


class InitialTrainingOutput(TypedDict):
    """Output values from the initial training forward pass."""

    W1: np.ndarray
    b1: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    predictions: np.ndarray
    Y_one_hot: np.ndarray
    loss: float


def run_initial_training_step(
    x_train: np.ndarray,
    y_train: np.ndarray,
    output_neurons: int = OUTPUT_LAYER_NEURONS,
) -> InitialTrainingOutput:
    """Run initialization, forward pass, prediction, one-hot encoding, and loss.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        output_neurons: Number of output neurons/classes.

    Returns:
        Dictionary containing W1, b1, Z, A, predictions, Y_one_hot, and loss.
    """
    parameters = initialize_weights_and_bias(
        x_train=x_train,
        h=output_neurons,
    )

    pre_activation = compute_pre_activation(
        x=x_train,
        w=parameters["W1"],
        b=parameters["b1"],
    )

    activation = softmax(logits=pre_activation["Z"])
    predictions = get_predictions(activation=activation["A"])
    y_one_hot = label_one_hot_representation(labels=y_train)

    loss = categorical_cross_entropy(
        y_one_hot=y_one_hot,
        y_pred=activation["A"],
    )

    return {
        "W1": parameters["W1"],
        "b1": parameters["b1"],
        "Z": pre_activation["Z"],
        "A": activation["A"],
        "predictions": predictions["predictions"],
        "Y_one_hot": y_one_hot,
        "loss": loss["loss"],
    }
