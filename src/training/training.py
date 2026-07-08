"""Training orchestration utilities for the Digit Recognizer model."""

from typing import TypedDict

import numpy as np

from training.backpropagation.backward_pass import run_backward_pass
from training.forward.forward_pass import run_forward_pass
from training.parameter.initialize import initialize_weights_and_bias

OUTPUT_LAYER_NEURONS = 10
DEFAULT_LEARNING_RATE = 1e-5


class InitialTrainingOutput(TypedDict):
    """Output values from one full training pass."""

    W1: np.ndarray
    b1: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    predictions: np.ndarray
    Y_one_hot: np.ndarray
    loss: float
    dZ: np.ndarray
    dW: np.ndarray
    db: np.ndarray
    updated_W1: np.ndarray
    updated_b1: np.ndarray


def run_initial_training_step(
    x_train: np.ndarray,
    y_train: np.ndarray,
    output_neurons: int = OUTPUT_LAYER_NEURONS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> InitialTrainingOutput:
    """Run one full training pass.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        output_neurons: Number of output neurons/classes.
        learning_rate: Step size used to update the parameters.

    Returns:
        Dictionary containing initial parameters, forward-pass outputs,
        backward-pass outputs, and updated parameters.
    """
    parameters = initialize_weights_and_bias(
        x_train=x_train,
        h=output_neurons,
    )

    forward_output = run_forward_pass(
        x_train=x_train,
        y_train=y_train,
        W1=parameters["W1"],
        b1=parameters["b1"],
    )

    backward_output = run_backward_pass(
        x_train=x_train,
        y_one_hot=forward_output["Y_one_hot"],
        activation=forward_output["A"],
        W1=parameters["W1"],
        b1=parameters["b1"],
        learning_rate=learning_rate,
    )

    return {
        "W1": parameters["W1"],
        "b1": parameters["b1"],
        "Z": forward_output["Z"],
        "A": forward_output["A"],
        "predictions": forward_output["predictions"],
        "Y_one_hot": forward_output["Y_one_hot"],
        "loss": backward_output["loss"],
        "dZ": backward_output["dZ"],
        "dW": backward_output["dW"],
        "db": backward_output["db"],
        "updated_W1": backward_output["updated_W1"],
        "updated_b1": backward_output["updated_b1"],
    }
