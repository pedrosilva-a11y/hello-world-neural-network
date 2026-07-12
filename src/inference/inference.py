"""Inference utilities for trained Digit Recognizer models."""

from typing import TypedDict

import numpy as np

from training.forward.forward_pass import run_forward_pass


class InferenceOutput(TypedDict):
    """Output values from model inference."""

    predictions: np.ndarray
    forward_output: dict[str, np.ndarray]


def run_inference(
    x_test_matrix: np.ndarray,
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
) -> InferenceOutput:
    """Run inference using trained model parameters.

    Args:
        x_test_matrix: Preprocessed test feature matrix.
        parameters: Dictionary containing trained weights and biases.
        neurons_profile: Quantity of neurons per layer, in order.

    Returns:
        Dictionary containing predicted labels and the full forward-pass output.
    """
    forward_output = run_forward_pass(
        x_train=x_test_matrix,
        parameters=parameters,
        neurons_profile=neurons_profile,
    )

    return {
        "predictions": forward_output["predictions"],
        "forward_output": forward_output,
    }
