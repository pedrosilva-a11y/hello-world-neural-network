"""Forward pass orchestration for the Digit Recognizer model."""

import numpy as np

from training.activation.pre_activation import compute_pre_activation
from training.activation.relu import relu
from training.activation.softmax import softmax
from training.encoding.label_one_hot_representation import label_one_hot_representation
from training.prediction.get_predictions import get_predictions


def run_forward_pass(
    x_train: np.ndarray,
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
    y_train: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the model forward pass.

    Args:
        x_train: Feature matrix.
        parameters: Dictionary containing weights and biases for each layer.
        neurons_profile: Quantity of neurons per layer, in order.
        y_train: Optional label array. If provided, one-hot labels are added to the output.

    Returns:
        Dictionary containing Z and A for all layers, predictions, and optionally
        the one-hot representation of the true labels.
    """
    forward_pass_results: dict[str, np.ndarray] = {}

    layers = len(neurons_profile)
    activation = x_train

    for i in range(layers):
        layer_number = i + 1

        pre_activation = compute_pre_activation(
            a=activation,
            w=parameters[f"W{layer_number}"],
            b=parameters[f"b{layer_number}"],
            layer_number=layer_number,
        )

        forward_pass_results[f"Z{layer_number}"] = pre_activation[f"Z{layer_number}"]

        if i != layers - 1:
            activation = relu(pre_activation=pre_activation[f"Z{layer_number}"])
        else:
            activation = softmax(logits=pre_activation[f"Z{layer_number}"])

        forward_pass_results[f"A{layer_number}"] = activation

    forward_pass_results["predictions"] = get_predictions(
        activation=forward_pass_results[f"A{layers}"],
    )

    if y_train is not None:
        forward_pass_results["Y_one_hot"] = label_one_hot_representation(labels=y_train)

    return forward_pass_results