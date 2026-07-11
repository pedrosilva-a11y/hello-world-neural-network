"""Forward pass orchestration for the Digit Recognizer model."""

import numpy as np

from training.activation.pre_activation import compute_pre_activation
from training.activation.relu import relu
from training.activation.softmax import softmax
from training.encoding.label_one_hot_representation import label_one_hot_representation
from training.prediction.get_predictions import get_predictions


def run_forward_pass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
) -> dict[str, np.ndarray]:
    """Run the model forward pass.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        parameters: Dictionary containing the weights and bias parameters for each layer.
        neurons_profile: Quantity of neurons per layer, in order.

    Returns:
        Dictionary containing Z, and A for all layers. Also predictions and
            the one-hot sparse representation of the true labels.
    """
    forward_pass_results: dict[str, np.ndarray] = {}

    layers = len(neurons_profile)
    activation=x_train

    for i in range(layers):

        pre_activation = compute_pre_activation(
                a=activation,
                w=parameters[f"W{i+1}"],
                b=parameters[f"b{i+1}"],
                layer_number=i+1,
            )

        forward_pass_results[f"Z{i+1}"] = pre_activation[f"Z{i+1}"]

        if i != (layers - 1):
            activation = relu(pre_activation=pre_activation[f"Z{i+1}"])
        else:
            activation = softmax(logits=pre_activation[f"Z{i+1}"])

        forward_pass_results[f"A{i+1}"] = activation

    predictions = get_predictions(activation=forward_pass_results[f"A{layers}"])
    y_one_hot = label_one_hot_representation(labels=y_train)

    forward_pass_results["predictions"] = predictions
    forward_pass_results["Y_one_hot"] = y_one_hot

    return forward_pass_results
