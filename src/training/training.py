"""Training orchestration utilities for the Digit Recognizer model."""

from typing import Any, TypedDict

import numpy as np

from evaluation.evaluation import run_evaluation
from training.backpropagation.backward_pass import run_backward_pass
from training.error.categorial_cross_entropy import categorical_cross_entropy
from training.forward.forward_pass import run_forward_pass
from training.parameter.initialize import initialize_weights_and_bias

SUPPORTED_MODEL_NAMES = {
    "single_layer_softmax_classifier",
    "one_hidden_layer_relu_classifier",
    "multi_layer_relu_classifier",
}
SUPPORTED_OPTIMIZER = "batch_gradient_descent"


class InitialTrainingOutput(TypedDict):
    """Output values from one full training pass."""

    initial_parameters: dict[str, np.ndarray]
    forward_output: dict[str, np.ndarray]
    backward_output: dict[str, Any]
    updated_parameters: dict[str, np.ndarray]


class TrainingIterationsOutput(TypedDict):
    """Output values from multiple training iterations."""

    final_parameters: dict[str, np.ndarray]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    train_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]
    validation_loss: list[float]


def validate_training_configuration(
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    """Validate supported model and training configuration values.

    Args:
        model_config: Model configuration section.
        training_config: Training configuration section.

    Raises:
        ValueError: If the configured model is unsupported.
        ValueError: If the configured optimizer is unsupported.
    """
    model_name = str(model_config["name"])
    optimizer = str(training_config["optimizer"])

    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model name: {model_name}")

    if optimizer != SUPPORTED_OPTIMIZER:
        raise ValueError(f"Unsupported optimizer: {optimizer}")


def _copy_parameters(
    parameters: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Create a copy of a parameter dictionary.

    Args:
        parameters: Dictionary containing weights and biases for each layer.

    Returns:
        Dictionary containing copied parameter arrays.
    """
    return {
        parameter_name: parameter_value.copy()
        for parameter_name, parameter_value in parameters.items()
    }


def run_initial_training_step(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> InitialTrainingOutput:
    """Run one full training step.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        model_config: Model configuration section.
        training_config: Training configuration section.

    Returns:
        Dictionary containing initial parameters, forward-pass outputs,
        backward-pass outputs, and updated parameters.
    """
    validate_training_configuration(
        model_config=model_config,
        training_config=training_config,
    )

    neurons_profile = list(model_config["neurons_profile"])
    learning_rate = float(training_config["learning_rate"])

    parameters = initialize_weights_and_bias(
        x_train=x_train,
        neurons_profile=neurons_profile,
    )
    initial_parameters = _copy_parameters(parameters)

    forward_output = run_forward_pass(
        x_train=x_train,
        y_train=y_train,
        parameters=parameters,
        neurons_profile=neurons_profile,
    )

    if not bool(training_config["regularization"]["enabled"]):
        lambda_coefficient = 0.0
    else:
        lambda_coefficient = float(
            training_config["regularization"]["lambda"],
        )

    backward_output = run_backward_pass(
        x_train=x_train,
        forward_pass_results=forward_output,
        parameters=parameters,
        neurons_profile=neurons_profile,
        lambda_coefficient=lambda_coefficient,
        learning_rate=learning_rate,
    )

    return {
        "initial_parameters": initial_parameters,
        "forward_output": forward_output,
        "backward_output": backward_output,
        "updated_parameters": backward_output["parameters"],
    }


def run_training_iterations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> TrainingIterationsOutput:
    """Run multiple training iterations.

    The model parameters are initialized once. Each iteration runs a forward
    pass, evaluates predictions, optionally evaluates validation performance,
    computes gradients using the training split, updates parameters, and carries
    the updated parameters into the next iteration.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        model_config: Model configuration section.
        training_config: Training configuration section.
        x_validation: Optional validation feature matrix.
        y_validation: Optional validation label array.

    Returns:
        Dictionary containing final parameters, final predictions, train loss
        history, train accuracy history, validation loss history, and
        validation accuracy history.

    Raises:
        ValueError: If num_iterations is less than 1.
        ValueError: If only one validation array is provided.
    """
    validate_training_configuration(
        model_config=model_config,
        training_config=training_config,
    )

    neurons_profile = list(model_config["neurons_profile"])
    learning_rate = float(training_config["learning_rate"])
    num_iterations = int(training_config["num_iterations"])

    if num_iterations < 1:
        raise ValueError("num_iterations must be at least 1.")

    if (x_validation is None) != (y_validation is None):
        raise ValueError("x_validation and y_validation must be provided together.")

    parameters = initialize_weights_and_bias(
        x_train=x_train,
        neurons_profile=neurons_profile,
    )

    train_loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    validation_loss_history: list[float] = []
    validation_accuracy_history: list[float] = []

    train_predictions = np.array([])
    validation_predictions = np.array([])

    output_layer = len(neurons_profile)

    if not bool(training_config["regularization"]["enabled"]):
        lambda_coefficient = 0.0
    else:
        lambda_coefficient = float(
            training_config["regularization"]["lambda"],
        )

    for _iteration in range(num_iterations):
        train_forward_output = run_forward_pass(
            x_train=x_train,
            y_train=y_train,
            parameters=parameters,
            neurons_profile=neurons_profile,
        )

        train_evaluation_output = run_evaluation(
            y=y_train,
            ypred=train_forward_output["predictions"],
        )

        if x_validation is not None and y_validation is not None:
            validation_forward_output = run_forward_pass(
                x_train=x_validation,
                y_train=y_validation,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

            validation_evaluation_output = run_evaluation(
                y=y_validation,
                ypred=validation_forward_output["predictions"],
            )

            validation_loss = categorical_cross_entropy(
                y_one_hot=validation_forward_output["Y_one_hot"],
                y_pred=validation_forward_output[f"A{output_layer}"],
            )

            validation_loss_history.append(validation_loss)
            validation_accuracy_history.append(
                validation_evaluation_output["accuracy"],
            )
            validation_predictions = validation_forward_output["predictions"]

        backward_output = run_backward_pass(
            x_train=x_train,
            forward_pass_results=train_forward_output,
            parameters=parameters,
            neurons_profile=neurons_profile,
            lambda_coefficient=lambda_coefficient,
            learning_rate=learning_rate,
        )

        parameters = backward_output["parameters"]

        train_loss_history.append(backward_output["loss"])
        train_accuracy_history.append(train_evaluation_output["accuracy"])
        train_predictions = train_forward_output["predictions"]

    return {
        "final_parameters": parameters,
        "train_predictions": train_predictions,
        "validation_predictions": validation_predictions,
        "train_loss": train_loss_history,
        "train_accuracy": train_accuracy_history,
        "validation_loss": validation_loss_history,
        "validation_accuracy": validation_accuracy_history,
    }
