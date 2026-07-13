"""Training orchestration utilities for the Digit Recognizer model."""

from typing import Any, TypedDict

import numpy as np

from evaluation.evaluation import run_evaluation
from training.backpropagation.backward_pass import run_backward_pass
from training.batching.batching import (
    SUPPORTED_BATCHING_STRATEGIES,
    get_batching_config,
    iter_mini_batches,
)
from training.error.categorial_cross_entropy import categorical_cross_entropy
from training.forward.forward_pass import run_forward_pass
from training.parameter.initialize import initialize_weights_and_bias
from training.regularization.weight_decay import weight_decay_loss_term

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
    best_parameters: dict[str, np.ndarray]
    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    train_loss: list[float]
    train_accuracy: list[float]
    validation_accuracy: list[float]
    validation_loss: list[float]
    best_train_loss: float | None
    best_train_accuracy: float | None
    best_validation_loss: float | None
    best_validation_accuracy: float | None
    best_iteration: int | None
    best_epoch: int | None
    checkpoint_metric: str


def _get_lambda_coefficient(
    training_config: dict[str, Any],
) -> float:
    """Get the active L2 regularization coefficient.

    Args:
        training_config: Training configuration section.

    Returns:
        L2 lambda coefficient. Returns 0.0 when regularization is disabled.
    """
    regularization_config = training_config["regularization"]

    if not bool(regularization_config["enabled"]):
        return 0.0

    return float(regularization_config["lambda"])


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
        ValueError: If the configured batching strategy is unsupported.
        ValueError: If iteration or epoch counts are invalid.
        ValueError: If mini-batch size is invalid.
    """
    model_name = str(model_config["name"])
    optimizer = str(training_config["optimizer"])
    batching_config = get_batching_config(training_config=training_config)
    batching_strategy = str(batching_config["strategy"])

    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported model name: {model_name}")

    if optimizer != SUPPORTED_OPTIMIZER:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if batching_strategy not in SUPPORTED_BATCHING_STRATEGIES:
        raise ValueError(f"Unsupported batching strategy: {batching_strategy}")

    if batching_strategy == "full_batch":
        num_iterations = int(training_config["num_iterations"])

        if num_iterations < 1:
            raise ValueError("num_iterations must be at least 1.")

    if batching_strategy == "mini_batch":
        num_epochs = int(training_config["num_epochs"])
        batch_size = int(batching_config["batch_size"])

        if num_epochs < 1:
            raise ValueError("num_epochs must be at least 1.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")


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


def _compute_training_objective_loss(
    x_train: np.ndarray,
    forward_output: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    neurons_profile: list[int],
    lambda_coefficient: float,
) -> float:
    """Compute full training objective loss.

    Args:
        x_train: Training feature matrix.
        forward_output: Forward-pass output for the training data.
        parameters: Current model parameters.
        neurons_profile: Quantity of neurons per layer, in order.
        lambda_coefficient: Weight decay coefficient.

    Returns:
        Training objective loss, including L2 penalty when enabled.
    """
    output_layer = len(neurons_profile)

    return categorical_cross_entropy(
        y_one_hot=forward_output["Y_one_hot"],
        y_pred=forward_output[f"A{output_layer}"],
    ) + weight_decay_loss_term(
        x_train=x_train,
        lambda_coefficient=lambda_coefficient,
        parameters=parameters,
    )


def _compute_validation_loss(
    forward_output: dict[str, np.ndarray],
    neurons_profile: list[int],
) -> float:
    """Compute validation categorical cross-entropy loss.

    Args:
        forward_output: Forward-pass output for the validation data.
        neurons_profile: Quantity of neurons per layer, in order.

    Returns:
        Validation categorical cross-entropy loss.
    """
    output_layer = len(neurons_profile)

    return categorical_cross_entropy(
        y_one_hot=forward_output["Y_one_hot"],
        y_pred=forward_output[f"A{output_layer}"],
    )


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

    lambda_coefficient = _get_lambda_coefficient(
        training_config=training_config,
    )

    backward_output = run_backward_pass(
        x_train=x_train,
        forward_pass_results=forward_output,
        parameters=parameters,
        neurons_profile=neurons_profile,
        lambda_coefficient=lambda_coefficient,
        learning_rate=learning_rate,
        regularization_sample_count=x_train.shape[0],
    )

    return {
        "initial_parameters": initial_parameters,
        "forward_output": forward_output,
        "backward_output": backward_output,
        "updated_parameters": backward_output["parameters"],
    }


def _run_full_batch_training_iterations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> TrainingIterationsOutput:
    """Run full-batch training iterations.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        model_config: Model configuration section.
        training_config: Training configuration section.
        x_validation: Optional validation feature matrix.
        y_validation: Optional validation label array.

    Returns:
        Dictionary containing final parameters, best parameters, and metric histories.
    """
    neurons_profile = list(model_config["neurons_profile"])
    learning_rate = float(training_config["learning_rate"])
    num_iterations = int(training_config["num_iterations"])

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

    best_parameters: dict[str, np.ndarray] | None = None
    best_train_loss: float | None = None
    best_train_accuracy: float | None = None
    best_validation_loss: float | None = None
    best_validation_accuracy: float | None = None
    best_iteration: int | None = None

    lambda_coefficient = _get_lambda_coefficient(
        training_config=training_config,
    )

    for iteration_index in range(num_iterations):
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

        candidate_parameters = _copy_parameters(parameters)
        validation_loss: float | None = None
        validation_accuracy: float | None = None

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

            validation_loss = _compute_validation_loss(
                forward_output=validation_forward_output,
                neurons_profile=neurons_profile,
            )
            validation_accuracy = validation_evaluation_output["accuracy"]

            validation_loss_history.append(validation_loss)
            validation_accuracy_history.append(validation_accuracy)
            validation_predictions = validation_forward_output["predictions"]

        backward_output = run_backward_pass(
            x_train=x_train,
            forward_pass_results=train_forward_output,
            parameters=parameters,
            neurons_profile=neurons_profile,
            lambda_coefficient=lambda_coefficient,
            learning_rate=learning_rate,
            regularization_sample_count=x_train.shape[0],
        )

        train_loss = backward_output["loss"]
        train_accuracy = train_evaluation_output["accuracy"]

        if validation_loss is not None and (
            best_validation_loss is None or validation_loss < best_validation_loss
        ):
            best_parameters = candidate_parameters
            best_train_loss = train_loss
            best_train_accuracy = train_accuracy
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            best_iteration = iteration_index + 1

        parameters = backward_output["parameters"]

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        train_predictions = train_forward_output["predictions"]

    if best_parameters is None:
        best_parameters = _copy_parameters(parameters)

    return {
        "final_parameters": parameters,
        "best_parameters": best_parameters,
        "train_predictions": train_predictions,
        "validation_predictions": validation_predictions,
        "train_loss": train_loss_history,
        "train_accuracy": train_accuracy_history,
        "validation_loss": validation_loss_history,
        "validation_accuracy": validation_accuracy_history,
        "best_train_loss": best_train_loss,
        "best_train_accuracy": best_train_accuracy,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "best_iteration": best_iteration,
        "best_epoch": None,
        "checkpoint_metric": "validation_loss",
    }


def _run_mini_batch_training_iterations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> TrainingIterationsOutput:
    """Run mini-batch training epochs.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        model_config: Model configuration section.
        training_config: Training configuration section.
        x_validation: Optional validation feature matrix.
        y_validation: Optional validation label array.

    Returns:
        Dictionary containing final parameters, best parameters, and metric histories.
    """
    neurons_profile = list(model_config["neurons_profile"])
    learning_rate = float(training_config["learning_rate"])
    num_epochs = int(training_config["num_epochs"])

    batching_config = get_batching_config(training_config=training_config)
    batch_size = int(batching_config["batch_size"])
    shuffle = bool(batching_config["shuffle"])
    random_seed = int(batching_config["random_seed"])

    random_generator = np.random.default_rng(random_seed)

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

    best_parameters: dict[str, np.ndarray] | None = None
    best_train_loss: float | None = None
    best_train_accuracy: float | None = None
    best_validation_loss: float | None = None
    best_validation_accuracy: float | None = None
    best_epoch: int | None = None

    lambda_coefficient = _get_lambda_coefficient(
        training_config=training_config,
    )

    regularization_sample_count = x_train.shape[0]

    for epoch_index in range(num_epochs):
        for x_batch, y_batch in iter_mini_batches(
            x_train=x_train,
            y_train=y_train,
            batch_size=batch_size,
            shuffle=shuffle,
            random_generator=random_generator,
        ):
            train_batch_forward_output = run_forward_pass(
                x_train=x_batch,
                y_train=y_batch,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

            backward_output = run_backward_pass(
                x_train=x_batch,
                forward_pass_results=train_batch_forward_output,
                parameters=parameters,
                neurons_profile=neurons_profile,
                lambda_coefficient=lambda_coefficient,
                learning_rate=learning_rate,
                regularization_sample_count=regularization_sample_count,
            )

            parameters = backward_output["parameters"]

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

        train_loss = _compute_training_objective_loss(
            x_train=x_train,
            forward_output=train_forward_output,
            parameters=parameters,
            neurons_profile=neurons_profile,
            lambda_coefficient=lambda_coefficient,
        )
        train_accuracy = train_evaluation_output["accuracy"]

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        train_predictions = train_forward_output["predictions"]

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

            validation_loss = _compute_validation_loss(
                forward_output=validation_forward_output,
                neurons_profile=neurons_profile,
            )
            validation_accuracy = validation_evaluation_output["accuracy"]

            validation_loss_history.append(validation_loss)
            validation_accuracy_history.append(validation_accuracy)
            validation_predictions = validation_forward_output["predictions"]

            if (
                best_validation_loss is None
                or validation_loss < best_validation_loss
            ):
                best_parameters = _copy_parameters(parameters)
                best_train_loss = train_loss
                best_train_accuracy = train_accuracy
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch_index + 1

    if best_parameters is None:
        best_parameters = _copy_parameters(parameters)

    return {
        "final_parameters": parameters,
        "best_parameters": best_parameters,
        "train_predictions": train_predictions,
        "validation_predictions": validation_predictions,
        "train_loss": train_loss_history,
        "train_accuracy": train_accuracy_history,
        "validation_loss": validation_loss_history,
        "validation_accuracy": validation_accuracy_history,
        "best_train_loss": best_train_loss,
        "best_train_accuracy": best_train_accuracy,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "best_iteration": None,
        "best_epoch": best_epoch,
        "checkpoint_metric": "validation_loss",
    }


def run_training_iterations(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    x_validation: np.ndarray | None = None,
    y_validation: np.ndarray | None = None,
) -> TrainingIterationsOutput:
    """Run training using the configured batching strategy.

    Args:
        x_train: Training feature matrix.
        y_train: Training label array.
        model_config: Model configuration section.
        training_config: Training configuration section.
        x_validation: Optional validation feature matrix.
        y_validation: Optional validation label array.

    Returns:
        Dictionary containing final parameters, best parameters, final predictions,
        train loss history, train accuracy history, validation loss history, and
        validation accuracy history.

    Raises:
        ValueError: If only one validation array is provided.
        ValueError: If the configured batching strategy is unsupported.
    """
    validate_training_configuration(
        model_config=model_config,
        training_config=training_config,
    )

    if (x_validation is None) != (y_validation is None):
        raise ValueError("x_validation and y_validation must be provided together.")

    batching_config = get_batching_config(training_config=training_config)
    batching_strategy = str(batching_config["strategy"])

    if batching_strategy == "full_batch":
        return _run_full_batch_training_iterations(
            x_train=x_train,
            y_train=y_train,
            model_config=model_config,
            training_config=training_config,
            x_validation=x_validation,
            y_validation=y_validation,
        )

    if batching_strategy == "mini_batch":
        return _run_mini_batch_training_iterations(
            x_train=x_train,
            y_train=y_train,
            model_config=model_config,
            training_config=training_config,
            x_validation=x_validation,
            y_validation=y_validation,
        )

    raise ValueError(f"Unsupported batching strategy: {batching_strategy}")
