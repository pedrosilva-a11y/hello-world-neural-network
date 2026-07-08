"""Pipeline orchestrator for the Digit Recognizer project."""

import numpy as np

from data_loading.data_loading import load_digit_recognizer_data
from preprocessing.preprocessing import preprocess_digit_recognizer_data
from training.training import run_initial_training_step

DigitRecognizerPreprocessedData = tuple[np.ndarray, np.ndarray, np.ndarray]


def run_digit_recognizer_preprocessing_pipeline() -> DigitRecognizerPreprocessedData:
    """Run the data-loading and preprocessing steps in sequence.

    Returns:
        Tuple containing the training feature matrix, training label array,
        and testing feature matrix.
    """
    loaded_data = load_digit_recognizer_data()

    return preprocess_digit_recognizer_data(loaded_data=loaded_data)


def main() -> None:
    """Run the current Digit Recognizer pipeline and print the output shapes."""
    print("Running Digit Recognizer pipeline...")

    x_train_matrix, y_train_array, x_test_matrix = (
        run_digit_recognizer_preprocessing_pipeline()
    )

    training_output = run_initial_training_step(
        x_train=x_train_matrix,
        y_train=y_train_array,
    )

    z = training_output["Z"]
    a = training_output["A"]
    predictions = training_output["predictions"]
    y_one_hot = training_output["Y_one_hot"]
    loss = training_output["loss"]
    dz = training_output["dZ"]
    dw = training_output["dW"]
    db = training_output["db"]
    w1 = training_output["W1"]
    b1 = training_output["b1"]
    updated_w1 = training_output["updated_W1"]
    updated_b1 = training_output["updated_b1"]

    print("Pipeline completed.")
    print("Train feature matrix shape:", x_train_matrix.shape)
    print("Train label array shape:", y_train_array.shape)
    print("Test feature matrix shape:", x_test_matrix.shape)

    print("Initial W1 shape:", w1.shape)
    print("Initial b1 shape:", b1.shape)

    print("Pre-activation Z shape:", z.shape)
    print("Pre-activation Z preview:")
    print(z[:5, :10])

    print("Softmax activation A shape:", a.shape)
    print("Softmax activation A preview:")
    print(a[:5, :10])

    print("Softmax row-sum preview:")
    print(np.sum(a[:5], axis=1))

    print("Predictions shape:", predictions.shape)
    print("Predictions preview:")
    print(predictions[:10])

    print("True labels preview:")
    print(y_train_array[:10])

    print("One-hot labels shape:", y_one_hot.shape)
    print("One-hot labels preview:")
    print(y_one_hot[:10, :10])

    print("Categorical cross-entropy loss:")
    print(loss)

    print("dZ shape:", dz.shape)
    print("dZ preview:")
    print(dz[:5, :10])

    print("dW shape:", dw.shape)
    print("dW preview:")
    print(dw[:5, :10])

    print("db shape:", db.shape)
    print("db preview:")
    print(db[:, :10])

    print("Updated W1 shape:", updated_w1.shape)
    print("Updated b1 shape:", updated_b1.shape)


if __name__ == "__main__":
    main()
