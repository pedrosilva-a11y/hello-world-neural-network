"""Tests for Digit Recognizer inference utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from inference import inference


class TestRunInference(unittest.TestCase):
    """Tests for run_inference."""

    def test_run_inference_coordinates_forward_pass_without_labels(self) -> None:
        """Run forward pass without labels and return predictions."""
        x_test_matrix = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
        )
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.01, 0.02]]),
        }
        neurons_profile = [2]

        predictions = np.array([1, 0])
        forward_output = {
            "Z1": np.array(
                [
                    [0.7, 1.0],
                    [1.5, 2.2],
                ],
            ),
            "A1": np.array(
                [
                    [0.4, 0.6],
                    [0.7, 0.3],
                ],
            ),
            "predictions": predictions,
        }

        with patch.object(
            inference,
            "run_forward_pass",
            return_value=forward_output,
        ) as mock_run_forward_pass:
            result = inference.run_inference(
                x_test_matrix=x_test_matrix,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["forward_output"], forward_output)

        mock_run_forward_pass.assert_called_once_with(
            x_train=x_test_matrix,
            parameters=parameters,
            neurons_profile=neurons_profile,
        )

    def test_run_inference_does_not_require_y_one_hot(self) -> None:
        """Return predictions even when forward output has no Y_one_hot."""
        x_test_matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
        )
        parameters = {
            "W1": np.zeros((3, 4)),
            "b1": np.zeros((1, 4)),
            "W2": np.zeros((4, 10)),
            "b2": np.zeros((1, 10)),
        }
        neurons_profile = [4, 10]

        predictions = np.array([0, 0])
        forward_output = {
            "Z1": np.zeros((2, 4)),
            "A1": np.zeros((2, 4)),
            "Z2": np.zeros((2, 10)),
            "A2": np.full((2, 10), 0.1),
            "predictions": predictions,
        }

        with patch.object(
            inference,
            "run_forward_pass",
            return_value=forward_output,
        ):
            result = inference.run_inference(
                x_test_matrix=x_test_matrix,
                parameters=parameters,
                neurons_profile=neurons_profile,
            )

        self.assertIs(result["predictions"], predictions)
        self.assertIs(result["forward_output"], forward_output)
        self.assertNotIn("Y_one_hot", result["forward_output"])


if __name__ == "__main__":
    unittest.main()
