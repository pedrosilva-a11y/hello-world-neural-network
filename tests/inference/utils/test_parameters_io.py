"""Tests for inference parameter IO utilities."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from inference.utils.parameters_io import load_parameters, save_parameters


class TestSaveParameters(unittest.TestCase):
    """Tests for save_parameters."""

    def test_save_parameters_creates_parent_directory_and_file(self) -> None:
        """Create parent directories and save parameters to an npz file."""
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.01, 0.02]]),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = (
                Path(temporary_directory)
                / "nested"
                / "experiment"
                / "parameters.npz"
            )

            save_parameters(
                parameters=parameters,
                file_path=file_path,
            )

            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.is_file())

            with np.load(file_path) as loaded_parameters:
                self.assertEqual(set(loaded_parameters.files), {"W1", "b1"})
                np.testing.assert_array_equal(
                    loaded_parameters["W1"],
                    parameters["W1"],
                )
                np.testing.assert_array_equal(
                    loaded_parameters["b1"],
                    parameters["b1"],
                )

    def test_save_parameters_supports_multiple_layers(self) -> None:
        """Save parameters for a multi-layer model."""
        parameters = {
            "W1": np.zeros((2, 3)),
            "b1": np.zeros((1, 3)),
            "W2": np.ones((3, 3)),
            "b2": np.ones((1, 3)),
            "W3": np.full((3, 2), 2.0),
            "b3": np.full((1, 2), 3.0),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "parameters.npz"

            save_parameters(
                parameters=parameters,
                file_path=file_path,
            )

            with np.load(file_path) as loaded_parameters:
                self.assertEqual(
                    set(loaded_parameters.files),
                    {"W1", "b1", "W2", "b2", "W3", "b3"},
                )

                for parameter_name, parameter_value in parameters.items():
                    np.testing.assert_array_equal(
                        loaded_parameters[parameter_name],
                        parameter_value,
                    )


class TestLoadParameters(unittest.TestCase):
    """Tests for load_parameters."""

    def test_load_parameters_returns_saved_parameter_dictionary(self) -> None:
        """Load parameters from an existing npz file."""
        expected_parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ),
            "b1": np.array([[0.01, 0.02]]),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "parameters.npz"
            np.savez_compressed(
                file_path,
                **cast(dict[str, Any], expected_parameters),
            )

            loaded_parameters = load_parameters(file_path=file_path)

        self.assertEqual(set(loaded_parameters), {"W1", "b1"})
        np.testing.assert_array_equal(
            loaded_parameters["W1"],
            expected_parameters["W1"],
        )
        np.testing.assert_array_equal(
            loaded_parameters["b1"],
            expected_parameters["b1"],
        )

    def test_load_parameters_raises_error_when_file_does_not_exist(self) -> None:
        """Raise FileNotFoundError when the parameter file does not exist."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "missing_parameters.npz"

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Parameter file not found",
            ):
                load_parameters(file_path=file_path)


class TestParameterRoundTrip(unittest.TestCase):
    """Tests for saving and then loading parameters."""

    def test_parameters_round_trip_preserves_names_shapes_and_values(self) -> None:
        """Save and load parameters without changing names, shapes, or values."""
        parameters = {
            "W1": np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
            ),
            "b1": np.array([[0.01, 0.02, 0.03]]),
            "W2": np.array(
                [
                    [0.7, 0.8],
                    [0.9, 1.0],
                    [1.1, 1.2],
                ],
            ),
            "b2": np.array([[0.04, 0.05]]),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "parameters.npz"

            save_parameters(
                parameters=parameters,
                file_path=file_path,
            )

            loaded_parameters = load_parameters(file_path=file_path)

        self.assertEqual(set(loaded_parameters), {"W1", "b1", "W2", "b2"})

        for parameter_name, parameter_value in parameters.items():
            self.assertEqual(
                loaded_parameters[parameter_name].shape,
                parameter_value.shape,
            )
            np.testing.assert_array_equal(
                loaded_parameters[parameter_name],
                parameter_value,
            )


if __name__ == "__main__":
    unittest.main()
