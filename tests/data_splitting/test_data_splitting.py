"""Tests for data-splitting orchestration utilities."""

import unittest
from unittest.mock import patch

import numpy as np

from data_splitting import data_splitting


class TestSplitDigitRecognizerTrainingData(unittest.TestCase):
    """Tests for split_digit_recognizer_training_data."""

    def test_split_digit_recognizer_training_data_coordinates_stratified_split(
        self,
    ) -> None:
        """Call stratified split with the provided data-splitting config."""
        data_splitting_config = {
            "enabled": True,
            "strategy": "stratified",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        x_train = np.array(
            [
                [1, 2],
                [3, 4],
            ],
        )
        y_train = np.array([0, 1])
        x_validation = np.array(
            [
                [5, 6],
                [7, 8],
            ],
        )
        y_validation = np.array([0, 1])

        with patch.object(
            data_splitting,
            "stratified_train_validation_split",
            return_value=(x_train, y_train, x_validation, y_validation),
        ) as mock_stratified_train_validation_split:
            result = data_splitting.split_digit_recognizer_training_data(
                x=x,
                y=y,
                data_splitting_config=data_splitting_config,
            )

        self.assertIs(result["x_train"], x_train)
        self.assertIs(result["y_train"], y_train)
        self.assertIs(result["x_validation"], x_validation)
        self.assertIs(result["y_validation"], y_validation)

        mock_stratified_train_validation_split.assert_called_once_with(
            x=x,
            y=y,
            validation_size=0.2,
            random_seed=42,
        )

    def test_split_digit_recognizer_training_data_returns_expected_keys(self) -> None:
        """Return a dictionary containing train and validation split arrays."""
        data_splitting_config = {
            "enabled": True,
            "strategy": "stratified",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        result = data_splitting.split_digit_recognizer_training_data(
            x=x,
            y=y,
            data_splitting_config=data_splitting_config,
        )

        self.assertEqual(
            set(result.keys()),
            {"x_train", "y_train", "x_validation", "y_validation"},
        )

    def test_split_digit_recognizer_training_data_returns_expected_shapes(self) -> None:
        """Return train and validation arrays with expected shapes."""
        data_splitting_config = {
            "enabled": True,
            "strategy": "stratified",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        result = data_splitting.split_digit_recognizer_training_data(
            x=x,
            y=y,
            data_splitting_config=data_splitting_config,
        )

        self.assertEqual(result["x_train"].shape, (16, 2))
        self.assertEqual(result["y_train"].shape, (16,))
        self.assertEqual(result["x_validation"].shape, (4, 2))
        self.assertEqual(result["y_validation"].shape, (4,))

    def test_split_digit_recognizer_training_data_preserves_distribution(self) -> None:
        """Preserve class distribution in train and validation splits."""
        data_splitting_config = {
            "enabled": True,
            "strategy": "stratified",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        result = data_splitting.split_digit_recognizer_training_data(
            x=x,
            y=y,
            data_splitting_config=data_splitting_config,
        )

        self.assertEqual(np.sum(result["y_train"] == 0), 8)
        self.assertEqual(np.sum(result["y_train"] == 1), 8)
        self.assertEqual(np.sum(result["y_validation"] == 0), 2)
        self.assertEqual(np.sum(result["y_validation"] == 1), 2)

    def test_split_digit_recognizer_training_data_raises_error_when_disabled(
        self,
    ) -> None:
        """Raise ValueError when data splitting is disabled."""
        data_splitting_config = {
            "enabled": False,
            "strategy": "stratified",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        with self.assertRaisesRegex(
            ValueError,
            "Data splitting must be enabled",
        ):
            data_splitting.split_digit_recognizer_training_data(
                x=x,
                y=y,
                data_splitting_config=data_splitting_config,
            )

    def test_split_digit_recognizer_training_data_raises_error_for_unsupported_strategy(
        self,
    ) -> None:
        """Raise ValueError when the configured split strategy is unsupported."""
        data_splitting_config = {
            "enabled": True,
            "strategy": "random",
            "validation_size": 0.2,
            "random_seed": 42,
        }
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported data-splitting strategy: random",
        ):
            data_splitting.split_digit_recognizer_training_data(
                x=x,
                y=y,
                data_splitting_config=data_splitting_config,
            )


if __name__ == "__main__":
    unittest.main()
