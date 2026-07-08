"""Tests for stratified train-validation splitting."""

import unittest

import numpy as np

from data_splitting.stratified_train_validation_split import (
    stratified_train_validation_split,
)


class TestStratifiedTrainValidationSplit(unittest.TestCase):
    """Tests for stratified_train_validation_split."""

    def test_stratified_split_returns_expected_shapes(self) -> None:
        """Return training and validation splits with expected shapes."""
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        x_train, y_train, x_validation, y_validation = stratified_train_validation_split(
            x=x,
            y=y,
            validation_size=0.2,
            random_seed=42,
        )

        self.assertEqual(x_train.shape, (16, 2))
        self.assertEqual(y_train.shape, (16,))
        self.assertEqual(x_validation.shape, (4, 2))
        self.assertEqual(y_validation.shape, (4,))

    def test_stratified_split_preserves_class_distribution(self) -> None:
        """Preserve class proportions in training and validation splits."""
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        _x_train, y_train, _x_validation, y_validation = (
            stratified_train_validation_split(
                x=x,
                y=y,
                validation_size=0.2,
                random_seed=42,
            )
        )

        self.assertEqual(np.sum(y_train == 0), 8)
        self.assertEqual(np.sum(y_train == 1), 8)
        self.assertEqual(np.sum(y_validation == 0), 2)
        self.assertEqual(np.sum(y_validation == 1), 2)

    def test_stratified_split_is_reproducible_with_same_seed(self) -> None:
        """Return the same split when the same random seed is used."""
        x = np.arange(40).reshape(20, 2)
        y = np.array([0] * 10 + [1] * 10)

        first_result = stratified_train_validation_split(
            x=x,
            y=y,
            validation_size=0.2,
            random_seed=42,
        )
        second_result = stratified_train_validation_split(
            x=x,
            y=y,
            validation_size=0.2,
            random_seed=42,
        )

        for first_array, second_array in zip(first_result, second_result, strict=True):
            np.testing.assert_array_equal(first_array, second_array)

    def test_stratified_split_has_no_overlap_between_train_and_validation(self) -> None:
        """Keep training and validation examples separate."""
        x = np.arange(20).reshape(20, 1)
        y = np.array([0] * 10 + [1] * 10)

        x_train, _y_train, x_validation, _y_validation = (
            stratified_train_validation_split(
                x=x,
                y=y,
                validation_size=0.2,
                random_seed=42,
            )
        )

        train_ids = set(x_train[:, 0].tolist())
        validation_ids = set(x_validation[:, 0].tolist())

        self.assertEqual(train_ids & validation_ids, set())
        self.assertEqual(train_ids | validation_ids, set(x[:, 0].tolist()))

    def test_stratified_split_raises_error_when_x_and_y_sizes_differ(self) -> None:
        """Raise ValueError when x and y have different example counts."""
        x = np.arange(12).reshape(6, 2)
        y = np.array([0, 0, 1, 1, 1])

        with self.assertRaisesRegex(
            ValueError,
            "x and y must contain the same number of examples.",
        ):
            stratified_train_validation_split(
                x=x,
                y=y,
            )

    def test_stratified_split_raises_error_for_zero_validation_size(self) -> None:
        """Raise ValueError when validation_size is zero."""
        x = np.arange(12).reshape(6, 2)
        y = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaisesRegex(
            ValueError,
            "validation_size must be greater than 0 and less than 1.",
        ):
            stratified_train_validation_split(
                x=x,
                y=y,
                validation_size=0.0,
            )

    def test_stratified_split_raises_error_for_one_validation_size(self) -> None:
        """Raise ValueError when validation_size is one."""
        x = np.arange(12).reshape(6, 2)
        y = np.array([0, 0, 0, 1, 1, 1])

        with self.assertRaisesRegex(
            ValueError,
            "validation_size must be greater than 0 and less than 1.",
        ):
            stratified_train_validation_split(
                x=x,
                y=y,
                validation_size=1.0,
            )

    def test_stratified_split_raises_error_when_class_has_one_example(self) -> None:
        """Raise ValueError when a class has fewer than two examples."""
        x = np.arange(6).reshape(3, 2)
        y = np.array([0, 1, 1])

        with self.assertRaisesRegex(
            ValueError,
            "Each class must contain at least two examples.",
        ):
            stratified_train_validation_split(
                x=x,
                y=y,
            )


if __name__ == "__main__":
    unittest.main()
