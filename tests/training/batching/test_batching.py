"""Tests for training batching utilities."""

import unittest

import numpy as np

from training.batching.batching import (
    SUPPORTED_BATCHING_STRATEGIES,
    get_batching_config,
    iter_mini_batches,
)


class TestBatchingConfig(unittest.TestCase):
    """Test batching configuration helpers."""

    def test_supported_batching_strategies(self) -> None:
        """Expose supported batching strategies."""
        self.assertEqual(
            SUPPORTED_BATCHING_STRATEGIES,
            {
                "full_batch",
                "mini_batch",
            },
        )

    def test_get_batching_config_returns_default_for_legacy_config(self) -> None:
        """Return full-batch defaults when batching config is missing."""
        training_config = {
            "optimizer": "batch_gradient_descent",
            "learning_rate": 0.1,
            "num_iterations": 1000,
        }

        result = get_batching_config(training_config=training_config)

        self.assertEqual(
            result,
            {
                "strategy": "full_batch",
                "batch_size": None,
                "shuffle": False,
                "random_seed": 42,
            },
        )

    def test_get_batching_config_returns_default_for_invalid_batching_config(
        self,
    ) -> None:
        """Return full-batch defaults when batching config is not a dictionary."""
        training_config = {
            "batching": "full_batch",
        }

        result = get_batching_config(training_config=training_config)

        self.assertEqual(
            result,
            {
                "strategy": "full_batch",
                "batch_size": None,
                "shuffle": False,
                "random_seed": 42,
            },
        )

    def test_get_batching_config_returns_configured_batching(self) -> None:
        """Return the configured batching section."""
        batching_config = {
            "strategy": "mini_batch",
            "batch_size": 16,
            "shuffle": True,
            "random_seed": 123,
        }
        training_config = {
            "batching": batching_config,
        }

        result = get_batching_config(training_config=training_config)

        self.assertEqual(result, batching_config)


class TestIterMiniBatches(unittest.TestCase):
    """Test mini-batch iteration."""

    def test_iter_mini_batches_preserves_order_when_shuffle_is_false(self) -> None:
        """Yield ordered mini-batches when shuffle is disabled."""
        x_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
        )
        y_train = np.array([0, 1, 2, 3])

        batches = list(
            iter_mini_batches(
                x_train=x_train,
                y_train=y_train,
                batch_size=2,
                shuffle=False,
                random_generator=np.random.default_rng(42),
            ),
        )

        self.assertEqual(len(batches), 2)

        np.testing.assert_array_equal(batches[0][0], x_train[:2])
        np.testing.assert_array_equal(batches[0][1], y_train[:2])
        np.testing.assert_array_equal(batches[1][0], x_train[2:])
        np.testing.assert_array_equal(batches[1][1], y_train[2:])

    def test_iter_mini_batches_creates_final_smaller_batch(self) -> None:
        """Yield a smaller final batch when sample count is not divisible."""
        x_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
        )
        y_train = np.array([0, 1, 2, 3, 4])

        batches = list(
            iter_mini_batches(
                x_train=x_train,
                y_train=y_train,
                batch_size=2,
                shuffle=False,
                random_generator=np.random.default_rng(42),
            ),
        )

        self.assertEqual(len(batches), 3)

        np.testing.assert_array_equal(batches[0][1], np.array([0, 1]))
        np.testing.assert_array_equal(batches[1][1], np.array([2, 3]))
        np.testing.assert_array_equal(batches[2][1], np.array([4]))

    def test_iter_mini_batches_handles_batch_size_larger_than_dataset(self) -> None:
        """Yield one batch when batch size is larger than sample count."""
        x_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
        )
        y_train = np.array([0, 1, 2])

        batches = list(
            iter_mini_batches(
                x_train=x_train,
                y_train=y_train,
                batch_size=10,
                shuffle=False,
                random_generator=np.random.default_rng(42),
            ),
        )

        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0][0], x_train)
        np.testing.assert_array_equal(batches[0][1], y_train)

    def test_iter_mini_batches_shuffles_deterministically(self) -> None:
        """Shuffle mini-batches deterministically with a fixed random generator."""
        x_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
            ],
        )
        y_train = np.array([0, 1, 2, 3, 4, 5])

        random_seed = 123
        expected_indices = np.arange(6)
        np.random.default_rng(random_seed).shuffle(expected_indices)

        batches = list(
            iter_mini_batches(
                x_train=x_train,
                y_train=y_train,
                batch_size=2,
                shuffle=True,
                random_generator=np.random.default_rng(random_seed),
            ),
        )

        observed_y_order = np.concatenate([batch_y for _batch_x, batch_y in batches])

        np.testing.assert_array_equal(observed_y_order, y_train[expected_indices])
        self.assertNotEqual(observed_y_order.tolist(), y_train.tolist())


if __name__ == "__main__":
    unittest.main()
