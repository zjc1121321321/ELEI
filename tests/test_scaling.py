"""Regression checks for zero-valued input columns and targets."""

import unittest

import numpy as np

from modules.scaling_utils import compute_scaling_factors


class ScalingTests(unittest.TestCase):
    def test_zero_columns_and_target_keep_unit_scale(self):
        X = np.column_stack([np.zeros(3), np.array([1e5, 2e5, 3e5])])
        y = np.zeros(3)
        scaled_X, scaled_y, x_scales, y_scale = compute_scaling_factors(X, y)
        self.assertEqual(x_scales, [1.0, 1e5])
        self.assertEqual(y_scale, 1.0)
        np.testing.assert_array_equal(scaled_X[:, 0], np.zeros(3))
        np.testing.assert_allclose(scaled_X[:, 1], [1, 2, 3])
        np.testing.assert_array_equal(scaled_y, y)


if __name__ == "__main__":
    unittest.main()
