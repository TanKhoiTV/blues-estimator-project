import numpy as np
import pytest
import sys
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import vif


class TestVIF:
    """Test suite for vif function."""

    def test_vif_independent_features(self):
        """Test that VIF ≈ 1 when features are completely independent (orthogonal)."""
        # Identity matrix represents perfectly independent features
        X_independent = np.eye(4)

        vif_values = vif(X_independent)

        # For orthogonal features, VIF should be exactly 1.0
        expected_vifs = np.ones(4)
        np.testing.assert_array_almost_equal(vif_values, expected_vifs, decimal=7)

    def test_vif_high_multicollinearity(self):
        """Test that VIF is high when one feature is a linear combination of others."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)
        # x3 is almost a linear combination of x1 and x2 plus tiny noise
        x3 = x1 + x2 + np.random.normal(0, 0.01, 100)

        X = np.column_stack([x1, x2, x3])
        vif_values = vif(X)

        # VIF for x3 should be very high (> 10)
        assert vif_values[2] > 10
        # VIFs should be identical to the reciprocal of (1 - R^2)
        assert np.all(vif_values >= 1.0)

    def test_against_statsmodels(self):
        """Cross-validate VIF calculation against statsmodels baseline."""
        np.random.seed(123)
        X = np.random.randn(50, 3)
        # Add some correlation
        X[:, 2] = X[:, 0] * 0.5 + X[:, 1] * 0.2 + np.random.randn(50) * 0.1

        vif_custom = vif(X)

        # Compute expected VIF using statsmodels
        expected_vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

        np.testing.assert_array_almost_equal(vif_custom, expected_vifs, decimal=5)

    def test_vif_perfect_collinearity(self):
        """Test behavior when columns are perfectly collinear."""
        x1 = np.array([1, 2, 3, 4, 5], dtype=float)
        x2 = x1 * 2  # Perfectly collinear
        X = np.column_stack([x1, x2])

        vif_values = vif(X)

        # VIF should be infinite or a very large number depending on pinv implementation
        assert np.isinf(vif_values).any() or np.all(vif_values > 1e10)

    def test_empty_input(self):
        """Test that the function raises ValueError for empty input."""
        with pytest.raises(ValueError):
            vif(np.array([]))
