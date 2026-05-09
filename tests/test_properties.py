import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import hat_matrix


class TestHatMatrix:
    """Test suite for hat_matrix function."""

    def test_idempotency_synthetic_data(self):
        """Test that H^2 ≈ H (idempotency property) with synthetic data.

        The hat matrix H is idempotent, meaning when multiplied by itself, it yields itself: H @ H = H.
        """
        np.random.seed(42)

        # Generate synthetic design matrix
        n_samples = 50
        n_features = 3
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1

        # Compute hat_matrix
        H = hat_matrix(X)

        # Verify idempotency
        H_squared = H @ H
        np.testing.assert_array_almost_equal(H_squared, H, decimal=10)

    def test_symmetry_synthetic_data(self):
        """Test that H is symmetric (H^T ≈ H) with synthetic data.

        The hat matrix is symmetric, meaning H.T = H.
        """
        np.random.seed(123)

        # Generate synthetic design matrix
        n_samples = 40
        n_features = 4
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1

        # Compute hat_matrix
        H = hat_matrix(X)

        # Verify symmetry
        H_squared = H @ H
        np.testing.assert_array_almost_equal(H, H.T, decimal=10)

    def test_known_baseline(self):
        """Test hat_matrix with a known baseline (simple 2-feature case).

        Use simple data where the hat matrix can be partially validated by hand calculations.
        """
        # Simple 3-sample, 2-feature (including intercept) case
        X = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]])

        H = hat_matrix(X)

        # Verify its properties
        assert H.shape == 3

        assert np.all(np.diag(H) >= 0)
        assert np.all(np.diag(H) <= 1)

        # Verify symmetry and idempotency
        np.testing.assert_array_almost_equal(H, H.T, decimal=10)
        np.testing.assert_array_almost_equal(H @ H, H, decimal=10)

    def test_trace_property(self):
        """Test that trace(H) equals the number of features (rank of X).

        For a hat matrix H = X(X^T X)^-1 X^T, trace(H) = rank(X) = p.
        """
        np.random.seed(789)

        # Generate synthetic design matrix
        n_samples = 60
        n_features = 5
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1

        # Compute hat_matrix
        H = hat_matrix(X)

        # Trace should be equal to the number of features
        trace_H = np.trace(H)
        assert trace_H == pytest.approx(n_features, abs=1e-10)

    def test_diagonal_bounds(self):
        """Test that all elements on the diagonal of H are bounded between 0 and 1.

        Key property of projection matrices.
        """
        np.random.seed(456)

        X = np.random.randn(100, 6)
        X[:, 0] = 1

        H = hat_matrix(X)

        # All diagonal elements must lies in [0, 1]
        diag_H = np.diag(H)
        assert np.all(diag_H >= -1e-10)
        assert np.all(diag_H <= 1 + 1e-10)
