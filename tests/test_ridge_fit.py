import numpy as np
from sklearn.linear_model import Ridge
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit
from part1.ridge_lasso import ridge_fit


class TestRidgeFit:
    """Test suite for ridge_fit function."""

    def test_lambda_zero_converges_to_ols(self):
        """Test that ridge_fit with lam=0 yields identical results to ols_fit.

        Verifies the fundamental property that Ridge regression without penalty
        collides with the Ordinary Least Squares solution.
        """
        np.random.seed(42)

        n_samples = 30
        n_features = 3
        beta_true = np.array([1.0, -2.0, 0.5, 1.5])
        noise_std = 0.1

        X = np.random.rand(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X_aug @ beta_true + epsilon

        beta_ols, _ = ols_fit(X, y)
        beta_ridge = ridge_fit(X, y, lam=0.0)

        np.testing.assert_allclose(beta_ridge, beta_ols, rtol=1e-6, atol=1e-6)

    def test_sklearn_ridge_parity(self):
        """Verify parity with scikit-learn Ridge implementation at a specific lambda."""
        np.random.seed(123)
        n_samples = 100
        n_features = 4
        beta_true = np.array([0.5, 2.0, -1.0, 0.8, -1.2])
        lam = 5.0

        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        epsilon = np.random.normal(0, 0.2, size=n_samples)
        y = X_aug @ beta_true + epsilon

        # Custom Ridge implementation
        beta_hat_custom = ridge_fit(X, y, lam=lam)

        # Scikit-learn Ridge verification
        # Note: sklearn applies penalty 'alpha' to coefficients but excludes intercept by default
        reg = Ridge(alpha=lam, fit_intercept=True)
        reg.fit(X, y)
        beta_sklearn = np.insert(reg.coef_, 0, reg.intercept_)

        np.testing.assert_allclose(beta_hat_custom, beta_sklearn, rtol=1e-6, atol=1e-6)

    def test_coefficient_shrinkage(self):
        """Test that increasing lambda strictly shrinks the magnitudes of coefficients (excluding intercept)."""
        np.random.seed(456)
        n_samples = 80
        n_features = 2
        beta_true = np.array([10.0, 5.0, -5.0])  # Large coefficients

        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        y = X_aug @ beta_true + np.random.normal(0, 0.1, size=n_samples)

        beta_lam_small = ridge_fit(X, y, lam=0.1)
        beta_lam_large = ridge_fit(X, y, lam=100.0)

        # Intercept shouldn't be penalized down to zero, but predictors must shrink
        assert np.abs(beta_lam_large[1]) < np.abs(beta_lam_small[1])
        assert np.abs(beta_lam_large[2]) < np.abs(beta_lam_small[2])
