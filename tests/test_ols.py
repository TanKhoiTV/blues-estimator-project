import numpy as np
from sklearn.linear_model import LinearRegression
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit


class TestOLSFit:
    """Test suite for ols_fit function."""

    def test_recover_parameters_synthetic_data(self):
        """Test that ols_fit accurately recover parameters from controlled synthetic data.

        Verifies that the estimated β̂  is within error tolerance of the true β
        when generated from y = Xβ + ε where ε ~ N(0, 0.1).
        """
        np.random.seed(42)

        # Define controlled parameters
        n_samples = 100
        n_features = 3 # intercept not included
        beta_true = np.array([1.5, 2.5, 1.0, -0.8]) # beta thực tế bao gồm cả intercept ở vị trí đầu tiên
        noise_std = 0.1

        # Generate design matrx
        X = np.random.randn(n_samples, n_features)

        # Generate y = Xβ + ε
        X_aug = np.column_stack([np.ones(n_samples), X]) # matches internal logic of ols_fit
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X_aug @ beta_true + epsilon

        # Fit OLS
        beta_hat, sigma2_hat = ols_fit(X, y)

        # Assert estimated β is close to true β
        # Reasonable tolerance with noise std=0.1 and n=100
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=1)

    def test_larger_sample_size(self):
        """Test that ols_fit recovers parameters better with larger sample size.

        Verifies robustness across different dataset sizes and noise levels.
        """
        np.random.seed(123)

        # Larger dataset for tighter parameter recovery
        n_samples = 500
        n_features = 4
        beta_true = np.array([0.5, 3.0, -1.5, 2.0, 0.5])
        noise_std = 0.1

        # Generate data
        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X_aug @ beta_true + epsilon

        # Fit OLS
        beta_hat, sigma2_hat = ols_fit(X, y)

        # Better recovery is expected with increased sample size
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=2)

    def test_residual_variance_estimate(self):
        """Test that the residual variance estimator σ̂² is reasonable."""
        np.random.seed(456)

        n_samples = 200
        n_features = 2
        beta_true = np.array([2.0, 1.0, 2.0])
        noise_std = 0.1

        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X_aug @ beta_true + epsilon

        beta_hat, sigma2_hat = ols_fit(X, y)
        n, p = X.shape

        rss = np.sum((y - (X_aug @ beta_hat)) ** 2)
        sigma2_expected = rss / (n - p - 1)
        assert sigma2_hat == pytest.approx(sigma2_expected, rel=1e-6)

    def test_metrics_numerical_correctness(self):
        """Verify TSS, R-squared, Adjusted R-squared, and F-statistic values analytically."""

        # Waiting for model_metrics to be review and merged

        pass

    def test_sklearn_parity(self):
        """Verify parity with scikit-learn OLS implementation at epsilon = 1e-6."""

        np.random.seed(789)

        n_samples = 80
        n_features = 5
        beta_true = np.array([2.0, -1.2, 0.5, 3.1, -0.4, 1.8])
        noise_std = 0.2

        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X_aug @ beta_true + epsilon

        beta_hat, sigma2_hat = ols_fit(X, y)

        # sklearn fit for ground truth verification
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)

        # Merge intercept and coefficients to an array like beta_hat
        beta_sklearn = np.insert(reg.coef_, 0, reg.intercept_)

        # Restore sigma2 from sklearn
        y_pred_sklearn = reg.predict(X)
        rss_sklearn = np.sum((y - y_pred_sklearn) ** 2)
        sigma2_sklearn = rss_sklearn / (n_samples - n_features - 1)

        np.testing.assert_allclose(beta_hat, beta_sklearn, rtol=1e-6, atol=1e-6)
        assert sigma2_hat == pytest.approx(sigma2_sklearn, abs=1e-6)
    
    def test_dof_edge_case_exact_boundary(self):
        """Test the exact boundary where n = p + 1. 
        
        Residual Degrees of Freedom (n - p - 1) equals 0. 
        The model perfectly interpolates points; RSS is 0, but variance is undefined (0/0).
        """

        # 2 samples, 1 predictor => n = 2, p = 1. DoF: 2 - 1 - 1 = 0
        X = np.array([[1.0], 
                      [2.0]])
        y = np.array([3.0, 5.0])
        
        with pytest.raises(ValueError, match="Not enough samples to compute variance"):
            ols_fit(X, y)

    def test_dof_edge_case_insufficient_samples(self):
        """Test extreme case where n < p + 1 (Overparameterized system)."""

        # 2 samples but with 3 predictors => System has infinite solutions, not enough DoF
        X = np.array([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0]])
        y = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError, match="Not enough samples to compute variance"):
            ols_fit(X, y)
