import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit, model_metrics, coef_inference, vif


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
        n_features = 3
        beta_true = np.array([2.5, 1.0, -0.8])
        noise_std = 0.1

        # Generate design matrix
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1  # Intercept col

        # Generate y = Xβ + ε
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X @ beta_true + epsilon

        # Fit OLS
        beta_hat, sigma2_hat = ols_fit(X, y)

        # Assert estimated β is close to true β
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=1)

    def test_larger_sample_size(self):
        """Test that ols_fit recovers parameters better with larger sample size.

        Verifies robustness across different dataset sizes and noise levels.
        """
        np.random.seed(123)

        # Larger dataset for tighter parameter recovery
        n_samples = 500
        n_features = 4
        beta_true = np.array([3.0, -1.5, 2.0, 0.5])
        noise_std = 0.1

        # Generate data
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X @ beta_true + epsilon

        # Fit OLS
        beta_hat, sigma2_hat = ols_fit(X, y)

        # Better recovery is expected with increased sample size
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=2)

    def test_residual_variance_estimate(self):
        """Test that the residual variance estimator σ̂² is reasonable."""
        np.random.seed(456)

        n_samples = 200
        n_features = 2
        beta_true = np.array([1.0, 2.0])
        noise_std = 0.1

        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X @ beta_true + epsilon

        beta_hat, sigma2_hat = ols_fit(X, y)

        assert sigma2_hat == pytest.approx(noise_std**2, rel=0.5)


class TestModelMetrics:
    """Test suite for model_metrics function."""

    def test_model_metrics_calculation(self):
        """Test if metrics are calculated correctly with simple data."""
        # Mock data setup
        y = np.array([3, -0.5, 2, 7])
        y_hat = np.array([2.5, 0.0, 2, 8])
        p = 1

        # Compute metrics
        metrics = model_metrics(y, y_hat, p)

        # Verify all expected keys are present
        assert "RSS" in metrics
        assert "TSS" in metrics
        assert "R2" in metrics
        assert "Adjusted_R2" in metrics
        assert "F_statistic" in metrics

        # Verify RSS calculation
        assert np.isclose(metrics["RSS"], 1.5)

    def test_model_metrics_validation(self):
        """Test if function correctly catches errors (different shapes or invalid degrees of freedom)."""
        # Test shape mismatch
        y = np.array([1, 2])
        y_hat = np.array([1, 2, 3])
        p = 1

        with pytest.raises(ValueError):
            model_metrics(y, y_hat, p)

        # Test insufficient degrees of freedom (p + 1 >= n)
        y_short = np.array([1, 2])
        y_hat_short = np.array([1, 2])
        p_large = 2

        with pytest.raises(ValueError):
            model_metrics(y_short, y_hat_short, p_large)


class TestCoefInference:
    """Test suite for coef_inference function."""

    def test_coef_inference_calculation(self):
        """Test coefficients statistical inference calculations (SE, t-stat, p-value, CI)."""
        np.random.seed(99)
        n, p = 50, 2
        X = np.random.randn(n, p)
        X[:, 0] = 1
        beta_true = np.array([2.0, 5.0])
        y = X @ beta_true + np.random.normal(0, 0.1, n)

        beta_hat, sigma2_hat = ols_fit(X, y)
        results = coef_inference(X, y, beta_hat, sigma2_hat)

        assert "SE" in results
        assert "t_stats" in results
        assert "p_values" in results
        assert "CI_lower" in results
        assert "CI_upper" in results
        assert np.all(results["SE"] > 0)
        assert np.all(results["CI_lower"] < results["CI_upper"])
        assert np.all(results["CI_lower"] <= beta_hat)
        assert np.all(beta_hat <= results["CI_upper"])


class TestVIF:
    """Test suite for vif function."""

    def test_vif_calculation(self):
        """Test VIF calculation with known collinearity properties."""
        np.random.seed(42)
        # n=100, p=3 (intercept + 2 features)
        X = np.random.rand(100, 3)
        X[:, 0] = 1.0

        # Independent features -> VIF should be close to 1
        vif_independent = vif(X)
        assert len(vif_independent) == 2
        assert np.allclose(vif_independent, [1.0, 1.0], atol=0.2)

        # Highly correlated features -> VIF should be high (> 10)
        X[:, 2] = X[:, 1] * 3.0 + np.random.normal(0, 0.01, 100)
        vif_correlated = vif(X)
        assert np.all(vif_correlated > 10.0)

    def test_vif_validation(self):
        """Test edge cases for VIF (e.g., matrix with only intercept)."""
        X_single = np.ones((10, 1))
        with pytest.raises(ValueError):
            vif(X_single)
