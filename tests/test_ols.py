import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit, vif


class TestOLSFit:
    """Test suite for ols_fit function."""

    def test_recover_parameters_synthetic_data(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        beta_true = np.array([2.5, 1.0, -0.8])
        noise_std = 0.1

        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X @ beta_true + epsilon

        beta_hat, sigma2_hat = ols_fit(X, y)
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=1)

    def test_larger_sample_size(self):
        np.random.seed(123)
        n_samples = 500
        n_features = 4
        beta_true = np.array([3.0, -1.5, 2.0, 0.5])
        noise_std = 0.1

        X = np.random.randn(n_samples, n_features)
        X[:, 0] = 1
        epsilon = np.random.normal(loc=0, scale=noise_std, size=n_samples)
        y = X @ beta_true + epsilon

        beta_hat, sigma2_hat = ols_fit(X, y)
        np.testing.assert_array_almost_equal(beta_hat, beta_true, decimal=2)

    def test_residual_variance_estimate(self):
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


class TestVIF:
    """Test suite for vif function."""

    def test_vif_calculation(self):
        np.random.seed(42)
        X = np.random.rand(100, 3)
        X[:, 0] = 1.0

        vif_independent = vif(X)
        assert len(vif_independent) == 2
        assert np.allclose(vif_independent, [1.0, 1.0], atol=0.2)

        X[:, 2] = X[:, 1] * 3.0 + np.random.normal(0, 0.01, 100)
        vif_correlated = vif(X)
        assert np.all(vif_correlated > 10.0)

    def test_vif_validation(self):
        X_single = np.ones((10, 1))
        with pytest.raises(ValueError):
            vif(X_single)
