import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Chỉ giữ lại import ols_fit vì phần test VIF đã có file test_vif.py riêng lo liệu
from part1.ols_implementation import ols_fit


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
