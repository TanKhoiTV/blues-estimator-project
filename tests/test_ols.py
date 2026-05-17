import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit, model_metrics


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
        """Test that the residual variance estimator is reasonable with a tight tolerance."""
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

        # Đfixed: Siết chặt độ lệch từ rel=0.5 xuống rel=0.2 theo đúng yêu cầu của CodeRabbit
        assert sigma2_hat == pytest.approx(noise_std**2, rel=0.2)


class TestModelMetrics:
    """Test suite for model_metrics function."""

    def test_model_metrics_calculation(self):
        y = np.array([3, -0.5, 2, 7])
        y_hat = np.array([2.5, 0.0, 2, 8])
        p = 1

        metrics = model_metrics(y, y_hat, p)

        assert "RSS" in metrics
        assert "TSS" in metrics
        assert "R2" in metrics
        assert "Adjusted_R2" in metrics
        assert "F_statistic" in metrics
        assert np.isclose(metrics["RSS"], 1.5)

    def test_model_metrics_validation(self):
        y = np.array([1, 2])
        y_hat = np.array([1, 2, 3])
        p = 1

        with pytest.raises(ValueError):
            model_metrics(y, y_hat, p)

        y_short = np.array([1, 2])
        y_hat_short = np.array([1, 2])
        p_large = 2

        with pytest.raises(ValueError):
            model_metrics(y_short, y_hat_short, p_large)

        with pytest.raises(TypeError):
            model_metrics(y_short, y_hat_short, "invalid_p_type")


class TestModelMetricsExpanded:
    """Bộ test mở rộng đáp ứng toàn bộ các tiêu chí nghiệm thu của Issue #25."""

    def setup_method(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        self.p = 2

    def test_perfect_fit(self):
        metrics = model_metrics(self.y_true, self.y_true, self.p)
        assert np.isclose(metrics["R2"], 1.0)

    def test_metrics_keys_and_outputs(self):
        metrics = model_metrics(self.y_true, self.y_pred, self.p)
        assert "Adj_R2" in metrics
        assert "F_stat" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics

    def test_adjusted_r2_penalty(self):
        """BỔ SUNG THEO ISSUE #25: Xác thực Adjusted R2 sẽ giảm khi thêm biến vô tri."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat = np.array([1.1, 1.9, 3.0, 4.2, 4.8])

        metrics_p1 = model_metrics(y, y_hat, p=1)
        metrics_p2 = model_metrics(y, y_hat, p=2)

        assert metrics_p2["Adj_R2"] < metrics_p1["Adj_R2"]
        assert metrics_p2["Adjusted_R2"] < metrics_p1["Adjusted_R2"]
