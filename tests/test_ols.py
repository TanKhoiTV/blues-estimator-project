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

        # Kiểm tra thêm lỗi sai kiểu dữ liệu của p theo ý CodeRabbit
        with pytest.raises(TypeError):
            model_metrics(y_short, y_hat_short, "invalid_p_type")


class TestModelMetricsExpanded:
    """Bộ test mở rộng tích hợp từ các yêu cầu mới của dự án."""

    def setup_method(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        self.p = 2

    def test_perfect_fit(self):
        """Test trường hợp dự báo khớp hoàn hảo 100%."""
        # ĐÃ SỬA TYPO: Đổi từ self.y_pred thành self.y_true ở tham số thứ 2 
        # để thực sự tạo ra kịch bản Perfect Fit đúng nghĩa toán học giúp R2 = 1.0
        metrics = model_metrics(self.y_true, self.y_true, self.p)
        assert np.isclose(metrics["R2"], 1.0)

    def test_metrics_keys_and_outputs(self):
        """Xác thực các key viết tắt mới (MAE, RMSE, Adj_R2, F_stat) có tồn tại."""
        metrics = model_metrics(self.y_true, self.y_pred, self.p)
        assert "Adj_R2" in metrics
        assert "F_stat" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert metrics["MAE"] > 0
        assert metrics["RMSE"] > 0