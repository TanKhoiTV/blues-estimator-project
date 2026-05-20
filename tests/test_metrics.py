import numpy as np
import pytest
import sys
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import model_metrics


class TestModelMetrics:
    """Test suite for model_metrics function."""

    def setup_method(self):
        """Initialize synthetic data for testing."""
        np.random.seed(42)
        self.y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
        self.y_pred = np.array([2.5, 0.0, 2.1, 7.8, 3.9])
        self.p = 2  # Number of features
        self.n = len(self.y_true)

    def test_perfect_fit(self):
        """Test metrics when predictions match ground truth exactly."""
        # ĐÃ FIX: Đổi self.y_pred thành self.y_true để RSS thực sự bằng 0
        metrics = model_metrics(self.y_true, self.y_true, self.p)

        assert metrics["RSS"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-10)
        assert metrics["MAE"] == pytest.approx(0.0, abs=1e-10)

    def test_against_sklearn_baseline(self):
        """Cross-validate R2, MAE, and RMSE with scikit-learn metrics."""
        metrics = model_metrics(self.y_true, self.y_pred, self.p)

        expected_r2 = r2_score(self.y_true, self.y_pred)
        expected_mae = mean_absolute_error(self.y_true, self.y_pred)
        expected_rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))

        # Verify custom implementation matches library standards
        assert metrics["R2"] == pytest.approx(expected_r2, rel=1e-5)
        assert metrics["MAE"] == pytest.approx(expected_mae, rel=1e-5)
        assert metrics["RMSE"] == pytest.approx(expected_rmse, rel=1e-5)

    def test_adjusted_r2_value(self):
        """Verify that Adjusted R2 correctly penalizes the number of features."""
        metrics = model_metrics(self.y_true, self.y_pred, self.p)
        r2 = metrics["R2"]

        # Manual calculation of Adjusted R2: 1 - ((1-R2)(n-1)/(n-p-1))
        expected_adj_r2 = 1 - ((1 - r2) * (self.n - 1) / (self.n - self.p - 1))

        assert metrics["Adj_R2"] == pytest.approx(expected_adj_r2, rel=1e-5)
        # Adjusted R2 should always be <= R2
        assert metrics["Adj_R2"] <= metrics["R2"]

    def test_f_test_significance(self):
        """Ensure the F-statistic is positive and calculated correctly."""
        metrics = model_metrics(self.y_true, self.y_pred, self.p)

        # F-statistic must be non-negative
        assert metrics["F_stat"] >= 0

        # If model is better than mean, F-stat should be significant (> 0)
        if metrics["R2"] > 0:
            assert metrics["F_stat"] > 0

    def test_invalid_degrees_of_freedom(self):
        """Test that the function raises ValueError if n <= p + 1."""
        y_small = np.array([1.0, 2.0])
        y_hat_small = np.array([1.1, 1.9])
        p_too_many = 2  # n=2, p=2 leads to n-p-1 = -1 (invalid)

        with pytest.raises(ValueError):
            model_metrics(y_small, y_hat_small, p_too_many)