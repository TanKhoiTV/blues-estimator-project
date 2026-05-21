import numpy as np
import pytest
import sys
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import coef_inference, ols_fit


class TestCoefInference:
    """Test suite for coef_inference function."""

    def setup_method(self):
        """Initialize controlled synthetic data for statistical testing."""
        np.random.seed(42)

        self.n_samples = 100
        self.n_features = 3

        # Design matrix with intercept
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.X[:, 0] = 1

        # Known coefficients: Beta1 is significant, Beta2 is nearly zero (insignificant)
        self.beta_true = np.array([10.0, 5.0, 0.01])
        self.noise_std = 0.5
        epsilon = np.random.normal(0, self.noise_std, self.n_samples)
        self.y = self.X @ self.beta_true + epsilon

        # Fit initial model to get prerequisites
        self.beta_hat, self.sigma2_hat = ols_fit(self.X, self.y)

    def test_standard_error_calculation(self):
        """Verify Standard Errors (SE) match theoretical expectations.

        SE(beta_j) = sqrt(sigma2 * [ (X^T X)^-1 ]_jj)
        """
        inference_results = coef_inference(
            self.X, self.y, self.beta_hat, self.sigma2_hat
        )
        se = inference_results["standard_errors"]

        # Manual calculation for verification
        xtx_inv = np.linalg.pinv(self.X.T @ self.X)
        expected_se = np.sqrt(self.sigma2_hat * np.diag(xtx_inv))

        np.testing.assert_array_almost_equal(se, expected_se, decimal=10)

    def test_t_statistics_logic(self):
        """Verify t-statistics are calculated as beta_hat / SE."""
        results = coef_inference(self.X, self.y, self.beta_hat, self.sigma2_hat)

        expected_t = self.beta_hat / results["standard_errors"]
        # ĐÃ SỬA LỖI TẠI ĐÂY: t_stats -> t_statistics
        np.testing.assert_array_almost_equal(
            results["t_statistics"], expected_t, decimal=10
        )

    def test_p_value_significance(self):
        """Verify that significant variables have low p-values.

        Beta[1] = 5.0 (High significance) should have p < 0.05.
        Beta[2] = 0.01 (Low significance) should likely have p > 0.05.
        """
        results = coef_inference(self.X, self.y, self.beta_hat, self.sigma2_hat)
        p_values = results["p_values"]

        assert p_values[1] < 0.01  # Strongly significant
        assert p_values[2] > 0.05  # Not significant at 5% level

    def test_confidence_intervals_95(self):
        """Verify the 95% Confidence Intervals (CI) are symmetric and properly scaled."""
        results = coef_inference(self.X, self.y, self.beta_hat, self.sigma2_hat)
        ci_lower = results["ci_lower"]
        ci_upper = results["ci_upper"]

        # CI should be centered around beta_hat
        ci_midpoint = (ci_lower + ci_upper) / 2
        np.testing.assert_array_almost_equal(ci_midpoint, self.beta_hat, decimal=10)

        # Verify CI width using t-critical value (n-p-1 degrees of freedom)
        dof = self.n_samples - self.n_features
        t_crit = stats.t.ppf(0.975, df=dof)
        expected_half_width = t_crit * results["standard_errors"]

        np.testing.assert_array_almost_equal(
            ci_upper - self.beta_hat, expected_half_width, decimal=7
        )

    def test_dimension_mismatch(self):
        """Test that the function raises ValueError for mismatched input dimensions."""
        wrong_beta = np.array([1.0, 2.0])  # Needs 3

        with pytest.raises(ValueError):
            coef_inference(self.X, self.y, wrong_beta, self.sigma2_hat)
