import matplotlib

matplotlib.use("Agg")

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import ols_fit
from part1.ridge_lasso import ridge_fit, plot_ridge_trace


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

    def test_large_lambda_shrinks_coefficients_to_zero(self):
        """Test coefficient shrinkage at extreme lambda values.

        When lambda approaches infinity, all penalized coefficients (excluding intercept)
        must converge to zero.
        """
        np.random.seed(0)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        beta_hat = ridge_fit(X, y, lam=1e10)

        np.testing.assert_allclose(beta_hat[1:], 0.0, atol=1e-4)

    def test_multicollinear_features_returns_stable_solution(self):
        """Test that Ridge regression handles multicollinearity stably.

        The model must return a finite, stable solution even when X is
        rank-deficient due to perfectly collinear features.
        """
        np.random.seed(0)
        X_raw = np.random.randn(30, 2)
        X = np.column_stack(
            [X_raw, X_raw[:, 0]]
        )  # column 3 = column 1 (perfect collinearity)
        y = np.random.randn(30)

        beta_hat = ridge_fit(X, y, lam=1.0)

        assert beta_hat.shape == (4,)  # 3 features + 1 intercept
        assert np.all(np.isfinite(beta_hat))

    def test_intercept_not_penalized(self):
        """Test that the intercept is excluded from the L2 penalty.

        When y is shifted by a constant c, beta_hat[0] (intercept) must increase
        by exactly c while all slope coefficients remain unchanged. This verifies that
        the intercept is excluded from the L2 penalty as intended by the I[0,0]=0 design.
        """
        np.random.seed(0)
        X = np.random.randn(40, 2)
        y = np.random.randn(40)
        c = 100.0

        beta_original = ridge_fit(X, y, lam=5.0)
        beta_shifted = ridge_fit(X, y + c, lam=5.0)

        np.testing.assert_allclose(beta_shifted[0], beta_original[0] + c, atol=1e-6)
        np.testing.assert_allclose(beta_shifted[1:], beta_original[1:], atol=1e-6)


class TestPlotRidgeTrace:
    """Test suite for plot_ridge_trace visualization function."""

    @pytest.fixture
    def sample_data(self):
        """Fixture to provide consistent toy data for plotting tests."""
        np.random.seed(42)
        n_samples = 40
        n_features = 3
        X = np.random.randn(n_samples, n_features)
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
        return X, y

    def test_plot_returns_figure_and_does_not_error(self, sample_data):
        """Verify that the function executes successfully and returns a matplotlib Figure."""
        X, y = sample_data

        fig = plot_ridge_trace(X, y)

        # Explicitly close the plot to prevent memory leaks during testing
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_elements_and_labels(self, sample_data):
        """Verify that axes labels, title, and scale type are configured correctly."""
        X, y = sample_data

        fig = plot_ridge_trace(X, y)
        ax = fig.gca()  # Get current axes

        try:
            # Check scale type
            assert ax.get_xscale() == "log"

            # Check that labels and titles are set (stripping spaces/quotes for robust checking)
            assert "lambda" in ax.get_xlabel().lower()
            assert "coefficient" in ax.get_ylabel().lower()
            assert "ridge trace" in ax.get_title().lower()
        finally:
            plt.close(fig)

    def test_plot_tracks_all_coefficients(self, sample_data):
        """Verify that the plot renders paths for all parameters (intercept + features)."""
        X, y = sample_data
        n_features = X.shape[1]

        expected_coef_lines = n_features + 1  # 3 features + 1 intercept

        fig = plot_ridge_trace(X, y)
        ax = fig.gca()

        try:
            # Check that the number of plotted lines matches our parameter count
            total_lines = ax.get_lines()

            # The total lines should equal our coefficients plus the 1 axhline baseline
            assert len(total_lines) == expected_coef_lines + 1
        finally:
            plt.close(fig)
