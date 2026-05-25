import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.gauss_markov_util import generate_synthetic_data


@pytest.fixture
def default_params():
    """Default parameter used for multiple test cases."""
    return {
        "beta": np.array([5.0, 2.5, -1.5]),
        "sigma": 2.0,
        "rng": np.random.default_rng(seed=42),
    }


class TestDesignMatrixRank:
    """GM2 — No Perfect Multicollinearity: rank(X) == number of columns."""

    def test_rank_equals_num_columns(self, default_params):
        """Verify that the design matrix has full column rank."""
        X, _, _ = generate_synthetic_data(n=100, **default_params)
        assert (
            np.linalg.matrix_rank(X) == X.shape[1]
        ), f"rank(X)={np.linalg.matrix_rank(X)} != p+1={X.shape[1]}"

    def test_intercept_column_is_ones(self, default_params):
        """The first column must be a constant vector of ones (intercept)."""
        X, _, _ = generate_synthetic_data(n=50, **default_params)
        np.testing.assert_array_equal(X[:, 0], np.ones(50))

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_rank_with_various_sample_sizes(self, n):
        """Ensure the matrix remains full rank across different sample sizes."""
        beta = np.array([1.0, -1.0, 0.5, 2.0])
        X, _, _ = generate_synthetic_data(
            n=n, beta=beta, sigma=1.0, rng=np.random.default_rng(0)
        )
        assert np.linalg.matrix_rank(X) == X.shape[1]


class TestNoiseProperties:
    """GM1 — E[ε] = 0 | GM3 — Var(ε) = σ²I (Homoscedasticity)."""

    N_LARGE = 50_000  # đủ lớn để sai số thống kê nhỏ
    ABS_TOLERANCE = 1e-2  # dung sai tuyệt đối cho E[ε] (so với 0)
    REL_TOLERANCE = 1e-2  # dung sai tương đối cho Var(ε) (so với σ²)

    @pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0])
    def test_noise_mean_approx_zero(self, sigma):
        """E[ε] must be approximately 0 within a standard error threshold.

        An absolute tolerance is used because the target value is always 0.
        A larger σ increases the standard error of the mean = σ/√n ≈ 5/224 ≈ 0.022,
        which is accounted for by scaling the threshold to 3 * std_of_mean.
        """
        beta = np.array([1.0, 1.0])
        _, _, epsilon = generate_synthetic_data(
            n=self.N_LARGE,
            beta=beta,
            sigma=sigma,
            rng=np.random.default_rng(seed=0),
        )
        # Chuẩn hóa sai số của mean theo σ để so sánh công bằng giữa các σ
        # Std(mean) = σ/√n → dùng |mean| / (σ/√n) ~ N(0,1), kỳ vọng < 3σ
        std_of_mean = sigma / np.sqrt(self.N_LARGE)
        assert (
            abs(np.mean(epsilon)) < 3 * std_of_mean
        ), f"E[ε]={np.mean(epsilon):.5f}, 3·std_of_mean={3*std_of_mean:.5f}"

    @pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0])
    def test_noise_variance_approx_sigma_squared(self, sigma):
        """Var(ε) must be approximately σ² within a relative tolerance of 1e-2.

        A relative tolerance is used because σ² varies across test cases.
        An absolute threshold of 1e-2 would be overly strict for large σ (e.g., σ=5 → σ²=25).
        """
        beta = np.array([1.0, 1.0])
        _, _, epsilon = generate_synthetic_data(
            n=self.N_LARGE,
            beta=beta,
            sigma=sigma,
            rng=np.random.default_rng(seed=0),
        )
        relative_error = abs(np.var(epsilon) - sigma**2) / sigma**2
        assert (
            relative_error < self.REL_TOLERANCE
        ), f"|Var(ε)/σ² - 1| = {relative_error:.5f} exceeded threshold {self.REL_TOLERANCE}"


class TestOutputShape:
    """Sanity-check for return shapes and data structures."""

    def test_output_shapes(self, default_params):
        """Verify that the shapes of X, y, and epsilon match the specification."""
        n = 80
        X, y, epsilon = generate_synthetic_data(n=n, **default_params)
        p_plus_1 = len(default_params["beta"])
        assert X.shape == (n, p_plus_1)
        assert y.shape == (n,)
        assert epsilon.shape == (n,)

    def test_linear_model_identity(self, default_params):
        """Verify the data generation identity: y = Xβ + ε."""
        X, y, epsilon = generate_synthetic_data(n=200, **default_params)
        y_reconstructed = X @ default_params["beta"] + epsilon
        np.testing.assert_allclose(y, y_reconstructed, rtol=1e-10)
