import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.residual_analysis import residual_plots
from part1.ols_implementation import ols_fit, hat_matrix


class TestResidualAnalysis:
    """Test suite for residual_plots function ensuring numerical correctness of underlying metrics."""

    def test_dimensions_and_figure_structure(self):
        """Verify that the returned object is a valid matplotlib Figure and maintains correct dimensions.

        Checks that the outputs generated internally have a length equal to n.
        """
        np.random.seed(42)
        n_samples = 15
        n_features = 3

        X = np.random.randn(n_samples, n_features)
        beta_true = np.array([1.0, 2.0, -0.5])
        y = X @ beta_true + np.random.normal(0, 0.1, size=n_samples)

        # Thực thi hàm sinh biểu đồ
        fig = residual_plots(X, y, beta_true)

        # Xác thực kiểu dữ liệu trả về và cấu trúc layout 2x2
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 4

        # Trích xuất dữ liệu từ Plot 1 (Residuals vs Fitted) để kiểm tra kích thước đầu ra (n)
        scatter_data = axes[0].collections[0].get_offsets()
        assert scatter_data.shape[0] == n_samples

        # Trích xuất dữ liệu từ Plot 4 (Cook's Distance) để kiểm tra kích thước đầu ra (n)
        stem_container = axes[3].containers[0]
        cooks_d_segments = stem_container.stemlines.get_segments()
        assert len(cooks_d_segments) == n_samples

        plt.close(fig)

    def test_residuals_properties_and_zero_sum(self):
        """Verify the algebraic properties of OLS residuals.

        Ensures that the sum of residuals approximates zero when an intercept is present
        and validates the calculation of y_hat and residuals.
        """
        np.random.seed(123)
        n_samples = 20
        n_features = 2

        X_raw = np.random.rand(n_samples, n_features - 1)
        beta_true = np.array([2.5, -1.2])

        X_full = np.column_stack([np.ones(n_samples), X_raw])
        y = X_full @ beta_true + np.random.normal(0, 0.05, size=n_samples)

        beta_hat, _ = ols_fit(X_raw, y)

        fig = residual_plots(X_full, y, beta_hat)
        axes = fig.get_axes()

        # Trích xuất y_hat và residuals từ Plot 1: Residuals vs Fitted
        scatter_data = axes[0].collections[0].get_offsets()
        y_hat_extracted = scatter_data[:, 0]
        residuals_extracted = scatter_data[:, 1]

        # Kiểm tra công thức cơ bản: y_hat = X @ beta_hat và residuals = y - y_hat
        np.testing.assert_array_almost_equal(
            y_hat_extracted, X_full @ beta_hat, decimal=7
        )
        np.testing.assert_array_almost_equal(
            residuals_extracted, y - y_hat_extracted, decimal=7
        )

        # Tính chất cốt lõi: Tổng các phần dư phải xấp xỉ bằng 0 (với sai số dấu phẩy động cực nhỏ)
        assert np.sum(residuals_extracted) == pytest.approx(0.0, abs=1e-10)

        plt.close(fig)

    def test_standardized_residuals_and_cooks_distance(self):
        """Verify the analytical correctness of Standardized Residuals and Cook's Distance formulas.

        Validates the math logic against an independent exact theoretical implementation.
        """
        np.random.seed(456)
        n_samples = 12
        n_features = 2  # Không tính intercept

        X_raw = np.random.rand(n_samples, n_features)
        X = np.column_stack([np.ones(n_samples), X_raw])
        beta_hat = np.array([0.5, 0.8, 1.5])  # intercept + 2 predictors
        y = X @ beta_hat + np.random.normal(0, 0.2, size=n_samples)

        fig = residual_plots(X, y, beta_hat)
        axes = fig.get_axes()

        # 1. Trích xuất dữ liệu Standardized Residuals thực tế từ trục Y của Plot 2 (Normal Q-Q) hoặc Plot 3
        scatter_data_p3 = axes[2].collections[0].get_offsets()
        sqrt_std_res_extracted = scatter_data_p3[:, 1]
        std_residuals_actual = (
            sqrt_std_res_extracted**2
        )  # Vì Plot 3 vẽ căn bậc hai trị tuyệt đối

        # 2. Trích xuất Cook's Distance thực tế từ Plot 4
        stem_container = axes[3].containers[0]
        cooks_d_segments = stem_container.stemlines.get_segments()
        cooks_d_actual = np.array([seg[1][1] for seg in cooks_d_segments])

        # 3. Tính toán lại theo công thức lý thuyết độc lập để đối chứng (Verification)
        n, p = X.shape
        y_hat_expected = X @ beta_hat
        res_expected = y - y_hat_expected

        # Tính toán ma trận Hat thủ công: H = X(X^TX)^-1X^T
        H_mat_expected = hat_matrix(X)
        h_ii_expected = np.diag(H_mat_expected)
        h_ii_expected = np.clip(h_ii_expected, 0, 0.9999)

        mse_expected = np.sum(res_expected**2) / (n - p)
        std_res_expected = res_expected / np.sqrt(mse_expected * (1 - h_ii_expected))
        cooks_d_expected = (std_res_expected**2 / p) * (
            h_ii_expected / (1 - h_ii_expected)
        )

        # 4. Assert đối chứng kiểm tra độ khớp chính xác tuyệt đối
        np.testing.assert_array_almost_equal(
            std_residuals_actual, np.abs(std_res_expected), decimal=7
        )
        np.testing.assert_array_almost_equal(
            cooks_d_actual, cooks_d_expected, decimal=7
        )

        # Kiểm tra điều kiện Cook's Distance phải luôn mang giá trị không âm
        assert np.all(cooks_d_actual >= 0.0)

        plt.close(fig)
