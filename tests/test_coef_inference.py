import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import coef_inference


class TestCoefInferenceExpanded:

    def setup_method(self):
        # Thiết lập ma trận mẫu chuẩn
        self.X = np.array([[1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
        self.beta_hat = np.array([1.0, 2.0])
        self.sigma2 = 0.5

    def test_output_keys(self):
        # Call site 1: Kiểm tra các key chính thức
        res = coef_inference(self.X, self.beta_hat, self.sigma2)
        assert "standard_errors" in res
        assert "t_stats" in res
        assert "p_values" in res

    def test_shapes(self):
        # Call site 2: Kiểm tra kích thước đầu ra khớp với số biến p
        res = coef_inference(self.X, self.beta_hat, self.sigma2)
        p = self.X.shape[1]
        assert len(res["standard_errors"]) == p
        assert len(res["t_stats"]) == p

    def test_confidence_intervals(self):
        # Call site 3: Kiểm tra tính logic của khoảng tin cậy
        res = coef_inference(self.X, self.beta_hat, self.sigma2)
        assert np.all(res["ci_lower"] <= self.beta_hat)
        assert np.all(res["ci_upper"] >= self.beta_hat)

    def test_se_non_negative(self):
        # Call site 4: Sai số chuẩn không được âm
        res = coef_inference(self.X, self.beta_hat, self.sigma2)
        assert np.all(res["standard_errors"] >= 0)

    def test_notebook_keys_compatibility(self):
        # Call site 5: Đảm bảo tương thích mượt mà với file Notebook demo
        res = coef_inference(self.X, self.beta_hat, self.sigma2)
        assert "SE" in res
        assert "CI_lower" in res
