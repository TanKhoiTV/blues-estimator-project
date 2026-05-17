import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.ols_implementation import model_metrics


class TestModelMetrics:
    def setup_method(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        self.p = 2

    def test_perfect_fit(self):
        """Test metrics when predictions match ground truth exactly."""
        # FIX 1: Đã đổi tham số thứ hai thành self.y_true theo đúng prompt yêu cầu
        metrics = model_metrics(self.y_true, self.y_true, self.p)

        assert metrics["RSS"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-10)

    def test_metrics_keys_and_outputs(self):
        """Verify that the expanded contract keys exist and return valid values."""
        # FIX 2: Đáp ứng trọn vẹn Option B của prompt đòi hỏi các key viết tắt
        metrics = model_metrics(self.y_true, self.y_pred, self.p)

        assert "Adj_R2" in metrics
        assert "F_stat" in metrics
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert metrics["MAE"] > 0
        assert metrics["RMSE"] > 0
