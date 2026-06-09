"""
Test Module for Data Pipeline.

This module contains unit tests for the DataPipeline class,
ensuring correct behavior of fitting, transforming, and missing value imputation
based on production standards.
"""

import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

# Thêm thư mục gốc vào hệ thống để có thể import từ part2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from part2.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Test suite cho class DataPipeline."""

    def setUp(self):
        """Khởi tạo pipeline và dữ liệu mẫu trước mỗi test case."""
        self.pipeline = DataPipeline()

        # Dữ liệu huấn luyện mẫu
        self.X_train = pd.DataFrame(
            {
                "nWBV": [0.75, 0.80, 0.72, 0.78],
                "Age": [65, 70, 75, 80],
                "EDUC": [12, 16, 12, 14],
                "SES": [3.0, 1.0, 4.0, np.nan],
                # Nhóm EDUC=12 có SES là 3.0 và 4.0 -> Median = 3.5
                # Nhóm EDUC=16 có SES là 1.0 -> Median = 1.0
                # Median toàn cục (skipna) của [3.0, 1.0, 4.0] là 3.0
                "M/F": ["M", "F", "F", "M"],
            }
        )

    def test_fit_stores_training_attributes(self):
        """Kiểm tra xem hàm fit có lưu đúng ses_global_median_ thay vì mode không."""
        self.pipeline.fit(self.X_train)

        # 1. Đảm bảo thuộc tính lỗi (mode) không còn tồn tại
        self.assertFalse(
            hasattr(self.pipeline, "ses_global_mode_"),
            "Lỗi: Pipeline vẫn còn sinh ra thuộc tính ses_global_mode_ cũ.",
        )

        # 2. Đảm bảo thuộc tính mới (median) được khởi tạo
        self.assertTrue(
            hasattr(self.pipeline, "ses_global_median_"),
            "Lỗi: Thiếu thuộc tính ses_global_median_.",
        )

        # 3. Kiểm tra giá trị global median phải là 3.0
        self.assertEqual(self.pipeline.ses_global_median_, 3.0)

        # 4. Kiểm tra dict median theo nhóm EDUC
        self.assertIn(12, self.pipeline.ses_educ_medians_)
        self.assertEqual(self.pipeline.ses_educ_medians_[12], 3.5)

    def test_handle_missing_values_with_fallback(self):
        """Kiểm tra logic điền khuyết SES cho nhóm đã biết và nhóm mới (Fallback)."""
        self.pipeline.fit(self.X_train)

        # Tạo tập test có chứa giá trị NaN ở biến SES
        X_test = pd.DataFrame(
            {
                "EDUC": [
                    12,
                    18,
                ],  # 12 là nhóm cũ (có median=3.5), 18 là nhóm mới hoàn toàn
                "SES": [np.nan, np.nan],
                "Age": [70, 72],
                "nWBV": [0.8, 0.7],
            }
        )

        X_transformed = self.pipeline.handle_missing_values(X_test)

        # Dòng 1 (EDUC=12): Phải dùng median của nhóm EDUC=12 là 3.5 -> ép kiểu int -> 3
        self.assertEqual(X_transformed["SES"].iloc[0], 3)

        # Dòng 2 (EDUC=18): Phải dùng global median fallback là 3.0 -> ép kiểu int -> 3
        self.assertEqual(X_transformed["SES"].iloc[1], 3)

    def test_preprocess_missing_columns_validation(self):
        """Kiểm tra hàm preprocess có báo lỗi chặt chẽ nếu dữ liệu bị thiếu cột bắt buộc."""
        df_invalid = pd.DataFrame(
            {"Age": [70], "EDUC": [12], "SES": [3], "MMSE": [28], "Subject ID": ["S1"]}
        )

        # Ghi đè phương thức load dữ liệu để trả về dữ liệu lỗi
        self.pipeline.load_dataset = lambda x: df_invalid

        with self.assertRaises(ValueError) as context:
            self.pipeline.preprocess("dummy_path.csv", target_column="MMSE")

        self.assertIn("nWBV", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
