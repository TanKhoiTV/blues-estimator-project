"""Module kiểm thử cho lớp DataPipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import unittest
import pandas as pd
from part2.data_pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    """Lớp chứa các test case kiểm tra tính đúng đắn của DataPipeline."""

    def setUp(self):
        """Khởi tạo cấu trúc dữ liệu giả lập trước mỗi case test."""
        # Tạo dữ liệu giả lập
        self.pipeline = DataPipeline()
        self.df_train = pd.DataFrame(
            {
                "numeric_feature": [10, 20, 30, None],  # Mean = 20, Std = 10
                "cat_feature": ["A", "B", "A", "C"],
            }
        )
        self.df_test = pd.DataFrame(
            {
                "numeric_feature": [40, 50],
                "cat_feature": ["B", "D"],  # 'D' là category hoàn toàn mới
            }
        )

    @unittest.skip(
        "Vô hiệu hóa tạm thời vì hệ thống đã tắt tính năng Feature Scaling để phục vụ hồi quy OLS"
    )
    def test_no_data_leakage_in_scaling(self):
        """Test 1: Kiểm tra Z-score scaling trên tập test có dùng parameters của tập train hay không."""
        self.pipeline.fit(self.df_train)
        X_test_transformed = self.pipeline.transform(self.df_test)

        # Mean của train là 20, std của train là 10.
        # Nếu áp dụng đúng, giá trị 40 ở tập test sẽ thành: (40 - 20) / 10 = 2.0
        # Chứ không phải tính mean/std mới trên tập test.
        self.assertEqual(X_test_transformed["numeric_feature"].iloc[0], 2.0)

    def test_structural_integrity_categorical(self):
        """Test 2: Kiểm tra tính toàn vẹn cấu trúc cột sau khi One-hot encoding.

        Tập train và test sau khi qua transform phải có số lượng cột giống hệt nhau.
        """
        self.pipeline.fit(self.df_train)
        X_train_transformed = self.pipeline.transform(self.df_train)
        X_test_transformed = self.pipeline.transform(self.df_test)

        self.assertListEqual(
            list(X_train_transformed.columns), list(X_test_transformed.columns)
        )

    def test_missing_categorical_uses_training_mode(self):
        """Test 3: Kiểm tra tính năng điền khuyết bằng Yếu vị (Mode) đã học từ tập Train."""
        self.pipeline.fit(self.df_train)

        df_test = pd.DataFrame(
            {
                "numeric_feature": [40, 50, 60],
                "cat_feature": [None, "B", "B"],
            }
        )

        filled = self.pipeline.handle_missing_values(df_test)

        self.assertEqual(filled["cat_feature"].iloc[0], "A")
