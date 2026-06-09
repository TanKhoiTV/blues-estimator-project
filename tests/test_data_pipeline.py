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
        """Khởi tạo cấu trúc dữ liệu giả lập chuẩn chỉnh trước mỗi case test."""
        self.pipeline = DataPipeline()

        # Thiết lập tập dữ liệu Train giả lập mang đúng tên cột y sinh để pass logic Pipeline
        self.df_train = pd.DataFrame(
            {
                "nWBV": [0.75, 0.80, 0.72, None],
                "Age": [65, 70, 75, 80],
                "EDUC": [12, 16, 12, 14],  # Phân nhóm theo học vấn
                "SES": [3.0, 1.0, 3.0, None],  # Yếu vị (Mode) toàn cục của SES là 3.0
                "Site": pd.Categorical(["A", "B", "A", "B"]),  # Biến phân loại object
                "MMSE": [27, 30, 26, 29],
            }
        )

        # Thiết lập tập dữ liệu Test giả lập để kiểm tra transform
        self.df_test = pd.DataFrame(
            {
                "nWBV": [0.78, 0.82],
                "Age": [68, 72],
                "EDUC": [
                    12,
                    18,
                ],  # Nhóm EDUC = 12 đã học ở Train, nhóm 18 là mới hoàn toàn
                "SES": [None, 2.0],  # Dòng đầu tiên bị khuyết SES để test điền khuyết
                "Site": pd.Categorical(["A", "C"]),  # "A" thấy ở Train, "C" là unseen
                "MMSE": [28, 29],
            }
        )

    # ──────────────────────────────────────────────────────────
    # Test 1 — Fit attributes
    # ──────────────────────────────────────────────────────────
    def test_fit_stores_training_attributes(self):
        """Test 1: Kiểm tra fit() lưu đúng các tham số thống kê từ tập Train."""
        df = self.df_train.drop(columns=["MMSE"])
        self.pipeline.fit(df)

        # Must store SES-EDUC group medians
        self.assertTrue(hasattr(self.pipeline, "ses_educ_medians_"))
        self.assertIn(12, self.pipeline.ses_educ_medians_)
        expected_ses_medians = df.groupby("EDUC")["SES"].median().to_dict()
        self.assertEqual(
            self.pipeline.ses_educ_medians_[12],
            expected_ses_medians.get(12),
        )

        # Must store global SES mode
        self.assertTrue(hasattr(self.pipeline, "ses_global_mode_"))
        expected_ses_mode = df["SES"].mode(dropna=True).iloc[0]
        self.assertEqual(self.pipeline.ses_global_mode_, expected_ses_mode)

        # Must store numeric column means/stds
        self.assertTrue(hasattr(self.pipeline, "numeric_means_"))
        self.assertIn("nWBV", self.pipeline.numeric_means_)
        self.assertAlmostEqual(
            self.pipeline.numeric_means_["nWBV"],
            df["nWBV"].mean(),
            places=6,
        )
        self.assertIn("nWBV", self.pipeline.numeric_stds_)
        self.assertAlmostEqual(
            self.pipeline.numeric_stds_["nWBV"],
            df["nWBV"].std(),
            places=6,
        )

        # Must store categorical column info
        self.assertIn("Site", self.pipeline.categorical_columns_)
        self.assertListEqual(
            sorted(self.pipeline.categorical_values_["Site"]), ["A", "B"]
        )

        # Must store encoded column names after transform
        X_train_transformed = self.pipeline.transform(df)
        self.assertIn("Site_A", X_train_transformed.columns)

    # ──────────────────────────────────────────────────────────
    # Test 2 — Column structural integrity
    # ──────────────────────────────────────────────────────────
    def test_structural_integrity_categorical(self):
        """Test 2: Kiểm tra tính toàn vẹn cấu trúc cột sau khi xử lý dữ liệu qua Pipeline.

        Tập train và test sau khi qua transform phải có cấu trúc các cột hoàn toàn đồng nhất.
        """
        X_train_only = self.df_train.drop(columns=["MMSE"])
        X_test_only = self.df_test.drop(columns=["MMSE"])

        self.pipeline.fit(X_train_only)
        X_train_transformed = self.pipeline.transform(X_train_only)
        X_test_transformed = self.pipeline.transform(X_test_only)

        self.assertListEqual(
            list(X_train_transformed.columns), list(X_test_transformed.columns)
        )

    # ──────────────────────────────────────────────────────────
    # Test 3 — One-hot encoding of categorical variables
    # ──────────────────────────────────────────────────────────
    def test_encode_categorical_produces_dummies(self):
        """Test 3: Kiểm tra encode_categorical() tạo đúng biến giả từ tập Train."""
        self.pipeline.fit(self.df_train.drop(columns=["MMSE"]))
        X_train_transformed = self.pipeline.transform(
            self.df_train.drop(columns=["MMSE"])
        )

        # Site (category) should be one-hot encoded to Site_A, Site_B
        self.assertIn("Site_A", X_train_transformed.columns)
        self.assertIn("Site_B", X_train_transformed.columns)

        # Row 0 has Site=A -> Site_A=1, Site_B=0
        self.assertEqual(X_train_transformed["Site_A"].iloc[0], 1)
        self.assertEqual(X_train_transformed["Site_B"].iloc[0], 0)

    # ──────────────────────────────────────────────────────────
    # Test 4 — Missing categorical values use training mode
    # ──────────────────────────────────────────────────────────
    def test_missing_categorical_uses_training_mode(self):
        """Test 4: Kiểm tra tính năng điền khuyết bằng Trung vị nhóm (EDUC) và cơ chế phòng vệ toàn cục."""
        # Bước 1: Cho Pipeline học tri thức từ ma trận Train (Học được ses_by_educ của nhóm 12 là 3.0)
        X_train_only = self.df_train.drop(columns=["MMSE"])
        self.pipeline.fit(X_train_only)

        # Bước 2: Tạo một tập Test giả lập chứa dòng khuyết SES tại nhóm EDUC = 12
        df_test_missing = pd.DataFrame(
            {
                "nWBV": [0.74, 0.76],
                "Age": [67, 85],
                "EDUC": [
                    12,
                    18,
                ],  # Dòng 1 thuộc nhóm EDUC=12 (có sẵn), Dòng 2 thuộc nhóm EDUC=18 (mới)
                "SES": [
                    None,
                    None,
                ],  # Cả 2 dòng đều bị khuyết SES để kiểm tra 2 tình huống điền khuyết
            }
        )

        # Bước 3: Chạy hàm điền khuyết
        filled = self.pipeline.handle_missing_values(df_test_missing)

        # KIỂM TRA 1: Dòng 1 (EDUC=12) phải lấy đúng Trung vị nhóm đã học ở Train là 3
        self.assertEqual(filled["SES"].iloc[0], 3)

        # KIỂM TRA 2: Dòng 2 (EDUC=18 là nhóm hoàn toàn mới ở Train) phải kích hoạt Fallback toàn cục (trung vị toàn tập Train là 3)
        self.assertEqual(filled["SES"].iloc[1], 3)

        # KIỂM TRA 3: Đảm bảo sau khi xử lý dữ liệu, cột SES được ép kiểu thành số nguyên an toàn sạch bóng NaN
        self.assertEqual(filled["SES"].dtype, "int64")
