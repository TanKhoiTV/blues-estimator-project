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
                "MMSE": [28, 29],
            }
        )

    @unittest.skip(
        "Vô hiệu hóa tạm thời vì hệ thống đã tắt tính năng Feature Scaling để phục vụ hồi quy OLS"
    )
    def test_no_data_leakage_in_scaling(self):
        """Test 1: Kiểm tra Z-score scaling trên tập test có dùng parameters của tập train hay không."""
        self.pipeline.fit(self.df_train.drop(columns=["MMSE"]))
        X_test_transformed = self.pipeline.transform(
            self.df_test.drop(columns=["MMSE"])
        )
        self.assertEqual(X_test_transformed["nWBV"].iloc[0], 2.0)

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

    def test_missing_categorical_uses_training_mode(self):
        """Test 3: Kiểm tra tính năng điền khuyết bằng Trung vị nhóm (EDUC) và cơ chế phòng vệ toàn cục."""
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
