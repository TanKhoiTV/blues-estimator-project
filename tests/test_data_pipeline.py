"""Data pipeline module for preprocessing OASIS longitudinal MRI data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataPipeline:
    """Hệ thống luồng dữ liệu (Data Pipeline) chuẩn mực chống Rò rỉ Dữ liệu.

    Tuân thủ nghiêm ngặt quy trình: Phân tách (Split) -> Học luật (Fit) -> Áp đặt (Transform).
    """

    def __init__(self):
        """Khởi tạo DataPipeline và các dictionary lưu trữ tham số."""
        self.ses_educ_medians_ = {}
        self.ses_global_mode_ = None
        self.median_values_ = {}

    def fit(self, df, y=None):
        """Pha HỌC (Fit): Khai phá các quy luật thống kê chỉ từ tập Huấn luyện (Train Set)."""
        if "EDUC" in df.columns and "SES" in df.columns:
            self.ses_educ_medians_ = df.groupby("EDUC")["SES"].median().to_dict()

            if not df["SES"].mode().empty:
                self.ses_global_mode_ = df["SES"].mode()[0]
            else:
                self.ses_global_mode_ = 2.0

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != "SES":
                self.median_values_[col] = df[col].median()

        return self

    def transform(self, df):
        """Pha ÁP ĐẶT (Transform): Sử dụng các quy luật đã học áp dụng lên tập dữ liệu.

        Tập dữ liệu này có thể là Train (tự áp dụng) hoặc Test (bị động).
        """
        df_clean = df.copy()

        if "EDUC" in df_clean.columns and "SES" in df_clean.columns:
            df_clean["SES"] = df_clean.apply(
                lambda row: (
                    self.ses_educ_medians_.get(row["EDUC"], row["SES"])
                    if pd.isna(row["SES"])
                    else row["SES"]
                ),
                axis=1,
            )
            df_clean["SES"] = df_clean["SES"].fillna(self.ses_global_mode_)

        for col, median_val in self.median_values_.items():
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col] = df_clean[col].fillna(median_val)

        return df_clean

    def preprocess(
        self,
        file_path: str,
        target_column: str = "MMSE",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """Hàm tích hợp chu trình hoàn chỉnh từ đọc file đến trả về tập Train/Test."""
        df = pd.read_csv(file_path)
        df_clean = df.copy()

        cols_to_drop = ["Subject ID", "MRI ID", "Hand"]
        existing_cols = [c for c in cols_to_drop if c in df_clean.columns]
        if existing_cols:
            df_clean = df_clean.drop(columns=existing_cols)

        df_clean = pd.get_dummies(df_clean, drop_first=True)
        df_clean = df_clean.dropna(subset=[target_column])

        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.fit(X_train)
        X_train_clean = self.transform(X_train)
        X_test_clean = self.transform(X_test)

        return X_train_clean, X_test_clean, y_train, y_test
