"""
Advanced Methods Module.

This module provides the AdvancedMethods class to implement
and evaluate advanced regression techniques including Kernel Ridge,
Random Forest, Gaussian Process, and GAMs.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Thư viện cho các mô hình mở rộng
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Thử import pygam, nếu không có sẽ đưa ra cảnh báo thay vì gây crash
try:
    from pygam import LinearGAM
except ImportError:
    LinearGAM = None
    print(
        "Cảnh báo: Chưa cài đặt 'pygam'. Mô hình GAM sẽ bị bỏ qua. Hãy chạy lệnh 'pip install pygam' để sử dụng."
    )


class KernelRidgeScratch:
    """Kernel Ridge Regression with built-in Z-score scaling."""

    def __init__(self, alpha: float = 1.0, gamma: float = 0.1) -> None:
        """Initialize the Kernel Ridge model with hyper-parameters."""
        self.alpha = alpha
        self.gamma = gamma
        self.X_train_map: Optional[np.ndarray] = None
        self.dual_coefficients_: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculate the Radial Basis Function kernel between two matrices."""
        s1 = np.sum(X1**2, axis=1, keepdims=True)
        s2 = np.sum(X2**2, axis=1)
        dist_matrix = s1 + s2 - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_matrix)

    def fit(self, X: Any, y: Any) -> "KernelRidgeScratch":
        """Fit the Kernel Ridge model on training data."""
        X_arr = np.array(X)
        y_arr = np.array(y)

        self.feature_means = np.mean(X_arr, axis=0)
        self.feature_stds = np.std(X_arr, axis=0) + 1e-8
        X_scaled = (X_arr - self.feature_means) / self.feature_stds

        self.X_train_map = X_scaled
        K = self._rbf_kernel(self.X_train_map, self.X_train_map)

        identity = np.eye(K.shape[0])
        self.dual_coefficients_ = np.linalg.solve(K + self.alpha * identity, y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using the trained model."""
        X_arr = np.array(X)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        K_test = self._rbf_kernel(X_scaled, self.X_train_map)
        return np.dot(K_test, self.dual_coefficients_)


class AdvancedMethods:
    """Implements and evaluates advanced regression models for complex datasets."""

    def __init__(self) -> None:
        """Initialize the AdvancedMethods object with a model cache."""
        self._cached_models: Optional[dict] = None
        self._cached_data: Optional[tuple] = None

    def _get_or_fit_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train and cache models once, reuse on subsequent calls."""
        data_key = (id(X_train), id(y_train))
        if self._cached_models is not None and self._cached_data == data_key:
            return self._cached_models

        # Khởi tạo danh sách các mô hình nâng cao
        models = {
            "Kernel Ridge (From Scratch)": self.train_kernel_regression(
                X_train, y_train
            ),
            "Random Forest Ensemble": self.train_random_forest(X_train, y_train),
            "Gaussian Process (GPR)": self.train_gaussian_process(X_train, y_train),
        }

        # Chỉ chèn GAM nếu thư viện đã được cài đặt
        if LinearGAM is not None:
            models["Generalized Additive (GAM)"] = self.train_gam(X_train, y_train)

        self._cached_models = models
        self._cached_data = data_key
        return models

    # ==========================================
    # CÁC HÀM HUẤN LUYỆN MÔ HÌNH ĐỘC LẬP
    # ==========================================

    def train_kernel_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Custom Kernel Ridge Regression model from scratch."""
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Random Forest Regressor to capture non-linear interactions."""
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_gaussian_process(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        model = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=2.0,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42,
            ),
        )
        model.fit(X_train, y_train)
        return model

    def train_gam(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Generalized Additive Model (GAM) using splines."""
        if LinearGAM is None:
            raise ImportError("Vui lòng cài đặt pygam để sử dụng hàm này.")
        model = LinearGAM()
        model.fit(X_train, y_train)
        return model

    # ==========================================
    # ĐÁNH GIÁ VÀ TRỰC QUAN HÓA
    # ==========================================

    def evaluate_advanced_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train and evaluate all models and compile their metrics."""
        models = self._get_or_fit_models(X_train, y_train)

        metrics_list = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics_list.append(
                {
                    "Model": name,
                    "R-squared": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                }
            )

        return pd.DataFrame(metrics_list)

    def plot_model_comparison(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Plot a bar chart comparing RMSE, MAE, and R² across all models."""
        metrics_df = self.evaluate_advanced_models(X_train, y_train, X_test, y_test)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Advanced Model Comparison on Test Set",
            fontsize=16,
            fontweight="bold",
        )

        for ax, metric in zip(axes, ["RMSE", "MAE", "R-squared"]):
            sns.barplot(
                data=metrics_df,
                x=metric,
                y="Model",
                palette="viridis",
                ax=ax,
            )
            ax.set_title(f"{metric} Comparison")
            ax.set_xlabel(metric)
            ax.set_ylabel("")

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Plot predicted vs actual values for all models in a grid layout."""
        models = self._get_or_fit_models(X_train, y_train)
        n_models = len(models)

        # Tự động tính toán số hàng và cột cho Subplots
        cols = 2
        rows = int(np.ceil(n_models / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
        fig.suptitle(
            "Predicted vs Actual Values — All Advanced Models",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        y_test_arr = np.array(y_test)
        # Làm phẳng array trục để dễ loop
        axes_flat = np.array(axes).flatten()

        for idx, (name, model) in enumerate(models.items()):
            ax = axes_flat[idx]
            y_pred = model.predict(X_test)

            ax.scatter(
                y_test_arr,
                y_pred,
                alpha=0.6,
                color="blue",
                edgecolors="k",
                label="Data Points",
            )

            lims = [
                min(y_test_arr.min(), y_pred.min()),
                max(y_test_arr.max(), y_pred.max()),
            ]
            ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Fit")

            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Actual MMSE Values")
            ax.set_ylabel("Predicted MMSE Values")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)

        # Ẩn các ô trống nếu số mô hình là số lẻ
        for i in range(n_models, len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        plt.show()
