"""
Advanced Methods From Scratch Module.

This module provides the AdvancedMethods class to implement
and evaluate advanced regression techniques completely from scratch
using pure NumPy matrix operations.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

        models = {
            "Kernel Ridge (From Scratch)": self.train_kernel_regression(
                X_train, y_train
            )
        }
        self._cached_models = models
        self._cached_data = data_key
        return models

    def train_kernel_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Custom Kernel Ridge Regression model from scratch."""
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(X_train, y_train)
        return model

    def evaluate_advanced_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train and evaluate Kernel Ridge regressor on the training data."""
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
        """Plot a bar chart comparing RMSE, MAE, and R² for the advanced model."""
        metrics_df = self.evaluate_advanced_models(X_train, y_train, X_test, y_test)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Advanced Model Comparison on Test Set",
            fontsize=14,
            fontweight="bold",
        )

        for ax, metric in zip(axes, ["RMSE", "MAE", "R-squared"]):
            sns.barplot(
                data=metrics_df,
                x="Model",
                y=metric,
                palette="Blues_d",
                ax=ax,
            )
            ax.set_title(f"{metric} by Model")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Plot predicted vs actual values for the advanced model."""
        models = self._get_or_fit_models(X_train, y_train)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(
            "Predicted vs Actual Values — Kernel Ridge",
            fontsize=14,
            fontweight="bold",
        )
        y_test_arr = np.array(y_test)

        name, model = list(models.items())[0]
        y_pred = model.predict(X_test)

        ax.scatter(y_test_arr, y_pred, alpha=0.6, edgecolors="k", label="Samples")
        lims = [
            min(y_test_arr.min(), y_pred.min()),
            max(y_test_arr.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()

        plt.tight_layout()
        plt.show()
