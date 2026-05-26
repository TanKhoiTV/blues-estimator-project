"""
Advanced Methods From Scratch Module.

This module provides the AdvancedMethods class to implement
and evaluate advanced regression techniques completely from scratch
using pure NumPy matrix operations.
"""

from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# 1. CÁC THUẬT TOÁN ĐƯỢC XÂY DỰNG TỪ ĐẦU (FROM SCRATCH)
# =====================================================================


class KernelRidgeScratch:
    """Kernel Ridge Regression with built-in Z-score scaling."""

    def __init__(self, alpha: float = 1.0, gamma: float = 0.1) -> None:
        """Initialize the Kernel Ridge model with hyper-parameters."""
        self.alpha = alpha
        self.gamma = gamma
        self.X_train_map: np.ndarray = np.array([])
        self.dual_coefficients_: np.ndarray = np.array([])
        self.feature_means: np.ndarray = np.array([])
        self.feature_stds: np.ndarray = np.array([])

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


class BayesianLinearRegressionScratch:
    """Bayesian Linear Regression using Gaussian Priors."""

    def __init__(self, alpha: float = 1.0, beta: float = 25.0) -> None:
        """Initialize the Bayesian Regression model with precision parameters."""
        self.alpha = alpha
        self.beta = beta
        self.w_mean: np.ndarray = np.array([])
        self.w_cov: np.ndarray = np.array([])

    def fit(self, X: Any, y: Any) -> "BayesianLinearRegressionScratch":
        """Fit the Bayesian linear regression and calculate posterior."""
        X_arr = np.array(X)
        y_arr = np.array(y)

        X_bias = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        num_features = X_bias.shape[1]

        precision = self.alpha * np.eye(num_features) + self.beta * np.dot(
            X_bias.T, X_bias
        )
        self.w_cov = np.linalg.inv(precision)

        self.w_mean = self.beta * np.dot(self.w_cov, np.dot(X_bias.T, y_arr))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values based on posterior mean."""
        X_arr = np.array(X)
        X_bias = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        return np.dot(X_bias, self.w_mean)


class DecisionTreeRegressorScratch:
    """Single Regression Tree used as building block for Random Forest."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        """Initialize the tree with constraints."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = -1
        self.threshold = 0.0
        self.value = 0.0
        self.left = None
        self.right = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> "DecisionTreeRegressorScratch":
        """Recursively build the decision tree structure."""
        num_samples, num_features = X.shape
        self.value = float(np.mean(y))

        if (
            depth >= self.max_depth
            or num_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return self

        best_mse = float("inf")

        for feat in range(num_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                mse = np.sum((y[left_mask] - np.mean(y[left_mask])) ** 2) + np.sum(
                    (y[right_mask] - np.mean(y[right_mask])) ** 2
                )

                if mse < best_mse:
                    best_mse = mse
                    self.feature_idx = feat
                    self.threshold = thresh

        if best_mse == float("inf"):
            return self

        left_idx = X[:, self.feature_idx] <= self.threshold
        self.left = DecisionTreeRegressorScratch(
            self.max_depth, self.min_samples_split
        ).fit(X[left_idx], y[left_idx], depth + 1)
        self.right = DecisionTreeRegressorScratch(
            self.max_depth, self.min_samples_split
        ).fit(X[~left_idx], y[~left_idx], depth + 1)
        return self

    def _predict_row(self, row: np.ndarray) -> float:
        """Traverse the tree to predict the value for a single row."""
        if self.left is None or self.right is None:
            return self.value
        if row[self.feature_idx] <= self.threshold:
            return self.left._predict_row(row)
        return self.right._predict_row(row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for multiple data points."""
        return np.array([self._predict_row(row) for row in X])


class RandomForestRegressorScratch:
    """Random Forest Regressor utilizing Bootstrapping."""

    def __init__(
        self, n_estimators: int = 50, max_depth: int = 5, seed: int = 42
    ) -> None:
        """Initialize the ensemble parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.trees: List[DecisionTreeRegressorScratch] = []

    def fit(self, X: Any, y: Any) -> "RandomForestRegressorScratch":
        """Build the forest of trees using bootstrapped subsets."""
        X_arr = np.array(X)
        y_arr = np.array(y)
        rng = np.random.default_rng(self.seed)
        num_samples = X_arr.shape[0]
        self.trees = []

        for _ in range(self.n_estimators):
            indices = rng.choice(num_samples, size=num_samples, replace=True)
            X_b, y_b = X_arr[indices], y_arr[indices]

            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth)
            tree.fit(X_b, y_b)
            self.trees.append(tree)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Aggregate predictions from all trees."""
        X_arr = np.array(X)
        tree_preds = np.array([tree.predict(X_arr) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


# =====================================================================
# 2. KHUNG QUẢN LÝ MÔ HÌNH YÊU CẦU CỦA ĐỒ ÁN (ADVANCED METHODS)
# =====================================================================


class AdvancedMethods:
    """Implements and evaluates advanced regression models for complex datasets."""

    def __init__(self) -> None:
        """Initialize the AdvancedMethods object."""
        pass

    def train_kernel_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, kernel: str = "rbf"
    ) -> Any:
        """Train a Custom Kernel Ridge Regression model from scratch.

        Scaling is implicitly handled inside the custom class.
        """
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(X_train, y_train)
        return model

    def train_bayesian_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Train a Custom Bayesian Linear Regression model from scratch."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(X_train, y_train)
        return model

    def train_random_forest_regressor(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 50
    ) -> Any:
        """Train a Custom Random Forest Regressor from scratch."""
        model = RandomForestRegressorScratch(
            n_estimators=n_estimators, max_depth=5, seed=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_advanced_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train and evaluate all custom advanced models and compile their metrics."""
        kernel_model = self.train_kernel_regression(X_train, y_train)
        bayes_model = self.train_bayesian_regression(X_train, y_train)
        rf_model = self.train_random_forest_regressor(X_train, y_train)

        models = {
            "Kernel Ridge (From Scratch)": kernel_model,
            "Bayesian Regression (From Scratch)": bayes_model,
            "Random Forest (From Scratch)": rf_model,
        }

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
