"""
Advanced Methods From Scratch Module.

This module provides the AdvancedMethods class to implement
and evaluate advanced regression techniques completely from scratch
using pure NumPy matrix operations.
"""

from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class KernelRidgeScratch:
    """Kernel Ridge Regression with built-in Z-score scaling."""

    def __init__(self, alpha: float = 1.0, gamma: float = 0.1) -> None:
        """Initialize the Kernel Ridge model with hyper-parameters.

        Parameters
        ----------
        alpha : float, optional
            Regularization strength. Larger values increase regularization.
            Default is 1.0.
        gamma : float, optional
            RBF kernel bandwidth parameter. Controls the width of the Gaussian
            kernel. Default is 0.1.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.X_train_map: Optional[np.ndarray] = None
        self.dual_coefficients_: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculate the Radial Basis Function kernel between two matrices.

        Computes K(X1, X2) = exp(-gamma * ||X1_i - X2_j||^2) for all pairs
        (i, j) using the expanded squared norm identity to avoid explicit loops.

        Parameters
        ----------
        X1 : np.ndarray of shape (n_samples_1, n_features)
            First input matrix.
        X2 : np.ndarray of shape (n_samples_2, n_features)
            Second input matrix.

        Returns
        -------
        np.ndarray of shape (n_samples_1, n_samples_2)
            Gram matrix of RBF kernel values between each pair of rows.
        """
        s1 = np.sum(X1**2, axis=1, keepdims=True)
        s2 = np.sum(X2**2, axis=1)
        dist_matrix = s1 + s2 - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_matrix)

    def fit(self, X: Any, y: Any) -> "KernelRidgeScratch":
        """Fit the Kernel Ridge model on training data.

        Scales features using Z-score normalization computed on the training
        set, builds the RBF kernel Gram matrix, and solves the dual form:
        (K + alpha * I) * alpha_hat = y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target vector.

        Returns
        -------
        KernelRidgeScratch
            The fitted model instance (self).
        """
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
        """Predict target values using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        X_arr = np.array(X)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        K_test = self._rbf_kernel(X_scaled, self.X_train_map)
        return np.dot(K_test, self.dual_coefficients_)


class BayesianLinearRegressionScratch:
    """Bayesian Linear Regression using Gaussian Priors with Z-score scaling."""

    def __init__(self, alpha: float = 1.0, beta: float = 25.0) -> None:
        """Initialize the Bayesian Regression model with precision parameters.

        Parameters
        ----------
        alpha : float, optional
            Prior precision (inverse variance) on the weight vector.
            Controls the strength of the zero-mean Gaussian prior.
            Default is 1.0.
        beta : float, optional
            Noise precision (inverse variance) of the likelihood.
            Equivalent to 1 / sigma^2 of the observation noise.
            Default is 25.0.
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean: Optional[np.ndarray] = None
        self.w_cov: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "BayesianLinearRegressionScratch":
        """Fit the Bayesian linear regression and calculate posterior.

        Z-score scales features first so the zero-mean Gaussian prior
        is applied uniformly across features regardless of their original
        scale. Augments features with a bias column, then computes the
        posterior precision matrix S^{-1} = alpha * I + beta * X^T X
        and posterior mean m = beta * S * X^T y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix (without bias column).
        y : array-like of shape (n_samples,)
            Training target vector.

        Returns
        -------
        BayesianLinearRegressionScratch
            The fitted model instance (self).
        """
        X_arr = np.array(X)
        y_arr = np.array(y)

        self.feature_means = np.mean(X_arr, axis=0)
        self.feature_stds = np.std(X_arr, axis=0) + 1e-8
        X_scaled = (X_arr - self.feature_means) / self.feature_stds

        X_bias = np.hstack([np.ones((X_arr.shape[0], 1)), X_scaled])
        num_features = X_bias.shape[1]

        precision = self.alpha * np.eye(num_features) + self.beta * np.dot(
            X_bias.T, X_bias
        )
        self.w_cov = np.linalg.inv(precision)
        self.w_mean = self.beta * np.dot(self.w_cov, np.dot(X_bias.T, y_arr))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values based on posterior mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values using the posterior mean weights.
        """
        X_arr = np.array(X)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        X_bias = np.hstack([np.ones((X_arr.shape[0], 1)), X_scaled])
        return np.dot(X_bias, self.w_mean)


class DecisionTreeRegressorScratch:
    """Single Regression Tree used as building block for Random Forest."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        """Initialize the tree with depth and split constraints.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree. Growth stops once this depth is reached.
            Default is 5.
        min_samples_split : int, optional
            Minimum number of samples required to attempt a split.
            Default is 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = -1
        self.threshold = 0.0
        self.value = 0.0
        self.left: Optional["DecisionTreeRegressorScratch"] = None
        self.right: Optional["DecisionTreeRegressorScratch"] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, depth: int = 0
    ) -> "DecisionTreeRegressorScratch":
        """Recursively build the decision tree structure.

        At each node, exhaustively searches all features and thresholds to
        find the split minimizing combined MSE of the two child nodes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current node.
        y : np.ndarray of shape (n_samples,)
            Target vector for the current node.
        depth : int, optional
            Current depth in the tree. Default is 0 (root).

        Returns
        -------
        DecisionTreeRegressorScratch
            The fitted node (self), which recursively contains child nodes.
        """
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
        """Traverse the tree to predict the value for a single row.

        Parameters
        ----------
        row : np.ndarray of shape (n_features,)
            A single sample to predict.

        Returns
        -------
        float
            Predicted target value for the given sample.
        """
        if self.left is None or self.right is None:
            return self.value
        if row[self.feature_idx] <= self.threshold:
            return self.left._predict_row(row)
        return self.right._predict_row(row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for multiple data points.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        return np.array([self._predict_row(row) for row in X])


class RandomForestRegressorScratch:
    """Random Forest Regressor with bootstrapping and feature subsampling."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        max_features: Any = "sqrt",
        seed: int = 42,
    ) -> None:
        """Initialize the ensemble parameters.

        Parameters
        ----------
        n_estimators : int, optional
            Number of decision trees to build. Default is 50.
        max_depth : int, optional
            Maximum depth for each individual tree. Default is 5.
        max_features : int, str, optional
            Number of features to consider at each split. Can be
            ``'sqrt'`` (default), ``'log2'``, or an int. ``None`` means
            all features.
        seed : int, optional
            Random seed for reproducible bootstrap sampling. Default is 42.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees: List[tuple] = []

    def fit(self, X: Any, y: Any) -> "RandomForestRegressorScratch":
        """Build the forest of trees using bootstrapped subsets.

        Each tree is trained on a bootstrap sample (sampling with replacement)
        of the training data and a random subset of features at each split,
        reducing variance via both bagging and feature subsampling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target vector.

        Returns
        -------
        RandomForestRegressorScratch
            The fitted ensemble model (self).
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        rng = np.random.default_rng(self.seed)
        num_samples, n_features = X_arr.shape[0], X_arr.shape[1]
        self.trees = []

        if self.max_features == "sqrt":
            m_features = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            m_features = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            m_features = max(1, min(self.max_features, n_features))
        else:
            m_features = n_features

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = rng.choice(num_samples, size=num_samples, replace=True)
            X_b, y_b = X_arr[indices], y_arr[indices]

            # Feature subsampling — each tree gets a random feature subset
            feature_subset = rng.choice(n_features, size=m_features, replace=False)
            X_b_sub = X_b[:, feature_subset]

            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth)
            # Pass a virtual index range so the tree uses all available features
            tree.fit(X_b_sub, y_b)
            self.trees.append((tree, feature_subset))

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Aggregate predictions from all trees.

        Each tree predicts using only its assigned feature subset.
        Final prediction is the mean across all trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Averaged predicted target values across all trees.
        """
        X_arr = np.array(X)
        tree_preds = np.array(
            [tree.predict(X_arr[:, feat_sub]) for tree, feat_sub in self.trees]
        )
        return np.mean(tree_preds, axis=0)


class AdvancedMethods:
    """Implements and evaluates advanced regression models for complex datasets."""

    def __init__(self) -> None:
        """Initialize the AdvancedMethods object with a model cache."""
        self._cached_models: Optional[dict] = None
        self._cached_data: Optional[tuple] = None

    def _get_or_fit_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """Train and cache models once, reuse on subsequent calls.

        Prevents retraining all models each time a plotting or evaluation
        method is called in the notebook. Cache is invalidated when new
        training data is passed.
        """
        data_key = (id(X_train), id(y_train))
        if self._cached_models is not None and self._cached_data == data_key:
            return self._cached_models

        models = {
            "Kernel Ridge (From Scratch)": self.train_kernel_regression(
                X_train, y_train
            ),
            "Bayesian Regression (From Scratch)": self.train_bayesian_regression(
                X_train, y_train
            ),
            "Random Forest (From Scratch)": self.train_random_forest_regressor(
                X_train, y_train
            ),
        }
        self._cached_models = models
        self._cached_data = data_key
        return models

    def train_kernel_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Custom Kernel Ridge Regression model from scratch.

        Uses an RBF kernel with built-in Z-score feature scaling.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_samples,)
            Training target vector.

        Returns
        -------
        KernelRidgeScratch
            Fitted Kernel Ridge Regression model.
        """
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(X_train, y_train)
        return model

    def train_bayesian_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Train a Custom Bayesian Linear Regression model from scratch.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_samples,)
            Training target vector.

        Returns
        -------
        BayesianLinearRegressionScratch
            Fitted Bayesian Linear Regression model.
        """
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(X_train, y_train)
        return model

    def train_random_forest_regressor(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 50
    ) -> Any:
        """Train a Custom Random Forest Regressor from scratch.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_samples,)
            Training target vector.
        n_estimators : int, optional
            Number of trees in the forest. Default is 50.

        Returns
        -------
        RandomForestRegressorScratch
            Fitted Random Forest Regression model.
        """
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
        """Train and evaluate all custom advanced models and compile their metrics.

        Fits Kernel Ridge, Bayesian Linear, and Random Forest regressors on
        the training data, evaluates each on the test set, and returns a
        summary DataFrame of R², RMSE, and MAE.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_train, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_train,)
            Training target vector.
        X_test : pd.DataFrame of shape (n_test, n_features)
            Test feature matrix.
        y_test : pd.Series of shape (n_test,)
            Test target vector.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["Model", "R-squared", "RMSE", "MAE"]``,
            one row per model.
        """
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
        """Plot a bar chart comparing RMSE, MAE, and R² across all models.

        Trains all three advanced models, computes their test metrics, and
        renders a grouped bar chart for visual comparison. Each metric is
        displayed in its own subplot.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_train, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_train,)
            Training target vector.
        X_test : pd.DataFrame of shape (n_test, n_features)
            Test feature matrix.
        y_test : pd.Series of shape (n_test,)
            Test target vector.

        Returns
        -------
        None
            Displays the comparison plot inline.
        """
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
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        plt.show()

    def plot_predictions_vs_actual(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Plot predicted vs actual values for each advanced model.

        Uses cached models from ``_get_or_fit_models`` to avoid
        retraining on every call.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_train, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_train,)
            Training target vector.
        X_test : pd.DataFrame of shape (n_test, n_features)
            Test feature matrix.
        y_test : pd.Series of shape (n_test,)
            Test target vector.

        Returns
        -------
        None
            Displays the predicted vs actual plot inline.
        """
        models = self._get_or_fit_models(X_train, y_train)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Predicted vs Actual Values — Advanced Models",
            fontsize=14,
            fontweight="bold",
        )
        y_test_arr = np.array(y_test)

        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            ax.scatter(y_test_arr, y_pred, alpha=0.6, edgecolors="k", label="Samples")
            lims = [
                min(y_test_arr.min(), y_pred.min()),
                max(y_test_arr.max(), y_pred.max()),
            ]
            ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
            ax.set_title(name)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.legend()

        plt.tight_layout()
        plt.show()
