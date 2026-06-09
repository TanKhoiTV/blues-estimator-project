"""
Advanced Methods Module.

This module provides the AdvancedMethods class to implement
and evaluate advanced regression techniques.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


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
        }
        self._cached_models = models
        self._cached_data = data_key
        return models

    def cv_select_kernel_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alphas: Optional[list] = None,
        gammas: Optional[list] = None,
        k: int = 5,
    ) -> tuple:
        """Select optimal alpha and gamma for Kernel Ridge via K-Fold CV.

        Performs an exhaustive grid search over all (alpha, gamma) combinations
        using K-Fold cross-validation on the training set. For each fold the
        model is fit on the fold's training split and evaluated on its
        validation split. The pair minimising mean CV MSE is returned.

        The internal Z-score scaling inside ``KernelRidgeScratch.fit`` ensures
        no leakage: scaling parameters are re-computed from scratch on each
        fold's training split, never from the validation split.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_samples,)
            Training target vector.
        alphas : list of float, optional
            Regularisation strengths to search over.
            Default is ``[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]``.
        gammas : list of float, optional
            RBF bandwidth values to search over.
            Default is ``[0.001, 0.01, 0.1, 0.5, 1.0]``.
        k : int, optional
            Number of cross-validation folds. Default is 5.

        Returns
        -------
        best_alpha : float
            Alpha value that minimises mean CV MSE.
        best_gamma : float
            Gamma value that minimises mean CV MSE.
        cv_results : pd.DataFrame
            Full grid-search results with columns
            ``["alpha", "gamma", "cv_mse"]``, sorted ascending by
            ``cv_mse``.
        """
        if alphas is None:
            alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        if gammas is None:
            gammas = [0.001, 0.01, 0.1, 0.5, 1.0]

        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        best_alpha, best_gamma = alphas[0], gammas[0]
        best_cv_mse = np.inf
        rows = []

        for alpha in alphas:
            for gamma in gammas:
                fold_mses = []
                for train_idx, val_idx in kf.split(X_arr):
                    X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                    y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

                    model = KernelRidgeScratch(alpha=alpha, gamma=gamma)
                    model.fit(X_tr, y_tr)
                    y_pred_val = model.predict(X_val)
                    fold_mses.append(mean_squared_error(y_val, y_pred_val))

                cv_mse = float(np.mean(fold_mses))
                rows.append({"alpha": alpha, "gamma": gamma, "cv_mse": cv_mse})

                if cv_mse < best_cv_mse:
                    best_cv_mse = cv_mse
                    best_alpha, best_gamma = alpha, gamma

        cv_results = pd.DataFrame(rows).sort_values("cv_mse").reset_index(drop=True)
        return best_alpha, best_gamma, cv_results

    def train_kernel_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alpha: float = 0.5,
        gamma: float = 0.1,
    ) -> Any:
        """Train a Custom Kernel Ridge Regression model from scratch.

        Uses an RBF kernel with built-in Z-score feature scaling.
        Pass ``alpha`` and ``gamma`` obtained from
        :meth:`cv_select_kernel_params` to avoid hard-coded hyper-parameters.

        Parameters
        ----------
        X_train : pd.DataFrame of shape (n_samples, n_features)
            Training feature matrix.
        y_train : pd.Series of shape (n_samples,)
            Training target vector.
        alpha : float, optional
            Regularisation strength. Default is 0.5.
        gamma : float, optional
            RBF kernel bandwidth. Default is 0.1.

        Returns
        -------
        KernelRidgeScratch
            Fitted Kernel Ridge Regression model.
        """
        model = KernelRidgeScratch(alpha=alpha, gamma=gamma)
        model.fit(X_train, y_train)
        return model

    def evaluate_advanced_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train and evaluate the Kernel Ridge model and compile its metrics.

        Fits Kernel Ridge on the training data, evaluates on the test set,
        and returns a summary DataFrame of R², RMSE, and MAE.

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
        """Plot predicted vs actual values for the Kernel Ridge model.

        Uses cached model from ``_get_or_fit_models`` to avoid
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

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle(
            "Predicted vs Actual Values — Kernel Ridge",
            fontsize=14,
            fontweight="bold",
        )
        y_test_arr = np.array(y_test)

        for name, model in models.items():
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
