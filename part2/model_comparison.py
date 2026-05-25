"""
Model Comparison Module.

This module provides the ModelComparison class to train, evaluate,
and compare multiple regression models.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from part1.ols_implementation import (
    ols_fit,
    coef_inference,
    vif,
)

class ModelComparison:
    """
    Trains and compares multiple regression models.

    Supports Linear Regression, Ridge Regression, Lasso Regression,
    and ElasticNet Regression.
    """

    def __init__(self) -> None:
        """Initialize the ModelComparison object."""
        pass

    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a Linear Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.

        Returns
        -------
        Any
            The trained Linear Regression model instance.
        """
        pass

    def train_ridge_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
    ) -> Any:
        """
        Train a Ridge Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Regularization strength (default is 1.0).

        Returns
        -------
        Any
            The trained Ridge Regression model instance.
        """
        pass

    def train_lasso_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
    ) -> Any:
        """
        Train a Lasso Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Regularization strength (default is 1.0).

        Returns
        -------
        Any
            The trained Lasso Regression model instance.
        """
        pass

    def train_elasticnet_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
    ) -> Any:
        """
        Train an ElasticNet Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        alpha : float, optional
            Constant that multiplies the penalty terms (default is 1.0).
        l1_ratio : float, optional
            The ElasticNet mixing parameter (default is 0.5).

        Returns
        -------
        Any
            The trained ElasticNet Regression model instance.
        """
        pass

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on testing data.

        Parameters
        ----------
        model : Any
            The trained regression model.
        X_test : pd.DataFrame
            The testing feature data.
        y_test : pd.Series
            The testing target data.

        Returns
        -------
        dict of str to float
            A dictionary containing evaluation metrics (e.g., MSE, R2).
        """
        pass

    def compare_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Train and evaluate all supported models and compile their metrics.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature data.
        y_train : pd.Series
            The training target data.
        X_test : pd.DataFrame
            The testing feature data.
        y_test : pd.Series
            The testing target data.

        Returns
        -------
        pd.DataFrame
            A dataframe containing comparison metrics for all trained models.
        """
        pass

    def generate_summary(self, metrics_df: pd.DataFrame) -> str:
        """
        Generate a text summary of the model comparison results.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            The dataframe containing model comparison metrics.

        Returns
        -------
        str
            A formatted summary string of the results.
        """
        pass

class OLSBaseline:
    """
    OLS baseline model using custom implementation from Part 1.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.beta_hat = None
        self.sigma2_hat = None

        self.metrics = None
        self.inference_table = None
        self.vif_table = None

        self.y_train_pred = None
        self.y_test_pred = None
        self.residuals = None

        self.X_train = None
        self.X_train_raw = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit OLS using custom implementation from Part 1.
        """

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        self.X_train_raw = X_train
        self.y_train = y_train

        n = X_train.shape[0]

        # augmented matrix with intercept
        self.X_train = np.column_stack([
            np.ones(n),
            X_train
        ])

        # ols_fit internally adds intercept
        self.beta_hat, self.sigma2_hat = ols_fit(
            X_train,
            y_train,
        )

        self.y_train_pred = (
            self.X_train @ self.beta_hat
        )

        self.residuals = (
            y_train - self.y_train_pred
        )

        return self

    def predict(self, X):
        """
        Predict using fitted OLS model.
        """

        X = np.asarray(X, dtype=float)

        n = X.shape[0]

        X_aug = np.column_stack([
            np.ones(n),
            X
        ])

        return X_aug @ self.beta_hat

    def evaluate(self, X_test, y_test):
        """
        Evaluate on held-out test set.
        """

        y_test = np.asarray(y_test, dtype=float)

        self.y_test_pred = self.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            self.y_test_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                self.y_test_pred,
            )
        )

        r2 = r2_score(
            y_test,
            self.y_test_pred,
        )

        self.metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "R2_test": r2,
        }

        return self.metrics

    def run_inference(self, feature_names):
        """
        Compute coefficient inference statistics.
        """
        feature_names = [
            "Intercept"
        ] + list(feature_names)

        result = coef_inference(
            self.X_train,
            self.y_train,
            self.beta_hat,
            self.sigma2_hat,
        )

        self.inference_table = pd.DataFrame({
            "feature": feature_names,
            "coefficient": self.beta_hat,
            "std_error": result["standard_errors"],
            "t_stat": result["t_statistics"],
            "p_value": result["p_values"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
        })

        return self.inference_table

    def compute_vif(self, X, feature_names):
        """
        Compute Variance Inflation Factors.
        """

        vif_values = vif(X)

        vif_scores = []

        for feature, vif_value in zip(feature_names, vif_values):  
            vif_scores.append({
                "feature": feature,
                "VIF": vif_value,
                "High Multicollinearity": (
                    vif_value > 10
                ),
            })

        self.vif_table = pd.DataFrame(
            vif_scores
        )

        return self.vif_table

    def diagnostic_plots(self):
        """
        Produce 4 residual diagnostic plots.
        """
        fitted = self.y_train_pred
        residuals = self.residuals

        residual_std = np.std(residuals)

        if residual_std == 0:
            standardized_residuals = residuals
        else:
            standardized_residuals = (
                residuals / residual_std
            )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10),
        )

        # 1. Residuals vs Fitted
        axes[0, 0].scatter(
            fitted,
            residuals,
        )

        axes[0, 0].axhline(
            y=0,
            color="red",
            linestyle="--",
        )

        axes[0, 0].set_title(
            "Residuals vs Fitted"
        )

        axes[0, 0].set_xlabel(
            "Fitted Values"
        )

        axes[0, 0].set_ylabel(
            "Residuals"
        )

        # 2. Q-Q Plot
        stats.probplot(
            standardized_residuals,
            dist="norm",
            plot=axes[0, 1],
        )

        axes[0, 1].set_title(
            "Q-Q Plot"
        )

        # 3. Scale-Location
        axes[1, 0].scatter(
            fitted,
            np.sqrt(
                np.abs(
                    standardized_residuals
                )
            ),
        )

        axes[1, 0].set_title(
            "Scale-Location"
        )

        axes[1, 0].set_xlabel(
            "Fitted Values"
        )

        axes[1, 0].set_ylabel(
            "Sqrt(|Standardized Residuals|)"
        )

        # 4. Cook's Distance
        xtx_inv = np.linalg.pinv(
            self.X_train.T @ self.X_train
        )

        leverage = np.diag(
            self.X_train
            @ xtx_inv
            @ self.X_train.T
        )

        cooks_d = (
            (
                residuals ** 2
            ) / (
                self.X_train.shape[1]
                * self.sigma2_hat
            )
        ) * (
            leverage
            / (1 - leverage) ** 2
        )

        axes[1, 1].stem(
            np.arange(len(cooks_d)),
            cooks_d,
        )

        axes[1, 1].set_title(
            "Cook's Distance"
        )

        axes[1, 1].set_xlabel(
            "Observation"
        )

        axes[1, 1].set_ylabel(
            "Cook's Distance"
        )

        plt.tight_layout()
        plt.show()
