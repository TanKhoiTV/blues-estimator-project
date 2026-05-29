"""
Model Comparison Module.

This module provides the ModelComparison class to train, evaluate,
and compare OLS and Ridge regression models.
"""

from typing import Dict, Any, List, Tuple
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
    model_metrics,
)

from part1.residual_analysis import residual_plots

from part1.ridge_lasso import (
    ridge_fit,
    plot_ridge_trace,
)


class ModelComparison:
    """Trains and compares OLS and Ridge regression models."""

    def __init__(self) -> None:
        """Initialize the ModelComparison object."""
        pass

    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train a Linear Regression model using raw values."""
        model = OLSBaseline()
        model.fit(X_train, y_train)
        return model

    def train_ridge_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
    ) -> Any:
        """Train a Ridge Regression model with manual internal Z-score scaling."""
        X = X_train.values
        y = y_train.values

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - X_mean) / X_std

        beta_hat = ridge_fit(X_scaled, y, lam=alpha)

        return {
            "beta_hat": beta_hat,
            "X_mean": X_mean,
            "X_std": X_std,
            "type": "ridge",
        }

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate a trained model on testing data."""
        if isinstance(model, OLSBaseline):
            return model.evaluate(X_test, y_test)

        if isinstance(model, dict) and model.get("type") == "ridge":
            X = np.asarray(X_test, dtype=float)

            X_mean = model["X_mean"]
            X_std = model["X_std"]
            X_scaled = (X - X_mean) / X_std

            X_aug = np.column_stack([np.ones(len(X_scaled)), X_scaled])
            y_pred = X_aug @ model["beta_hat"]
        else:
            y_pred = model.predict(X_test)

        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2_test": r2_score(y_test, y_pred),
        }

    def cv_select_alpha(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        alphas=None,
        k: int = 5,
    ) -> tuple:
        """Select the optimal alpha via k-fold cross-validation with fold-level scaling."""
        if alphas is None:
            alphas = np.logspace(-3, 3, 100)

        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
        n_samples = X_arr.shape[0]

        if k < 2 or k > n_samples:
            raise ValueError(f"k must be between 2 and {n_samples}, got {k}")

        indices = np.arange(n_samples)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        folds = np.array_split(indices, k)

        mean_mses = []

        for alpha in alphas:
            fold_mses = []

            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

                X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
                y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

                X_tr_mean = np.mean(X_tr, axis=0)
                X_tr_std = np.std(X_tr, axis=0) + 1e-8
                X_tr_scaled = (X_tr - X_tr_mean) / X_tr_std

                X_val_scaled = (X_val - X_tr_mean) / X_tr_std

                beta = ridge_fit(X_tr_scaled, y_tr, lam=alpha)
                X_val_aug = np.column_stack([np.ones(len(X_val_scaled)), X_val_scaled])
                y_pred = X_val_aug @ beta

                fold_mses.append(mean_squared_error(y_val, y_pred))

            mean_mses.append(np.mean(fold_mses))

        mean_mses = np.array(mean_mses)
        best_alpha = alphas[np.argmin(mean_mses)]
        return best_alpha, mean_mses

    def plot_cv_error_curve(
        self,
        alphas,
        mse_values,
        model_name="Ridge",
    ):
        """Plot cross-validation error curve."""
        plt.figure(figsize=(8, 5))
        plt.plot(alphas, mse_values, marker="o")
        plt.xscale("log")
        plt.xlabel("Lambda (α)")
        plt.ylabel("Cross-Validation MSE")
        plt.title(f"{model_name} CV Error Curve")
        plt.grid(True)
        plt.show()

    def compare_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """Train and evaluate OLS and Ridge models and compile their metrics."""
        alphas = np.logspace(-3, 3, 100)
        results = {}

        ols = self.train_linear_regression(X_train, y_train)
        results["OLS"] = self.evaluate_model(ols, X_test, y_test)

        best_ridge_alpha, _ = self.cv_select_alpha(X_train, y_train, alphas)
        ridge = self.train_ridge_regression(X_train, y_train, alpha=best_ridge_alpha)
        results[f"Ridge (λ={best_ridge_alpha:.4f})"] = self.evaluate_model(
            ridge, X_test, y_test
        )

        return pd.DataFrame(results).T

    def select_features_by_vif(
        self,
        X_train: pd.DataFrame,
        threshold: float = 10.0,
        iterative: bool = True,
    ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        """Select features by removing predictors with high VIF."""
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        selected_features = list(X_train.columns)
        removed_features = []

        while len(selected_features) > 1:
            current_X = X_train[selected_features]
            current_vif = vif(current_X.to_numpy(dtype=float))
            max_idx = int(np.argmax(current_vif))
            max_vif = current_vif[max_idx]

            if not np.isfinite(max_vif) or max_vif > threshold:
                feature_to_remove = selected_features[max_idx]
                removed_features.append(
                    {
                        "feature": feature_to_remove,
                        "VIF": max_vif,
                        "removed": True,
                    }
                )
                selected_features.pop(max_idx)

                if iterative:
                    continue

            break

        final_vif_values = vif(X_train[selected_features].to_numpy(dtype=float))
        final_vif_table = pd.DataFrame(
            {
                "feature": selected_features,
                "VIF": final_vif_values,
                "High Multicollinearity": final_vif_values > threshold,
                "removed": False,
            }
        )

        if removed_features:
            removed_table = pd.DataFrame(removed_features)
            removed_table["High Multicollinearity"] = True
            final_vif_table = pd.concat(
                [final_vif_table, removed_table],
                ignore_index=True,
            )

        final_vif_table = final_vif_table.sort_values(
            ["removed", "VIF"],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

        return X_train[selected_features].copy(), selected_features, final_vif_table

    def compare_metrics_with_variable_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        vif_threshold: float = 10.0,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select variables with VIF filtering, then compare OLS and Ridge models."""
        X_train_selected, selected_features, vif_table = self.select_features_by_vif(
            X_train,
            threshold=vif_threshold,
            iterative=True,
        )
        if isinstance(X_test, pd.DataFrame):
            X_test_selected = X_test[selected_features].copy()
        else:
            X_test_arr = np.asarray(X_test, dtype=float)
            selected_indices = [
                (
                    X_train.columns.get_loc(feature)
                    if isinstance(X_train, pd.DataFrame)
                    else int(feature)
                )
                for feature in selected_features
            ]
            X_test_selected = X_test_arr[:, selected_indices]

        metrics_df = self.compare_metrics(
            X_train_selected,
            y_train,
            X_test_selected,
            y_test,
        )

        selection_info = {
            "X_train_selected": X_train_selected,
            "X_test_selected": X_test_selected,
            "feature_names_selected": selected_features,
            "vif_table": vif_table,
            "removed_features": vif_table.loc[vif_table["removed"], "feature"].tolist(),
        }

        return metrics_df, selection_info

    def generate_summary(self, metrics_df: pd.DataFrame) -> str:
        """Generate a text summary of the model comparison results."""
        best_rmse = metrics_df["RMSE"].idxmin()
        best_r2 = metrics_df["R2_test"].idxmax()
        best_mae = metrics_df["MAE"].idxmin()

        lines = [
            "=" * 50,
            "         MODEL COMPARISON SUMMARY",
            "=" * 50,
            f"  Best RMSE : {best_rmse} ({metrics_df.loc[best_rmse, 'RMSE']:.4f})",
            f"  Best R²   : {best_r2} ({metrics_df.loc[best_r2, 'R2_test']:.4f})",
            f"  Best MAE  : {best_mae} ({metrics_df.loc[best_mae, 'MAE']:.4f})",
            "=" * 50,
        ]
        return "\n".join(lines)


class OLSBaseline:
    """OLS baseline model using custom implementation from Part 1."""

    def __init__(self, random_state=42):
        self.random_state = random_state
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
        """Fit OLS using custom implementation from Part 1."""
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        self.X_train_raw = X_train
        self.y_train = y_train

        n = X_train.shape[0]
        self.X_train = np.column_stack([np.ones(n), X_train])

        self.beta_hat, self.sigma2_hat = ols_fit(X_train, y_train)
        self.y_train_pred = self.X_train @ self.beta_hat
        self.residuals = y_train - self.y_train_pred
        return self

    def predict(self, X):
        """Predict using fitted OLS model."""
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        X_aug = np.column_stack([np.ones(n), X])
        return X_aug @ self.beta_hat

    def evaluate(self, X_test, y_test):
        """Evaluate on held-out test set."""
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float)

        self.y_test_pred = self.predict(X_test)

        self.metrics = {
            "MAE": mean_absolute_error(y_test, self.y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, self.y_test_pred)),
            "R2_test": r2_score(y_test, self.y_test_pred),
        }
        return self.metrics

    def run_inference(self, feature_names):
        """Compute coefficient inference statistics."""
        feature_names = ["Intercept"] + list(feature_names)

        result = coef_inference(
            self.X_train,
            self.y_train,
            self.beta_hat,
            self.sigma2_hat,
        )

        self.inference_table = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": self.beta_hat,
                "std_error": result["standard_errors"],
                "t_stat": result["t_statistics"],
                "p_value": result["p_values"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
            }
        )
        return self.inference_table

    def compute_vif(self, X, feature_names):
        """Compute Variance Inflation Factors."""
        vif_values = vif(X)
        vif_scores = []

        for feature, vif_value in zip(feature_names, vif_values):
            vif_scores.append(
                {
                    "feature": feature,
                    "VIF": vif_value,
                    "High Multicollinearity": (vif_value > 10),
                }
            )

        self.vif_table = pd.DataFrame(vif_scores)
        return self.vif_table

    def diagnostic_plots(self):
        """Generate residual diagnostic plots."""
        return residual_plots(
            self.X_train,
            self.y_train,
            self.beta_hat,
        )


class OLSWithVariables(OLSBaseline):
    """OLS baseline with VIF-based variable selection before fitting."""

    def __init__(self, vif_threshold=10.0, random_state=42):
        super().__init__(random_state=random_state)
        self.vif_threshold = vif_threshold
        self.feature_names_selected = None
        self.feature_indices_selected = None
        self.removed_features = None
        self.selection_vif_table = None

    def select_features(self, X_train, feature_names=None):
        """Select variables by iteratively removing the largest VIF above threshold."""
        if isinstance(X_train, pd.DataFrame):
            X_df = X_train.copy()
        else:
            X_df = pd.DataFrame(X_train, columns=feature_names)

        comparison = ModelComparison()
        X_selected, selected_features, vif_table = comparison.select_features_by_vif(
            X_df,
            threshold=self.vif_threshold,
            iterative=True,
        )

        self.feature_names_selected = selected_features
        self.feature_indices_selected = [
            X_df.columns.get_loc(feature) for feature in selected_features
        ]
        self.removed_features = vif_table.loc[vif_table["removed"], "feature"].tolist()
        self.selection_vif_table = vif_table

        return X_selected, selected_features, vif_table

    def fit(self, X_train, y_train, feature_names=None):
        """Select variables by VIF, then fit using the inherited OLSBaseline logic."""
        X_selected, _, _ = self.select_features(X_train, feature_names)
        return super().fit(X_selected, y_train)

    def transform(self, X):
        """Keep the selected features from new data."""
        if self.feature_names_selected is None:
            raise ValueError("Model must be fitted before calling transform.")

        if isinstance(X, pd.DataFrame):
            return X[self.feature_names_selected].copy()

        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.feature_indices_selected]

    def predict(self, X):
        """Predict using selected variables only."""
        return super().predict(self.transform(X))
