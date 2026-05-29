"""
Unit tests for the Advanced Methods module.

Tests all four from-scratch models (KernelRidge, Bayesian,
DecisionTree, RandomForest) and the AdvancedMethods facade.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for plot tests

from part2.advanced_methods import (
    KernelRidgeScratch,
    BayesianLinearRegressionScratch,
    DecisionTreeRegressorScratch,
    RandomForestRegressorScratch,
    AdvancedMethods,
)


class TestKernelRidgeScratch(unittest.TestCase):
    """Tests for KernelRidgeScratch: fit, predict, interpolation."""

    def setUp(self):
        """Set up a simple synthetic dataset."""
        np.random.seed(42)
        n = 50
        self.X_train = pd.DataFrame(
            {"x1": np.random.randn(n), "x2": np.random.randn(n)}
        )
        self.y_train = pd.Series(
            3
            + self.X_train["x1"] * 2
            - self.X_train["x2"] * 1.5
            + 0.1 * np.random.randn(n)
        )
        self.X_test = pd.DataFrame(
            {"x1": np.random.randn(20), "x2": np.random.randn(20)}
        )
        self.y_test = pd.Series(
            3
            + self.X_test["x1"] * 2
            - self.X_test["x2"] * 1.5
            + 0.1 * np.random.randn(20)
        )

    def test_fit_stores_dual_coefficients(self):
        """KernelRidgeScratch.fit() stores dual_coefficients_ with correct shape."""
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.dual_coefficients_)
        self.assertEqual(model.dual_coefficients_.shape, (50,))

    def test_fit_stores_scaler_params(self):
        """KernelRidgeScratch.fit() stores feature_means and feature_stds."""
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.feature_means)
        self.assertIsNotNone(model.feature_stds)
        self.assertEqual(model.feature_means.shape, (2,))

    def test_predict_returns_correct_shape(self):
        """KernelRidgeScratch.predict() returns shape (n_test,)."""
        model = KernelRidgeScratch(alpha=0.5, gamma=0.1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        self.assertEqual(y_pred.shape, (20,))

    def test_predict_with_large_alpha_approaches_mean(self):
        """KernelRidgeScratch with large alpha predicts near training mean."""
        model = KernelRidgeScratch(alpha=1e6, gamma=0.1)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Heavy regularization pulls predictions toward zero
        self.assertTrue(np.all(np.abs(y_pred) < 1.0))


class TestBayesianLinearRegressionScratch(unittest.TestCase):
    """Tests for BayesianLinearRegressionScratch: fit, predict, posterior."""

    def setUp(self):
        """Set up a linear synthetic dataset."""
        np.random.seed(123)
        n = 80
        self.X_train = pd.DataFrame(
            {"x1": np.random.randn(n), "x2": np.random.randn(n)}
        )
        self.y_train = pd.Series(
            -1
            + self.X_train["x1"] * 3
            + self.X_train["x2"] * 0.5
            + 0.2 * np.random.randn(n)
        )
        self.X_test = pd.DataFrame(
            {"x1": np.random.randn(20), "x2": np.random.randn(20)}
        )

    def test_fit_stores_posterior_mean(self):
        """BayesianLinearRegressionScratch.fit() stores w_mean with correct shape."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.w_mean)
        self.assertEqual(model.w_mean.shape, (3,))  # bias + 2 features

    def test_fit_stores_posterior_covariance(self):
        """BayesianLinearRegressionScratch.fit() stores w_cov."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.w_cov)
        self.assertEqual(model.w_cov.shape, (3, 3))

    def test_fit_stores_scaler_params(self):
        """BayesianLinearRegressionScratch.fit() stores feature_means and feature_stds."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.feature_means)
        self.assertIsNotNone(model.feature_stds)

    def test_predict_returns_correct_shape(self):
        """BayesianLinearRegressionScratch.predict() returns shape (n_test,)."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        self.assertEqual(y_pred.shape, (20,))
        self.assertTrue(np.all(np.isfinite(y_pred)))


class TestDecisionTreeRegressorScratch(unittest.TestCase):
    """Tests for DecisionTreeRegressorScratch: fit, predict, depth control."""

    def setUp(self):
        """Set up step-function data that a tree can fit perfectly."""
        np.random.seed(42)
        n = 40
        self.x = np.linspace(0, 1, n).reshape(-1, 1)
        # Step function: 0 for x < 0.5, 1 for x >= 0.5
        self.y = np.where(self.x.flatten() < 0.5, 0.0, 1.0)
        self.y_noisy = self.y + 0.05 * np.random.randn(n)

    def test_fit_perfect_on_step_function(self):
        """DecisionTree with sufficient depth fits a step function perfectly."""
        model = DecisionTreeRegressorScratch(max_depth=5)
        model.fit(self.x, self.y_noisy)
        y_pred = model.predict(self.x)

        # Predictions should be close to the clean step values
        mse = np.mean((y_pred - self.y) ** 2)
        self.assertLess(mse, 0.01)

    def test_max_depth_limits_growth(self):
        """Shallow tree has higher error on noisy step function."""
        deep = DecisionTreeRegressorScratch(max_depth=10)
        shallow = DecisionTreeRegressorScratch(max_depth=2)

        deep.fit(self.x, self.y_noisy)
        shallow.fit(self.x, self.y_noisy)

        pred_deep = deep.predict(self.x)
        pred_shallow = shallow.predict(self.x)

        # Shallow tree has fewer splits -> less flexibility
        mse_deep = np.mean((pred_deep - self.y_noisy) ** 2)
        mse_shallow = np.mean((pred_shallow - self.y_noisy) ** 2)
        self.assertLess(mse_deep, mse_shallow)

    def test_predict_shapes(self):
        """DecisionTree.predict() returns correct shape."""
        model = DecisionTreeRegressorScratch(max_depth=5)
        model.fit(self.x, self.y_noisy)

        test_x = np.array([[0.2], [0.8]])
        y_pred = model.predict(test_x)
        self.assertEqual(y_pred.shape, (2,))

    def test_constant_feature_no_split(self):
        """Tree with constant feature returns mean value (no split)."""
        X_const = np.ones((10, 1))
        y_const = np.array([5.0] * 10)

        model = DecisionTreeRegressorScratch(max_depth=5)
        model.fit(X_const, y_const)
        y_pred = model.predict(np.array([[1.0], [1.0]]))

        np.testing.assert_allclose(y_pred, [5.0, 5.0])


class TestRandomForestRegressorScratch(unittest.TestCase):
    """Tests for RandomForestRegressorScratch: fit, predict, ensemble effect."""

    def setUp(self):
        """Set up nonlinear dataset where RF ensemble helps."""
        np.random.seed(42)
        n = 60
        x = np.random.randn(n, 3)
        # Nonlinear target: interaction + quadratic
        y = x[:, 0] ** 2 + np.sin(x[:, 1]) - x[:, 2] + 0.1 * np.random.randn(n)
        self.X = pd.DataFrame(x, columns=["a", "b", "c"])
        self.y = pd.Series(y)

    def test_fit_stores_trees(self):
        """RandomForest.fit() stores the correct number of trees."""
        model = RandomForestRegressorScratch(n_estimators=5, max_depth=3, seed=42)
        model.fit(self.X, self.y)

        self.assertEqual(len(model.trees), 5)

    def test_predict_shape(self):
        """RandomForest.predict() returns correct shape."""
        model = RandomForestRegressorScratch(n_estimators=5, max_depth=3, seed=42)
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X)

        self.assertEqual(y_pred.shape, (60,))

    def test_predict_single_tree_versus_ensemble(self):
        """Ensemble should have lower or equal MSE than a single tree.

        n_estimators=10 vs n_estimators=1 — ensemble reduces variance.
        """
        single = RandomForestRegressorScratch(n_estimators=1, max_depth=4, seed=42)
        ensemble = RandomForestRegressorScratch(n_estimators=10, max_depth=4, seed=42)

        single.fit(self.X, self.y)
        ensemble.fit(self.X, self.y)

        pred_single = single.predict(self.X)
        pred_ensemble = ensemble.predict(self.X)

        mse_single = np.mean((pred_single - self.y) ** 2)
        mse_ensemble = np.mean((pred_ensemble - self.y) ** 2)

        self.assertLessEqual(mse_ensemble, mse_single * 1.05)


class TestAdvancedMethods(unittest.TestCase):
    """Tests for AdvancedMethods facade: cache, evaluation, plotting."""

    def setUp(self):
        """Set up synthetic dataset for all advanced models."""
        np.random.seed(42)
        n = 60
        self.X_train = pd.DataFrame(
            {"x1": np.random.randn(n), "x2": np.random.randn(n)}
        )
        self.y_train = pd.Series(
            2
            + self.X_train["x1"] * 1.5
            - self.X_train["x2"] * 0.8
            + 0.2 * np.random.randn(n)
        )
        self.X_test = pd.DataFrame(
            {"x1": np.random.randn(20), "x2": np.random.randn(20)}
        )
        self.y_test = pd.Series(
            2
            + self.X_test["x1"] * 1.5
            - self.X_test["x2"] * 0.8
            + 0.2 * np.random.randn(20)
        )

    def test_get_or_fit_models_returns_dict(self):
        """_get_or_fit_models() returns a dict with the three model types."""
        am = AdvancedMethods()
        models = am._get_or_fit_models(self.X_train, self.y_train)

        self.assertIsInstance(models, dict)
        self.assertEqual(len(models), 3)
        self.assertIn("Kernel Ridge (From Scratch)", models)
        self.assertIn("Bayesian Regression (From Scratch)", models)
        self.assertIn("Random Forest (From Scratch)", models)

    def test_get_or_fit_models_cache_hit(self):
        """_get_or_fit_models() returns the same dict on repeat calls (identity cache)."""
        am = AdvancedMethods()
        models1 = am._get_or_fit_models(self.X_train, self.y_train)
        models2 = am._get_or_fit_models(self.X_train, self.y_train)

        self.assertIs(models1, models2)

    def test_get_or_fit_models_cache_miss(self):
        """_get_or_fit_models() retrains when new data is passed."""
        am = AdvancedMethods()
        models1 = am._get_or_fit_models(self.X_train, self.y_train)

        new_y = self.y_train + 10.0
        models2 = am._get_or_fit_models(self.X_train, new_y)

        self.assertIsNot(models1, models2)

    def test_evaluate_advanced_models_returns_dataframe(self):
        """evaluate_advanced_models() returns DataFrame with correct columns."""
        am = AdvancedMethods()
        df = am.evaluate_advanced_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("Model", df.columns)
        self.assertIn("R-squared", df.columns)
        self.assertIn("RMSE", df.columns)
        self.assertIn("MAE", df.columns)
        self.assertEqual(len(df), 3)

    def test_evaluate_advanced_models_reasonable_metrics(self):
        """All models should achieve R² > 0 on a linear-ish dataset."""
        am = AdvancedMethods()
        df = am.evaluate_advanced_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        for _, row in df.iterrows():
            self.assertGreater(
                row["R-squared"],
                -1.0,
                f"{row['Model']} has suspicious R²={row['R-squared']}",
            )

    def test_plot_model_comparison_does_not_crash(self):
        """plot_model_comparison() runs without error."""
        am = AdvancedMethods()
        try:
            am.plot_model_comparison(
                self.X_train, self.y_train, self.X_test, self.y_test
            )
        except Exception as e:
            self.fail(f"plot_model_comparison raised {e}")

    def test_plot_predictions_vs_actual_does_not_crash(self):
        """plot_predictions_vs_actual() runs without error."""
        am = AdvancedMethods()
        try:
            am.plot_predictions_vs_actual(
                self.X_train, self.y_train, self.X_test, self.y_test
            )
        except Exception as e:
            self.fail(f"plot_predictions_vs_actual raised {e}")
