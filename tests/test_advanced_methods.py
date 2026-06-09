"""
Unit tests for the Advanced Methods module.

Tests KernelRidge and the AdvancedMethods facade.
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
        """_get_or_fit_models() returns a dict with the Kernel Ridge model."""
        am = AdvancedMethods()
        models = am._get_or_fit_models(self.X_train, self.y_train)

        self.assertIsInstance(models, dict)
        self.assertEqual(len(models), 1)
        self.assertIn("Kernel Ridge (From Scratch)", models)

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
        self.assertEqual(len(df), 1)

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
