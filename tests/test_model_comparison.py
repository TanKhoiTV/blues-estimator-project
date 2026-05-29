"""
Unit tests for the ModelComparison module.

Tests both OLSBaseline (fit, predict, evaluate, inference) and
ModelComparison (trainers, CV, VIF selection, metrics table).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from part2.model_comparison import ModelComparison, OLSBaseline, OLSWithVariables


class TestOLSBaseline(unittest.TestCase):
    """Tests for OLSBaseline: fitting, prediction, evaluation, and inference."""

    def setUp(self):
        """Create a simple synthetic dataset with known coefficients."""
        np.random.seed(42)
        n = 200

        self.true_beta = np.array([5.0, -2.5, 1.2, 0.0])  # intercept, x1, x2, x3 (zero)

        # Independent features for recovery tests
        X_raw = np.random.randn(n, 3)

        self.X_train = pd.DataFrame(X_raw[:150], columns=["x1", "x2", "x3"])
        self.y_train = pd.Series(
            self.true_beta[0]
            + self.X_train["x1"] * self.true_beta[1]
            + self.X_train["x2"] * self.true_beta[2]
            + self.X_train["x3"] * self.true_beta[3]
            + 0.2 * np.random.randn(150)
        )

        self.X_test = pd.DataFrame(X_raw[150:], columns=["x1", "x2", "x3"])
        self.y_test = pd.Series(
            self.true_beta[0]
            + self.X_test["x1"] * self.true_beta[1]
            + self.X_test["x2"] * self.true_beta[2]
            + self.X_test["x3"] * self.true_beta[3]
            + 0.2 * np.random.randn(50)
        )

    # ── Fit ──────────────────────────────────────────────────────

    def test_ols_fit_recovers_parameters(self):
        """OLSBaseline.fit() returns beta_hat close to true coefficients."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)

        # Coefficient recovery (allow some noise)
        np.testing.assert_allclose(model.beta_hat[0], self.true_beta[0], atol=0.3)
        np.testing.assert_allclose(model.beta_hat[1], self.true_beta[1], atol=0.3)
        np.testing.assert_allclose(model.beta_hat[2], self.true_beta[2], atol=0.3)

    def test_ols_fit_stores_residuals(self):
        """OLSBaseline.fit() stores residuals with correct shape."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)

        self.assertIsNotNone(model.residuals)
        self.assertEqual(model.residuals.shape, (150,))

    def test_ols_fit_sklearn_parity(self):
        """OLSBaseline coefficients match sklearn LinearRegression."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)

        sk = LinearRegression()
        sk.fit(self.X_train, self.y_train)

        sk_coefs = np.concatenate([[sk.intercept_], sk.coef_])
        np.testing.assert_allclose(model.beta_hat, sk_coefs, atol=1e-6)

    # ── Predict ──────────────────────────────────────────────────

    def test_ols_predict_shape_and_values(self):
        """OLSBaseline.predict() returns correct shape and sensible values."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        self.assertEqual(y_pred.shape, (50,))
        # Predictions should correlate well with true test values
        corr = np.corrcoef(y_pred, self.y_test)[0, 1]
        self.assertGreater(corr, 0.95)

    # ── Evaluate ─────────────────────────────────────────────────

    def test_ols_evaluate_returns_metrics(self):
        """OLSBaseline.evaluate() returns MAE, RMSE, R2_test."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)
        metrics = model.evaluate(self.X_test, self.y_test)

        self.assertIn("MAE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertIn("R2_test", metrics)
        self.assertGreater(metrics["R2_test"], 0.9)

    def test_ols_evaluate_perfect_fit(self):
        """OLSBaseline.evaluate() gives R²=1 for a perfect linear fit."""
        model = OLSBaseline()

        # Construct perfect data: y = 3 + 2*x
        X_perfect = pd.DataFrame({"x": np.arange(10)})
        y_perfect = pd.Series(3 + 2 * np.arange(10))
        model.fit(X_perfect, y_perfect)
        metrics = model.evaluate(X_perfect, y_perfect)

        self.assertAlmostEqual(metrics["R2_test"], 1.0, places=10)
        self.assertAlmostEqual(metrics["RMSE"], 0.0, places=10)

    # ── Inference ────────────────────────────────────────────────

    def test_ols_run_inference_returns_table(self):
        """OLSBaseline.run_inference() returns DataFrame with expected columns."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)
        table = model.run_inference(list(self.X_train.columns))

        expected_cols = [
            "feature",
            "coefficient",
            "std_error",
            "t_stat",
            "p_value",
            "ci_lower",
            "ci_upper",
        ]
        for col in expected_cols:
            self.assertIn(col, table.columns)

        self.assertEqual(len(table), 4)  # intercept + 3 features

    def test_ols_run_inference_significant_features(self):
        """Significant features (beta != 0) have low p-values."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)
        table = model.run_inference(list(self.X_train.columns))

        # x1 (true beta=-2.5) should be highly significant
        p_x1 = table.loc[table["feature"] == "x1", "p_value"].values[0]
        self.assertLess(p_x1, 0.01)

        # Intercept (true beta=5.0) should be highly significant
        p_intercept = table.loc[table["feature"] == "Intercept", "p_value"].values[0]
        self.assertLess(p_intercept, 0.01)

    # ── VIF ──────────────────────────────────────────────────────

    def test_ols_compute_vif_returns_table(self):
        """OLSBaseline.compute_vif() returns correct VIF table."""
        model = OLSBaseline()
        model.fit(self.X_train, self.y_train)
        vif_df = model.compute_vif(self.X_train.values, list(self.X_train.columns))

        self.assertIn("feature", vif_df.columns)
        self.assertIn("VIF", vif_df.columns)
        self.assertEqual(len(vif_df), 3)  # 3 features
        # Independent features -> VIF close to 1
        self.assertAlmostEqual(vif_df["VIF"].mean(), 1.0, delta=0.3)


class TestModelComparison(unittest.TestCase):
    """Tests for ModelComparison: trainers, CV, VIF selection, metrics."""

    def setUp(self):
        """Synthetic dataset for model comparison."""
        np.random.seed(123)
        n = 100
        X_raw = np.random.randn(n, 4)
        coefs = np.array([1.5, -0.8, 0.0, 2.3])  # x3 has zero effect
        y_raw = 3.0 + X_raw @ coefs + 0.2 * np.random.randn(n)

        self.X_train = pd.DataFrame(X_raw[:70], columns=["a", "b", "c", "d"])
        self.y_train = pd.Series(y_raw[:70])
        self.X_test = pd.DataFrame(X_raw[70:], columns=["a", "b", "c", "d"])
        self.y_test = pd.Series(y_raw[70:])

    # ── Trainers ─────────────────────────────────────────────────

    def test_train_linear_regression_returns_ols(self):
        """train_linear_regression() returns OLSBaseline with fitted attributes."""
        comp = ModelComparison()
        model = comp.train_linear_regression(self.X_train, self.y_train)

        self.assertIsInstance(model, OLSBaseline)
        self.assertIsNotNone(model.beta_hat)

    def test_train_ridge_sklearn_parity(self):
        """Ridge with small alpha approximates OLS."""
        comp = ModelComparison()
        ols = comp.train_linear_regression(self.X_train, self.y_train)
        ridge = comp.train_ridge_regression(self.X_train, self.y_train, alpha=1e-6)

        # Both are dicts with beta_hat
        self.assertIn("beta_hat", ridge)
        np.testing.assert_allclose(ridge["beta_hat"], ols.beta_hat, atol=0.1)

    def test_train_ridge_large_lambda_shrinkage(self):
        """Ridge with large lambda shrinks coefficients toward zero."""
        comp = ModelComparison()
        ridge = comp.train_ridge_regression(self.X_train, self.y_train, alpha=1e6)

        # All non-intercept coefficients should be near zero
        self.assertLess(np.max(np.abs(ridge["beta_hat"][1:])), 0.1)

    def test_train_lasso_returns_sklearn_model(self):
        """train_lasso_regression() returns a fitted sklearn Lasso."""
        comp = ModelComparison()
        # On this branch, Lasso returns the sklearn model directly
        model = comp.train_lasso_regression(self.X_train, self.y_train, alpha=0.1)

        # Check it's an sklearn Lasso with fitted coef_
        self.assertTrue(hasattr(model, "coef_"))
        self.assertEqual(len(model.coef_), 4)

    def test_train_lasso_shrinks_weak_features(self):
        """Lasso with large alpha shrinks weak features to exactly zero."""
        comp = ModelComparison()
        model = comp.train_lasso_regression(self.X_train, self.y_train, alpha=10.0)

        # Feature 'c' has true coefficient 0.0 — should be zero
        self.assertEqual(model.coef_[2], 0.0)

    def test_train_elasticnet_returns_sklearn_model(self):
        """train_elasticnet_regression() returns a fitted sklearn ElasticNet."""
        comp = ModelComparison()
        model = comp.train_elasticnet_regression(self.X_train, self.y_train, alpha=0.1)

        self.assertTrue(hasattr(model, "coef_"))
        self.assertEqual(len(model.coef_), 4)

    # ── Compare metrics ──────────────────────────────────────────

    def test_compare_metrics_returns_dataframe(self):
        """compare_metrics() returns a DataFrame with MAE, RMSE, R2_test."""
        comp = ModelComparison()
        df = comp.compare_metrics(self.X_train, self.y_train, self.X_test, self.y_test)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("MAE", df.columns)
        self.assertIn("RMSE", df.columns)
        self.assertIn("R2_test", df.columns)

    def test_compare_metrics_includes_all_models(self):
        """compare_metrics() contains OLS, Ridge, Lasso, ElasticNet."""
        comp = ModelComparison()
        df = comp.compare_metrics(self.X_train, self.y_train, self.X_test, self.y_test)

        model_names = df.index.tolist()
        self.assertTrue(any("OLS" in n for n in model_names))
        self.assertTrue(any("Ridge" in n for n in model_names))
        self.assertTrue(any("Lasso" in n for n in model_names))
        self.assertTrue(any("Elastic" in n for n in model_names))

    # ── CV ───────────────────────────────────────────────────────

    def test_cv_select_alpha_best_alpha_finite(self):
        """cv_select_alpha() returns a finite positive best alpha."""
        comp = ModelComparison()
        best_alpha, cv_mse = comp.cv_select_alpha(
            self.X_train, self.y_train, alphas=np.logspace(-2, 2, 10), k=5
        )

        self.assertGreater(best_alpha, 0)
        self.assertTrue(np.isfinite(best_alpha))
        self.assertEqual(len(cv_mse), 10)

    def test_cv_select_alpha_invalid_k_raises_error(self):
        """cv_select_alpha() raises ValueError for invalid k."""
        comp = ModelComparison()
        with self.assertRaises(ValueError):
            comp.cv_select_alpha(self.X_train, self.y_train, k=1)

    # ── VIF selection ────────────────────────────────────────────

    def test_select_features_by_vif_drops_high_vif(self):
        """select_features_by_vif() removes features with VIF > threshold."""
        # Create data with extreme collinearity
        np.random.seed(7)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 + 0.01 * np.random.randn(n)  # almost perfectly correlated
        x3 = np.random.randn(n)

        X_collinear = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        y = pd.Series(1 + x1 * 2 + x3 * 3 + 0.1 * np.random.randn(n))

        comp = ModelComparison()
        X_sel, features, vif_table = comp.select_features_by_vif(
            X_collinear, threshold=5.0
        )

        # x1 or x2 should be removed (they are highly correlated)
        removed = vif_table.loc[vif_table["removed"], "feature"].tolist()
        self.assertGreater(len(removed), 0)

        # The remaining features should have lower VIF
        remaining = vif_table.loc[~vif_table["removed"], "feature"].tolist()
        for feat in remaining:
            row = vif_table.loc[vif_table["feature"] == feat]
            self.assertLess(row["VIF"].values[0], 5.0)

    def test_select_features_by_vif_returns_shapes(self):
        """select_features_by_vif() returns correct shapes."""
        comp = ModelComparison()
        X_sel, features, vif_table = comp.select_features_by_vif(
            self.X_train, threshold=10.0
        )

        self.assertEqual(X_sel.shape[0], self.X_train.shape[0])
        self.assertEqual(len(features), X_sel.shape[1])
        self.assertIn("removed", vif_table.columns)

    # ── Compare metrics with VIF selection ───────────────────────

    def test_compare_metrics_with_variable_selection(self):
        """compare_metrics_with_variable_selection() returns (df, dict) tuple."""
        comp = ModelComparison()
        metrics_df, selection_info = comp.compare_metrics_with_variable_selection(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertIn("feature_names_selected", selection_info)
        self.assertIn("vif_table", selection_info)
        self.assertIn("removed_features", selection_info)

    # ── Generate summary ─────────────────────────────────────────

    def test_generate_summary_string(self):
        """generate_summary() returns a non-empty string with best model info."""
        comp = ModelComparison()
        df = comp.compare_metrics(self.X_train, self.y_train, self.X_test, self.y_test)
        summary = comp.generate_summary(df)

        self.assertIsInstance(summary, str)
        self.assertIn("Best RMSE", summary)
        self.assertIn("Best R²", summary)
        self.assertIn("Best MAE", summary)


class TestOLSWithVariables(unittest.TestCase):
    """Tests for OLSWithVariables: VIF-based selection before fitting."""

    def setUp(self):
        """Set up data with collinear features for selection tests."""
        np.random.seed(99)
        n = 50
        x1 = np.random.randn(n)
        x2 = x1 + 0.02 * np.random.randn(n)  # highly correlated with x1
        x3 = np.random.randn(n)

        self.X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        self.y = pd.Series(10 + x1 * 2 + x3 * 5 + 0.5 * np.random.randn(n))

    def test_select_features_removes_collinear(self):
        """select_features() removes correlated features before fitting."""
        model = OLSWithVariables(vif_threshold=5.0)
        X_sel, features, vif_tbl = model.select_features(self.X)

        # Should have removed either x1 or x2
        self.assertLess(len(features), 3)
        self.assertGreater(len(model.removed_features), 0)

    def test_transform_selects_same_features(self):
        """transform() keeps only the features selected during fit()."""
        model = OLSWithVariables(vif_threshold=5.0)
        model.fit(self.X, self.y, feature_names=list(self.X.columns))

        new_data = pd.DataFrame({"x1": [0.5], "x2": [0.6], "x3": [-0.3]})
        transformed = model.transform(new_data)

        self.assertEqual(list(transformed.columns), model.feature_names_selected)

    def test_predict_uses_selected_features(self):
        """predict() uses only VIF-selected features for prediction."""
        model = OLSWithVariables(vif_threshold=5.0)
        model.fit(self.X, self.y, feature_names=list(self.X.columns))

        # Predict should work without error
        y_pred = model.predict(self.X)
        self.assertEqual(y_pred.shape, (50,))
