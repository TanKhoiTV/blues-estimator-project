"""Tests for archived out-of-scope models.

Corresponding tests for the archived out-of-scope models. These tests
were originally part of test_model_comparison.py and
test_advanced_methods.py.

They are preserved here to verify that the archived code remains functional.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd

from part2.model_comparison import ModelComparison
from part2.archive.lasso_elasticnet_models import (
    train_lasso_regression,
    train_elasticnet_regression,
)
from part2.archive.bayesian_random_forest_models import (
    BayesianLinearRegressionScratch,
    DecisionTreeRegressorScratch,
    RandomForestRegressorScratch,
)


class TestArchivedLassoElasticNet(unittest.TestCase):
    """Tests for archived Lasso and ElasticNet implementations."""

    def setUp(self):
        np.random.seed(123)
        n = 100
        X_raw = np.random.randn(n, 4)
        coefs = np.array([1.5, -0.8, 0.0, 2.3])
        y_raw = 3.0 + X_raw @ coefs + 0.2 * np.random.randn(n)
        self.X_train = pd.DataFrame(X_raw[:70], columns=["a", "b", "c", "d"])
        self.y_train = pd.Series(y_raw[:70])

    def test_lasso_returns_sklearn_model(self):
        """train_lasso_regression() returns a dict with an sklearn Lasso."""
        result = train_lasso_regression(self.X_train, self.y_train, alpha=0.1)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "lasso")
        self.assertTrue(hasattr(result["model"], "coef_"))

    def test_lasso_shrinks_weak_features(self):
        """Lasso with large alpha shrinks weak features to exactly zero."""
        result = train_lasso_regression(self.X_train, self.y_train, alpha=10.0)
        self.assertEqual(result["model"].coef_[2], 0.0)

    def test_elasticnet_returns_sklearn_model(self):
        """train_elasticnet_regression() returns a dict with an sklearn ElasticNet."""
        result = train_elasticnet_regression(self.X_train, self.y_train, alpha=0.1)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["type"], "elasticnet")
        self.assertTrue(hasattr(result["model"], "coef_"))


class TestArchivedBayesianRegression(unittest.TestCase):
    """Tests for archived BayesianLinearRegressionScratch."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.X_train = pd.DataFrame(
            {"x1": np.random.randn(n), "x2": np.random.randn(n)}
        )
        self.y_train = pd.Series(
            2.0
            + self.X_train["x1"] * 1.5
            + self.X_train["x2"] * (-0.8)
            + 0.2 * np.random.randn(n)
        )

    def test_fit_stores_posterior_mean(self):
        """fit() stores w_mean with correct shape."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.w_mean)
        self.assertEqual(model.w_mean.shape[0], 2)

    def test_fit_stores_posterior_covariance(self):
        """fit() stores w_cov."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.w_cov)
        self.assertEqual(model.w_cov.shape, (2, 2))

    def test_fit_stores_scaler_params(self):
        """fit() stores feature_means and feature_stds."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)
        self.assertIsNotNone(model.feature_means)
        self.assertIsNotNone(model.feature_stds)
        self.assertEqual(len(model.feature_means), 2)

    def test_predict_returns_correct_shape(self):
        """predict() returns shape (n_test,)."""
        model = BayesianLinearRegressionScratch(alpha=1.0, beta=25.0)
        model.fit(self.X_train, self.y_train)
        X_test = pd.DataFrame({"x1": np.random.randn(20), "x2": np.random.randn(20)})
        preds = model.predict(X_test)
        self.assertEqual(preds.shape, (20,))


class TestArchivedDecisionTree(unittest.TestCase):
    """Tests for archived DecisionTreeRegressorScratch."""

    def setUp(self):
        np.random.seed(42)
        self.X_step = np.array([[1], [2], [3], [4], [5], [6]])
        self.y_step = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_perfect_on_step_function(self):
        """DecisionTree with sufficient depth fits a step function perfectly."""
        model = DecisionTreeRegressorScratch(max_depth=5)
        model.fit(self.X_step, self.y_step)
        preds = model.predict(self.X_step)
        np.testing.assert_array_equal(preds, self.y_step)

    def test_max_depth_limits_growth(self):
        """Shallow tree has higher error than deep tree on non-linear data."""
        np.random.seed(7)
        X = np.random.randn(50, 1)
        y = np.sin(X[:, 0] * 3) + 0.1 * np.random.randn(50)
        deep = DecisionTreeRegressorScratch(max_depth=10)
        shallow = DecisionTreeRegressorScratch(max_depth=2)
        deep.fit(X, y)
        shallow.fit(X, y)
        deep_preds = deep.predict(X)
        shallow_preds = shallow.predict(X)
        deep_mse = np.mean((y - deep_preds) ** 2)
        shallow_mse = np.mean((y - shallow_preds) ** 2)
        self.assertLess(deep_mse, shallow_mse)

    def test_constant_feature_no_split(self):
        """Tree handles constant features without crashing."""
        X = np.column_stack([np.ones(20), np.random.randn(20)])
        y = np.random.randn(20)
        model = DecisionTreeRegressorScratch(max_depth=3)
        model.fit(X, y)  # should not raise

    def test_predict_shapes(self):
        """predict() returns correct shape."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        model = DecisionTreeRegressorScratch(max_depth=5)
        model.fit(X, y)
        preds = model.predict(np.random.randn(5, 2))
        self.assertEqual(preds.shape, (5,))


class TestArchivedRandomForest(unittest.TestCase):
    """Tests for archived RandomForestRegressorScratch."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(50, 3)
        y_true = self.X[:, 0] * 2 + self.X[:, 1] * (-1) + 0.2 * np.random.randn(50)
        self.y = y_true

    def test_fit_stores_trees(self):
        """RandomForest.fit() stores the correct number of trees."""
        model = RandomForestRegressorScratch(n_estimators=5, max_depth=3, seed=42)
        model.fit(self.X, self.y)
        self.assertEqual(len(model.trees), 5)

    def test_predict_shape(self):
        """RandomForest.predict() returns correct shape."""
        model = RandomForestRegressorScratch(n_estimators=5, max_depth=3, seed=42)
        model.fit(self.X, self.y)
        preds = model.predict(np.random.randn(10, 3))
        self.assertEqual(preds.shape, (10,))

    def test_predict_single_tree_versus_ensemble(self):
        """Ensemble with more trees has lower MSE than a single tree."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = X[:, 0] * 2 + X[:, 1] * (-1) + 0.3 * np.random.randn(n)
        single = RandomForestRegressorScratch(n_estimators=1, max_depth=4, seed=42)
        ensemble = RandomForestRegressorScratch(n_estimators=10, max_depth=4, seed=42)
        single.fit(X, y)
        ensemble.fit(X, y)
        single_preds = single.predict(X)
        ensemble_preds = ensemble.predict(X)
        single_mse = np.mean((y - single_preds) ** 2)
        ensemble_mse = np.mean((y - ensemble_preds) ** 2)
        self.assertLess(ensemble_mse, single_mse)


if __name__ == "__main__":
    unittest.main()
