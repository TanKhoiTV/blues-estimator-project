"""Archived Bayesian, Decision Tree, and Random Forest models.

These from-scratch implementations were removed from the main
advanced_methods module because the spec requires only Kernel Ridge
(or Bayesian) as a bonus technique — not both. Kernel Ridge was kept
as the chosen advanced model.

This archive preserves the working, tested code for reference.
"""

from typing import Any, List, Optional

import numpy as np


class BayesianLinearRegressionScratch:
    """Bayesian Linear Regression using Gaussian Priors with Z-score scaling."""

    def __init__(self, alpha: float = 1.0, beta: float = 25.0) -> None:
        """Initialize the Bayesian Regression model with precision parameters.

        Parameters
        ----------
        alpha : float, optional
            Prior precision on coefficients. Controls the strength of the
            Gaussian prior on weights. Larger values shrink coefficients
            more strongly toward zero. Default is 1.0.
        beta : float, optional
            Noise precision (inverse variance). Models the expected noise
            level in the target variable. Default is 25.0.
        """
        self.alpha = alpha
        self.beta = beta
        self.w_mean: Optional[np.ndarray] = None
        self.w_cov: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "BayesianLinearRegressionScratch":
        """Compute the posterior over weights given training data.

        Uses the closed-form Bayesian update for linear regression with
        a Gaussian prior (conjugate prior):  posterior N(w_mean, w_cov).

        ``w_cov = (alpha * I + beta * X^T X)^{-1}``
        ``w_mean = beta * w_cov * X^T y``

        Features are Z-score normalized before fitting; the stored
        ``feature_means`` and ``feature_stds`` are reused at predict time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
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

        n_features = X_scaled.shape[1]
        XtX = X_scaled.T @ X_scaled
        self.w_cov = np.linalg.inv(self.alpha * np.eye(n_features) + self.beta * XtX)
        self.w_mean = self.beta * self.w_cov @ X_scaled.T @ y_arr
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using the posterior mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values (posterior mean).
        """
        X_arr = np.array(X)
        X_scaled = (X_arr - self.feature_means) / self.feature_stds
        return X_scaled @ self.w_mean


class DecisionTreeRegressorScratch:
    """Single Regression Tree used as building block for Random Forest."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        """Initialize the tree with depth and leaf-size constraints.

        Parameters
        ----------
        max_depth : int, optional
            Maximum depth of the tree. Default is 5.
        min_samples_split : int, optional
            Minimum number of samples required to split a node. Default is 2.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree: Optional[dict] = None

    def fit(self, X: Any, y: Any) -> "DecisionTreeRegressorScratch":
        """Build the decision tree by recursive binary splitting.

        Splits are chosen by minimizing the Mean Squared Error (MSE)
        of the target within each child region. Stops when max_depth
        is reached or a node has fewer than min_samples_split samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training target vector.

        Returns
        -------
        DecisionTreeRegressorScratch
            The fitted tree model (self).
        """
        X_arr = np.array(X)
        y_arr = np.array(y).flatten()
        self.tree = self._build_tree(X_arr, y_arr, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """Recursively build tree nodes."""
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or depth >= self.max_depth:
            return {"leaf": True, "mean": np.mean(y)}

        best_mse = float("inf")
        best_feat, best_thresh = None, None

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left = y[X[:, feat] <= t]
                right = y[X[:, feat] > t]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = np.var(left) * len(left) + np.var(right) * len(right)
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = t

        if best_feat is None:
            return {"leaf": True, "mean": np.mean(y)}

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values by traversing the tree for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for which to generate predictions.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted target values.
        """
        X_arr = np.array(X)
        return np.array([self._predict_row(row, self.tree) for row in X_arr])

    def _predict_row(self, row: np.ndarray, node: dict) -> float:
        """Traverse the tree for a single row."""
        if node.get("leaf"):
            return node["mean"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])


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
            indices = rng.choice(num_samples, size=num_samples, replace=True)
            X_b, y_b = X_arr[indices], y_arr[indices]

            feature_subset = rng.choice(n_features, size=m_features, replace=False)
            X_b_sub = X_b[:, feature_subset]

            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth)
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
