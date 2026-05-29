"""Archived Lasso and ElasticNet models.

These implementations were removed from the main model_comparison module
because the spec requires only Ridge (or Lasso) — not both. Ridge was kept
as the primary regularized model.

This archive preserves the working, tested code for reference.
"""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


def train_lasso_regression(
    X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0
) -> dict:
    """
    Train a Lasso Regression model with internal Z-score scaling.

    Scales features internally via ``StandardScaler`` and stores the
    scaler alongside the fitted model so evaluation can apply the same
    transformation on test data.

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
    dict
        A dict with ``"model"``, ``"type"``, ``"scaler"``, and
        ``"feature_names"`` for use by ``evaluate_model``.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.asarray(X_train, dtype=float))

    model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    model.fit(X_scaled, y_train)

    return {
        "model": model,
        "type": "lasso",
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }


def train_elasticnet_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> dict:
    """
    Train an ElasticNet Regression model with internal Z-score scaling.

    Scales features internally via ``StandardScaler`` and stores the
    scaler alongside the fitted model so evaluation can apply the same
    transformation on test data.

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
    dict
        A dict with ``"model"``, ``"type"``, ``"scaler"``, and
        ``"feature_names"`` for use by ``evaluate_model``.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.asarray(X_train, dtype=float))

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
    model.fit(X_scaled, y_train)

    return {
        "model": model,
        "type": "elasticnet",
        "scaler": scaler,
        "feature_names": list(X_train.columns),
    }
