"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np


def ols_fit(X, y):
    """Compute OLS solution (beta_hat) and Residual Variance Estimator."""
    pass


def hat_matrix(X):
    """Compute the Hat Matrix $H = X(X^T X)^{-1} X^T$."""
    X = np.array(X)
    if X.size == 0:
        raise ValueError("Input matrix X cannot be empty.")
    xtx_inv = np.linalg.pinv(X.T @ X)
    h_mat = X @ xtx_inv @ X.T
    return h_mat


def model_metrics(y, y_hat, p):
    """Compute various metrics: MAE, RMSE, R-squared."""
    pass


def coef_inference(X, y, beta_hat, sigma2):
    """Compute SE, t-stat, p-value and Confidence Intervals."""
    pass


def vif(X):
    """Compute Variance Inflation Factor (VIF)."""
    pass
