"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np


def ols_fit(X, y):
    """Compute OLS solution (beta_hat) and Residual Variance Estimator."""
    X = np.asarray(X)
    y = np.asarray(y)

    # 1. Validation: Empty arrays
    if X.size == 0 or y.size == 0:
        raise ValueError("Inputs X and y cannot be empty.")

    # 2. Validation: Dimension mismatch (y must be 1D and match X rows)
    if y.ndim != 1:
        raise ValueError("Target vector y must be one-dimensional.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Dimension mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} elements."
        )

    n, p = X.shape

    # 3. Validation: Degrees of freedom (n > p) to avoid division by zero
    if n <= p:
        raise ValueError(
            f"Insufficient degrees of freedom: n ({n}) must be greater than p ({p})."
        )

    # Compute beta_hat using (X^T X)^{-1} X^T y
    # np.linalg.pinv handles singular/collinear X
    beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y

    # Compute Residual Variance Estimator (sigma^2)
    y_hat = X @ beta_hat
    residuals = y - y_hat
    sigma2 = np.sum(residuals**2) / (n - p)

    return beta_hat, sigma2


def hat_matrix(X):
    """Compute the Hat Matrix H = X(X^T X)^{-1} X^T.
    Uses pseudoinverse to handle rank-deficient cases.
    """
    X = np.asarray(X)

    # Validation: Require a 2D numeric array
    if X.ndim != 2:
        raise TypeError("X must be a 2D array.")

    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError("Dimensions of X cannot be zero.")

    # Compute hat matrix
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
