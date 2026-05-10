"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np


def ols_fit(X, y):
    """
    Compute OLS solution and Residual Variance Estimator.

    Formulas:
    - beta_hat: $\\hat{\\beta} = (X^T X)^{-1} X^T y$
    - sigma2: $\\hat{\\sigma}^2 = \\frac{RSS}{n - p - 1}$
    """
    X = np.array(X)
    y = np.array(y)

    if X.size == 0 or y.size == 0:
        raise ValueError("Input matrices X and y cannot be empty.")

    n, p = X.shape

    # Tính beta_hat bằng pseudo-inverse để đảm bảo ổn định số học
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_hat = xtx_inv @ X.T @ y

    # Tính Residual Sum of Squares (RSS)
    y_hat = X @ beta_hat
    rss = np.sum((y - y_hat) ** 2)

    # Tính sigma^2 theo đúng yêu cầu: n - p - 1
    sigma2 = rss / (n - p - 1)

    return beta_hat, sigma2


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
