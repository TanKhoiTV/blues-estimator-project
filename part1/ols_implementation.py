"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np


def ols_fit(X, y):
    r"""
    Compute OLS solution and Residual Variance Estimator.

    Formulas:
    - beta_hat: $\hat{\beta} = (X^T X)^{-1} X^T y$
    - sigma2: $\hat{\sigma}^2 = \frac{RSS}{n - p}$
    """
    X = np.array(X)
    y = np.array(y)

    if X.size == 0 or y.size == 0:
        raise ValueError("Input matrices X and y cannot be empty.")

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, but got {X.ndim}D.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, but got {y.ndim}D.")

    n, p = X.shape

    if n != y.shape[0]:
        raise ValueError(f"Mismatch: X has {n} samples, y has {y.shape[0]} samples.")

    if n - p <= 0:
        raise ValueError(
            f"Not enough samples to compute variance. n ({n}) must be strictly greater than p ({p})."
        )

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_hat = xtx_inv @ X.T @ y

    y_hat = X @ beta_hat
    rss = np.sum((y - y_hat) ** 2)

    sigma2 = rss / (n - p)

    return beta_hat, sigma2


def hat_matrix(X):
    """Compute the Hat Matrix H = X(X^T X)^{-1} X^T."""
    X = np.array(X)

    if X.size == 0:
        raise ValueError("Input matrix X cannot be empty.")

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, but got {X.ndim}D.")

    xtx_inv = np.linalg.pinv(X.T @ X)
    h_mat = X @ xtx_inv @ X.T

    return h_mat


def model_metrics(y, y_hat, p):
    """Compute statistical metrics for evaluating OLS model performance."""
    pass


def vif(X):
    """Compute Variance Inflation Factor (VIF) for each feature."""
    X = np.asarray(X)
    n, p = X.shape

    if p < 2:
        raise ValueError(
            "X must have at least two columns (intercept and one feature) to compute VIF."
        )

    vif_values = []

    for j in range(1, p):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)

        xtx_inv = np.linalg.pinv(X_j.T @ X_j)
        beta_j = xtx_inv @ X_j.T @ y_j
        y_j_hat = X_j @ beta_j

        rss = np.sum((y_j - y_j_hat) ** 2)
        tss = np.sum((y_j - np.mean(y_j)) ** 2)

        if tss == 0:
            vif_values.append(1.0)
            continue

        r2_j = 1.0 - (rss / tss)

        if np.isclose(r2_j, 1.0):
            vif_values.append(float("inf"))
        else:
            vif_values.append(1.0 / (1.0 - r2_j))

    return np.array(vif_values)
