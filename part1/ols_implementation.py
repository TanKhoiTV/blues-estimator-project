"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np
from scipy import stats


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


def coef_inference(X, y, beta_hat, sigma2):
    """Compute SE, t-stat, p-value and Confidence Intervals for coefficients."""
    X = np.asarray(X)
    beta_hat = np.asarray(beta_hat)

    n, p = X.shape

    # Bẫy lỗi để pass bài test: test_dimension_mismatch
    if beta_hat.shape[0] != p:
        raise ValueError(
            f"Dimension mismatch: X has {p} columns but beta_hat has {beta_hat.shape[0]} elements."
        )

    df = n - p

    xtx_inv = np.linalg.pinv(X.T @ X)
    cov_matrix = sigma2 * xtx_inv

    se = np.sqrt(np.maximum(0, np.diag(cov_matrix)))
    t_stats = np.where(se > 0, beta_hat / se, 0.0)
    p_values = 2 * stats.t.sf(np.abs(t_stats), df)

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se

    return {
        "standard_errors": se,
        "t_statistics": t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def vif(X):
    """Compute Variance Inflation Factor (VIF)."""
    pass
