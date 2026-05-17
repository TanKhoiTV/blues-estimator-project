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
    y = np.asarray(y)
    beta_hat = np.asarray(beta_hat)

    # FIX CODECOBBIT: Sử dụng biến y để kiểm tra shape, triệt tiêu lỗi "Unused parameter y"
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples."
        )

    n, p = X.shape
    df = n - p

    # Tính toán ma trận hiệp biến bằng giả nghịch đảo
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov_matrix = sigma2 * xtx_inv

    # Tính toán các đại lượng thống kê suy diễn
    se = np.sqrt(np.maximum(0, np.diag(cov_matrix)))
    t_stats = np.where(se > 0, beta_hat / se, 0.0)
    p_values = 2 * stats.t.sf(np.abs(t_stats), df)

    t_crit = stats.t.ppf(0.975, df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se

    return {
        # Định dạng key phục vụ bộ test (tests/test_coef_inference.py)
        "standard_errors": se,
        "t_statistics": t_stats,
        "t_stats": t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        # Định dạng key viết hoa phục vụ file demo Notebook (part1_notebook.ipynb)
        "SE": se,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    }


def vif(X):
    """Compute Variance Inflation Factor (VIF)."""
    pass
