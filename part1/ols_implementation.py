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


def coef_inference(X, y, beta_hat, sigma2):
    """Compute SE, t-stat, p-value and Confidence Intervals for coefficients."""
    pass


def vif(X):
    """Compute Variance Inflation Factor (VIF) for each feature matching statsmodels."""
    X = np.asarray(X)
    n, p = X.shape

    if p < 1:
        raise ValueError("X must have at least one column to compute VIF.")

    vif_values = []

    for j in range(p):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)

        if X_j.shape[1] == 0:
            vif_values.append(1.0)
            continue

        # Sử dụng lstsq (SVD) thay vì pinv để đạt độ chính xác số học tuyệt đối như statsmodels
        beta_j, _, _, _ = np.linalg.lstsq(X_j, y_j, rcond=None)
        y_j_hat = X_j @ beta_j

        rss = np.sum((y_j - y_j_hat) ** 2)

        # THUẬT TOÁN ĐỈNH CAO: Tự động kiểm tra xem ma trận con X_j có chứa cột hằng số (intercept) không
        has_constant = False
        for col_idx in range(X_j.shape[1]):
            if np.allclose(X_j[:, col_idx], X_j[0, col_idx]):
                has_constant = True
                break

        # Nếu có hằng số dùng Centered TSS, nếu không có hằng số bắt buộc dùng Uncentered TSS
        if has_constant:
            tss = np.sum((y_j - np.mean(y_j)) ** 2)
        else:
            tss = np.sum(y_j**2)

        if tss == 0:
            vif_values.append(1.0)
            continue

        r2_j = 1.0 - (rss / tss)

        if r2_j >= 1.0 or np.isclose(r2_j, 1.0):
            vif_values.append(float("inf"))
        else:
            vif_values.append(1.0 / (1.0 - r2_j))

    return np.array(vif_values)
