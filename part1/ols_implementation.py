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
    """Compute statistical metrics for evaluating OLS model performance.

    Metrics include: RSS, TSS, R-squared, Adjusted R-squared, and F-statistic.
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    if y.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y {y.shape} and y_hat {y_hat.shape} must be identical."
        )

    n = len(y)

    if n <= p + 1:
        raise ValueError(
            f"Sample size n ({n}) must be strictly greater than p + 1 ({p + 1}) to compute Adjusted R^2 and F-stat."
        )

    if p <= 0:
        raise ValueError(
            "p must be a strictly positive integer to compute F-statistic."
        )

    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    if tss == 0:
        r2 = 1.0 if rss == 0 else 0.0
        adj_r2 = r2
        f_stat = float("nan")
    else:
        r2 = 1 - (rss / tss)
        adj_r2 = 1 - ((n - 1) / (n - p - 1)) * (1 - r2)
        f_stat = ((tss - rss) / p) / (rss / (n - p - 1))

    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "F_statistic": f_stat,
    }


def coef_inference(X, y, beta_hat, sigma2):
    """Compute SE, t-stat, p-value and Confidence Intervals for coefficients.

    Uses scipy.stats.t to derive p-values and 95% CI with n - p degrees of freedom.
    """
    X = np.asarray(X)
    n, p_total = X.shape
    df = n - p_total

    cov_matrix = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_matrix))
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

    t_crit = stats.t.ppf(0.975, df=df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se

    return {
        "SE": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    }


def vif(X):
    """Compute Variance Inflation Factor (VIF) for each feature.

    Returns an array of VIF values for each column in X (excluding the intercept).
    """
    X = np.asarray(X)
    n, p = X.shape

    if p < 2:
        raise ValueError(
            "X must have at least two columns (intercept and one feature) to compute VIF."
        )

    vif_values = []

    # Lặp qua từng cột đặc trưng (bỏ qua cột 0 là intercept)
    for j in range(1, p):
        y_j = X[:, j]
        # X_j là ma trận X sau khi đã loại bỏ cột j
        X_j = np.delete(X, j, axis=1)

        # Hồi quy OLS: y_j theo X_j (Dùng pseudo-inverse cho lẹ và ổn định)
        xtx_inv = np.linalg.pinv(X_j.T @ X_j)
        beta_j = xtx_inv @ X_j.T @ y_j
        y_j_hat = X_j @ beta_j

        # Tính R^2_j
        rss = np.sum((y_j - y_j_hat) ** 2)
        tss = np.sum((y_j - np.mean(y_j)) ** 2)

        if tss == 0:
            # Nếu cột này là hằng số, VIF coi như = 1
            vif_values.append(1.0)
            continue

        r2_j = 1.0 - (rss / tss)

        # Tính VIF và chặn lỗi chia cho 0 nếu R^2_j = 1 (đa cộng tuyến hoàn hảo)
        if np.isclose(r2_j, 1.0):
            vif_values.append(float("inf"))
        else:
            vif_values.append(1.0 / (1.0 - r2_j))

    return np.array(vif_values)
