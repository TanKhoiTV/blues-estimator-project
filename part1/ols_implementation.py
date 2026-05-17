"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np
from scipy import stats


def ols_fit(X, y):
    """
    Compute OLS solution and Residual Variance Estimator.

    Formulas:
    - beta_hat: $\\hat{\\beta} = (X^T X)^{-1} X^T y$
    - sigma2: $\\hat{\\sigma}^2 = \\frac{RSS}{n - p - 1}$
    """
    X = np.array(X)
    y = np.array(y)

    # 1. Kiểm tra mảng rỗng
    if X.size == 0 or y.size == 0:
        raise ValueError("Input matrices X and y cannot be empty.")

    # 2. Kiểm tra số chiều dữ liệu (Dimensionality check)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, but got {X.ndim}D.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, but got {y.ndim}D.")

    n, p = X.shape

    # Kiểm tra tính tương thích giữa X và y
    if n != y.shape[0]:
        raise ValueError(f"Mismatch: X has {n} samples, y has {y.shape[0]} samples.")

    # 3. Chặn lỗi chia cho 0 (Guard residual DoF)
    if n - p - 1 <= 0:
        raise ValueError(
            f"Not enough samples to compute variance. n ({n}) must be strictly greater than p + 1 ({p + 1})."
        )

    # Tính beta_hat bằng pseudo-inverse
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta_hat = xtx_inv @ X.T @ y

    # Tính Residual Sum of Squares (RSS)
    y_hat = X @ beta_hat
    rss = np.sum((y - y_hat) ** 2)

    # Tính sigma^2
    sigma2 = rss / (n - p - 1)

    return beta_hat, sigma2


def hat_matrix(X):
    """Compute the Hat Matrix $H = X(X^T X)^{-1} X^T$."""
    X = np.array(X)

    if X.size == 0:
        raise ValueError("Input matrix X cannot be empty.")

    # Kiểm tra số chiều dữ liệu cho Hat Matrix
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

    # Validation to prevent runtime errors
    if y.shape != y_hat.shape:
        raise ValueError(
            f"Shape mismatch: y {y.shape} and y_hat {y_hat.shape} must be identical."
        )

    n = len(y)

    # Validation for degrees of freedom
    if n <= p + 1:
        raise ValueError(
            f"Sample size n ({n}) must be strictly greater than p + 1 ({p + 1}) to compute Adjusted R^2 and F-stat."
        )

    # 1. Compute RSS and TSS
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)

    # 2. Compute R-squared
    r2 = 1 - (rss / tss)

    # 3. Compute Adjusted R-squared
    adj_r2 = 1 - ((n - 1) / (n - p - 1)) * (1 - r2)

    # 4. Compute F-statistic
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
    df = n - p_total  # Bậc tự do (Degrees of freedom)

    # 1. Compute covariance matrix of coefficients: Var(beta_hat) = sigma^2 * (X^T X)^-1
    # Using pinv for numerical stability as requested
    cov_matrix = sigma2 * np.linalg.pinv(X.T @ X)

    # 2. Extract Standard Errors (SE) from the diagonal
    se = np.sqrt(np.diag(cov_matrix))

    # 3. Calculate t-statistics
    t_stats = beta_hat / se

    # 4. Calculate p-values (two-tailed test)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

    # 5. Calculate 95% Confidence Intervals
    t_crit = stats.t.ppf(0.975, df=df)  # Giá trị t tới hạn ở mức ý nghĩa 5% (2 phía)
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
    """Compute Variance Inflation Factor (VIF)."""
    pass
