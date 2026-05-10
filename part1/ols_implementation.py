"""Module for OLS implementation, hat matrix, and statistical inference."""

import numpy as np


def ols_fit(X, y):
    """Compute OLS solution (beta_hat) and Residual Variance Estimator."""
    X = np.array(X)
    y = np.array(y)
    
    # 1. Tính beta_hat bằng công thức: (X^T * X)^-1 * X^T * y
    # Sử dụng np.linalg.pinv để xử lý cả trường hợp ma trận bị suy biến (đa cộng tuyến)
    beta_hat = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    # 2. Tính toán sigma^2 (phương sai sai số)
    n, p = X.shape
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    # Công thức: RSS / (n - p)
    # n - p là bậc tự do (degrees of freedom)
    sigma2 = np.sum(residuals**2) / (n - p)
    
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
