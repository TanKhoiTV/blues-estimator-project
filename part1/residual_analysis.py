"""Module for residual analysis and diagnostic plots."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from part1.ols_implementation import hat_matrix


def residual_plots(X, y, beta_hat):
    """
    Generate 4 residual diagnostic plots for OLS regression.

    Plots included:
    1. Residuals vs Fitted
    2. Normal Q-Q Plot
    3. Scale-Location
    4. Cook's Distance

    Args:
        X (array-like): Design matrix INCLUDING intercept column.
            Must match the X used to compute beta_hat
            NOTE: Unlike ridge_fit/ols_fit which prepend intercept internally,
            this function expects X to already contain the intercept column.
            Example: X_aug = np.column_stack([np.ones(n), X_raw])
        y (array-like): Target vector.
        beta_hat (array-like): Estimated OLS coefficients.

    Returns
    -------
        matplotlib.figure.Figure: The generated figure containing the 4 subplots.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    beta_hat = np.asarray(beta_hat)

    n, p = X.shape

    # 1. Tính toán giá trị dự báo và phần dư
    y_hat = X @ beta_hat
    residuals = y - y_hat

    # 2. Tính ma trận Hat (Leverage) để chuẩn hóa phần dư và tính Cook's Distance
    # h_ii là các phần tử trên đường chéo của ma trận Hat
    H_mat = hat_matrix(X)
    h_ii = np.diag(H_mat)

    # Tránh lỗi chia cho 0 do sai số dấu phẩy động
    h_ii = np.clip(h_ii, 0, 0.9999)

    # Tính MSE (Mean Squared Error)
    mse = np.sum(residuals**2) / (n - p)

    # 3. Chuẩn hóa phần dư (Standardized Residuals)
    std_residuals = residuals / np.sqrt(mse * (1 - h_ii))

    # 4. Tính khoảng cách Cook (Cook's Distance)
    cooks_d = (std_residuals**2 / p) * (h_ii / (1 - h_ii))

    # ==========================================
    # VẼ 4 BIỂU ĐỒ (2x2 Layout)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Residual Diagnostics Plots", fontsize=16, fontweight="bold")

    # --- Plot 1: Residuals vs Fitted ---
    axes[0, 0].scatter(y_hat, residuals, alpha=0.7, edgecolors="k", label="Residuals")
    axes[0, 0].axhline(0, color="red", linestyle="--", label="Zero line")
    axes[0, 0].set_title("Residuals vs Fitted")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].legend()

    # --- Plot 2: Normal Q-Q Plot ---
    stats.probplot(std_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_markeredgecolor("k")
    axes[0, 1].get_lines()[0].set_alpha(0.7)
    axes[0, 1].get_lines()[1].set_color("red")
    axes[0, 1].get_lines()[1].set_linestyle("--")
    axes[0, 1].set_title("Normal Q-Q")
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Standardized Residuals")
    # Tự tạo legend cho Q-Q plot
    axes[0, 1].plot([], [], "o", color="blue", label="Data quantiles")
    axes[0, 1].plot([], [], color="red", linestyle="--", label="Normal line")
    axes[0, 1].legend()

    # --- Plot 3: Scale-Location ---
    sqrt_std_res = np.sqrt(np.abs(std_residuals))
    axes[1, 0].scatter(
        y_hat, sqrt_std_res, alpha=0.7, edgecolors="k", label=r"$\sqrt{|Std. Res|}$"
    )
    axes[1, 0].set_title("Scale-Location")
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel(r"$\sqrt{|Standardized\ Residuals|}$")
    axes[1, 0].legend()

    # --- Plot 4: Cook's Distance ---
    axes[1, 1].stem(range(n), cooks_d, markerfmt=",", basefmt=" ", label="Cook's dist")
    axes[1, 1].axhline(4 / n, color="red", linestyle="--", label="Threshold (4/n)")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig
