"""Module for Ridge and Lasso regression implementations."""

import numpy as np
import matplotlib.pyplot as plt


def ridge_fit(X, y, lam):
    """
    Compute Ridge Regression (L2 Regularization) solution using closed-form formula.

    Args:
        X (array-like): Design matrix.
        y (array-like): Target vector.
        lam (float): Regularization parameter (lambda).

    Returns
    -------
        numpy.ndarray: Estimated coefficients beta_hat.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n, p = X.shape

    # Tạo ma trận đơn vị I
    I = np.eye(p)

    # Công thức đóng: beta_hat = (X^T X + lambda * I)^-1 X^T y
    xtx = X.T @ X
    ridge_matrix = xtx + lam * I

    # Dùng np.linalg.pinv (giả nghịch đảo) để tính toán an toàn hơn với số học
    beta_hat = np.linalg.pinv(ridge_matrix) @ X.T @ y

    return beta_hat


def plot_ridge_trace(X, y, lambdas=None):
    """
    Plot the Ridge Trace to visualize how coefficients change with lambda.

    Args:
        X (array-like): Design matrix.
        y (array-like): Target vector.
        lambdas (array-like, optional): Array of lambda values. Defaults to logspace(-3, 3).

    Returns
    -------
        matplotlib.figure.Figure: The generated Ridge Trace plot.
    """
    if lambdas is None:
        # Tạo mảng giá trị lambda từ 10^-3 đến 10^3
        lambdas = np.logspace(-3, 3, 200)

    X = np.asarray(X)
    y = np.asarray(y)

    betas = []
    for lam in lambdas:
        beta = ridge_fit(X, y, lam)
        betas.append(beta)

    betas = np.array(betas)

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))

    for j in range(betas.shape[1]):
        ax.plot(lambdas, betas[:, j], label=f"$\\beta_{{{j}}}$")

    ax.set_xscale("log")
    ax.set_xlabel("$\lambda$ (Log Scale)", fontsize=12)
    ax.set_ylabel("Coefficients ($\\beta$)", fontsize=12)
    ax.set_title(
        "Ridge Trace: Coefficient Paths vs. Regularization Strength",
        fontsize=14,
        fontweight="bold",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Chỉ hiển thị legend nếu số lượng biến không quá lớn
    if betas.shape[1] <= 10:
        ax.legend(loc="best")

    ax.grid(True, alpha=0.3)

    return fig
