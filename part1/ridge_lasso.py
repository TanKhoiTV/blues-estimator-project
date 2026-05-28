"""Module for Ridge and Lasso regression implementations."""

import numpy as np
import matplotlib.pyplot as plt


def ridge_fit(X, y, lam):
    """
    Compute Ridge Regression (L2 Regularization) solution using closed-form formula.

    An intercept column is prepended internally. Do NOT include one in X.

    Parameters
    ----------
        X: array-like of shape (n, p)
           Design matrix of predictors, without intercept column.
        y: array-like of shape (n, )
           Response vector.
        lam: float
           Regularization parameter (lambda).

    Returns
    -------
        beta_hat: ndarray of shape (p + 1, )
            Estimated coefficients, where beta_hat[0] is the unpenalized intercept.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.size == 0 or y.size == 0:
        raise ValueError("Input matrices X and y cannot be empty.")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, but got {X.ndim}D.")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, but got {y.ndim}D.")

    n, p = X.shape

    if n != y.shape[0]:
        raise ValueError(f"Mismatch: X has {n} samples, y has {y.shape[0]} samples.")

    X_aug = np.column_stack([np.ones(n), X])

    # Tạo ma trận đơn vị I
    I = np.eye(p + 1)
    I[0, 0] = 0.0

    # Công thức đóng: beta_hat = (X^T X + lambda * I)^-1 X^T y
    xtx = X_aug.T @ X_aug
    ridge_matrix = xtx + lam * I

    # Dùng np.linalg.solve (giả nghịch đảo) để tính toán an toàn hơn với số học
    beta_hat = np.linalg.solve(ridge_matrix, X_aug.T @ y)

    return beta_hat


def plot_ridge_trace(X, y, lambdas=None):
    """
    Plot the Ridge Trace to visualize how coefficients change with lambda.

    Parameters
    ----------
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
    ax.set_xlabel(r"$\lambda$ (Log Scale)", fontsize=12)
    ax.set_ylabel(r"Coefficients ($\beta$)", fontsize=12)
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
