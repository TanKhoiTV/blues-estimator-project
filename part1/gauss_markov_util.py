import numpy as np


def generate_synthetic_data(
    n: int,
    beta: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
):
    """
    Sinh dữ liệu giả lập thỏa mãn các giả thiết Gauss-Markov từ GM1 đến GM5.

    Parameters
    ----------
    n     : số quan sát
    beta  : vector hệ số thực, shape (p+1,); phần tử đầu là intercept
    sigma : độ lệch chuẩn của nhiễu (> 0)
    rng   : numpy Generator để kiểm soát seed (tuỳ chọn)

    Returns
    -------
    X       : ma trận thiết kế (n, p+1), cột đầu là hằng số 1  — GM2: full rank
    y       : vector phản hồi (n,)
    epsilon : vector nhiễu (n,)  — GM1: E[ε]=0, GM3: Var(ε)=σ²I, GM4: không tự tương quan
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n < len(beta):
        raise ValueError("n must be >= len(beta) for full rank design matrix")

    if rng is None:
        rng = np.random.default_rng()

    p_plus_1 = len(beta)  # số cột = số hệ số (gồm intercept)
    p = p_plus_1 - 1  # số biến giải thích

    # --- GM2: X là ma trận cố định, không ngẫu nhiên, đạt hạng đầy đủ ---
    # Sinh phần biến giải thích từ N(0,1); cột đầu là intercept = 1
    X = np.random.randn(n, p) if rng is None else rng.standard_normal((n, p))
    X = np.column_stack([np.ones(n), X])  # thêm cột intercept

    # --- GM1 & GM3 & GM4: ε ~ i.i.d. N(0, σ²) ---
    epsilon = rng.normal(loc=0.0, scale=sigma, size=n)

    # --- Mô hình tuyến tính: y = Xβ + ε ---
    y = X @ beta + epsilon

    return X, y, epsilon
