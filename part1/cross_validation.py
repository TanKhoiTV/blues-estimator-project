"""Module for Cross-Validation techniques."""

import numpy as np
from part1.ols_implementation import ols_fit


def kfold_cv(X, y, k=5, random_state=None):
    """
    Evaluate model generalization using k-Fold Cross-Validation.

    Args:
        X (array-like): Design matrix.
        y (array-like): Target vector.
        k (int): Number of folds. Default is 5.
        random_state (int, optional): Seed for random shuffling. Default is None.

    Returns
    -------
        tuple: (Array of MSE for each fold, Mean CV Score)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = X.shape[0]
    if n_samples != y.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {n_samples} samples, but y has {y.shape[0]} samples."
        )
    if k < 2:
        raise ValueError(f"k must be greater than 1. Received k={k}.")
    if k > n_samples:
        raise ValueError(
            f"k cannot exceed number of samples. Received k={k}, n_samples={n_samples}."
        )

    # Tạo mảng chỉ số và thiết lập xáo trộn ngẫu nhiên có fixed seed
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    # Chia đều mảng chỉ số thành k phần (folds)
    folds = np.array_split(indices, k)

    mse_list = []

    for i in range(k):
        # Lấy fold thứ i làm tập validation
        val_idx = folds[i]

        # Lấy k-1 folds còn lại gộp thành tập training
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Huấn luyện mô hình
        beta_hat, _ = ols_fit(X_train, y_train)

        # Kiểm tra xem ols_fit có tự động thêm 1 cột Intercept vào beta_hat hay không
        if beta_hat.shape[0] == X_val.shape[1] + 1:
            # Nếu có, ta phải gắn thêm 1 cột số 1 vào X_val để kích thước khớp nhau
            X_val_padded = np.column_stack((np.ones(X_val.shape[0]), X_val))
            y_val_pred = X_val_padded @ beta_hat
        else:
            # Nếu kích thước đã khớp (ols_fit không thêm), thì nhân bình thường
            y_val_pred = X_val @ beta_hat

        # Tính Mean Squared Error (MSE) cho fold hiện tại
        mse = np.mean((y_val - y_val_pred) ** 2)
        mse_list.append(mse)

    mse_array = np.array(mse_list)
    cv_score = np.mean(mse_array)

    return mse_array, cv_score
