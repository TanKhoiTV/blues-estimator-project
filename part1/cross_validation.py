"""Module for implementing k-fold cross-validation techniques."""

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

    # 1. Tạo mảng chỉ số và thiết lập xáo trộn ngẫu nhiên có fixed seed
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)

    # 2. Chia đều mảng chỉ số thành k phần (folds)
    folds = np.array_split(indices, k)

    mse_list = []

    # 3. Chạy vòng lặp k lần để train và evaluate
    for i in range(k):
        # Lấy fold thứ i làm tập validation (test set)
        val_idx = folds[i]

        # Lấy k-1 folds còn lại gộp thành tập training
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Huấn luyện mô hình trên tập train bằng ols_fit (lấy beta_hat, bỏ qua sigma2)
        beta_hat, _ = ols_fit(X_train, y_train)

        # Dự đoán trên tập validation
        y_val_pred = X_val @ beta_hat

        # Tính Mean Squared Error (MSE) cho fold hiện tại
        mse = np.mean((y_val - y_val_pred) ** 2)
        mse_list.append(mse)

    # 4. Tính toán mảng MSE và điểm trung bình CV Score
    mse_array = np.array(mse_list)
    cv_score = np.mean(mse_array)

    return mse_array, cv_score
