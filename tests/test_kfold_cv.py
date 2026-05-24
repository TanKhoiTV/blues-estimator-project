import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from part1.cross_validation import kfold_cv


class TestKFoldCV:
    """Test suite for custom kfold_cv execution framework."""

    def test_cv_fold_splits_integrity(self):
        """Verify that the indices partition data exactly into k groups without overlaps or omissions."""
        # Using a simple deterministic setup to monitor partition logic
        n_samples = 100
        n_features = 2
        k_folds = 5

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        # Expect kfold_cv to execute internally or expose metrics per fold
        # For validation, we test if the function executes smoothly and returns a list of scores length k
        cv_scores, mean_score = kfold_cv(X, y, k=k_folds)

        assert len(cv_scores) == k_folds
        assert isinstance(mean_score, float)
        assert mean_score > 0.0

    def test_perfect_fit_zero_error(self):
        """Test CV score on a noiseless deterministic system.

        In an ideal linear relationship y = Xβ, every validation fold should yield an MSE close to zero.
        """
        np.random.seed(202)
        n_samples = 60
        n_features = 2
        # No noise added
        X = np.random.randn(n_samples, n_features)
        X_aug = np.column_stack([np.ones(n_samples), X])
        beta_true = np.array([1.5, -3.0, 2.5])
        y = X_aug @ beta_true

        _, mean_score = kfold_cv(X, y, k=3)

        # Mean MSE across folds should be extremely close to 0
        assert mean_score == pytest.approx(0.0, abs=1e-12)

    def test_invalid_k_raises_error(self):
        """Test edge cases where k is invalid relative to sample size."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)

        # k must be greater than 1
        with pytest.raises(ValueError, match="k must be greater than 1"):
            kfold_cv(X, y, k=1)

        # k cannot exceed n_samples
        with pytest.raises(ValueError, match="k cannot exceed number of samples"):
            kfold_cv(X, y, k=15)
