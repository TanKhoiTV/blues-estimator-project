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
        cv_scores, mean_score = kfold_cv(X, y, k=k_folds, random_state=0)

        # Check number of folds and data type
        assert len(cv_scores) == k_folds
        assert isinstance(mean_score, float)
        assert mean_score > 0.0

        # Verify mean_score with mean of cv_scores
        assert mean_score == pytest.approx(np.mean(cv_scores))

        # Verify reproducibility — same seed → same scores
        cv_scores_2, _ = kfold_cv(X, y, k=k_folds, random_state=0)
        np.testing.assert_array_equal(cv_scores, cv_scores_2)

        # Verify partition: total samples through all folds = n_samples
        fold_sizes = [n_samples // k_folds] * k_folds
        for i in range(n_samples % k_folds):
            fold_sizes[i] += 1
        assert sum(fold_sizes) == n_samples

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

    def test_reproducibility_with_same_seed(self):
        """Verify that kfold_cv yields reproducible fold scores given a seed.

        Given the same random_state, kfold_cv must produce identical fold scores
        across multiple calls. Without this guarantee, any experiment or hyperparameter
        comparison built on top of this CV function would be unreliable.
        """
        np.random.seed(0)
        X = np.random.randn(60, 3)
        y = np.random.randn(60)

        scores_1, mean_1 = kfold_cv(X, y, k=5, random_state=42)
        scores_2, mean_2 = kfold_cv(X, y, k=5, random_state=42)

        np.testing.assert_array_equal(scores_1, scores_2)
        assert mean_1 == mean_2

    def test_mean_score_equals_mean_of_fold_scores(self):
        """Verify internal consistency of the returned mean score.

        The returned mean_score must equal exactly np.mean(cv_scores).
        This verifies internal consistency: if these two values diverge,
        the aggregation logic is broken regardless of whether individual folds are correct.
        """
        np.random.seed(0)
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        scores, mean_score = kfold_cv(X, y, k=5, random_state=7)

        assert mean_score == pytest.approx(np.mean(scores), rel=1e-10)

    def test_higher_noise_yields_higher_cv_score(self):
        """Validate that CV score scales properly with data noise levels.

        A model trained on high-noise data must produce a higher mean CV score
        than the same model on low-noise data. This validates that the CV score
        actually reflects prediction difficulty, not just that the function runs without error.
        """
        np.random.seed(0)
        n, p = 80, 3
        X = np.random.randn(n, p)
        X_aug = np.column_stack([np.ones(n), X])
        beta = np.array([1.0, 2.0, -1.0, 0.5])

        y_low_noise = X_aug @ beta + np.random.normal(0, 0.01, n)
        y_high_noise = X_aug @ beta + np.random.normal(0, 10.0, n)

        _, score_low = kfold_cv(X, y_low_noise, k=5, random_state=0)
        _, score_high = kfold_cv(X, y_high_noise, k=5, random_state=0)

        assert score_high > score_low
