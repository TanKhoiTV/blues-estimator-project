import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))



class TestGaussMarkov:
    def test_design_matrix_full_rank():
        """
        Test case 1: Khẳng định rank(X) == X.shape[1] (Thỏa mãn GM2: Không đa cộng tuyến hoàn hảo)
        """
        n_samples = 100
        beta = np.array([5.0, 2.5, -1.5]) # p = 2 (chưa tính intercept), tổng cộng p+1 = 3 hệ số
        sigma = 2.0
        
        X, y, epsilon = generate_synthetic_data(n_samples, beta, sigma)
        
        expected_rank = X.shape[1]  # p + 1
        actual_rank = np.linalg.matrix_rank(X)
        
        assert actual_rank == expected_rank, f"Ma trận X bị đa cộng tuyến, rank thực tế là {actual_rank} thay vì {expected_rank}"

    def test_epsilon_assumptions():
        """
        Test case 2: Sinh lượng mẫu lớn (n = 50000) để kiểm tra kỳ vọng nhiễu xấp xỉ 0 
        và phương sai nhiễu xấp xỉ sigma^2 với dung sai < 1e-2.
        """
        n_samples = 50000
        beta = np.array([1.0, -2.0])
        sigma = 1.5
        tolerance = 1e-2
        
        # Khóa seed để đảm bảo tính tái lập ổn định khi chạy test
        np.random.seed(42)
        
        _, _, epsilon = generate_synthetic_data(n_samples, beta, sigma)
        
        empirical_mean = np.mean(epsilon)
        empirical_var = np.var(epsilon)
        expected_var = sigma ** 2
        
        # Kiểm tra kỳ vọng sai số xấp xỉ 0
        assert abs(empirical_mean - 0.0) < tolerance, \
            f"Kỳ vọng thực nghiệm của nhiễu bằng {empirical_mean}, vượt quá dung sai {tolerance}"
            
        # Kiểm tra phương sai sai số xấp xỉ sigma^2
        assert abs(empirical_var - expected_var) < tolerance, \
            f"Phương sai thực nghiệm của nhiễu bằng {empirical_var}, lệch khỏi kỳ vọng {expected_var} vượt quá dung sai {tolerance}"
