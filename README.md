# Dự án Blues Estimator

Nghiên cứu và triển khai thực nghiệm các Bộ ước lượng Tuyến tính Không chệch Tốt nhất (BLUE) từ nền tảng gốc — bao gồm OLS, Ridge và Lasso — hoàn toàn độc lập với các thư viện hỗ trợ như scikit-learn. Từ việc tái hiện các chứng minh lý thuyết của định lý Gauss-Markov đến việc xây dựng pipeline xử lý dữ liệu thực tế, quá trình này làm nổi bật tính quy luật chặt chẽ của toán học và ý nghĩa thống kê sâu sắc của các phần dư (residuals). Giải pháp được tối ưu hóa 100% bằng thuật toán thuần, đảm bảo tính toàn vẹn và khả năng kiểm soát hệ thống từ cốt lõi.
## Quick Start

Làm theo các bước sau để bắt đầu. Xem [CONTRIBUTING](CONTRIBUTING.md) để có hướng dẫn đầy đủ.

1. Clone repository.
2. Cài đặt `uv`.
3. Cài đặt dependencies bằng `uv sync`.
4. Chạy bất kỳ code nào bằng `uv run`.
5. Chạy `uv run pytest` để chạy tất cả các test.

## Cấu trúc Dự án

```bash
blues-estimator-project         # Sẽ được đổi tên thành Group_<ID>
├── report/
│   ├── report.pdf              # Bản PDF cuối cùng (LaTeX/Markdown)  
│   └── report.tex              # File nguồn cho báo cáo
├── part1/
│   ├── ols_implementation.py   # OLS cốt lõi & toán học từ đầu
│   ├── ridge_lasso.py          # Logic regularization
│   ├── residual_analysis.py    # Code cho các đồ thị chẩn đoán
│   ├── cross_validation.py     # Implement k-fold CV
│   └── part1_notebook.ipynb    # Demo lý thuyết & chứng minh
├── part2/
│   ├── data/
│   │   └── dataset.csv         # Dữ liệu thực tế gốc
│   ├── data_pipeline.py        # Class DataPipeline để làm sạch dữ liệu
│   ├── model_comparison.py     # So sánh 3+ mô hình
│   ├── advanced_methods.py     # Bonus: Kernel/Bayesian
│   └── part2_notebook.ipynb    # Ứng dụng thực tế & thảo luận
├── README.md                   # Tổng quan dự án & hướng dẫn sử dụng
├── pyproject.toml              # Danh sách dependencies "nguồn sự thật"
└── requirements.txt            # Danh sách dependencies, export từ `uv`
```

## Lộ trình Dự án

### Phần 1

#### Implement

Với mỗi hàm trong danh sách:

1. Trình bày công thức toán học.
2. Implement bằng Python thuần.
3. Minh họa với dữ liệu kiểm thử.
4. Kiểm tra và so sánh với `NumPy` hoặc `sklearn`.

Các hàm cần implement:

```bash
ols_fit(X, y)                   # Tính nghiệm OLS (beta_hat) và ước lượng phương sai phần dư (squared_sigma_hat)
hat_matrix(X)                   # Tính ma trận H và kiểm tra tính lũy đẳng
model_metrics(y, y_hat, p)      # Tính các chỉ số đánh giá mô hình
coef_inference(X, y, beta_hat, sigma2)  # Tính SE, t-stat, p-value và CI
vif(X)                          # Tính VIF
ridge_fit(X, y, lam)            # Tính Ridge Regression, vẽ Ridge Trace
residual_plots(X, y, beta_hat)  # Vẽ bốn đồ thị phần dư
kfold_cv(X, y, k)               # k-fold cross-validation, tính CV score
```

Cuối cùng, minh họa định lý Gauss-Markov bằng mô phỏng Monte Carlo để xác minh **tính không chệch** ($\mathbb{E}[\hat{\beta}] = \beta$) và **phương sai tối thiểu**.

#### Tiêu chí Chấp nhận

- [ ] **Độ chính xác**: Tất cả các hàm tự viết (ví dụ: `ols_fit`, `ridge_fit`) phải cho ra các hệ số khớp với output của `scikit-learn` hoặc `statsmodels` với dung sai $\varepsilon = 10^{-6}$.
- [ ] **Kiểm chứng Thực nghiệm Định lý Gauss-Markov**: Mô phỏng Monte Carlo phải chứng minh thành công rằng ước lượng OLS là không chệch ($\mathbb{E}[\hat{\beta}] = \beta$) và có phương sai thấp nhất trong số các ước lượng tuyến tính không chệch theo Gauss-Markov.
- [ ] **Tính toàn vẹn Chẩn đoán**: Hàm `residual_plots` phải tạo đủ bốn đồ thị chuẩn (Residuals vs. Fitted, Normal Q-Q, Scale-Location, và Residuals vs. Leverage) với nhãn trục đầy đủ.
- [ ] **Unit Test**: Mỗi hàm toán học phải vượt qua ít nhất hai unit test độc lập, sử dụng cả dữ liệu tổng hợp lẫn dữ liệu baseline đã biết trước.
- [ ] **Xác thực Ma trận**: Hàm `hat_matrix` phải xác minh bằng số tính lũy đẳng ($H^2 \approx H$) và tính đối xứng ($H^T \approx H$).

### Phần 2

#### Tóm tắt Nghiên cứu Dữ liệu & Chiến lược

**Nguồn**
- Kaggle: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers

**1. Xác nhận dataset đã chọn**
* **Tên Dataset:** OASIS Longitudinal Brain MRI Dataset (Bệnh Alzheimer's).
* **Mô tả:** Dataset thực tế gồm một tập hợp dọc của 150 đối tượng trong độ tuổi 60 đến 96. Mỗi đối tượng được quét ít nhất hai lần, cung cấp các chỉ số lâm sàng và nhận thức cùng dữ liệu thể tích não.
* **Mục tiêu:** Dự đoán điểm suy giảm nhận thức `MMSE` (Mini-Mental State Examination, thang điểm 0-30) bằng hồi quy, dựa trên tuổi, trình độ học vấn và teo não.

**2. Tóm tắt các feature chính và kiểu dữ liệu**
*(Tổng số hàng: n = 373)*

| Tên Feature | Kiểu | Liên tục/Rời rạc | Giá trị Thiếu | Vai trò | Mô tả |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Subject_ID` | Categorical | N/A | 0 (0%) | Index/Group | Định danh duy nhất cho mỗi bệnh nhân. |
| **`MMSE`** | **Số** | **Liên tục*** | **2 (~0.5%)** | **Target** | Điểm Mini-Mental State Examination. |
| `Age` | Số | Liên tục | 0 (0%) | Feature | Tuổi bệnh nhân tại thời điểm thăm khám. |
| `nWBV` | Số | **Liên tục** | 0 (0%) | Feature | Normalized Whole Brain Volume (teo não). |
| `eTIV` | Số | **Liên tục** | 0 (0%) | Feature | Estimated Total Intracranial Volume. |
| `SES` | Ordinal | Rời rạc | **19 (~5.09%)** | Feature | Địa vị Kinh tế Xã hội (1 = cao nhất, 5 = thấp nhất). |

*\*Lưu ý: Mặc dù MMSE về mặt toán học là số nguyên hữu hạn rời rạc (0-30), nó được chấp nhận rộng rãi và xử lý như một thang đo nhận thức liên tục trong mô hình hồi quy lâm sàng.*

**3. Tỉ lệ chia train/test và `random_state` đề xuất**
* **Tỉ lệ Chia:** 80% Train / 20% Test.
* **Chiến lược Chia:** **Group Shuffle Split** (nhóm theo `Subject_ID`).
    * *Lý do:* Đây là dữ liệu dọc — cùng một bệnh nhân xuất hiện nhiều lần qua các lần thăm khám khác nhau. Nếu chia ngẫu nhiên thông thường, Visit 1 của Patient A có thể nằm trong tập train còn Visit 2 lại nằm trong tập test, gây data leakage nghiêm trọng. Group bởi Subject ID đảm bảo một bệnh nhân chỉ xuất hiện ở đúng một trong hai tập.
* **`random_state`:** 42 (đặt cố định để đảm bảo tái tạo kết quả baseline OLS).

**4. Các feature được đánh dấu là ứng viên tiềm năng để loại bỏ**
* **Được đánh dấu vì Missing Values tự nhiên:** `SES` (Địa vị Kinh tế Xã hội) và `MMSE` trong một số lần theo dõi.
    * *Lý do:* Đáp ứng điều kiện $\ge 5\%$ giá trị thiếu (cụ thể ~5.09% cho SES). Trong nghiên cứu về người cao tuổi và chứng mất trí, "Natural Missingness" xảy ra thường xuyên do bệnh nhân nhầm lẫn, không nhớ được thu nhập/học vấn quá khứ, hoặc từ chối/không thể hoàn thành bài kiểm tra nhận thức đầy đủ trong buổi thăm khám.
    * *Hành động:* **Không loại bỏ.** Sẽ xử lý bằng Median Imputation hoặc KNN Imputation để bảo toàn các bản ghi thể tích não (`nWBV`, `eTIV`) có giá trị cao của những lần thăm khám đó phục vụ baseline OLS.

#### Implement

1. Thiết kế và implement class `DataPipeline` có khả năng xử lý missing values, encoding, chuẩn hóa, `fit` trên tập train, `transform` trên tập test.
2. So sánh MAE, RMSE, $R^2$ trên tập test.
3. Dùng $k$-fold với $k = 5$ hoặc $k = 10$ để chọn hyperparameter $\lambda$ cho Ridge hoặc Lasso.
4. Thực hiện phân tích phần dư với mô hình tốt nhất, vẽ bốn đồ thị phân tích.
5. Khảo sát các feature bằng cách vẽ đồ thị normalized regression coefficient.
6. Rút ra các kết luận phù hợp.

#### Tiêu chí Chấp nhận

- [ ] **Pipeline**: Class `DataPipeline` phải xử lý đúng các phép biến đổi có trạng thái, đảm bảo `fit` chỉ được gọi trên tập train và `transform` được áp dụng cho tập test để tránh data leakage.
- [ ] **Preprocessing**: Pipeline phải xử lý thành công 5% missing values qua imputation và chuyển đổi tất cả các biến categorical sang định dạng số mà không cần làm sạch thủ công trước.
- [ ] **Tối ưu hóa Mô hình**: Hyperparameter $\lambda$ cho Ridge và Lasso phải được chọn dựa trên MSE/RMSE trung bình thấp nhất qua 5-fold hoặc 10-fold cross-validation.
- [ ] **Phân tích So sánh**: Báo cáo cuối cùng phải có bảng so sánh MAE, RMSE và $R^2$ của ít nhất ba mô hình khác nhau.
- [ ] **Tái tạo kết quả**: Chạy pipeline với `random_state` cố định phải cho ra kết quả giống nhau trên các máy khác nhau khi dùng môi trường `uv`.

### Báo cáo

Báo cáo bao gồm các mục sau:

1. Trang bìa
2. Mục lục
3. Phần 1: Lý thuyết và Trực quan hóa
4. Phần 2: Ứng dụng trên Dataset Thực tế
5. Kết luận
6. Tài liệu Tham khảo
7. Phụ lục

## Các Yêu cầu Khác

- Code Python phải được format và lint bằng `Black` và `Ruff`.
- Docstring coverage đạt 90%.
- Tất cả các đồ thị phải có tiêu đề, nhãn trục và chú giải.
- Giải thích tất cả các quyết định liên quan đến dữ liệu bằng toán học.
- Đặt `random_state` và `seed` cố định để có kết quả tái tạo được.
- Mỗi hàm cần ít nhất _hai_ unit test với các dataset đã biết trước.

## Giấy phép

Dự án này được cấp phép theo **giấy phép MIT**.

Xem [LICENSE](LICENSE.md) để biết thêm thông tin.

## Tài liệu Tham khảo

### Cài đặt Repo

#### README

- <https://www.makeareadme.com/>
- <https://utrechtuniversity.github.io/workshop-computational-reproducibility/chapters/readme-files.html>
- <https://github.com/matiassingers/awesome-readme>

#### Git Attributes

- <https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes>
- <https://github.com/gitattributes/gitattributes>
- <https://richienb.github.io/gitattributes-generator/>

#### `uv` — Package and Project Manager

- <https://docs.astral.sh/uv/>
- <https://docs.astral.sh/uv/getting-started/installation/>

#### `pytest` — Unit Testing

- <https://docs.pytest.org/en/stable/>
- <https://docs.pytest.org/en/stable/getting-started.html>

### Nền tảng Toán học và Lý thuyết

#### Seeing Theory (Đại học Brown)

- <https://seeing-theory.brown.edu/index.html>
  - Xem "Regression Analysis"

#### 3Blue1Brown

- <https://www.3blue1brown.com>
  - Xem các bài về Đại số Tuyến tính, Giải tích, Xác suất

#### StatQuest với Josh Starmer (YouTube)

- <https://www.youtube.com/@statquest/playlists>
  - Xem playlist "Linear Regression"

### Implement Source Code

#### NeuralNine: Linear Regression từ Đầu (Video & Code)

- <https://www.youtube.com/watch?v=jMpNdxnDaXg>

#### Kaggle: "Linear Regression From Scratch & Its Types" (Interactive Notebook)

- <https://www.kaggle.com/code/egazakharenko/linear-regression-from-scratch-its-types>

#### Stackademic: Ridge & Lasso từ Đầu

- <https://blog.stackademic.com/ridge-lasso-regression-from-scratch-bfd320ea3a83>

#### CodeSignal: Implement Multiple Linear Regression từ Đầu

- <https://codesignal.com/learn/courses/regression-and-gradient-descent/lessons/implementing-multiple-linear-regression-from-scratch>

#### The Algorithm - Python (Github)

- <https://github.com/TheAlgorithms/Python>

## Changelog

- 2026-05-01: Dự án được tạo.
