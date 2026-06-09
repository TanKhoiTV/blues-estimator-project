# Contributing cho Dự án Blues Estimator

Để bắt đầu contributing, hãy làm theo các bước dưới đây.

> [!NOTE]
> Một số lệnh dưới đây có tham số dấu chấm `.` chỉ thư mục hiện tại. Trong hầu hết các trường hợp, hãy chạy các lệnh này từ thư mục gốc và chú ý không bỏ sót dấu chấm đó.

## Clone Repository

Clone repository bằng phương thức tuỳ chọn.

## Cài đặt Dependencies

### Cài đặt `uv`

Dự án này sử dụng `uv` để quản lý package nhanh và có tính tái tạo cao. Cài đặt bằng lệnh sau:

* **macOS/Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

* **Windows (PowerShell)**:

```PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

* **Cách thay thế (pip)**:

```bash
pip install uv
```

### Setup và Sync Dependencies

Sau khi đã cài đặt `uv`, bạn không cần tạo virtual environment thủ công bằng `venv`.
Lệnh `sync` xử lý việc tạo môi trường, quản lý phiên bản Python và cài đặt package trong một bước duy nhất.

Từ thư mục gốc của repository, chạy:

```bash
uv sync
```

Tùy chọn, bạn có thể cài đặt **pre-commit hook** để tự động format và lint code mỗi lần commit. Xem phần [**Thực hiện Commit**](#thực-hiện-commit) ở cuối tài liệu. Bước này chỉ cần cài một lần.

```bash
uv run pre-commit install
```

### Chạy Code

Để đảm bảo bạn đang dùng môi trường biệt lập của dự án, hãy thêm tiền tố `uv run` trước các lệnh hoặc activate môi trường thủ công:

* **Chạy một script**: `uv run python main.py`
* **Khởi động Jupyter**: `uv run jupyter notebook`
* **Activate thủ công**: `source .venv/bin/activate` (macOS/Linux) hoặc `.venv\Scripts\activate` (Windows)

## Sync Repository

Nếu bạn đã làm việc với repo này trước đây, hãy sync nhánh `main` local với remote bằng:

```bash
git fetch --all
git checkout main
git pull origin main --rebase
```

Nếu bạn có working branch riêng, hãy sync nó sau khi đã sync nhánh `main`:

```bash
git checkout <tên-nhánh-của-bạn>
git rebase main
```

## Tạo Working Branch

Dự án có rule bảo vệ nhánh `main` khỏi commit trực tiếp, vì vậy bạn phải tạo branch riêng để contributing.

> [!IMPORTANT]
> Hãy nhớ **sync** nhánh `main` local trước! Sau đó từ nhánh `main` local:

```bash
git checkout -b <tên-nhánh-của-bạn>
```

Mỗi branch chỉ nên do một người sở hữu, đó là _bạn_. Đừng để ai khác commit lên branch của bạn. Nếu cần hỗ trợ, hãy cân nhắc chia nhỏ công việc thành các branch nhỏ hơn.

## Quy ước Đặt tên Branch và Commit

### Branch

Tên branch tuân theo pattern `<type>/<mô-tả-ngắn>`, trong đó các từ được phân tách bằng dấu gạch ngang.

```bash
feat/sql-parser
fix/crc32-overflow
chore/pr-templates
ci/release-workflow
```

**Các type:**

| Type       | Dùng cho                                                    |
| ---------- | ----------------------------------------------------------- |
| `feat`     | Tính năng hoặc khả năng mới                                 |
| `fix`      | Sửa lỗi                                                     |
| `refactor` | Tái cấu trúc code mà không thay đổi hành vi                 |
| `docs`     | Chỉ thay đổi tài liệu                                       |
| `test`     | Thêm hoặc cập nhật test                                     |
| `chore`    | Bảo trì repo/dự án (config, template, tooling)              |
| `ci`       | Thay đổi CI/CD pipeline và automation                       |

---

### Commit

Commit tuân theo [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>[scope tùy chọn]: <mô tả ngắn>
```

Phần mô tả viết thường, dùng thì mệnh lệnh và không có dấu chấm ở cuối.

```bash
feat(sql): add recursive descent parser for SELECT
fix(kv): handle CRC-32 overflow on large entries
docs: document Schema::compute_metadata contract
chore(github): add issue and PR templates
ci(build): pin actions/checkout to SHA
```

**Scope** là tùy chọn nhưng được khuyến khích dùng cho codebase lớn. Dùng tên module hoặc subsystem: `core`, `kv`, `table`, `sql`, `ci`, `github`.

**Breaking changes** được đánh dấu bằng `!` trước dấu hai chấm và được giải thích trong phần thân commit:

```bash
feat(sql)!: replace token enum with std::variant

BREAKING CHANGE: TokenKind is no longer a plain enum.
Callers must update match arms to use the new variant types.
```

### Scopes

| Scope    | Dùng cho                                                                      |
| :------- | :---------------------------------------------------------------------------- |
| `math`   | Implement logic OLS, Normal Equations, hoặc Hat Matrix cốt lõi               |
| `stat`   | Kiểm định thống kê (t-test, p-values, F-test) và toán phân phối               |
| `pipe`   | Class `DataPipeline`, logic imputation và encoding                            |
| `viz`    | Code vẽ đồ thị (Residual plots, EDA heatmaps, feature importance)             |
| `proof`  | Mô phỏng Monte Carlo hoặc minh họa định lý Gauss-Markov                       |
| `report` | Thay đổi các file LaTeX/Markdown trong thư mục `report/`                      |

## Thực hiện Commit

### Format và Lint Local

Dùng `Black` và `Ruff` (đã cài cùng `uv`), chạy các lệnh sau từ thư mục gốc để sửa các vấn đề về style:

```bash
uv run black .
uv run ruff check --fix .
```

Nếu không muốn chạy thủ công mỗi lần, hãy cài pre-commit hook. Chạy lệnh này một lần, tốt nhất là ngay sau khi clone repo:

```bash
uv run pre-commit install
```

### Quy tắc Docstring

Docstring phải tuân theo **quy ước NumPy** và được kiểm tra bởi ruleset `D` của Ruff. Kiểm tra này chạy tự động cùng với các lint rule khác khi bạn chạy:

```bash
uv run ruff check --fix .
```

Một docstring hợp lệ tối thiểu trông như sau:

```python
def fit(X: np.ndarray, y: np.ndarray) -> None:
    """Fit the OLS model using the normal equations.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix.
    y : np.ndarray of shape (n_samples,)
        Response vector.

    Returns
    -------
    None
        Coefficients are stored in ``self.beta_``.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have incompatible shapes.
    """
```

Các quy tắc cần lưu ý:

* Dòng tóm tắt là một câu duy nhất ở **thì mệnh lệnh**: "Fit", "Compute", "Return", không phải "Fits", "Computes", "Returns".
* Để một dòng trống giữa phần tóm tắt và section `Parameters`.
* Tiêu đề section (`Parameters`, `Returns`, `Raises`) được theo sau bởi gạch chân `----------` có độ dài khớp chính xác với tiêu đề.
* Mỗi parameter được viết dạng `tên : kiểu` trên một dòng, với mô tả thụt lề ở dòng tiếp theo.

### Kiểm tra Docstring Coverage

Dùng `interrogate` để kiểm tra docstring coverage có đạt ngưỡng tối thiểu không. Từ thư mục gốc:

```bash
uv run interrogate .
```

Lệnh này chạy tự động mỗi khi bạn commit ở verbosity cấp 1 (hiển thị coverage theo từng file) nếu bạn đã cài **pre-commit hook**.

Để xem các hàm nào đang thiếu docstring, chạy với verbosity cao hơn:

```bash
uv run interrogate -v 2 .
```

### Chạy Tests

Nếu code của bạn có test liên quan, hãy chạy chúng.
Nếu chưa có test nào, hãy tự thêm vào trong cùng commit.

Dùng `pytest.approx()` từ `pytest` để so sánh số dấu phẩy động.

Để chạy toàn bộ test suite:

```bash
uv run pytest
```

Để chạy test cho một phần cụ thể của dự án:

```bash
uv run pytest tests/test_file.py
```

### Commit Thay đổi của Bạn

Tuân theo hướng dẫn đặt tên commit ở trên.

> [!NOTE]
> Nếu bạn đã cài **pre-commit** hook, commit có thể thất bại do các pre-commit check. Nếu nguyên nhân là format và lint, chỉ cần commit lại. Nếu nguyên nhân là docstring coverage, hãy thêm docstring trước khi thử lại.

## Issues và Pull Requests

### Làm việc với Issues

Có hai template được cấu hình sẵn để chọn khi mở issue mới.

#### Tạo Task

Loại này định nghĩa một task mới cho team thực hiện, giúp dự án hoàn thành lộ trình. Khi mở issue _tạo task_, có một số phần cần điền:

* **Description**: Mô tả rõ ràng và súc tích về task.
* **Definition of Done**: Mô tả rõ ràng và súc tích về những gì cần có để task được coi là _xong_. Nên tạo sub-issue để hỗ trợ hoàn thành task.
* **Proposed Implementation**: Giải thích cách thực hiện task. Đề xuất các file bị ảnh hưởng.
* **Additional Context**: Thêm thông tin hoặc ảnh chụp màn hình nếu cần.

#### Bug Report

Dùng template này khi phát hiện các vấn đề cần giải quyết.

* **Describe the bug/problem**: Mô tả rõ ràng và súc tích về lỗi.
* **To Reproduce**: Các bước cần thiết để tái hiện lỗi nếu cần.
* **Expected behavior**: Mô tả rõ ràng và súc tích về điều bạn mong đợi sẽ xảy ra.
* **Screenshots**: Nếu có, thêm ảnh chụp màn hình để giải thích vấn đề.
* **Additional Context**: Thêm thông tin về vấn đề, ví dụ vị trí có thể xảy ra lỗi.

#### Blank Issue

Chỉ dùng loại này nếu issue của bạn thực sự không phải task cũng không phải bug — ví dụ, một câu hỏi về định hướng dự án hoặc vấn đề về quy trình. Nếu bất kỳ template nào phù hợp dù chỉ một phần, hãy ưu tiên dùng nó.

> [!NOTE]
> **Feature request** nằm ngoài phạm vi dự án này. Nếu có đề xuất, hãy nêu lên trong kênh nhóm. Nó có thể được xem xét cho phiên bản tương lai.

Dù chọn template nào, hãy càng cụ thể càng tốt. Sử dụng tag phù hợp để phân loại issue.

### Tạo Pull Requests

Nhánh `main` remote được bảo vệ khỏi việc merge trực tiếp. Để merge working branch của bạn, hãy tạo một **pull request**. Làm theo các bước sau để có PR thành công!

#### Publish Working Branch

Nếu chưa làm, hãy publish working branch của bạn lên remote.

> [!IMPORTANT]
> Nếu bạn chưa là collaborator của repository, hãy fork nó trước và publish branch lên fork đó. Mở PR từ branch trên fork của bạn nhắm vào nhánh `main` của upstream.

Checkout branch của bạn bằng `git checkout <tên-nhánh>`, sau đó:

```bash
git push -u origin <tên-nhánh>
```

Flag `-u` thiết lập upstream tracking, để các lệnh `git push` và `git pull` sau này trên branch đó hoạt động mà không cần chỉ định remote.

> [!NOTE]
> Tài liệu tham khảo Git nhanh:
> **Tạo Branch Local và Publish lên Remote**
>
> ```bash
> git checkout -b <branch>      # tạo + chuyển sang
> git push -u origin <branch>   # publish + thiết lập tracking
> ```
>
> **Xem các branch**
>
> ```bash
> git branch                    # local branches
> git branch -r                 # remote branches
> git branch -a                 # cả hai
> git branch -vv                # local branches + remote tracking của chúng
> ```
>
> **Thiết lập/gỡ bỏ tracking**
>
> ```bash
> git branch -u origin/<tên-nhánh>      # thiết lập upstream cho branch hiện tại
> git branch -u origin/<remote> <local> # thiết lập upstream cho branch cụ thể
> git branch --unset-upstream           # gỡ bỏ cho branch hiện tại
> ```
>
> **Push, Pull, Fetch**
>
> ```bash
> git fetch                     # tải thay đổi remote, không merge
> git fetch origin <branch>     # chỉ fetch một branch cụ thể
> git pull                      # fetch + merge (hoặc rebase nếu đã cấu hình)
> git pull --rebase             # fetch + rebase thay vì merge
> git push                      # push branch hiện tại lên upstream
> git push origin <branch>      # push một branch cụ thể
> git push --force-with-lease   # force push, nhưng hủy nếu remote có commit mới
> ```
>
> **Xóa branch**
>
> ```bash
> git branch -d <branch>        # xóa local (an toàn — từ chối nếu chưa merge)
> git branch -D <branch>        # force xóa local
> git push origin -d <branch>   # xóa remote
> ```
>

#### Tạo Pull Request

Khi code sẵn sàng để review, mở PR nhắm vào nhánh `main` bằng PR Template được cung cấp.

* **Cập nhật branch của bạn** thường xuyên để tránh merge conflict, đặc biệt ngay sau khi mở PR.
* **Giữ PR tập trung** vào một tính năng hoặc fix duy nhất để review thuận tiện hơn.
* **Reference issue** bằng `Closes #<số-issue>` trong phần mô tả PR.
* **Assign reviewer** để PR được review nhanh hơn.
* **Giải quyết tất cả CI check** và các conversation để PR được accept.
  Nếu một check thất bại, hãy sửa local theo phần liên quan trong hướng dẫn này ([Format và Lint Local](#format-và-lint-local), [Kiểm tra Docstring Coverage](#kiểm-tra-docstring-coverage), [Chạy Tests](#chạy-tests)), sau đó push lại.
* **Xóa local branch** sau khi merge thành công. GitHub sẽ tự động xóa remote branch. Để dọn dẹp local:

```bash
  git checkout main
  git branch -d <tên-nhánh>
```
