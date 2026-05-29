# Code Context — Blues Estimator Project

## Project Overview

**Purpose:** Pedagogical implementation of Best Linear Unbiased Estimators (OLS, Ridge, Lasso) from scratch in Python — no `sklearn` training wheels allowed for the core math. Part 1 builds the theory from Gauss-Markov proofs to diagnostic plots. Part 2 applies the pipeline to a real-world regression dataset (OASIS Brain MRI → predict MMSE cognitive score).

**Deadline:** May 30, 2026 (end of day).

**Python:** 3.12+ required. Managed by `uv`.

---

## Directory Layout

```
blues-estimator-project/
├── .github/
│   ├── ISSUE_TEMPLATE/          # bug_report.md, task-creation.md
│   ├── PULL_REQUEST_TEMPLATE.md # Checklist for PRs
│   └── workflows/ci.yml        # 3 jobs: lint, test, docstring-check
├── .internal_docs/
│   ├── data-sources.md          # ADR-001: Dataset selection (House Prices proposed, but OASIS used)
│   ├── projected-task-list.md   # Sprint 1 & 2 task breakdown
│   └── role-assignment.md       # Team roles: PO (TK), SM (LTD), Dev A (NPD), Dev B (CVC), Dev C (MPNH)
├── part1/
│   ├── assets/                  # Images for notebook
│   ├── cross_validation.py      # kfold_cv() — custom k-fold CV
│   ├── gauss_markov_util.py     # generate_synthetic_data() — Monte Carlo data gen
│   ├── ols_implementation.py    # Core: ols_fit, hat_matrix, model_metrics, coef_inference, vif
│   ├── part1_notebook.ipynb     # Theory demo notebook (exists, non-empty)
│   ├── residual_analysis.py     # residual_plots() — 4 diagnostic plots
│   └── ridge_lasso.py           # ridge_fit(), plot_ridge_trace()
├── part2/
│   ├── data/
│   │   ├── data.csv             # OASIS dataset (not the House Prices from ADR)
│   │   └── oasis_longitudinal.csv
│   ├── advanced_methods.py      # Stub class: AdvancedMethods (Kernel, Bayesian, RF)
│   ├── data_pipeline.py         # DataPipeline class (fit/transform, imputation, encoding)
│   ├── model_comparison.py      # ModelComparison class (stub methods + some real implementation)
│   └── part2_notebook.ipynb     # Application notebook (exists, non-empty)
├── report/
│   ├── chapters/                # LaTeX chapter files
│   ├── images/                  # Report figures
│   ├── report.pdf               # Final compiled PDF
│   ├── report.tex               # Main LaTeX source
│   ├── reportDoCS2025.cls       # Custom LaTeX class
│   └── utilities.tex            # LaTeX helpers
├── tests/
│   ├── test_coef_inference.py   # 5 tests — SE, t-stat, p-values, CI, dim mismatch
│   ├── test_data_pipeline.py    # 3 tests (1 skipped) — structural integrity, imputation
│   ├── test_gauss_markov.py     # 7 tests — rank, intercept, noise mean/var, shapes, identity
│   ├── test_kfold_cv.py         # 6 tests — splits, perfect fit, k errors, reproducibility, noise
│   ├── test_metrics.py          # 5 tests — perfect fit, sklearn parity, adj R2, F-stat, DoF
│   ├── test_ols.py              # 6 tests — parameter recovery, sigma², metrics, sklearn parity, edge cases
│   ├── test_placeholder.py      # Single placeholder test (passes)
│   ├── test_properties.py       # Hat matrix: idempotency, symmetry, trace, bounds, collinearity, fat matrix, zeros
│   ├── test_residual_analysis.py# 6 tests — figure structure, residual properties, standardized residuals, perfect fit, Cook's threshold, high leverage
│   ├── test_ridge_fit.py        # 6+3 tests — lambda=0→OLS, sklearn parity, shrinkage, large lambda, multicollinearity, intercept invariance + plot tests
│   └── test_vif.py              # 5 tests — independence, multicollinearity, statsmodels parity, perfect collinearity, empty
├── main.py                      # Entry point: prints "Hello from blues-estimator-project!"
├── pyproject.toml               # Project config, dependencies, ruff/black/interrogate settings
├── README.md                    # Full project docs
├── CONTRIBUTING.md              # Branch + commit conventions, dev workflow
├── pyproject.toml               # (duplicate key — already listed)
├── .pre-commit-config.yaml      # pre-commit: black, ruff --fix, interrogate
├── .python-version              # "3.12"
├── LICENSE                      # MIT
└── uv.lock                      # Locked dependencies
```

---

## Key Files and Code

### 1. `pyproject.toml` (lines 1-44)

- **Dependencies:** `matplotlib>=3.10.9`, `numpy>=2.4.4`, `pandas>=3.0.2`, `scipy>=1.17.1`, `seaborn>=0.13.2`
- **Dev dependencies:** `black>=26.3.1`, `interrogate>=1.7.0`, `jupyter>=1.1.1`, `pre-commit>=4.6.0`, `pytest>=9.0.3`, `ruff>=0.15.12`, `scikit-learn>=1.8.0`, `statsmodels>=0.14.6`
- **Black:** target-version py312
- **Ruff:** lint select D2, D4; pydocstyle convention = "numpy"
- **Interrogate:** fail-under 90%, exclude `setup.py`, `tests`, `docs`

### 2. `part1/ols_implementation.py` (236 lines)

Core OLS math module containing:
- **`ols_fit(X, y)`** — Prepends intercept, computes β̂ = pinv(XᵀX)Xᵀy, returns σ̂² = RSS/(n-p-1). Raises `ValueError` if n ≤ p, empty input, or shape mismatch.
- **`hat_matrix(X)`** — Returns H = X·pinv(XᵀX)·Xᵀ. Validated for empty/2D.
- **`model_metrics(y, y_hat, p)`** — Returns dict: RSS, TSS, R², Adjusted_R², Adj_R² (alias), F_statistic, F_stat (alias), MAE, RMSE.
- **`coef_inference(X, y, beta_hat, sigma2)`** — SE = √(σ̂²·diag(pinv(XᵀX))), t-stats, p-values (2-sided), 95% CI via t-distribution.
- **`vif(X)`** — VIF = 1/(1-R²ⱼ). Handles centered/uncentered TSS (detects constant column). Uses `lstsq` for numerical parity with `statsmodels`.

### 3. `part1/ridge_lasso.py` (108 lines)

- **`ridge_fit(X, y, lam)`** — β̂ = solve(XᵀX + λ·I*, Xᵀy) where I*[0,0]=0 (intercept not penalized). Returns (p+1,) coefficients.
- **`plot_ridge_trace(X, y, lambdas)`** — Plots coefficient paths over λ on log scale. Returns matplotlib Figure.

### 4. `part1/residual_analysis.py` (111 lines)

- **`residual_plots(X, y, beta_hat)`** — 4 diagnostic plots in 2×2 layout:
  1. Residuals vs Fitted
  2. Normal Q-Q (standardized residuals)
  3. Scale-Location (√|std residuals| vs fitted)
  4. Cook's Distance with threshold at 4/n

### 5. `part1/cross_validation.py` (70 lines)

- **`kfold_cv(X, y, k=5, random_state=None)`** — Shuffles indices, splits into k folds, trains OLS on k-1 folds, evaluates on held-out fold. Returns (mse_array, cv_score).

### 6. `part1/gauss_markov_util.py` (50 lines)

- **`generate_synthetic_data(n, beta, sigma, rng)`** — Generates X ∼ N(0,1) with intercept column, y = Xβ + ε with ε ∼ N(0, σ²). Satisfies GM1-GM5 assumptions.

### 7. `part2/data_pipeline.py` (345 lines)

- **`DataPipeline`** class with fit/transform/fit_transform/preprocess lifecycle.
  - `fit(X)`: Learns numeric means/stds, categorical modes, SES-by-EDUC group medians.
  - `transform(X)`: Applies handle_missing_values → encode_categorical → (scale_features commented out).
  - `preprocess(file_path, target_column)`: Loads, deduplicates, drops target NaNs, filters to required columns, splits (train_test_split with group shuffle), fits on train, transforms both.
  - **Notable:** `handle_missing_values` has a bug — it references `self.categorical_values_.get("ses_by_educ", {})` but `fit()` stores the median dict in `self.ses_educ_medians_`, not in `categorical_values_`.
  - `scale_features` is commented out in `transform()` but test expects it active.

### 8. `part2/model_comparison.py` (178 lines)

- **`ModelComparison`** class: stubs for `train_linear_regression`, `train_ridge_regression`, `train_lasso_regression`, `train_elasticnet_regression`, `evaluate_model`, `compare_metrics`, `generate_summary`. All methods currently `pass`.

### 9. `part2/advanced_methods.py` (110 lines)

- **`AdvancedMethods`** class: stubs for `train_kernel_regression`, `train_bayesian_regression`, `train_random_forest_regressor`, `evaluate_advanced_models`. All methods `pass`.

---

## Architecture and Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        part1/                                  │
│  ols_fit ← hat_matrix ← model_metrics ← coef_inference ← vif  │
│     ↕                          ↕                              │
│  ridge_fit  ←──────────────── kfold_cv                        │
│  (ridge_lasso.py)             (cross_validation.py)            │
│     ↕                                                         │
│  residual_plots (residual_analysis.py) ← uses hat_matrix      │
│     ↕                                                         │
│  gauss_markov_util.py — generates synthetic data for Monte    │
│                        Carlo simulation                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                        part2/                                  │
│  DataPipeline.preprocess() → (X_train, X_test, y_train,        │
│                                y_test)                          │
│       ↓                                                        │
│  ModelComparison (stub) — train + evaluate multiple models     │
│       ↓                                                        │
│  AdvancedMethods (stub) — optional kernel/bayesian/RF          │
└────────────────────────────────────────────────────────────────┘

Tests import from PROJECT_ROOT via sys.path.append(str(Path(__file__).resolve().parent.parent))
```

**Key conventions:**
- `ols_fit` and `ridge_fit` **prepend intercept internally** — callers pass raw X without intercept column.
- `residual_plots` and `coef_inference` expect X to **already include the intercept column** (augmented matrix).
- `kfold_cv` calls `ols_fit` internally (so X should be raw, no intercept).
- All functions use `np.linalg.pinv` or `np.linalg.solve` for numerical stability.

---

## Git State

- **Current branch:** `main` (up to date with `origin/main`)
- **Working tree:** clean, nothing to commit
- **Remote origin:** `https://github.com/TanKhoiTV/blues-estimator-project.git`
- **Author:** Tran Van Tan Khoi <tranvantankhoi@gmail.com>
- **Last commit:** `939cc16 feat(part1): complete theory and code demo in notebook for part 1 (#71)`

### Remote Feature Branches (4 available, all diverge from main)

| Branch | Latest Commit | Delta from main |
|--------|--------------|-----------------|
| `origin/feat/advanced-method` | `9fc7db3` — feat: implement advanced methods | Heavy changes in `advanced_methods.py`, `data_pipeline.py`, notebooks, tests |
| `origin/feat/advanced-regression-methods` | `24e7a7d` — feat: advanced regression methods | Changes in `advanced_methods.py`, `model_comparison.py`, notebooks, tests |
| `origin/feat/ols-baseline-model` | `4580093` — fix: black formatting | Changes in `model_comparison.py`, tests, notebooks (3.2k insertions) |
| `origin/feat/ridge-lasso-cv-lambda-selection` | `1713dfd` — feat: model_comparison validation | Changes only in `model_comparison.py` (+454/-51 lines) |

**None of the 4 feature branches have been merged to main.** They represent Part 2 development work that is still in progress.

### Branch Naming Convention (from CONTRIBUTING.md)

`<type>/<short-description>` with hyphens: `feat/`, `fix/`, `refactor/`, `docs/`, `test/`, `chore/`, `ci/`

### Commit Convention (Conventional Commits)

`<type>[optional scope]: <short description>` — lowercase, imperative, no trailing period.

**Scopes used:** `math`, `stat`, `pipe`, `viz`, `proof`, `report`, `github`, `ci`, `part1`, `part2`

Breaking changes: `!` before colon + BREAKING CHANGE in body.

---

## Testing

- **Test framework:** `pytest` (68 tests collected, all pass on main)
- **Import pattern:** `sys.path.append(str(PROJECT_ROOT))` before importing from `part1.*`
- **Floating point:** `pytest.approx()` and `np.testing.assert_array_almost_equal` / `assert_allclose`
- **Matplotlib tests:** Use `matplotlib.use("Agg")` at top of test file; close figures with `plt.close(fig)` in `finally` blocks
- **Skipped tests:** `test_data_pipeline.py::test_no_data_leakage_in_scaling` — skipped because `scale_features` is commented out in `transform()`
- **Coverage threshold:** 90% (enforced by `interrogate`, checked in CI)
- **CI:** 3 jobs on push/PR to main: lint (black —check + ruff check), test (pytest), docstring-check (interrogate)

---

## Coding Conventions

1. **Docstrings:** NumPy convention, enforced by Ruff D2/D4 ruleset
   - Summary in imperative mood (single sentence)
   - Sections: `Parameters`, `Returns`, `Raises` with `----------` underline
   - Each param: `name : type` on one line, description indented below
2. **Formatting:** Black (target-version py312), all code must pass `black --check`
3. **Linting:** Ruff (D2, D4 rules), run `ruff check --fix`
4. **Docstring coverage:** ≥ 90% (excludes tests, setup.py, docs)
5. **Random seeds:** Always set explicit `random_state` / `np.random.seed()` for reproducibility
6. **Imports:** Project modules import via `from part1.<module> import <func>` pattern (relies on project root being in sys.path or working directory)
7. **Pre-commit:** Optional hook runs black, ruff --fix, interrogate on each commit

---

## Known Issues / Risks

1. **DataPipeline bug:** `handle_missing_values()` references `self.categorical_values_.get("ses_by_educ", {})` but `fit()` stores SES-EDUC medians in `self.ses_educ_medians_` — so the fallback to global median always triggers, bypassing the group-median logic.
2. **Feature scaling disabled:** `scale_features()` is fully implemented but commented out in `transform()`. The corresponding test is `@unittest.skip`.
3. **ModelComparison and AdvancedMethods are stubs** — most methods are `pass`. Feature branches contain real implementations but are not merged.
4. **Dataset mismatch:** ADR-001 proposed House Prices dataset, but the actual data in `part2/data/` is the OASIS longitudinal MRI dataset. The OASIS dataset has n=373 and features matching the README description.
5. **Part 2 notebook** exists but may be stale relative to the feature branches.

---

## Start Here

If you are new to the project and need to understand the core:

1. **`part1/ols_implementation.py`** — the heart of the project. Read `ols_fit` first, then `model_metrics`, then `coef_inference`. These form the mathematical foundation that everything else (ridge, CV, residual plots, VIF) builds on.

2. **`tests/test_ols.py`** — shows how each function is validated, the synthetic data generation pattern, and the sklearn parity assertions.

3. **`pyproject.toml`** — understand the dependency and tooling setup before making any changes.

4. **`README.md`** — full project roadmap, acceptance criteria, and dataset details.

For Part 2, start with **`part2/data_pipeline.py`** — it's the most complex file with the known bug mentioned above.

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `uv sync` | Install/update dependencies |
| `uv run pytest` | Run all tests |
| `uv run pytest tests/test_ols.py -v` | Run specific test file verbose |
| `uv run black .` | Format all Python files |
| `uv run ruff check --fix .` | Lint and auto-fix |
| `uv run interrogate .` | Check docstring coverage |
| `uv run pre-commit install` | Install git pre-commit hook |
| `uv run jupyter notebook` | Launch Jupyter |
| `uv run python main.py` | Run entry point |
