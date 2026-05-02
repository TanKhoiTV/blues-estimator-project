# Blues Estimator Project

Implementation of Best Linear Unbiased Estimators from scratch. Including OLS, Ridge, and Lasso without 'sklearn' training wheels. From Gauss-Markov proofs to real-world data pipelines, the math is rhythmic, and the residuals have soul. 100% manual, 0% mechanical failure.

## Quick Start

Follow these steps to get started. See [CONTRIBUTING](CONTRIBUTING.md) for the complete guide.

1. Clone the repository.
2. Install `uv`.
3. Install dependencies with `uv sync`.
4. Run any code with `uv run`.
5. Run `uv run pytest` to run all test scripts.

## Project Structure

```bash
blues-estimator-project         # To be replaced with Group_<ID>
├── report/
│   ├── report.pdf              # The final PDF (LaTeX/Markdown)  
│   └── report.tex              # Source file for the report
├── part1/
│   ├── ols_implementation.py   # Core OLS & Math from scratch
│   ├── ridge_lasso.py          # Regularization logic
│   ├── residual_analysis.py    # Code for diagnostic plots
│   ├── cross_validation.py     # k-fold CV implementation
│   └── part1_notebook.ipynb    # Theoretical demo & proofs
├── part2/
│   ├── data/
│   │   └── dataset.csv         # Original real-world data
│   ├── data_pipeline.py        # DataPipeline class for cleaning
│   ├── model_comparison.py     # Comparison of 3+ models
│   ├── advanced_methods.py     # Bonus: Kernel/Bayesian
│   └── part2_notebook.ipynb    # Real-world application & discussion
├── README.md                   # Project overview & usage instructions
├── pyproject.toml              # Package dependencies "Source of Truth"
└── requirements.txt            # Package dependencies, exported from `uv`
```

## Project Roadmap

### Part 1

#### Implementation

For each function in the list:

1. Present the mathematical formula.
2. Implement with vanilla Python.
3. Demonstrate with test data.
4. Test and compare with `NumPy` or `sklearn`.

Functions to be implemented:

```bash
ols_fit(X, y)                   # Compute OLS solution (beta_hat) and Residual Variance Estimator (squared_sigma_hat)
hat_matrix(X)                   # Compute matrix H and check for idempotency
model_metrics(y, y_hat, p)      # Compute various metrics
coef_inference(X, y, beta_hat, sigma2)  # Compute SE, t-stat, p-value and CI
vif(X)                          # Compute VIF
ridge_fit(X, y, lam)            # Compute Ridge Regression, draw Ridge Trace
residual_plots(X, y, beta_hat)  # Draw four residual plots
kfold_cv(X, y, k)               # k-fold cross-validation, compute CV score
```

Lastly, demonstrate Gauss-Markov by simulating Monte Carlo to verify **unbiasedness** ($\mathbb{E}[\hat{\beta}] = \beta$) and **minimum variance**.

#### Acceptance Criteria

- [ ] **Mathematical Parity**: All custom functions (e.g., `ols_fit`, `ridge_fit`) must produce coefficients that match `scikit-learn` or `statsmodels` output with a tolerance of $\varepsilon = 10^{-6}$.
- [ ] **Proof of Property**: The Monte Carlo simulation must successfully demonstrate that the OLS estimator is unbiased ($\mathbb{E}[\hat{\beta}] = \beta$) and maintains the lowest variance among linear unbiased estimators as per Gauss-Markov.
- [ ] **Diagnostic Integrity**: The `residual_plots` function must generate all four standard plots (Residuals vs Fitted, Normal Q-Q, Scale-Location, and Residuals vs Leverage) with proper axis labeling.
- [ ] **Unit Testing**: Each mathematical function must pass at least two unique unit tests using both synthetic and known baseline datasets.
- [ ] **Matrix Validation**: The `hat_matrix` function must numerically verify idempotency ($H^2 \approx H$) and symmetry ($H^T \approx H$).

### Part 2

#### Implementation

1. Design and implement `DataPipeline` class that can handle missing values, encoding, normalization, `fit` on train data, `transform` on test data.
2. Compare MAE, RMSE, $R^2$ on test dataset.
3. Use $k$-fold with $k = 5$ or $k = 10$ to pick hyperparameter $\lambda$ for Ridge or Lasso.
4. Perform residual analysis with the best model, drawing four analysis plots.
5. Investigate features by drawing normalized regression coefficient plot.
6. Draw appropriate conclusions.

#### Acceptance Criteria

- [ ] **Pipeline Encapsulation**: The `DataPipeline` class must correctly handle stateful transformations, ensuring that `fit` is only called on training data and `transform` is applied to test data to prevent data leakage.
- [ ] **Preprocessing Robustness**: The pipeline must successfully handle 5% missing values via imputation and convert all categorical variables into numerical formats without manual pre-cleaning.
- [ ] **Model Optimization**: Hyperparameter $\lambda$ for Ridge and Lasso must be selected based on the lowest average MSE/RMSE across a 5-fold or 10-fold cross-validation split.
- [ ] **Comparative Analysis**: The final report must include a side-by-side comparison table of MAE, RMSE, and $R^2$ for at least three distinct models.
- [ ] **Reproducibility**: Running the pipeline with a fixed `random_state` must produce identical results across different machines using the `uv` environment.

### Report

The report should include the following sections:

1. Cover page
2. Table of Contents
3. Part 1: Theory and Visualization
4. Part 2: Application on a Real-world Dataset
5. Conclusions
6. Bibliography and References
7. Appendices

## Other Specifications

- Python code must be formatted and linted with `Black` and `Ruff`
- Docstring coverage reaches 90%.
- All plots drawn must include a title, axis labels, and legends.
- Explain all data-related decisions with math.
- Set static `random_state` and `seed` to get reproducible results.
- Each function needs at least _two_ unit tests with known datasets.

## License

This project is licensed under the **MIT license**.

See [LICENSE](LICENSE.md) for more information.

## References

### Repo setup

#### README

- <https://www.makeareadme.com/>
- <https://utrechtuniversity.github.io/workshop-computational-reproducibility/chapters/readme-files.html>
- <https://github.com/matiassingers/awesome-readme>

#### Git Attributes

- <https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes>
- <https://github.com/gitattributes/gitattributes>
- <https://richienb.github.io/gitattributes-generator/>

#### `uv` Package and Project Manager

- <https://docs.astral.sh/uv/>
- <https://docs.astral.sh/uv/getting-started/installation/>

#### `pytest` unit testing

- <https://docs.pytest.org/en/stable/>
- <https://docs.pytest.org/en/stable/getting-started.html>

### Math and Theory Foundation

#### Seeing Theory (Brown University)

- <https://seeing-theory.brown.edu/index.html>
  - See "Regression Analysis"

#### 3Blue1Brown

- <https://www.3blue1brown.com>
  - See lessons in Linear Algebra, Calculus, Probability

#### StatQuest with Josh Starmer (YouTube)

- <https://www.youtube.com/@statquest/playlists>
  - See "Linear Regression" playlist

### Source Code Implementation

#### NeuralNine: Linear Regression from Scratch (Video & Code)

- <https://www.youtube.com/watch?v=jMpNdxnDaXg>

#### Kaggle: "Linear Regression From Scratch & Its Types" (Interactive Notebook)

- <https://www.kaggle.com/code/egazakharenko/linear-regression-from-scratch-its-types>

#### Stackademic: Ridge & Lasso From Scratch

- <https://blog.stackademic.com/ridge-lasso-regression-from-scratch-bfd320ea3a83>

#### CodeSignal: Implementing Multiple Linear Regression from Scratch

- <https://codesignal.com/learn/courses/regression-and-gradient-descent/lessons/implementing-multiple-linear-regression-from-scratch>

#### The Algorithm - Python (Github)

- <https://github.com/TheAlgorithms/Python>

## Changelog

- 2026-05-01: Project created.
