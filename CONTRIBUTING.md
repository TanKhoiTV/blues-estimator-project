# Contributing to Blues Estimator Project

To start contributing, follow the steps.

> [!NOTE]
> Some commands shown below have a dot `.` parameter indicating the current directory. In most cases run these commands from root and be mindful to not miss it.

## Clone the Repository

Clone the repository using your preferred method.

## Install Dependencies

### Install `uv`

This project uses `uv` for extremely fast, reproducible package management. Install it using the following command:

* **macOS/Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

* **Windows (PowerShell)**:

```PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

* **Alternative (pip)**:

```bash
pip install uv
```

### Setup and Sync Dependencies

Once `uv` is installed, you do not need to create a virtual environment manually using `venv`.
The `sync` command handles environment creation, Python versioning, and package installation in one step.

From the root of the repository, run:

```bash
uv sync
```

Optionally, you can install the **pre-commit hook** to automatically format and lint your code every time you commit. See [**Making a Commit** section](#making-a-commit) at the end. This requires a separate one-time installation step.

```bash
uv run pre-commit install
```

### Running Code

To ensure you are using the project's isolated environment, prefix your commands with `uv run` or activate the environment:

* **Run a script**: `uv run python main.py`
* **Launch Jupyter**: `uv run jupyter notebook`
* **Manual Activation**: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

## Sync the Repository

If you worked on this repo before, sync your local `main` branch to the upstream branch with:

```bash
git fetch --all
git checkout main
git pull origin main --rebase
```

If you have a working branch, sync it as well, after syncing the `main` branch:

```bash
git checkout <your-branch-name>
git rebase main
```

## Create Your Working Branch

This project has a rule to protect the `main` branch from direct commits, so you have to create your own branch if you want to contribute.

> [!IMPORTANT]
> Remember to **sync** your local `main` branch first! Then from the local `main` branch:

```bash
git checkout -b <your-branch-name>
```

Your branches should only be owned by one person, and that is _you_. Don't let anybody commit onto them. If you find yourself needing some help with your branches, consider splitting the work across smaller branches.

## Branch and Commit Naming

### Branches

Branch names follow the pattern `<type>/<short-description>`, where words are separated by hyphens.

```bash
feat/sql-parser
fix/crc32-overflow
chore/pr-templates
ci/release-workflow
```

**Types:**

| Type       | Use for                                               |
| ---------- | ----------------------------------------------------- |
| `feat`     | New feature or capability                             |
| `fix`      | Bug fix                                               |
| `refactor` | Code restructuring with no behavior change            |
| `docs`     | Documentation only                                    |
| `test`     | Adding or updating tests                              |
| `chore`    | Repo/project maintenance (config, templates, tooling) |
| `ci`       | CI/CD pipeline and automation changes                 |

---

### Commits

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
<type>[optional scope]: <short description>
```

The description is lowercase, imperative mood, and has no trailing period.

```bash
feat(sql): add recursive descent parser for SELECT
fix(kv): handle CRC-32 overflow on large entries
docs: document Schema::compute_metadata contract
chore(github): add issue and PR templates
ci(build): pin actions/checkout to SHA
```

**Scopes** are optional but encouraged for larger codebases. Use the
module or subsystem name: `core`, `kv`, `table`, `sql`, `ci`, `github`.

**Breaking changes** are marked with a `!` before the colon and explained in the commit body:

```bash
feat(sql)!: replace token enum with std::variant

BREAKING CHANGE: TokenKind is no longer a plain enum.
Callers must update match arms to use the new variant types.
```

### Scopes

| Scope    | Use for                                                                |
| :------- | :--------------------------------------------------------------------- |
| `math`   | Implementation of core OLS, Normal Equations, or Hat Matrix logic      |
| `stat`   | Statistical inference (t-test, p-values, F-test) and distribution math |
| `pipe`   | The `DataPipeline` class, imputation logic, and encoding               |
| `viz`    | Plotting code (Residual plots, EDA heatmaps, feature importance)       |
| `proof`  | Monte Carlo simulations or Gauss-Markov theorem illustrations          |
| `report` | Changes to the LaTeX/Markdown files in the `report/` folder            |

## Making a Commit

### Format and Lint Locally

Using `Black` and `Ruff` (already installed with `uv`), run these commands from root to fix styling issues:

```bash
uv run black .
uv run ruff check --fix .
```

If you get tired of running the commands every time, you can install the pre-commit hook. Run this once, preferably after you just cloned the repo:

```bash
uv run pre-commit install
```

### Docstring Style

Docstrings must follow the **NumPy convention** and are enforced by Ruff's `D` ruleset. This is checked automatically alongside other lint rules when you run:

```bash
uv run ruff check --fix .
```

A minimal compliant docstring should look like this:

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

Key rules to keep in mind:

* Summary line is a single sentence in **imperative mood**: "Fit", "Compute", "Return", not "Fits", "Computes", "Returns"
* Leave a blank line between the summary and the `Parameters` section.
* Section headers (`Parameters`, `Returns`, `Raises`) are followed by a `----------` underline matching the header length exactly.
* Each parameter is documented as `name : type` on one line with the description indented on the line below.

### Check for Documentation Coverage

Use `interrogate` to check if docstring coverage passes the minimum threshold. From root:

```bash
uv run interrogate .
```

This is run automatically every time you commit with level-1 verbosity (showing the coverage per-file) if you have **pre-commit hook** installed.

To see which functions are missing docstrings, run with higher verbosity:

```bash
uv run interrogate -v 2 .
```

### Run the Tests

If your code has any associated tests, run the test scripts.
If there isn't any test, add the test yourself within the same commit.

Use `pytest.approx()` from `pytest` for floating point comparisons.

To run the entire suite:

```bash
uv run pytest
```

To run tests for a specific part of the project:

```bash
uv run pytest tests/test_file.py
```

### Commit Your Changes

Follow the commit naming guide above.

> [!NOTE]
> If you have `pre-commit` hook installed, your commit can fail because of pre-commit checks. If it is due to formatting and linting, just re-commit. If it is due to docstring coverage fail, add the docstrings before trying again.

## Issues and Pull Requests

### Working with Issues

There are two pre-configured templates you can choose when opening a new issue.

#### Task Creation

This defines a new task for the team to work on, letting the project completes its roadmap. Upon opening a _task creation_ issue, there are a few sections to fill in.

* **Description**: A clear and concise description of what the task is about.
* **Definition of Done**: A clear and concise description of what is needed for the task to be considered _finished_. Creating sub-issues to help completing the task is recommended.
* **Proposed Implementation**: An explanation of how the task can be done. Suggest which files are affected.
* **Additional context**: Add any other context or screenshots about the task.

#### Bug Report

Use this template when bringing potential problems that need to be solved.

* **Describe the bug/problem**: A clear and concise description of what the bug is.
* **To Reproduce**: The steps needed to reproduce the bug if needed.
* **Expected behavior**: A clear and concise description of what you expected to happen.
* **Screenshots**: If applicable, add screenshots to help explain your problem.
* **Additional context**: Add any other context about the problem here e.g. where the bug might be.

#### Blank Issue

Use this only if your issue is genuinely neither a task nor a bug; for example, a question about project direction or a process concern. If either template fits even partially, prefer it over a blank issue.

> [!NOTE]
> **Feature requests** are out of scope for this project. If you have a suggestion, raise it in the team channel instead. It may be considered for a future iteration.

Whichever template you choose, please be as specific as possible with your issues. Use proper tags to categorize the issue.

### Creating Pull Requests

The remote `main` branch is protected against direct merges. To merge your working branch, create a **pull request**. Follow these steps for a successful PR!

#### Publish Your Working Branch

If you didn't already, publish your working branch to remote.

> [!IMPORTANT]
> If you are not yet a collaborator on the repository, fork it first and publish your branch to your fork instead. Open the PR from your fork's branch against the upstream `main` branch.

Checkout your branch with `git checkout <branch-name>`, then:

```bash
git push -u origin <branch-name>
```

The `-u` flag sets the upstream tracking, so future `git push` and `git pull` commands in that branch work without specifying the remote.

> [!NOTE]
> Here's a quick Git reference:
> **Creating a Local Branch and Publish to Remote**
>
> ```bash
> git checkout -b <branch>      # create + switch
> git push -u origin <branch>   # publish + set tracking
> ```
>
> **Viewing branches**
>
> ```bash
> git branch                    # local branches
> git branch -r                 # remote branches
> git branch -a                 # both
> git branch -vv                # local branches + their tracking remotes
> ```
>
> **Set/unset tracking**
>
> ```bash
> git branch -u origin/<branch-name>    # set upstream for current branch
> git branch -u origin/<remote> <local> # set upstream for a specific branch
> git branch --unset-upstream           # unset for current branch
> ```
>
> **Push, Pull, Fetch**
>
> ```bash
> git fetch                     # download remote changes, don't merge
> git fetch origin <branch>     # fetch a specific branch only
> git pull                      # fetch + merge (or rebase if configured)
> git pull --rebase             # fetch + rebase instead of merge
> git push                      # push current branch to its upstream
> git push origin <branch>      # push a specific branch explicitly
> git push --force-with-lease   # force push, but abort if remote has new commits
> ```
>
> **Deleting branches**
>
> ```bash
> git branch -d <branch>        # delete local (safe — refuses if unmerged)
> git branch -D <branch>        # force delete local
> git push origin -d <branch>   # delete remote
> ```
>

#### Create a Pull Request

When your code is ready for review, open a PR against the `main` branch using the provided PR Template.

* **Update your branch** frequently to prevent merge conflicts, especially right after opening a PR.
* **Keep your PR focused** on a single feature or fix to make review smoother.
* **Reference the issue** using `Closes #<issue-number>` in your PR description.
* **Assign a reviewer** to get your PR reviewed quicker.
* **Resolve all CI checks** and conversations in order to get your PR accepted.
  If a check fails, fix it locally using the relevant section of this guide ([Format and Lint Locally](#format-and-lint-locally), [Check for Documentation Coverage](#check-for-documentation-coverage), [Run the Tests](#run-the-tests)), then push again.
* **Delete your local branch** after a successful merge. GitHub will automatically delete the remote branch. To clean up locally:

```bash
  git checkout main
  git branch -d <branch-name>
```
