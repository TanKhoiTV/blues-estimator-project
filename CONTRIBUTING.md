# Contributing to Blues Estimator Project

To start contributing, follow the steps.

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

Optionally, you can install the **pre-commit hook** to automatically format and lint your code every time you commit.
This requires a separate one-time installation step.

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
git pull upstream main --rebase
```

If you have a working branch, sync it as well, after syncing the `main` branch:

```bash
git checkout <your-branch-name>
git rebase main
```

## Create Your Working Branch

This project has a rule to protect the `main` branch from direct commits, so you have to create your own branch if you want to contribute.

Remember to **sync** your local `main` branch first! Then from the local `main` branch:

```bash
git checkout -b <your-branch-name>
```

Your branches should only be owned by only person, and that is _you_. Don't let anybody commit onto them.
If you find yourself needing some help with your branches, consider the branches are overachieving.

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

Using **Black** and **Ruff** (already installed with `uv`), run these commands from the root directory to fix styling issues:

```bash
uv run black .
uv run ruff check --fix .
```

If you get tired of running the commands every time, you can install the **pre-commit hook**.
Run this once, preferably after you just cloned the repo:

```bash
uv run pre-commit install
```

### Run the Tests

If your code has any associated tests, run the test scripts.
If there isn't any test, add the test yourself within the same commit.

### Commit Your Changes

Follow the commit naming guide above.
