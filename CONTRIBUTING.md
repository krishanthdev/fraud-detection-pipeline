# Contributing and Git workflow

This repository uses a feature to develop to main workflow. The rules below are the ones
the project follows, and CI enforces the parts it can.

## Branches

| Branch | Purpose | Protected |
| --- | --- | --- |
| `main` | Always releasable. Every merge is tagged with a version. | Yes |
| `develop` | Integration branch. Features land here first. | Yes |
| `feature/<name>` | One unit of work, branched from `develop`. | No |
| `fix/<name>` | A bug fix, branched from `develop`. | No |
| `docs/<name>` | Documentation only. | No |

Nothing is ever committed straight to `main` or `develop`. The pre commit hook blocks it
locally and branch protection blocks it on GitHub.

## The loop

1. Update your local `develop`.
   ```bash
   git checkout develop && git pull
   ```
2. Branch off it.
   ```bash
   git checkout -b feature/my-thing
   ```
3. Commit in small steps using Conventional Commits (see below).
4. Push and open a pull request into `develop`.
   ```bash
   git push -u origin feature/my-thing
   gh pr create --base develop --fill
   ```
5. CI has to pass. Then merge with a merge commit, and delete the branch.
6. When `develop` reaches a working milestone, open a pull request from `develop` into
   `main`, merge it, and tag a release.
   ```bash
   git tag -a v1.0.0 -m "First complete pipeline" && git push --tags
   ```

## Commit messages

Conventional Commits, lower case, present tense, no full stop at the end.

```
feat: add rolling amount aggregates per card
fix: stop the scaler from being fitted on the test split
docs: add the results table to the readme
test: cover the threshold tuning helper
chore: pin lightgbm to 4.5.0
refactor: move the metric helpers into evaluate.py
```

Scope is optional and goes in brackets, for example `feat(features): ...`.

## The planned feature branches

The project is sliced into one pull request per branch, in this order.

| Order | Branch | What lands |
| --- | --- | --- |
| 1 | `feature/project-scaffold` | Repo structure, config, CLI, tests, CI |
| 2 | `feature/data-ingest-validate` | Stages 1 and 2 |
| 3 | `feature/eda` | Exploration notebook and figures |
| 4 | `feature/feature-engineering` | Stage 3 |
| 5 | `feature/baseline-models` | Stage 4, logistic regression and random forest |
| 6 | `feature/advanced-models` | Stage 4, XGBoost, LightGBM, neural net |
| 7 | `feature/evaluation-and-results-table` | Stages 5 and 6 |
| 8 | `feature/serving-api` | Stage 7, FastAPI |
| 9 | `feature/web-app` | Stage 7, Streamlit dashboard |
| 10 | `feature/dockerize` | Stage 8 |
| 11 | `feature/docs-readme` | Final README, then tag v1.0.0 |
| 12 | `feature/ieee-cis-dataset` | Second dataset pass |

## Before you push

```bash
pre-commit run --all-files
pytest
```

On Windows, `.\tasks.ps1 lint` and `.\tasks.ps1 test` do the same thing.

## Notebooks keep their outputs

Most repositories strip notebook outputs with `nbstripout`, and this one deliberately does not.
It is a portfolio repository, so the notebook is something a reader opens on GitHub and reads
in ten seconds. Stripped, it renders as a page of empty cells.

The usual objection does not apply. The seven figures come to about 540 KB embedded, the
outputs are otherwise text tables, and the diffs stay readable.

**The tradeoff is staleness.** Outputs can drift from the code that produced them, so re-run a
notebook before committing a change to it:

```bash
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=fraud-pipeline \
  --ExecutePreprocessor.timeout=1800 notebooks/01_eda.ipynb
```

Two details that cost time the first time round:

- Register the venv's kernel once, or the notebook runs against whatever Python owns the
  `python3` kernelspec and fails on the first import:
  ```bash
  python -m ipykernel install --sys-prefix --name fraud-pipeline
  ```
- Call `python -m nbconvert`, not `python -m jupyter nbconvert`. The `jupyter` command finds
  subcommands by searching PATH for a `jupyter-nbconvert` executable, which can be a different
  Python entirely.

Executing the notebook is also a real check: `--execute` fails on any cell that raises, so a
notebook that has drifted out of step with the modules cannot be committed looking fine. CI
cannot do this, because the dataset is not in the repository.

## Code style

Ruff handles both linting and formatting. The line length is 100. Import sorting, pyupgrade
rules and pathlib rules are switched on, so use `Path` rather than `os.path`.

Type hints are expected on anything public. Docstrings explain why a thing exists, not what
each line does.

## Tests

Tests live in `tests/` and mirror the module they cover. Two markers exist:

- `slow` for anything that trains a model
- `needs_data` for anything that reads the real dataset

CI skips `needs_data` because the dataset is not in the repository.
