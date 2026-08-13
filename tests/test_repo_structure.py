"""Guard the repository layout required by the project standards.

If someone (including a future me) deletes a directory or forgets a file, this test fails
before the missing piece causes a confusing error deeper in the pipeline.
"""

from __future__ import annotations

import pytest

from fraud_pipeline.paths import repo_root

REQUIRED_DIRECTORIES = [
    "app",
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "models",
    "notebooks",
    "reports/figures",
    "reports/tables",
    "scripts",
    "src/fraud_pipeline",
    "tests",
    ".github/workflows",
]

REQUIRED_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".pre-commit-config.yaml",
    "configs/config.yaml",
    ".github/workflows/ci.yml",
]


@pytest.mark.parametrize("relative", REQUIRED_DIRECTORIES)
def test_required_directory_exists(relative: str) -> None:
    assert (repo_root() / relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative: str) -> None:
    assert (repo_root() / relative).is_file(), f"missing file: {relative}"


def test_data_directories_are_ignored_by_git() -> None:
    """Raw data must never reach the repository. The .gitignore has to say so."""
    ignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/*", "data/interim/*", "data/processed/*", "models/*"):
        assert pattern in ignore, f"{pattern} is not in .gitignore"


def test_secrets_are_ignored_by_git() -> None:
    ignore = (repo_root() / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "kaggle.json"):
        assert pattern in ignore, f"{pattern} is not in .gitignore"
