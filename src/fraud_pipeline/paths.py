"""Repository path resolution.

Every other module asks this one where things live, so nothing anywhere else needs to
guess a working directory. The rule is simple: a path in the config file is relative to
the repository root unless it is already absolute.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT_MARKER = "pyproject.toml"
_ENV_ROOT = "FRAUD_PIPELINE_ROOT"


def repo_root() -> Path:
    """Return the repository root.

    Resolution order:

    1. the ``FRAUD_PIPELINE_ROOT`` environment variable, which the Docker image sets
    2. the nearest parent directory of this file that contains ``pyproject.toml``
    3. the current working directory, as a last resort
    """
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / _ROOT_MARKER).is_file():
            return parent

    return Path.cwd().resolve()


def resolve(path: str | Path) -> Path:
    """Turn a config path into an absolute path anchored at the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (repo_root() / candidate).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and its parents) if it is missing, then return it."""
    target = resolve(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def default_config_path() -> Path:
    """Path to the config file the command line uses when none is given."""
    return repo_root() / "configs" / "config.yaml"
