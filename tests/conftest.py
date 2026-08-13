"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fraud_pipeline.config import Config, load_config
from fraud_pipeline.paths import default_config_path


@pytest.fixture(scope="session")
def config_path() -> Path:
    """The real config file that ships with the repository."""
    return default_config_path()


@pytest.fixture(scope="session")
def config(config_path: Path) -> Config:
    """The real config, validated. If this fails, the shipped config file is broken."""
    return load_config(config_path)


@pytest.fixture
def config_tree(config_path: Path) -> dict:
    """A fresh mutable copy of the raw config dictionary, for override tests."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
