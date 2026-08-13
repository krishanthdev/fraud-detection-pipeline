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


@pytest.fixture
def sandbox_config(config_path: Path, tmp_path: Path) -> Config:
    """The real config, redirected at a throwaway directory.

    Every path points inside ``tmp_path``, so a test can run the full ingest and validate
    stages without touching the repository or the real dataset. Thresholds that assume a
    284,807 row file are lowered to suit the small synthetic frames.
    """
    return load_config(
        config_path,
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.data_processed={(tmp_path / 'processed').as_posix()}",
            f"paths.models={(tmp_path / 'models').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            "validation.min_rows=100",
            "validation.min_fraud_rows_per_split=1",
            "ingest.chunk_size=0",
        ],
    )


@pytest.fixture
def real_data_config(config: Config) -> Config:
    """The real config, skipped when the dataset has not been downloaded."""
    raw_file = config.paths.raw() / config.dataset.spec().raw_file
    if not raw_file.is_file():
        pytest.skip(f"dataset not present at {raw_file}")
    return config
