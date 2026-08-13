"""Tests for repository path resolution."""

from __future__ import annotations

from pathlib import Path

from fraud_pipeline import paths


def test_repo_root_contains_pyproject() -> None:
    root = paths.repo_root()
    assert (root / "pyproject.toml").is_file()


def test_repo_root_honours_the_environment_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_PIPELINE_ROOT", str(tmp_path))
    assert paths.repo_root() == tmp_path.resolve()


def test_relative_paths_anchor_at_the_repo_root() -> None:
    assert paths.resolve("data/raw") == (paths.repo_root() / "data" / "raw").resolve()


def test_absolute_paths_are_left_alone(tmp_path) -> None:
    absolute = tmp_path / "elsewhere"
    assert paths.resolve(absolute) == absolute


def test_ensure_dir_creates_missing_directories(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_PIPELINE_ROOT", str(tmp_path))
    created = paths.ensure_dir("a/b/c")

    assert created.is_dir()
    assert created == (tmp_path / "a" / "b" / "c").resolve()


def test_ensure_dir_is_safe_to_call_twice(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRAUD_PIPELINE_ROOT", str(tmp_path))
    first = paths.ensure_dir("repeat")
    second = paths.ensure_dir("repeat")
    assert first == second


def test_default_config_path_points_at_the_shipped_file() -> None:
    default = paths.default_config_path()
    assert default.is_file()
    assert default.name == "config.yaml"
    assert default.parent.name == "configs"


def test_config_paths_resolve_to_real_directories(config) -> None:
    for directory in (
        config.paths.raw(),
        config.paths.interim(),
        config.paths.processed(),
        config.paths.model_dir(),
        config.paths.figures_dir(),
        config.paths.tables_dir(),
    ):
        assert isinstance(directory, Path)
        assert directory.is_dir(), f"{directory} is missing from the repository"
