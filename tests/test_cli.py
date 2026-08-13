"""Tests for the command line interface.

The stage commands are wired up before they are implemented, so these tests check that the
wiring is right and that an unimplemented stage says so plainly instead of failing in a
confusing way.
"""

from __future__ import annotations

from typer.testing import CliRunner

from fraud_pipeline import __version__
from fraud_pipeline.cli import app

runner = CliRunner()

STAGE_COMMANDS = [
    "ingest",
    "validate",
    "features",
    "train",
    "evaluate",
    "register",
    "serve",
    "run-all",
]


def test_help_lists_every_stage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in STAGE_COMMANDS:
        assert command in result.output


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_show_config_prints_the_active_dataset() -> None:
    result = runner.invoke(app, ["show-config"])
    assert result.exit_code == 0
    assert "ulb" in result.output


def test_show_config_applies_an_override() -> None:
    result = runner.invoke(app, ["--set", "project.seed=99", "show-config"])
    assert result.exit_code == 0
    assert "99" in result.output


def test_bad_override_fails_with_a_non_zero_exit() -> None:
    result = runner.invoke(app, ["--set", "project.not_a_key=1", "show-config"])
    assert result.exit_code != 0


def test_unimplemented_stages_exit_with_code_two() -> None:
    for command in STAGE_COMMANDS:
        result = runner.invoke(app, [command])
        assert result.exit_code == 2, f"{command} should report that it is not built yet"
        assert "not implemented yet" in result.output
