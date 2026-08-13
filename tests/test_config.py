"""Tests for configuration loading and validation.

The point of these tests is that a broken config fails loudly at load time, not halfway
through a training run.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from fraud_pipeline.config import Config, apply_overrides, load_config


def test_shipped_config_loads(config: Config) -> None:
    assert config.project.name == "fraud-detection-pipeline"
    assert config.project.seed == 42
    assert config.dataset.active == "ulb"


def test_dataset_spec_builds_column_names(config: Config) -> None:
    spec = config.dataset.spec()
    pca = spec.pca_columns()

    assert len(pca) == 28
    assert pca[0] == "V1"
    assert pca[-1] == "V28"
    # Time, 28 components, Amount, Class
    assert len(spec.expected_columns()) == 31
    assert spec.expected_columns()[0] == "Time"
    assert spec.expected_columns()[-1] == "Class"


def test_enabled_models_covers_every_technique_in_the_brief(config: Config) -> None:
    enabled = set(config.enabled_models())
    required = {"logistic_regression", "random_forest", "xgboost", "lightgbm", "neural_net"}
    assert required <= enabled


def test_model_params_exclude_the_enabled_flag(config: Config) -> None:
    params = config.models["xgboost"].params()
    assert "enabled" not in params
    assert params["max_depth"] == 6


def test_imbalance_strategies_cover_the_comparison(config: Config) -> None:
    assert set(config.imbalance.strategies) == {"none", "class_weight", "smote"}


def test_accuracy_is_not_a_reported_metric(config: Config) -> None:
    """The brief is explicit that accuracy alone is not acceptable on this problem."""
    assert "accuracy" not in config.evaluation.report_metrics
    assert config.evaluation.primary_metric == "average_precision"


def test_override_changes_a_nested_value(config_path, config_tree) -> None:
    updated = apply_overrides(config_tree, ["project.seed=7", "models.xgboost.max_depth=9"])
    parsed = Config.model_validate(updated)

    assert parsed.project.seed == 7
    assert parsed.models["xgboost"].params()["max_depth"] == 9


def test_override_parses_types_not_just_strings(config_tree) -> None:
    updated = apply_overrides(
        config_tree,
        [
            "features.amount_log=false",
            "split.train_fraction=0.8",
            "features.rolling_windows=[5, 20]",
        ],
    )
    assert updated["features"]["amount_log"] is False
    assert updated["split"]["train_fraction"] == 0.8
    assert updated["features"]["rolling_windows"] == [5, 20]


def test_override_of_an_unknown_key_is_rejected(config_tree) -> None:
    with pytest.raises(KeyError):
        apply_overrides(config_tree, ["project.nonexistent=1"])


def test_override_without_equals_is_rejected(config_tree) -> None:
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(config_tree, ["project.seed"])


def test_split_fractions_must_sum_to_one(config_tree) -> None:
    config_tree["split"]["train_fraction"] = 0.9
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Config.model_validate(config_tree)


def test_unknown_config_key_is_rejected(config_tree) -> None:
    config_tree["project"]["typo_key"] = True
    with pytest.raises(ValidationError):
        Config.model_validate(config_tree)


def test_unknown_imbalance_strategy_is_rejected(config_tree) -> None:
    config_tree["imbalance"]["strategies"] = ["none", "magic"]
    with pytest.raises(ValidationError, match="unknown imbalance strategies"):
        Config.model_validate(config_tree)


def test_unknown_threshold_strategy_is_rejected(config_tree) -> None:
    config_tree["evaluation"]["threshold_strategy"] = "vibes"
    with pytest.raises(ValidationError):
        Config.model_validate(config_tree)


def test_missing_config_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_non_mapping_config_file_raises(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_load_config_applies_overrides_from_disk(config_path, tmp_path) -> None:
    copy = tmp_path / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        copy.write_text(handle.read(), encoding="utf-8")

    parsed = load_config(copy, overrides=["project.seed=1234"])
    assert parsed.project.seed == 1234

    # The file on disk is untouched. Overrides live only in memory.
    on_disk = yaml.safe_load(copy.read_text(encoding="utf-8"))
    assert on_disk["project"]["seed"] == 42
