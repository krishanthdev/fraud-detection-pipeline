"""Tests for stage 6, promotion.

The promotion rule is pure and separated from MLflow on purpose, so the thing that decides
whether a model reaches production can be tested without a tracking server anywhere near it.

The rule that matters is the margin. Without one, every rerun swaps the model whenever the
number moves at all, and with about fifty fraud cases in the scoring split it moves for no
reason.
"""

from __future__ import annotations

import pytest

from fraud_pipeline import registry
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.registry import RegistryError, decide


@pytest.fixture
def cost_basis(config_path) -> Config:
    return load_config(
        config_path,
        overrides=["registry.promotion_basis=cost", "registry.promotion_min_improvement=10.0"],
    )


@pytest.fixture
def metric_basis(config_path) -> Config:
    return load_config(
        config_path,
        overrides=["registry.promotion_basis=metric", "registry.promotion_min_improvement=0.005"],
    )


# --------------------------------------------------------------------------------------
# The first model
# --------------------------------------------------------------------------------------


def test_the_first_model_takes_the_title_unopposed(cost_basis) -> None:
    decision = decide("challenger", 500.0, None, None, cost_basis)

    assert decision.promoted
    assert "unopposed" in decision.reason
    assert decision.margin_achieved is None


def test_a_champion_without_a_score_counts_as_no_champion(cost_basis) -> None:
    """A registry entry with no recorded score cannot be compared against."""
    decision = decide("challenger", 500.0, "old_run", None, cost_basis)

    assert decision.promoted
    assert decision.champion_score is None


# --------------------------------------------------------------------------------------
# Cost, where lower is better
# --------------------------------------------------------------------------------------


def test_a_clearly_cheaper_challenger_is_promoted(cost_basis) -> None:
    decision = decide("new", 400.0, "old", 500.0, cost_basis)

    assert decision.promoted
    assert decision.margin_achieved == pytest.approx(100.0)
    assert "clears" in decision.reason


def test_a_more_expensive_challenger_is_rejected(cost_basis) -> None:
    """The sign has to be right. Getting it backwards would promote the worse model."""
    decision = decide("new", 600.0, "old", 500.0, cost_basis)

    assert not decision.promoted
    assert decision.margin_achieved == pytest.approx(-100.0)
    assert "does not beat" in decision.reason


def test_a_marginally_cheaper_challenger_is_not_worth_a_swap(cost_basis) -> None:
    """Better, but inside the margin. This is the case the margin exists for.

    Without it every rerun swaps the model on noise, and on fifty fraud cases the noise is
    larger than this difference.
    """
    decision = decide("new", 495.0, "old", 500.0, cost_basis)

    assert not decision.promoted
    assert decision.margin_achieved == pytest.approx(5.0)
    assert "inside the" in decision.reason


def test_exactly_at_the_margin_is_promoted(cost_basis) -> None:
    decision = decide("new", 490.0, "old", 500.0, cost_basis)

    assert decision.promoted
    assert decision.margin_achieved == pytest.approx(10.0)


def test_an_identical_challenger_is_not_promoted(cost_basis) -> None:
    decision = decide("new", 500.0, "old", 500.0, cost_basis)

    assert not decision.promoted
    assert decision.margin_achieved == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# A metric, where higher is better
# --------------------------------------------------------------------------------------


def test_the_direction_flips_for_a_metric(metric_basis) -> None:
    """Cost is minimised and a metric is maximised, and the same numbers mean the opposite."""
    on_metric = decide("new", 0.90, "old", 0.85, metric_basis)

    assert on_metric.promoted
    assert on_metric.margin_achieved == pytest.approx(0.05)


def test_a_lower_metric_is_rejected(metric_basis) -> None:
    decision = decide("new", 0.80, "old", 0.85, metric_basis)

    assert not decision.promoted
    assert decision.margin_achieved == pytest.approx(-0.05)


def test_the_same_numbers_give_opposite_verdicts_on_the_two_bases(cost_basis, metric_basis) -> None:
    """The clearest way to state why the sign matters."""
    as_cost = decide("new", 0.90, "old", 0.85, cost_basis)
    as_metric = decide("new", 0.90, "old", 0.85, metric_basis)

    assert not as_cost.promoted, "a higher cost is worse"
    assert as_metric.promoted, "a higher metric is better"


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def test_an_unknown_promotion_basis_is_rejected(config_tree) -> None:
    from pydantic import ValidationError

    config_tree["registry"]["promotion_basis"] = "vibes"
    with pytest.raises(ValidationError, match="promotion_basis must be one of"):
        Config.model_validate(config_tree)


def test_the_decision_records_everything_needed_to_audit_it(cost_basis) -> None:
    """A promotion nobody can reconstruct later is not much of a record."""
    payload = decide("new", 400.0, "old", 500.0, cost_basis).as_dict()

    for key in (
        "promoted",
        "reason",
        "basis",
        "challenger_run",
        "challenger_score",
        "champion_run",
        "champion_score",
        "margin_required",
        "margin_achieved",
    ):
        assert key in payload


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


@pytest.fixture
def evaluated(tmp_path, config_path):
    """A written evaluation summary, which is all stage 6 reads.

    Built by hand rather than by running the pipeline, because the promotion rule does not
    care how the summary was produced and a full run would make this file minutes long.
    """
    import json

    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    summary = {
        "champion": {
            "run_name": "random_forest__smote__selected",
            "label": "Random Forest",
            "imbalance": "smote",
            "feature_set": "selected",
            "considered": 12,
        },
        "threshold": 0.6543,
        "validation": {"cost": 250.0, "average_precision": 0.87},
        "test": {
            "recall": 0.64,
            "value_recall": 0.51,
            "true_positives": 33,
            "false_negatives": 19,
            "false_positives": 3,
            "rows": 42560,
        },
    }
    (tables / "evaluation_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    config = load_config(
        config_path,
        overrides=[
            f"paths.tables={tables.as_posix()}",
            f"paths.models={(tmp_path / 'models').as_posix()}",
            "registry.enabled=false",
        ],
    )
    return config, summary


def test_the_stage_promotes_the_first_model_and_writes_a_report(evaluated) -> None:
    config, _ = evaluated
    decision = registry.run(config)

    assert decision.promoted
    assert decision.challenger_run == "random_forest__smote__selected"

    report = (config.paths.tables_dir() / config.registry.report_file).read_text(encoding="utf-8")
    assert "# Model registry" in report
    assert "Why a margin" in report
    assert "value recall" in report


def test_the_stage_promotes_on_validation_cost_not_the_test_number(evaluated) -> None:
    """Promoting on test would fold the held out estimate into the decision it judges."""
    config, summary = evaluated
    decision = registry.run(config)

    assert decision.challenger_score == pytest.approx(summary["validation"]["cost"])


def test_the_stage_can_promote_on_a_metric_instead(evaluated, config_path) -> None:
    config, summary = evaluated
    on_metric = load_config(
        config_path,
        overrides=[
            f"paths.tables={config.paths.tables_dir().as_posix()}",
            f"paths.models={config.paths.model_dir().as_posix()}",
            "registry.enabled=false",
            "registry.promotion_basis=metric",
            "registry.promotion_metric=average_precision",
        ],
    )
    decision = registry.run(on_metric)

    assert decision.challenger_score == pytest.approx(summary["validation"]["average_precision"])


def test_nothing_is_written_to_a_registry_when_mlflow_is_off(evaluated) -> None:
    config, _ = evaluated
    assert registry.run(config).model_version is None


def test_the_stage_needs_an_evaluation_first(tmp_path, config_path) -> None:
    config = load_config(
        config_path,
        overrides=[f"paths.tables={(tmp_path / 'empty').as_posix()}", "registry.enabled=false"],
    )
    with pytest.raises(RegistryError, match="Run the evaluate stage first"):
        registry.run(config)


def test_the_report_states_the_verdict_either_way(evaluated) -> None:
    config, _ = evaluated
    decision = registry.run(config)
    report = registry.build_report(
        decision,
        {
            "threshold": 0.5,
            "test": {
                "recall": 0.6,
                "value_recall": 0.5,
                "true_positives": 3,
                "false_negatives": 2,
                "false_positives": 1,
                "rows": 100,
            },
        },
        config,
    )

    assert ("Promoted" in report) or ("Not promoted" in report)
    assert "Why cost rather than the headline metric" in report


def test_a_disabled_registry_is_not_consulted(evaluated, monkeypatch) -> None:
    """Switching tracking off must mean not touching the store.

    Otherwise a run that cannot write to the registry still reads a champion out of it, and
    compares against a title it has no way to take. This test exists because it happened: a
    stage test in a temporary directory silently read the repository's own database.
    """
    config, _ = evaluated

    def explode(*_args, **_kwargs):
        raise AssertionError("the registry was consulted while disabled")

    monkeypatch.setattr(registry, "_client", explode)

    assert registry.current_champion(config) == (None, None, None)
    assert registry.run(config).promoted
