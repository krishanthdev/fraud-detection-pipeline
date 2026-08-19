"""Tests for stage 5, the cost model and the operating threshold.

Two things carry the weight here.

The exact threshold sweep must agree with brute force. It replaced a quantile thinned grid
that silently missed the optimum, so an approximation that looks close is exactly the failure
mode being guarded against.

The test split must not be read before the threshold is frozen. That is a one way door, and
once it has informed a choice there is no way to undo it or to measure the optimism it added.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_pipeline import evaluate
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.evaluate import CostModel, EvaluationError


@pytest.fixture
def cost_config(config_path) -> Config:
    return load_config(config_path)


@pytest.fixture
def scored():
    """A small scored split with known amounts, so costs can be checked by hand."""
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    p = np.array([0.10, 0.20, 0.30, 0.85, 0.40, 0.90, 0.95])
    amounts = np.array([10.0, 10.0, 10.0, 10.0, 1000.0, 5.0, 5.0])
    return y, p, amounts


# --------------------------------------------------------------------------------------
# The cost model
# --------------------------------------------------------------------------------------


def test_a_flat_cost_charges_every_missed_fraud_the_same(scored) -> None:
    y, p, amounts = scored
    cost = CostModel("flat", 120.0, 5.0)

    # Flag nothing: three frauds missed, no false alarms.
    total = cost.total(y, np.zeros(len(y), dtype=bool), amounts)

    assert total == pytest.approx(360.0)


def test_an_amount_cost_charges_each_missed_fraud_its_own_value(scored) -> None:
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    total = cost.total(y, np.zeros(len(y), dtype=bool), amounts)

    # 1000 + 5 + 5, not 3 x 120. This is the whole point of the amount model.
    assert total == pytest.approx(1010.0)


def test_false_alarms_are_charged_at_the_configured_price(scored) -> None:
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    # Flag everything: no missed fraud, four false alarms.
    total = cost.total(y, np.ones(len(y), dtype=bool), amounts)

    assert total == pytest.approx(20.0)


def test_a_perfect_model_costs_nothing(scored) -> None:
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    assert cost.total(y, y == 1, amounts) == pytest.approx(0.0)


def test_the_amount_model_refuses_to_run_without_amounts(scored) -> None:
    y, _, _ = scored
    cost = CostModel("amount", 120.0, 5.0)

    with pytest.raises(EvaluationError, match="needs transaction amounts"):
        cost.total(y, np.zeros(len(y), dtype=bool), None)


def test_the_flat_model_does_not_need_amounts(scored) -> None:
    y, _, _ = scored
    cost = CostModel("flat", 120.0, 5.0)

    assert cost.total(y, np.zeros(len(y), dtype=bool), None) == pytest.approx(360.0)


# --------------------------------------------------------------------------------------
# The exact sweep
# --------------------------------------------------------------------------------------


def _brute_force_min(y, p, amounts, cost) -> float:
    """The obvious slow answer, used to check the fast one."""
    options = [cost.total(y, p >= t, amounts) for t in np.unique(p)]
    options.append(cost.total(y, np.zeros(len(y), dtype=bool), amounts))
    return min(options)


@pytest.mark.parametrize("model", ["flat", "amount"])
@pytest.mark.parametrize("fp_cost", [1.0, 5.0, 50.0])
def test_the_exact_sweep_matches_brute_force(scored, model, fp_cost) -> None:
    y, p, amounts = scored
    cost = CostModel(model, 120.0, fp_cost)

    _, costs = evaluate.cost_curve(y, p, amounts, cost)

    assert costs.min() == pytest.approx(_brute_force_min(y, p, amounts, cost))


def test_the_exact_sweep_matches_brute_force_on_random_data() -> None:
    """The hand built case is too tidy to catch an off by one in the cumulative counts."""
    rng = np.random.default_rng(0)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = (rng.random(200) < 0.1).astype(int)
        p = rng.random(200)
        amounts = rng.lognormal(3, 1.5, 200)
        cost = CostModel("amount", 120.0, float(rng.choice([1, 5, 25])))

        _, costs = evaluate.cost_curve(y, p, amounts, cost)

        assert costs.min() == pytest.approx(_brute_force_min(y, p, amounts, cost))


def test_flagging_nothing_is_always_an_option() -> None:
    """Sometimes the cheapest thing a model can do is stay quiet, and the sweep must see it.

    The fixture cannot show this: its two highest scoring rows are frauds, so flagging them
    costs no false alarms and silence is never optimal there. This needs a model whose top
    score is a legitimate transaction.
    """
    y = np.array([0, 1, 1])
    p = np.array([0.9, 0.4, 0.3])
    amounts = np.array([10.0, 1.0, 1.0])
    # A false alarm priced far above the fraud it would have to catch first.
    cost = CostModel("amount", 120.0, 100_000.0)

    thresholds, costs = evaluate.cost_curve(y, p, amounts, cost)
    best = thresholds[int(np.argmin(costs))]

    assert best > p.max(), "silence should win when a false alarm costs more than the fraud"
    assert costs.min() == pytest.approx(2.0)


def test_tied_scores_do_not_produce_a_threshold_that_cannot_reproduce(scored) -> None:
    """A cut inside a run of equal scores is not something a threshold can express."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.5, 0.5, 0.5, 0.9])
    amounts = np.array([10.0, 10.0, 10.0, 10.0])
    cost = CostModel("amount", 120.0, 5.0)

    thresholds, _ = evaluate.cost_curve(y, p, amounts, cost)

    # Only the distinct scores, plus the flag nothing option.
    assert len(thresholds) == 3


def test_the_sweep_handles_a_split_with_no_fraud() -> None:
    y = np.zeros(20, dtype=int)
    p = np.linspace(0.01, 0.99, 20)
    amounts = np.full(20, 10.0)
    cost = CostModel("amount", 120.0, 5.0)

    thresholds, costs = evaluate.cost_curve(y, p, amounts, cost)

    # Best is to flag nothing, which costs nothing.
    assert costs.min() == pytest.approx(0.0)
    assert thresholds[int(np.argmin(costs))] > p.max()


# --------------------------------------------------------------------------------------
# Threshold strategies
# --------------------------------------------------------------------------------------


def test_the_cost_strategy_finds_the_cheapest_threshold(scored, cost_config) -> None:
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    point = evaluate.tune_threshold(y, p, amounts, cost, cost_config, "cost")

    assert point.cost == pytest.approx(_brute_force_min(y, p, amounts, cost))


def test_the_amount_model_prefers_catching_the_expensive_fraud(scored, cost_config) -> None:
    """The 1000 fraud scores 0.40 and the two 5s score above it.

    A flat cost is indifferent about which three it catches. An amount cost is not, and this
    is the behaviour the whole cost model exists to produce.
    """
    y, p, amounts = scored

    flat = evaluate.tune_threshold(
        y, p, amounts, CostModel("flat", 120.0, 5.0), cost_config, "cost"
    )
    amount = evaluate.tune_threshold(
        y, p, amounts, CostModel("amount", 120.0, 5.0), cost_config, "cost"
    )

    assert amount.threshold <= 0.40, "the amount model must reach down to the expensive fraud"
    assert amount.value_caught >= flat.value_caught


def test_max_f1_ignores_the_cost_model(scored, cost_config) -> None:
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    point = evaluate.tune_threshold(y, p, amounts, cost, cost_config, "max_f1")

    assert point.strategy == "max_f1"
    assert point.f1 > 0


def test_fixed_recall_reaches_the_target(scored, config_path) -> None:
    config = load_config(config_path, overrides=["evaluation.fixed_recall_target=1.0"])
    y, p, amounts = scored
    cost = CostModel("amount", 120.0, 5.0)

    point = evaluate.tune_threshold(y, p, amounts, cost, config, "fixed_recall")

    assert point.recall >= 1.0


def test_an_impossible_recall_target_is_rejected_by_the_config(config_tree) -> None:
    """Writing the unreachable case as a test showed it cannot happen at run time.

    Flagging every transaction is one of the candidate thresholds, so any target at or below
    1.0 is always reachable. The only unreachable target is one above 1.0, and that is a
    config error rather than a modelling outcome, so it is caught at load.
    """
    from pydantic import ValidationError

    for bad in (1.5, 0.0, -0.2):
        config_tree["evaluation"]["fixed_recall_target"] = bad
        with pytest.raises(ValidationError, match="fixed_recall_target"):
            Config.model_validate(config_tree)


def test_full_recall_is_always_reachable(scored, config_path) -> None:
    """Because flagging everything is always on the menu."""
    config = load_config(config_path, overrides=["evaluation.fixed_recall_target=1.0"])
    y, p, amounts = scored

    point = evaluate.tune_threshold(
        y, p, amounts, CostModel("amount", 120.0, 5.0), config, "fixed_recall"
    )

    assert point.recall == pytest.approx(1.0)


def test_an_unknown_strategy_is_rejected(scored, cost_config) -> None:
    y, p, amounts = scored
    with pytest.raises(EvaluationError, match="unknown threshold strategy"):
        evaluate.tune_threshold(y, p, amounts, CostModel("amount", 1.0, 1.0), cost_config, "vibes")


# --------------------------------------------------------------------------------------
# Operating point arithmetic
# --------------------------------------------------------------------------------------


def test_the_confusion_counts_add_up(scored) -> None:
    y, p, amounts = scored
    point = evaluate.confusion_at(y, p, 0.5, amounts, CostModel("amount", 120.0, 5.0))

    total = (
        point.true_positives + point.false_positives + point.false_negatives + point.true_negatives
    )
    assert total == len(y)
    assert point.rows == len(y)


def test_value_recall_is_not_case_recall(scored) -> None:
    """The distinction the whole report turns on.

    At this threshold the model catches two of three frauds, so case recall is 0.667. Those
    two are worth 5 each and the one it missed was worth 1000, so value recall is 0.010.
    """
    y, p, amounts = scored
    point = evaluate.confusion_at(y, p, 0.5, amounts, CostModel("amount", 120.0, 5.0))

    assert point.recall == pytest.approx(2 / 3)
    assert point.value_recall == pytest.approx(10 / 1010, abs=1e-6)


def test_alerts_are_what_an_analyst_would_see(scored) -> None:
    y, p, amounts = scored
    point = evaluate.confusion_at(y, p, 0.5, amounts, CostModel("amount", 120.0, 5.0))

    assert point.alerts == point.true_positives + point.false_positives


# --------------------------------------------------------------------------------------
# The guard that matters
# --------------------------------------------------------------------------------------


def test_the_config_refuses_to_report_on_anything_but_test(config_tree) -> None:
    """The mirror of the guard on training.

    Training refuses to touch test. This stage exists to open it, so the mistake to prevent
    here is the opposite one: reporting final numbers from the split the threshold was tuned
    on, and calling them held out.
    """
    from pydantic import ValidationError

    config_tree["evaluation"]["test_split"] = "validation"
    with pytest.raises(ValidationError, match="reports on the test split"):
        Config.model_validate(config_tree)


def test_an_unknown_cost_model_is_rejected(config_tree) -> None:
    from pydantic import ValidationError

    config_tree["evaluation"]["cost"]["model"] = "guesswork"
    with pytest.raises(ValidationError, match="cost model must be one of"):
        Config.model_validate(config_tree)


def test_describe_value_separates_caught_from_missed(scored) -> None:
    y, p, amounts = scored
    summary = evaluate.describe_value(y, p, amounts, 0.5, "example")

    assert summary["frauds"] == 3
    assert summary["caught"] == 2
    assert summary["missed"] == 1
    assert summary["largest_missed"] == pytest.approx(1000.0)
    assert summary["value_recall"] == pytest.approx(10 / 1010, abs=1e-6)


# --------------------------------------------------------------------------------------
# The stage end to end
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """Run the pipeline far enough that stage 5 has something to evaluate.

    Module scoped, because getting here means ingest, validate, eda, features and a full
    training sweep.
    """
    from fraud_pipeline import eda, features, ingest, train, validation
    from fraud_pipeline.paths import default_config_path

    from .synthetic import make_transactions, write_raw_csv

    tmp_path = tmp_path_factory.mktemp("evaluation")
    config = load_config(
        default_config_path(),
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.data_processed={(tmp_path / 'processed').as_posix()}",
            f"paths.models={(tmp_path / 'models').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            "validation.min_rows=100",
            "validation.min_fraud_rows_per_split=1",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=800",
            "eda.figures=false",
            "training.feature_sets=[selected]",
            "training.bootstrap_samples=30",
            "models.logistic_regression.max_iter=200",
            "models.random_forest.n_estimators=10",
            "models.random_forest.max_depth=4",
            "models.xgboost.n_estimators=20",
            "models.lightgbm.n_estimators=20",
            "models.neural_net.epochs=3",
            "models.neural_net.hidden_sizes=[8]",
            "registry.enabled=false",
        ],
    )

    write_raw_csv(config, make_transactions(config, rows=6000, fraud_rows=45))
    ingest.run(config)
    validation.run(config)
    eda.run(config)
    features.run(config)
    train.run(config)
    return config, evaluate.run(config)


def test_the_stage_writes_its_report_and_summary(evaluated) -> None:
    import json

    config, result = evaluated
    tables = config.paths.tables_dir()

    assert (tables / config.evaluation.report_file).is_file()
    assert (tables / "threshold_sensitivity.csv").is_file()

    payload = json.loads((tables / config.evaluation.summary_file).read_text(encoding="utf-8"))
    assert payload["threshold"] == pytest.approx(result.threshold)
    assert payload["champion"]["run_name"] == result.champion.run_name
    assert "test" in payload and "validation" in payload


def test_the_champion_comes_from_the_healthy_runs(evaluated) -> None:
    config, result = evaluated
    healthy = set(evaluate.healthy_runs(config)["run_name"])

    assert result.champion.run_name in healthy
    assert result.champion.considered == len(healthy)


def test_the_threshold_used_on_test_is_the_one_tuned_on_validation(evaluated) -> None:
    """The whole point of the stage: the threshold is frozen before test is opened."""
    _, result = evaluated

    assert result.test.threshold == pytest.approx(result.validation.threshold)
    assert result.test.strategy == "frozen"


def test_test_and_validation_are_different_splits(evaluated) -> None:
    config, result = evaluated
    from fraud_pipeline.features import load_engineered

    assert result.test.rows == len(load_engineered(config, "test"))
    assert result.validation.rows == len(load_engineered(config, "validation"))


def test_the_report_states_the_order_it_was_done_in(evaluated) -> None:
    config, _ = evaluated
    report = (config.paths.tables_dir() / config.evaluation.report_file).read_text(encoding="utf-8")

    assert "The order this was done in" in report
    assert "opened once" in report
    assert "Value recall" in report


def test_the_value_breakdown_covers_both_splits(evaluated) -> None:
    _, result = evaluated
    assert set(result.value_breakdown["split"]) == {"validation", "test"}


def test_the_stage_is_reproducible(evaluated) -> None:
    config, first = evaluated
    second = evaluate.run(config)

    assert second.champion.run_name == first.champion.run_name
    assert second.threshold == pytest.approx(first.threshold)
    assert second.test.recall == pytest.approx(first.test.recall)


def test_a_pinned_champion_is_honoured(evaluated) -> None:
    config, result = evaluated
    healthy = list(evaluate.healthy_runs(config)["run_name"])
    other = next(n for n in healthy if n != result.champion.run_name)

    pinned = load_config(
        None,
        overrides=[
            f"paths.data_interim={config.paths.interim().as_posix()}",
            f"paths.data_processed={config.paths.processed().as_posix()}",
            f"paths.models={config.paths.model_dir().as_posix()}",
            f"paths.tables={config.paths.tables_dir().as_posix()}",
            f"evaluation.champion={other}",
            "registry.enabled=false",
        ],
    )
    assert evaluate.run(pinned).champion.run_name == other


def test_an_unknown_pinned_champion_is_rejected(evaluated) -> None:
    config, _ = evaluated
    pinned = load_config(
        None,
        overrides=[
            f"paths.data_interim={config.paths.interim().as_posix()}",
            f"paths.data_processed={config.paths.processed().as_posix()}",
            f"paths.models={config.paths.model_dir().as_posix()}",
            f"paths.tables={config.paths.tables_dir().as_posix()}",
            "evaluation.champion=no_such_run",
            "registry.enabled=false",
        ],
    )
    with pytest.raises(EvaluationError, match="not a healthy run"):
        evaluate.run(pinned)


def test_misaligned_predictions_are_caught(evaluated) -> None:
    """A silent misalignment would attach the wrong amount to every fraud."""

    config, result = evaluated
    from fraud_pipeline.features import load_engineered

    validation = load_engineered(config, "validation")
    predictions = evaluate.load_run_predictions(config, result.champion.run_name)

    shuffled = predictions.iloc[::-1].reset_index(drop=True)
    with pytest.raises(EvaluationError, match="do not line up"):
        evaluate.aligned_amounts(shuffled, validation, config)

    truncated = predictions.iloc[:-5]
    with pytest.raises(EvaluationError, match="rows"):
        evaluate.aligned_amounts(truncated, validation, config)

    assert isinstance(evaluate.aligned_amounts(predictions, validation, config), np.ndarray)


def test_missing_predictions_explain_themselves(evaluated) -> None:
    config, _ = evaluated
    with pytest.raises(EvaluationError, match="Run the training stage first"):
        evaluate.load_run_predictions(config, "a_run_that_was_never_trained")
