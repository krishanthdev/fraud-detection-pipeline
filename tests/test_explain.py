"""Tests for the SHAP explanations.

An explanation nobody checks is decoration. Two things are worth asserting: that the values
line up with the features they claim to describe, and that the explainer chosen actually suits
the model, because the champion changes between runs and a tree explainer pointed at a neural
network fails in a confusing way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_pipeline import explain
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.explain import ExplanationError


@pytest.fixture
def explain_config(config_path) -> Config:
    return load_config(
        config_path,
        overrides=[
            "explainability.shap_background_samples=40",
            "explainability.shap_background_clusters=5",
            "explainability.shap_explain_samples=12",
            "explainability.shap_top_features=5",
            "explainability.shap_examples=3",
        ],
    )


@pytest.fixture(scope="module")
def toy():
    """A small separable problem where one feature obviously matters most."""
    rng = np.random.default_rng(0)
    rows = 400
    frame = pd.DataFrame(
        {
            "strong": rng.normal(0, 1, rows),
            "weak": rng.normal(0, 1, rows),
            "noise": rng.normal(0, 1, rows),
            "Amount": rng.lognormal(3, 1, rows).round(2),
        }
    )
    frame["Class"] = (frame["strong"] > 1.2).astype(int)
    features = ["strong", "weak", "noise", "Amount"]
    return frame, features


def _fit(model, toy):
    frame, features = toy
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipeline.fit(frame[features].to_numpy(dtype=float), frame["Class"].to_numpy(dtype=int))
    return pipeline


# --------------------------------------------------------------------------------------
# Choosing the explainer
# --------------------------------------------------------------------------------------


def test_a_tree_model_gets_the_exact_explainer(toy) -> None:
    pipeline = _fit(RandomForestClassifier(n_estimators=5, random_state=0), toy)
    assert explain.choose_method(pipeline) == "tree"


def test_anything_else_gets_the_model_agnostic_one(toy) -> None:
    """Including the neural network that is currently champion."""
    pipeline = _fit(LogisticRegression(max_iter=100), toy)
    assert explain.choose_method(pipeline) == "kernel"


def test_the_final_estimator_is_found_past_the_scaler(toy) -> None:
    pipeline = _fit(LogisticRegression(max_iter=100), toy)
    assert isinstance(explain.final_estimator(pipeline), LogisticRegression)


# --------------------------------------------------------------------------------------
# The background set
# --------------------------------------------------------------------------------------


def test_the_background_is_a_sample_of_the_requested_size() -> None:
    features = np.arange(1000, dtype=float).reshape(500, 2)
    assert len(explain.background_sample(features, 50, seed=0)) == 50


def test_asking_for_more_background_than_exists_returns_everything() -> None:
    features = np.arange(40, dtype=float).reshape(20, 2)
    assert len(explain.background_sample(features, 200, seed=0)) == 20


def test_the_background_sample_is_reproducible() -> None:
    features = np.arange(1000, dtype=float).reshape(500, 2)
    first = explain.background_sample(features, 30, seed=7)
    second = explain.background_sample(features, 30, seed=7)

    np.testing.assert_array_equal(first, second)


def test_summarising_the_background_shrinks_it() -> None:
    """This is the setting that decides whether the stage takes minutes or hours."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4))

    summarised = explain.summarise_background(features, 8)

    assert summarised.data.shape[0] == 8
    assert summarised.data.shape[1] == 4


def test_summarising_is_skipped_when_it_would_not_help() -> None:
    features = np.random.default_rng(0).normal(size=(5, 3))
    assert explain.summarise_background(features, 50) is features


# --------------------------------------------------------------------------------------
# The values themselves
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def explained(toy):
    """Explain a tree model once, since the exact explainer is fast."""
    frame, features = toy
    config = load_config(
        None,
        overrides=[
            "explainability.shap_background_samples=40",
            "explainability.shap_background_clusters=5",
            "explainability.shap_top_features=4",
        ],
    )
    pipeline = Pipeline([("model", RandomForestClassifier(n_estimators=12, random_state=0))])
    pipeline.fit(frame[features].to_numpy(dtype=float), frame["Class"].to_numpy(dtype=int))

    result = explain.explain(
        pipeline,
        frame[features].to_numpy(dtype=float),
        frame.head(25),
        features,
        config,
    )
    return result, frame, features, config


def test_every_feature_gets_a_contribution_for_every_row(explained) -> None:
    result, _, features, _ = explained

    assert result.contributions.shape == (25, len(features))
    assert list(result.contributions.columns) == features


def test_importance_is_ranked_and_sums_to_one(explained) -> None:
    result, _, _, _ = explained

    assert list(result.importance["mean_abs_shap"]) == sorted(
        result.importance["mean_abs_shap"], reverse=True
    )
    assert result.importance["share"].sum() == pytest.approx(1.0)


def test_the_feature_that_defines_the_label_ranks_first(explained) -> None:
    """The label is a threshold on `strong`, so anything else winning would be wrong."""
    result, _, _, _ = explained

    assert result.importance.iloc[0]["feature"] == "strong"


def test_a_useless_feature_contributes_almost_nothing(explained) -> None:
    result, _, _, _ = explained
    importance = result.importance.set_index("feature")["mean_abs_shap"]

    assert importance["noise"] < importance["strong"] / 10


def test_top_returns_the_requested_number(explained) -> None:
    result, _, _, _ = explained
    assert len(result.top(2)) == 2


# --------------------------------------------------------------------------------------
# Local explanations
# --------------------------------------------------------------------------------------


def test_a_local_explanation_names_the_features_and_their_values(explained) -> None:
    result, frame, _, config = explained

    local = explain.explain_one(result, 0, config, "Amount", "Class")

    assert local["row"] == 0
    assert local["amount"] == pytest.approx(frame.iloc[0]["Amount"])
    assert local["actual"] == int(frame.iloc[0]["Class"])
    assert len(local["reasons"]) == config.explainability.shap_top_features


def test_local_reasons_are_ordered_by_how_much_they_moved_the_score(explained) -> None:
    result, _, _, config = explained
    reasons = explain.explain_one(result, 0, config, "Amount", "Class")["reasons"]

    sizes = [abs(r["contribution"]) for r in reasons]
    assert sizes == sorted(sizes, reverse=True)


def test_each_reason_says_which_way_it_pushed(explained) -> None:
    result, _, _, config = explained
    reasons = explain.explain_one(result, 0, config, "Amount", "Class")["reasons"]

    for reason in reasons:
        towards = reason["contribution"] > 0
        assert reason["direction"] == ("towards fraud" if towards else "away from fraud")


# --------------------------------------------------------------------------------------
# Choosing examples
# --------------------------------------------------------------------------------------


def test_examples_cover_the_failures_not_only_the_successes() -> None:
    """A report showing only correct predictions teaches a reviewer nothing."""
    frame = pd.DataFrame({"Class": [1, 1, 0, 0, 1, 0]})
    probabilities = np.array([0.9, 0.8, 0.95, 0.1, 0.05, 0.02])

    picked = explain.pick_examples(frame, probabilities, 0.5, "Class", 3)
    kinds = {kind for kind, _ in picked}

    assert "caught fraud" in kinds
    assert "false alarm" in kinds
    assert "missed fraud" in kinds


def test_examples_are_capped_at_the_requested_count() -> None:
    frame = pd.DataFrame({"Class": [1] * 10 + [0] * 10})
    probabilities = np.concatenate([np.full(10, 0.9), np.full(10, 0.8)])

    assert len(explain.pick_examples(frame, probabilities, 0.5, "Class", 4)) <= 4


def test_examples_survive_a_split_with_no_false_alarms() -> None:
    frame = pd.DataFrame({"Class": [1, 1, 0]})
    probabilities = np.array([0.9, 0.1, 0.05])

    picked = explain.pick_examples(frame, probabilities, 0.5, "Class", 3)

    assert picked
    assert {kind for kind, _ in picked} <= {"caught fraud", "missed fraud"}


# --------------------------------------------------------------------------------------
# Shapes and failures
# --------------------------------------------------------------------------------------


def test_the_positive_class_is_pulled_from_either_shap_layout() -> None:
    """The shap API returns a list in some versions and a 3d array in others."""
    as_list = [np.zeros((4, 3)), np.ones((4, 3))]
    as_array = np.stack([np.zeros((4, 3)), np.ones((4, 3))], axis=-1)

    np.testing.assert_array_equal(explain._positive_class_values(as_list), np.ones((4, 3)))
    np.testing.assert_array_equal(explain._positive_class_values(as_array), np.ones((4, 3)))


def test_a_two_dimensional_result_is_passed_through() -> None:
    plain = np.ones((4, 3))
    np.testing.assert_array_equal(explain._positive_class_values(plain), plain)


def test_a_mismatched_shape_is_rejected(toy, explain_config, monkeypatch) -> None:
    """Silently misaligned SHAP values would attach every reason to the wrong feature."""
    frame, features = toy
    pipeline = _fit(RandomForestClassifier(n_estimators=5, random_state=0), toy)

    monkeypatch.setattr(explain, "_positive_class_values", lambda _: np.zeros((3, 99)))

    with pytest.raises(ExplanationError, match="cannot be lined up"):
        explain.explain(
            pipeline,
            frame[features].to_numpy(dtype=float),
            frame.head(3),
            features,
            explain_config,
        )


# --------------------------------------------------------------------------------------
# The stages end to end
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run everything up to and including evaluation, so stages 6 and SHAP have inputs."""
    from fraud_pipeline import eda, evaluate, features, ingest, train, validation
    from fraud_pipeline.paths import default_config_path

    from .synthetic import make_transactions, write_raw_csv

    tmp_path = tmp_path_factory.mktemp("explainreg")
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
            "eda.permutation_shuffles=4",
            "eda.mutual_info_sample=600",
            "eda.figures=false",
            "training.feature_sets=[selected]",
            "training.bootstrap_samples=20",
            "models.logistic_regression.enabled=false",
            "models.xgboost.enabled=false",
            "models.neural_net.enabled=false",
            "models.lightgbm.enabled=false",
            "models.random_forest.n_estimators=8",
            "models.random_forest.max_depth=4",
            "explainability.shap_background_samples=30",
            "explainability.shap_background_clusters=4",
            "explainability.shap_explain_samples=15",
            "explainability.shap_examples=3",
            "registry.enabled=false",
        ],
    )

    write_raw_csv(config, make_transactions(config, rows=5000, fraud_rows=40))
    ingest.run(config)
    validation.run(config)
    eda.run(config)
    features.run(config)
    train.run(config)
    result = evaluate.run(config)
    return config, result


def test_the_explain_stage_writes_its_report(pipeline_run) -> None:
    config, _ = pipeline_run
    result = explain.run(config)

    assert result is not None
    tables = config.paths.tables_dir()
    assert (tables / config.explainability.report_file).is_file()
    assert (tables / "shap_importance.csv").is_file()

    report = (tables / config.explainability.report_file).read_text(encoding="utf-8")
    assert "What the model is actually using" in report
    assert "Worked examples" in report


def test_the_explain_stage_can_be_switched_off(pipeline_run) -> None:
    config, _ = pipeline_run
    off = load_config(
        None,
        overrides=[
            f"paths.data_processed={config.paths.processed().as_posix()}",
            f"paths.data_interim={config.paths.interim().as_posix()}",
            f"paths.tables={config.paths.tables_dir().as_posix()}",
            f"paths.models={config.paths.model_dir().as_posix()}",
            "explainability.shap_enabled=false",
        ],
    )
    assert explain.run(off) is None


def test_explaining_without_an_evaluation_explains_itself(tmp_path) -> None:
    empty = load_config(None, overrides=[f"paths.tables={(tmp_path / 'nope').as_posix()}"])
    with pytest.raises(ExplanationError, match="Run the evaluate stage first"):
        explain.run(empty)


def test_the_explained_sample_includes_every_alert(pipeline_run) -> None:
    """A random sample at this base rate would contain almost no alerts."""
    config, evaluation = pipeline_run
    result = explain.run(config)

    flagged = int(evaluation.test.true_positives + evaluation.test.false_positives)
    assert len(result.explained) >= flagged
