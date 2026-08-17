"""Tests for the model registry and how imbalance strategies are wired in.

The test that matters here is `test_smote_never_resamples_the_data_being_scored`. Resampling
the evaluation set is the classic way to produce a fraud model that looks excellent and is
worthless, and it does not announce itself: the metrics simply come back high.
"""

from __future__ import annotations

import numpy as np
import pytest
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_pipeline import models
from fraud_pipeline.config import load_config
from fraud_pipeline.models import ModelError


@pytest.fixture
def model_config(config_path):
    return load_config(config_path)


def _imbalanced_data(rows: int = 800, positives: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.zeros(rows, dtype=int)
    y[rng.choice(rows, size=positives, replace=False)] = 1
    x = rng.normal(0, 1, (rows, 6))
    x[y == 1] += 1.5
    return x, y


def _tiny_params(name: str) -> dict:
    """Enough of each model to build, small enough that a test fits in a second."""
    return {
        "logistic_regression": {"max_iter": 50},
        "random_forest": {"n_estimators": 5},
        "xgboost": {"n_estimators": 5},
        "lightgbm": {"n_estimators": 5},
        "neural_net": {"hidden_sizes": [8], "epochs": 2, "early_stopping_patience": 0},
    }[name]


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


def test_every_configured_model_is_now_implemented(model_config) -> None:
    """The PLANNED mechanism is empty, so nothing in the config gets silently skipped."""
    from fraud_pipeline import models as models_module

    assert models_module.PLANNED == {}
    assert set(models.available_models(model_config)) == set(model_config.enabled_models())


def test_all_five_models_are_registered(model_config) -> None:
    available = models.available_models(model_config)

    assert set(available) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "neural_net",
    }


def test_only_the_network_and_the_linear_model_need_scaling(model_config) -> None:
    """Trees split on order, so scaling them is wasted work with no effect on the result."""
    scaled = {name: d.needs_scaling for name, d in models.available_models(model_config).items()}

    assert scaled["logistic_regression"] is True
    assert scaled["neural_net"] is True
    assert scaled["random_forest"] is False
    assert scaled["xgboost"] is False
    assert scaled["lightgbm"] is False


# --------------------------------------------------------------------------------------
# The two routes to weighting the rare class
# --------------------------------------------------------------------------------------


def test_xgboost_weights_through_scale_pos_weight(model_config) -> None:
    """The one model without scikit learn's class_weight keyword."""
    definition = models.REGISTRY["xgboost"]
    assert definition.weighting == "scale_pos_weight"

    pipeline = models.build_pipeline(
        definition, {"n_estimators": 5}, "class_weight", model_config, pos_weight=99.0
    )

    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == 99.0


def test_the_others_weight_through_the_class_weight_keyword(model_config) -> None:
    for name in ("logistic_regression", "random_forest", "lightgbm", "neural_net"):
        definition = models.REGISTRY[name]
        assert definition.weighting == "class_weight", name

        pipeline = models.build_pipeline(
            definition, _tiny_params(name), "class_weight", model_config, pos_weight=99.0
        )
        assert pipeline.named_steps["model"].get_params()["class_weight"] == "balanced", name


def test_xgboost_refuses_to_guess_a_missing_weight(model_config) -> None:
    """Silently training unweighted would look like a working run and quietly change it."""
    with pytest.raises(models.ModelError, match="needs pos_weight"):
        models.build_pipeline(
            models.REGISTRY["xgboost"], {"n_estimators": 5}, "class_weight", model_config
        )


def test_the_positive_weight_is_the_class_ratio() -> None:
    labels = np.array([0] * 990 + [1] * 10)
    assert models.positive_class_weight(labels) == pytest.approx(99.0)


def test_the_positive_weight_needs_a_positive_class() -> None:
    with pytest.raises(models.ModelError, match="no positive rows"):
        models.positive_class_weight(np.zeros(50, dtype=int))


def test_scale_pos_weight_is_not_set_for_the_other_strategies(model_config) -> None:
    for strategy in ("none", "smote"):
        pipeline = models.build_pipeline(
            models.REGISTRY["xgboost"],
            {"n_estimators": 5},
            strategy,
            model_config,
            pos_weight=99.0,
        )
        params = pipeline.named_steps["model"].get_params()
        assert params.get("scale_pos_weight") in (None, 1, 1.0), strategy


def test_a_planned_model_is_skipped_rather_than_failing(model_config, monkeypatch) -> None:
    """The mechanism that let this sweep grow one branch at a time.

    Nothing is planned any more, so the behaviour is tested with a name that is configured and
    deliberately not implemented. It is kept because it is how the next model gets added
    without restructuring the sweep, and because being skipped with a message naming the branch
    is very different from crashing on a name nobody recognises.
    """
    spec_type = type(model_config.models["xgboost"])
    monkeypatch.setitem(model_config.models, "future_model", spec_type(enabled=True))
    monkeypatch.setitem(models.PLANNED, "future_model", "feature/some-later-branch")

    available = models.available_models(model_config)

    assert "future_model" in model_config.enabled_models()
    assert "future_model" not in available
    # Everything already implemented still comes through.
    assert "xgboost" in available


def test_an_implemented_model_wins_over_a_stale_planned_entry(model_config, monkeypatch) -> None:
    """Forgetting to clear a PLANNED entry must not silently drop a working model."""
    monkeypatch.setitem(models.PLANNED, "xgboost", "feature/already-done")

    assert "xgboost" in models.available_models(model_config)


def test_an_unknown_model_in_the_config_is_an_error(config_path) -> None:
    config = load_config(config_path)
    config.models["mystery_model"] = type(config.models["xgboost"])(enabled=True)

    with pytest.raises(ModelError, match="unknown"):
        models.available_models(config)


def test_a_disabled_model_is_not_offered(config_path) -> None:
    config = load_config(config_path, overrides=["models.random_forest.enabled=false"])
    assert "random_forest" not in models.available_models(config)


def test_run_names_are_stable_and_descriptive(model_config) -> None:
    definition = models.REGISTRY["logistic_regression"]
    name = models.describe(definition, "smote", "selected")

    assert name == "logistic_regression__smote__selected"
    assert models.describe(definition, "smote", "selected") == name


# --------------------------------------------------------------------------------------
# Pipeline assembly
# --------------------------------------------------------------------------------------


def test_logistic_regression_gets_a_scaler(model_config) -> None:
    """Amount runs to 25,000 while the components sit near zero, so scaling is not optional."""
    pipeline = models.build_pipeline(
        models.REGISTRY["logistic_regression"], {"max_iter": 100}, "none", model_config
    )

    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_random_forest_does_not_get_a_scaler(model_config) -> None:
    """A tree splits on order, so scaling changes nothing and only costs time."""
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "none", model_config
    )

    assert "scaler" not in pipeline.named_steps
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)


def test_class_weight_is_passed_to_the_estimator(model_config) -> None:
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "class_weight", model_config
    )
    assert pipeline.named_steps["model"].class_weight == "balanced"


def test_the_none_strategy_leaves_the_weights_alone(model_config) -> None:
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "none", model_config
    )
    assert pipeline.named_steps["model"].class_weight is None


def test_smote_is_a_pipeline_step_not_a_preprocessing_call(model_config) -> None:
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "smote", model_config
    )

    assert isinstance(pipeline, ImbalancedPipeline)
    assert "smote" in pipeline.named_steps


def test_a_pipeline_without_smote_is_a_plain_sklearn_pipeline(model_config) -> None:
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "none", model_config
    )
    assert isinstance(pipeline, Pipeline)
    assert not isinstance(pipeline, ImbalancedPipeline)


def test_smote_does_not_also_apply_class_weights(model_config) -> None:
    """Resampling and reweighting together would count the rare class twice."""
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 10}, "smote", model_config
    )
    assert pipeline.named_steps["model"].class_weight is None


def test_an_unknown_strategy_is_rejected(model_config) -> None:
    with pytest.raises(ModelError, match="unknown imbalance strategy"):
        models.build_pipeline(models.REGISTRY["random_forest"], {}, "magic", model_config)


def test_a_model_that_cannot_weight_rejects_the_weight_strategy(model_config) -> None:
    definition = models.ModelDefinition(
        name="unweightable",
        label="Unweightable",
        build=lambda params, seed: LogisticRegression(random_state=seed, **params),
        needs_scaling=False,
        weighting=None,
    )

    assert not definition.supports("class_weight")
    with pytest.raises(ModelError, match="cannot use"):
        models.build_pipeline(definition, {}, "class_weight", model_config)
    # SMOTE is still available to it, which is the point of tracking this separately.
    assert definition.supports("smote")


# --------------------------------------------------------------------------------------
# The one that matters
# --------------------------------------------------------------------------------------


def test_smote_never_resamples_the_data_being_scored(model_config) -> None:
    """Resampling the evaluation set produces a model that looks excellent and is worthless.

    An imblearn pipeline applies a sampler during fit and skips it during predict. This
    asserts that, so the guarantee is structural rather than a matter of remembering.
    """
    x, y = _imbalanced_data()
    pipeline = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 20}, "smote", model_config
    )
    pipeline.fit(x, y)

    probabilities = pipeline.predict_proba(x)

    # One prediction per input row. A resampled prediction step would return more.
    assert len(probabilities) == len(x)
    assert probabilities.shape == (len(x), 2)


def test_smote_actually_balances_the_training_data(model_config) -> None:
    """The other half of the guarantee: it does resample during fit."""
    x, y = _imbalanced_data(rows=600, positives=20)
    sampler = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 5}, "smote", model_config
    ).named_steps["smote"]

    x_resampled, y_resampled = sampler.fit_resample(x, y)

    assert len(x_resampled) > len(x)
    assert y_resampled.mean() > y.mean()


def test_every_pipeline_fits_and_predicts(model_config) -> None:
    """A smoke test across the whole grid, so an assembly mistake cannot hide."""
    x, y = _imbalanced_data()

    for name, definition in models.REGISTRY.items():
        for strategy in models.IMBALANCE_STRATEGIES:
            pipeline = models.build_pipeline(
                definition,
                _tiny_params(name),
                strategy,
                model_config,
                pos_weight=models.positive_class_weight(y),
            )
            pipeline.fit(x, y)
            probabilities = pipeline.predict_proba(x)[:, 1]

            assert len(probabilities) == len(y)
            assert np.isfinite(probabilities).all()
            assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_the_same_seed_gives_the_same_model(model_config) -> None:
    x, y = _imbalanced_data()

    first = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 15}, "smote", model_config
    ).fit(x, y)
    second = models.build_pipeline(
        models.REGISTRY["random_forest"], {"n_estimators": 15}, "smote", model_config
    ).fit(x, y)

    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
