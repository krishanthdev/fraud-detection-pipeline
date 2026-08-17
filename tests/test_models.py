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


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


def test_the_baselines_are_registered(model_config) -> None:
    available = models.available_models(model_config)

    assert "logistic_regression" in available
    assert "random_forest" in available


def test_unimplemented_models_are_skipped_not_failed(model_config) -> None:
    """XGBoost and friends are enabled in the config but arrive on a later branch."""
    available = models.available_models(model_config)

    for planned in ("xgboost", "lightgbm", "neural_net"):
        assert planned in model_config.enabled_models()
        assert planned not in available


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
    small = {"logistic_regression": {"max_iter": 200}, "random_forest": {"n_estimators": 10}}

    for name, definition in models.REGISTRY.items():
        for strategy in models.IMBALANCE_STRATEGIES:
            pipeline = models.build_pipeline(definition, small[name], strategy, model_config)
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
