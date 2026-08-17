"""Tests for the torch network wrapper.

Two things are being checked. That it is a well behaved scikit learn estimator, because the
whole point of the wrapper is that the sweep can treat it like any other model. And that it
actually learns, because a network that runs and returns noise is worse than one that crashes:
it produces a number, the number goes in a results table, and nothing announces the problem.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_pipeline.neural import TorchMLPClassifier


@pytest.fixture(scope="module")
def imbalanced_data():
    """A separable but rare positive class, shaped like the real problem."""
    X, y = make_classification(
        n_samples=6000,
        n_features=16,
        n_informative=8,
        weights=[0.98],
        random_state=0,
    )
    return X[:4500], y[:4500], X[4500:], y[4500:]


def _quick(**overrides) -> TorchMLPClassifier:
    """A small, fast network. The defaults are for the real sweep, not for tests."""
    settings = {
        "hidden_sizes": [16],
        "epochs": 6,
        "batch_size": 256,
        "early_stopping_patience": 0,
        "random_state": 0,
    }
    settings.update(overrides)
    return TorchMLPClassifier(**settings)


# --------------------------------------------------------------------------------------
# It behaves like a scikit learn estimator
# --------------------------------------------------------------------------------------


def test_it_can_be_cloned_like_any_estimator() -> None:
    """clone() reads the constructor arguments back, so they must round trip exactly."""
    model = TorchMLPClassifier(hidden_sizes=[8, 4], dropout=0.25, learning_rate=0.005)
    copy = clone(model)

    assert copy.hidden_sizes == [8, 4]
    assert copy.dropout == 0.25
    assert copy.learning_rate == 0.005


def test_get_params_exposes_everything_the_constructor_takes() -> None:
    params = TorchMLPClassifier().get_params()

    for name in (
        "hidden_sizes",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "class_weight",
        "random_state",
    ):
        assert name in params


def test_predict_proba_has_the_scikit_learn_shape(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data
    model = _quick().fit(train_x, train_y)

    probabilities = model.predict_proba(test_x)

    assert probabilities.shape == (len(test_x), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_returns_labels(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data
    model = _quick().fit(train_x, train_y)

    predictions = model.predict(test_x)

    assert predictions.shape == (len(test_x),)
    assert set(np.unique(predictions)) <= {0, 1}


def test_it_records_the_classes_it_saw(imbalanced_data) -> None:
    train_x, train_y, _, _ = imbalanced_data
    model = _quick().fit(train_x, train_y)

    assert list(model.classes_) == [0, 1]
    assert model.n_features_in_ == train_x.shape[1]


def test_it_fits_inside_a_pipeline_with_a_scaler(imbalanced_data) -> None:
    """How it is actually used. The network is the only model here that needs the scaler.

    Trained properly rather than for the six epochs the other structural tests use, because
    this one asserts the result is useful and six epochs is not enough to learn anything.
    """
    train_x, train_y, test_x, test_y = imbalanced_data
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", _quick(epochs=25, hidden_sizes=[32, 16]))]
    )

    pipeline.fit(train_x, train_y)
    scores = pipeline.predict_proba(test_x)[:, 1]

    assert average_precision_score(test_y, scores) > test_y.mean() * 4


# --------------------------------------------------------------------------------------
# It actually learns
# --------------------------------------------------------------------------------------


def test_it_beats_the_base_rate_by_a_wide_margin(imbalanced_data) -> None:
    """A network that runs and returns noise is the failure this catches.

    Average precision equal to the positive rate is what random scores achieve, so the bar is
    comfortably above it rather than merely above chance.
    """
    train_x, train_y, test_x, test_y = imbalanced_data
    model = _quick(epochs=25, hidden_sizes=[32, 16]).fit(train_x, train_y)

    scores = model.predict_proba(test_x)[:, 1]
    base_rate = test_y.mean()

    assert average_precision_score(test_y, scores) > base_rate * 4


def test_it_separates_the_two_classes(imbalanced_data) -> None:
    train_x, train_y, test_x, test_y = imbalanced_data
    model = _quick(epochs=25, hidden_sizes=[32, 16]).fit(train_x, train_y)

    scores = model.predict_proba(test_x)[:, 1]

    assert scores[test_y == 1].mean() > scores[test_y == 0].mean()


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_predictions(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data

    first = _quick(random_state=7).fit(train_x, train_y).predict_proba(test_x)
    second = _quick(random_state=7).fit(train_x, train_y).predict_proba(test_x)

    np.testing.assert_allclose(first, second)


def test_a_different_seed_gives_a_different_network(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data

    first = _quick(random_state=1).fit(train_x, train_y).predict_proba(test_x)
    second = _quick(random_state=2).fit(train_x, train_y).predict_proba(test_x)

    assert not np.allclose(first, second)


# --------------------------------------------------------------------------------------
# Class weighting
# --------------------------------------------------------------------------------------


def test_class_weighting_raises_the_predicted_probabilities(imbalanced_data) -> None:
    """The same effect logistic regression showed, and the reason Brier score is reported.

    Weighting the rare class up improves ranking and pushes every probability upwards, so the
    scores stop being probabilities in any useful sense.
    """
    train_x, train_y, test_x, _ = imbalanced_data

    plain = _quick(epochs=15).fit(train_x, train_y).predict_proba(test_x)[:, 1]
    weighted = (
        _quick(epochs=15, class_weight="balanced").fit(train_x, train_y).predict_proba(test_x)[:, 1]
    )

    assert weighted.mean() > plain.mean() * 3


def test_an_unknown_class_weight_value_is_ignored_rather_than_guessed(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data

    none_weighted = _quick(class_weight=None).fit(train_x, train_y).predict_proba(test_x)
    other = _quick(class_weight="something_else").fit(train_x, train_y).predict_proba(test_x)

    np.testing.assert_allclose(none_weighted, other)


# --------------------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------------------


def test_early_stopping_can_finish_before_the_last_epoch(imbalanced_data) -> None:
    train_x, train_y, _, _ = imbalanced_data
    model = TorchMLPClassifier(
        hidden_sizes=[8],
        epochs=40,
        batch_size=256,
        early_stopping_patience=2,
        random_state=0,
    ).fit(train_x, train_y)

    assert model.n_epochs_run_ <= 40
    assert not np.isnan(model.best_score_)


def test_early_stopping_is_declined_when_positives_are_too_scarce() -> None:
    """Stopping on a score computed from two positives is stopping on noise."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 6))
    target = np.zeros(400, dtype=int)
    target[:3] = 1

    model = TorchMLPClassifier(
        hidden_sizes=[8], epochs=4, batch_size=128, early_stopping_patience=2, random_state=0
    ).fit(features, target)

    assert model.n_epochs_run_ == 4, "it should train every epoch rather than stop on noise"
    assert np.isnan(model.best_score_)


def test_patience_of_zero_disables_early_stopping(imbalanced_data) -> None:
    train_x, train_y, _, _ = imbalanced_data
    model = _quick(epochs=5, early_stopping_patience=0).fit(train_x, train_y)

    assert model.n_epochs_run_ == 5
    assert np.isnan(model.best_score_)


# --------------------------------------------------------------------------------------
# Refusing to guess
# --------------------------------------------------------------------------------------


def test_predicting_before_fitting_is_an_error(imbalanced_data) -> None:
    _, _, test_x, _ = imbalanced_data

    with pytest.raises(RuntimeError, match="not been fitted"):
        TorchMLPClassifier().predict_proba(test_x)


def test_the_wrong_number_of_features_is_an_error(imbalanced_data) -> None:
    train_x, train_y, test_x, _ = imbalanced_data
    model = _quick().fit(train_x, train_y)

    with pytest.raises(ValueError, match="expected 16 features"):
        model.predict_proba(test_x[:, :5])


def test_mismatched_lengths_are_an_error() -> None:
    with pytest.raises(ValueError, match="rows"):
        _quick().fit(np.zeros((10, 3)), np.zeros(7))


def test_a_one_dimensional_feature_matrix_is_an_error() -> None:
    with pytest.raises(ValueError, match="2d feature matrix"):
        _quick().fit(np.zeros(10), np.zeros(10))
