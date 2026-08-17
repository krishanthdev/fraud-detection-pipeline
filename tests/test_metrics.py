"""Tests for scoring and uncertainty.

The metrics are thin wrappers, so what gets tested here is the behaviour that matters on an
imbalanced problem: that a useless model scores like a useless model, that the interval
widens when there is less to go on, and that accuracy never appears.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_pipeline import metrics
from fraud_pipeline.metrics import Interval, MetricError


def _labels_and_scores(
    rows: int = 2000, positives: int = 40, separation: float = 3.0, seed: int = 0
):
    """Labels plus probabilities.

    Squashed through a sigmoid rather than left as raw normal scores, because that is what a
    classifier actually returns and what the Brier score needs.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(rows, dtype=int)
    y[rng.choice(rows, size=positives, replace=False)] = 1
    logits = rng.normal(0, 1, rows) + y * separation
    return y, 1.0 / (1.0 + np.exp(-logits))


# --------------------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------------------


def test_a_good_ranking_scores_well() -> None:
    y, scores = _labels_and_scores(separation=4.0)
    result = metrics.score(y, scores)

    assert result.roc_auc > 0.95
    assert result.average_precision > 0.5
    assert result.positives == 40
    assert result.rows == 2000


def test_a_useless_ranking_scores_like_chance() -> None:
    """A random score should land near the base rate on PR AUC, not near zero."""
    y, _ = _labels_and_scores()
    noise = np.random.default_rng(1).random(len(y))

    result = metrics.score(y, noise)
    base_rate = y.mean()

    assert 0.4 < result.roc_auc < 0.6
    assert result.average_precision < base_rate * 4


def test_distinct_scores_counts_what_the_model_actually_emitted() -> None:
    """The health check that caught a collapsed LightGBM run.

    A model that has stopped splitting still returns valid probabilities and still produces a
    plausible ROC AUC. The only thing that gives it away is how few different values it emits.
    """
    labels = np.array([0, 0, 0, 0, 1, 1])

    varied = metrics.score(labels, np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))
    collapsed = metrics.score(labels, np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9]))

    assert varied.distinct_scores == 6
    assert collapsed.distinct_scores == 2


def test_distinct_scores_ignores_floating_point_dust() -> None:
    """Two scores differing in the fifteenth decimal are the same score.

    Without rounding, a collapsed model would report as many distinct scores as it has rows
    and the check would never fire.
    """
    labels = np.array([0, 0, 0, 1])
    almost_identical = np.array([0.5, 0.5 + 1e-15, 0.5 - 1e-15, 0.9])

    assert metrics.score(labels, almost_identical).distinct_scores == 2


def test_scoring_rejects_scores_that_are_not_probabilities() -> None:
    """A decision function passed in by mistake would silently corrupt the Brier score."""
    y = np.array([0, 0, 1, 1])

    with pytest.raises(MetricError, match="must be probabilities"):
        metrics.score(y, np.array([-2.0, 0.5, 3.4, 8.1]))


def test_scoring_rejects_non_finite_scores() -> None:
    y = np.array([0, 0, 1, 1])
    with pytest.raises(MetricError, match="missing or infinite"):
        metrics.score(y, np.array([0.1, 0.2, np.nan, 0.9]))


def test_a_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 0, 1, 1])
    result = metrics.score(y, np.array([0.1, 0.2, 0.3, 0.9, 0.95]))

    assert result.average_precision == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)


def test_an_inverted_ranking_scores_badly() -> None:
    y = np.array([0, 0, 0, 1, 1])
    result = metrics.score(y, np.array([0.95, 0.9, 0.3, 0.2, 0.1]))

    assert result.roc_auc < 0.2


def test_brier_score_rewards_calibration() -> None:
    """Two models can rank identically and still differ in how honest their numbers are."""
    y = np.array([0, 0, 0, 0, 1])
    confident = np.array([0.01, 0.02, 0.03, 0.04, 0.99])
    hedging = np.array([0.3, 0.31, 0.32, 0.33, 0.6])

    assert metrics.score(y, confident).brier_score < metrics.score(y, hedging).brier_score
    # Yet both rank the single positive top, so PR AUC cannot tell them apart.
    assert metrics.score(y, confident).average_precision == pytest.approx(
        metrics.score(y, hedging).average_precision
    )


def test_precision_at_recall_is_reachable() -> None:
    y, scores = _labels_and_scores(separation=3.0)
    result = metrics.score(y, scores)

    # Demanding more recall cannot buy more precision.
    assert result.precision_at_50_recall >= result.precision_at_80_recall
    assert 0.0 <= result.precision_at_80_recall <= 1.0


def test_precision_at_recall_is_zero_when_the_recall_is_unreachable() -> None:
    y = np.array([0, 1])
    assert metrics.precision_at_recall(y, np.array([0.5, 0.5]), 1.01) == 0.0


def test_scoring_rejects_data_with_no_positives() -> None:
    with pytest.raises(MetricError, match="no positive cases"):
        metrics.score(np.zeros(10, dtype=int), np.random.default_rng(0).random(10))


def test_scoring_rejects_data_with_only_positives() -> None:
    with pytest.raises(MetricError, match="every case is positive"):
        metrics.score(np.ones(10, dtype=int), np.random.default_rng(0).random(10))


def test_scoring_rejects_a_length_mismatch() -> None:
    with pytest.raises(MetricError, match="length mismatch"):
        metrics.score(np.array([0, 1]), np.array([0.5]))


def test_scoring_rejects_empty_input() -> None:
    with pytest.raises(MetricError, match="no rows"):
        metrics.score(np.array([], dtype=int), np.array([]))


def test_accuracy_is_not_among_the_scores() -> None:
    """At a 0.18 percent positive rate, accuracy is not a measurement."""
    y, scores = _labels_and_scores()
    assert "accuracy" not in metrics.score(y, scores).as_dict()


# --------------------------------------------------------------------------------------
# The interval
# --------------------------------------------------------------------------------------


def test_the_interval_contains_the_point_estimate() -> None:
    y, scores = _labels_and_scores(separation=3.0)
    point = metrics.score(y, scores).average_precision
    interval = metrics.bootstrap_interval(y, scores, samples=200, seed=7)

    assert interval.low <= point <= interval.high


def test_the_interval_widens_when_there_are_fewer_positives() -> None:
    """The whole reason the interval is reported. Fewer fraud cases, less certainty."""
    many_y, many_scores = _labels_and_scores(rows=4000, positives=400, separation=2.0)
    few_y, few_scores = _labels_and_scores(rows=4000, positives=20, separation=2.0)

    many = metrics.bootstrap_interval(many_y, many_scores, samples=200, seed=3)
    few = metrics.bootstrap_interval(few_y, few_scores, samples=200, seed=3)

    assert few.width > many.width


def test_the_interval_is_reproducible() -> None:
    y, scores = _labels_and_scores()
    first = metrics.bootstrap_interval(y, scores, samples=100, seed=11)
    second = metrics.bootstrap_interval(y, scores, samples=100, seed=11)

    assert first.low == second.low
    assert first.high == second.high


def test_a_different_seed_gives_a_slightly_different_interval() -> None:
    y, scores = _labels_and_scores()
    first = metrics.bootstrap_interval(y, scores, samples=100, seed=1)
    second = metrics.bootstrap_interval(y, scores, samples=100, seed=2)

    assert (first.low, first.high) != (second.low, second.high)


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    y, scores = _labels_and_scores()
    narrow = metrics.bootstrap_interval(y, scores, samples=300, confidence=0.5, seed=5)
    wide = metrics.bootstrap_interval(y, scores, samples=300, confidence=0.99, seed=5)

    assert wide.width > narrow.width


def test_the_interval_reports_how_many_samples_it_used() -> None:
    y, scores = _labels_and_scores()
    interval = metrics.bootstrap_interval(y, scores, samples=150, seed=0)
    assert interval.samples <= 150
    assert interval.samples > 100


def test_intervals_overlap_detects_overlap() -> None:
    a = Interval(0.10, 0.30, 0.95, 100)
    b = Interval(0.25, 0.45, 0.95, 100)
    c = Interval(0.40, 0.60, 0.95, 100)

    assert metrics.intervals_overlap(a, b)
    assert metrics.intervals_overlap(b, a)
    assert not metrics.intervals_overlap(a, c)


def test_intervals_that_touch_count_as_overlapping() -> None:
    a = Interval(0.1, 0.3, 0.95, 100)
    b = Interval(0.3, 0.5, 0.95, 100)
    assert metrics.intervals_overlap(a, b)


def test_interval_formats_readably() -> None:
    assert Interval(0.1234, 0.5678, 0.95, 100).format(3) == "[0.123, 0.568]"
