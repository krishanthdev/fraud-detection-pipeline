"""Scoring, and how much of a score to believe.

The metrics themselves are one line each from scikit learn. The part worth writing carefully
is the uncertainty, because this is where an imbalanced problem misleads people.

Validation holds 55 fraud cases. A model that catches one more of them gains about two
points of recall, so two models three points apart may not differ at all. Reporting a single
number to four decimals invites reading it as solid, so every primary score here comes with a
bootstrap interval attached, and the results table shows the interval next to the number.

Accuracy is absent on purpose. At a 0.18 percent positive rate a model that never predicts
fraud scores 99.82 percent, and there is a test in the suite that fails if anyone adds it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricError(RuntimeError):
    """A score cannot be computed on the data given."""


@dataclass(frozen=True)
class Interval:
    """A bootstrap interval on a metric."""

    low: float
    high: float
    confidence: float
    samples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def format(self, decimals: int = 4) -> str:
        return f"[{self.low:.{decimals}f}, {self.high:.{decimals}f}]"


@dataclass(frozen=True)
class Scores:
    """Threshold free scores for one set of predictions.

    Everything here is computed from the ranking or the calibration of the probabilities, so
    none of it depends on a decision threshold. Choosing that threshold is a separate
    question with a cost model attached, and it belongs in stage 5. Fitting a model and
    deciding when to block a card are different jobs and mixing them makes both harder to
    reason about.
    """

    average_precision: float
    roc_auc: float
    brier_score: float
    positives: int
    rows: int
    #: Precision at the recall the cost model is likely to want. A cheap early read on
    #: whether a model is usable at all, without committing to a threshold.
    precision_at_50_recall: float
    precision_at_80_recall: float
    #: How many different values the model actually emitted.
    #:
    #: Not a quality metric, a health check. A tree ensemble that has stopped splitting still
    #: returns valid probabilities and still produces a plausible looking AUC, but it emits
    #: only a handful of distinct values, so most rows are tied and the ranking is arbitrary.
    #: Nothing else here reveals that. See the build log for the run where it mattered.
    distinct_scores: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _check(y_true: np.ndarray, y_score: np.ndarray, *, probabilities: bool = False) -> None:
    if len(y_true) != len(y_score):
        raise MetricError(f"length mismatch: {len(y_true)} labels, {len(y_score)} scores")
    if len(y_true) == 0:
        raise MetricError("no rows to score")

    positives = int(np.sum(y_true))
    if positives == 0:
        raise MetricError("no positive cases in the labels, so nothing can be measured")
    if positives == len(y_true):
        raise MetricError("every case is positive, so nothing can be measured")

    if not np.isfinite(y_score).all():
        raise MetricError("scores contain missing or infinite values")

    if probabilities and (y_score.min() < 0.0 or y_score.max() > 1.0):
        # Ranking metrics work on any monotonic score, but the Brier score is a squared
        # error against the label, so it only means anything for a calibrated probability.
        # Saying that here is clearer than letting scikit learn raise about it three frames
        # down, and it catches a decision function being passed in by mistake.
        raise MetricError(
            f"scores must be probabilities between 0 and 1 for calibration to be measured, "
            f"got a range of [{y_score.min():.4g}, {y_score.max():.4g}]. "
            f"Pass predict_proba output rather than a decision function."
        )


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float) -> float:
    """The best precision available at or above a given recall.

    Read off the precision recall curve rather than by picking a threshold, so it answers
    "if this model had to catch 80 percent of fraud, how many false alarms would that cost"
    without committing to how the threshold gets chosen.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    reachable = precision[recall >= target_recall]
    return float(reachable.max()) if len(reachable) else 0.0


def score(y_true: np.ndarray, y_score: np.ndarray) -> Scores:
    """Compute every threshold free metric for one set of predictions.

    ``y_score`` must be probabilities between 0 and 1, that is ``predict_proba`` output. The
    ranking metrics would accept any monotonic score, but the Brier score is a squared error
    against the label and is meaningless on an unbounded one.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    _check(y_true, y_score, probabilities=True)

    return Scores(
        average_precision=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        brier_score=float(brier_score_loss(y_true, y_score)),
        positives=int(y_true.sum()),
        rows=int(len(y_true)),
        precision_at_50_recall=precision_at_recall(y_true, y_score, 0.50),
        precision_at_80_recall=precision_at_recall(y_true, y_score, 0.80),
        # Rounded, because floating point noise in the last digits would count two
        # indistinguishable scores as different and hide exactly the collapse this detects.
        distinct_scores=int(np.unique(np.round(y_score, 9)).size),
    )


def bootstrap_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    samples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """A percentile bootstrap interval on average precision.

    Rows are resampled with replacement and the metric recomputed each time. The spread of
    those values is how much the score would move on a different sample of the same size,
    which is the question worth asking when the positive class is this small.

    Resampling rows rather than the two classes separately is deliberate. The number of fraud
    cases that happen to fall in a validation window is itself uncertain, and a stratified
    bootstrap would hold it fixed and report an interval that is too narrow.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    _check(y_true, y_score)

    rng = np.random.default_rng(seed)
    rows = len(y_true)
    values = []

    for _ in range(samples):
        index = rng.integers(0, rows, size=rows)
        resampled = y_true[index]
        # A draw with no fraud at all cannot be scored. At 55 positives in 42,558 rows this
        # is astronomically unlikely, but skipping it is cheaper than reasoning about it.
        if resampled.sum() == 0:
            continue
        values.append(average_precision_score(resampled, y_score[index]))

    if not values:
        raise MetricError("every bootstrap sample was unusable")

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return Interval(low=float(low), high=float(high), confidence=confidence, samples=len(values))


def intervals_overlap(first: Interval, second: Interval) -> bool:
    """Whether two intervals overlap.

    Used to decide whether the results table is allowed to call one model better than
    another. Overlapping intervals do not prove two models are equal, but they do mean the
    data cannot tell them apart, which is the honest thing to report.
    """
    return first.low <= second.high and second.low <= first.high
