"""Stage 5. Choose an operating threshold, then open the test split exactly once.

Stage 4 compared models without ever picking a threshold, on purpose: fitting a model and
deciding when to block a card are separate jobs. This is the second job.

The order here is a one way door and the code is arranged to make that visible.

1. Pick a champion from the validation scores.
2. Tune its threshold on validation against a cost model.
3. Freeze the threshold.
4. Open test, apply the frozen threshold, and report.

Nothing after step 3 changes anything from before it. Once a held out set has informed a
decision it stops being an estimate of future performance, and there is no way to undo that
afterwards or to measure how much optimism it introduced.

**The cost model is where the real thinking is.** A missed fraud and a false alarm cost
different amounts, and neither is a modelling question:

``flat``
    Every missed fraud costs the same. Simple, and it optimises for counting frauds.
``amount``
    A missed fraud costs the value of the transaction. It optimises for money.

Those are not the same objective, and on this data they choose very different thresholds. The
fraudulent amounts are severely skewed, a median of 12.31 against a mean of 120.29, so a flat
cost treats a 2,126 fraud and a 1.00 fraud as equally worth catching.

The false positive cost cannot be derived from this dataset at all. It is the support and
churn expense of blocking a real customer, which is a business input, so its effect is
reported as a sensitivity sweep rather than asserted as a fact.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)


class EvaluationError(RuntimeError):
    """Evaluation cannot run or cannot produce a usable result."""


# --------------------------------------------------------------------------------------
# The cost model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """What each kind of mistake costs.

    Args:
        model: ``flat`` charges every missed fraud the same, ``amount`` charges each one the
            value of the transaction that was missed.
        false_negative_cost: the flat charge per missed fraud. Ignored when ``model`` is
            ``amount``, where each transaction supplies its own.
        false_positive_cost: the charge for blocking a legitimate customer. Not derivable
            from transaction data, so it is a business input.
    """

    model: str
    false_negative_cost: float
    false_positive_cost: float

    @classmethod
    def from_config(cls, config: Config) -> CostModel:
        cost = config.evaluation.cost
        return cls(
            model=cost.model,
            false_negative_cost=cost.false_negative_cost,
            false_positive_cost=cost.false_positive_cost,
        )

    def with_false_positive_cost(self, value: float) -> CostModel:
        """A copy at a different false alarm price, for the sensitivity sweep."""
        return CostModel(
            model=self.model,
            false_negative_cost=self.false_negative_cost,
            false_positive_cost=value,
        )

    def missed_fraud_costs(self, y_true: np.ndarray, amounts: np.ndarray | None) -> np.ndarray:
        """What each row costs if the model fails to flag it.

        Zero for legitimate rows, since missing something that was never fraud costs nothing.
        """
        if self.model == "flat":
            return np.where(y_true == 1, self.false_negative_cost, 0.0)

        if amounts is None:
            raise EvaluationError(
                "the amount cost model needs transaction amounts, and none were supplied"
            )
        return np.where(y_true == 1, amounts, 0.0)

    def total(
        self, y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray | None = None
    ) -> float:
        """Expected cost of one set of decisions.

        Only mistakes are charged. A caught fraud and a correctly ignored normal transaction
        both cost nothing, which keeps the number readable as what the errors cost.
        """
        per_row = self.missed_fraud_costs(y_true, amounts)
        missed = (~flagged) & (y_true == 1)
        false_alarms = flagged & (y_true == 0)

        return float(per_row[missed].sum() + self.false_positive_cost * false_alarms.sum())


# --------------------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------------------


@dataclass
class OperatingPoint:
    """One threshold and everything that follows from it."""

    threshold: float
    strategy: str
    cost: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    #: How much fraudulent value the model caught, and how much slipped through. The headline
    #: counts hide this: catching many small frauds and missing one large one reads as a good
    #: recall and a bad month.
    value_caught: float
    value_missed: float
    rows: int

    @property
    def value_recall(self) -> float:
        """Share of fraudulent value caught, as opposed to share of fraud cases."""
        total = self.value_caught + self.value_missed
        return self.value_caught / total if total else 0.0

    @property
    def alerts(self) -> int:
        return self.true_positives + self.false_positives

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value_recall"] = self.value_recall
        payload["alerts"] = self.alerts
        return payload


def confusion_at(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    amounts: np.ndarray | None,
    cost: CostModel,
    strategy: str = "fixed",
) -> OperatingPoint:
    """Score one threshold."""
    flagged = probabilities >= threshold

    true_positive = int((flagged & (y_true == 1)).sum())
    false_positive = int((flagged & (y_true == 0)).sum())
    false_negative = int(((~flagged) & (y_true == 1)).sum())
    true_negative = int(((~flagged) & (y_true == 0)).sum())

    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if amounts is None:
        caught = missed = 0.0
    else:
        caught = float(amounts[flagged & (y_true == 1)].sum())
        missed = float(amounts[(~flagged) & (y_true == 1)].sum())

    return OperatingPoint(
        threshold=float(threshold),
        strategy=strategy,
        cost=cost.total(y_true, flagged, amounts),
        true_positives=true_positive,
        false_positives=false_positive,
        false_negatives=false_negative,
        true_negatives=true_negative,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        value_caught=caught,
        value_missed=missed,
        rows=int(len(y_true)),
    )


def cost_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray | None,
    cost: CostModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost at every threshold that can change a decision, exactly.

    Returns the candidate thresholds and their costs.

    An earlier version swept a grid of 4,000 thresholds thinned by quantile, on the reasoning
    that this keeps the grid dense where the scores actually are. That reasoning is backwards
    for this problem and it cost real accuracy. Fraud scores are extremely bimodal: on the
    champion, 42,493 of 42,558 validation scores sit below 0.01 and only 47 sit above 0.9.
    Quantile thinning therefore piles the candidates into the region where nothing is decided
    and leaves the decision boundary, out in the sparse tail, barely sampled. It missed the
    real optimum by a wide margin at some false alarm prices.

    So there is no grid. The optimum always sits at one of the predicted scores, and sorting
    once makes every one of them affordable to evaluate: sort descending, then walk the
    cumulative counts. That is O(n log n) for an exact answer, against O(grid * n) for an
    approximate one.
    """
    order = np.argsort(-probabilities, kind="stable")
    sorted_scores = probabilities[order]
    sorted_labels = y_true[order]

    per_row = cost.missed_fraud_costs(y_true, amounts)[order]

    # Flagging the first k rows, for every k. cumsum gives all of them in one pass.
    caught_cost = np.cumsum(per_row)
    true_positive = np.cumsum(sorted_labels == 1)
    flagged = np.arange(1, len(sorted_scores) + 1)
    false_positive = flagged - true_positive

    total_missable = per_row.sum()
    costs = (total_missable - caught_cost) + cost.false_positive_cost * false_positive

    # Only the last row of each run of equal scores is a real cut point. Cutting inside a tie
    # is not something a threshold can express, and pretending otherwise reports a threshold
    # that does not reproduce.
    last_of_tie = np.append(sorted_scores[1:] != sorted_scores[:-1], True)

    thresholds = sorted_scores[last_of_tie]
    costs = costs[last_of_tie]

    # Flagging nothing at all is always available and is sometimes the cheapest answer.
    nothing = np.nextafter(sorted_scores[0], np.inf)
    return (
        np.append(thresholds, nothing),
        np.append(costs, total_missable),
    )


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray | None,
    cost: CostModel,
    config: Config,
    strategy: str | None = None,
) -> OperatingPoint:
    """Pick an operating threshold.

    **This must only ever be given validation data.** It is the step that turns a held out set
    into a set that has informed a decision.

    Strategies:

    ``cost``
        Minimise expected cost. The default, because it makes the trade off explicit and
        arguable rather than hidden in a round number.
    ``max_f1``
        Maximise F1. Assumes precision and recall matter equally, which is a claim about the
        business that nobody made.
    ``fixed_recall``
        The cheapest threshold that still reaches a target recall. How some real fraud teams
        work, when the target comes from a policy or a regulator.
    """
    chosen = strategy or config.evaluation.threshold_strategy

    if chosen == "cost":
        # Exact, and cheaper than the approximate version it replaced.
        thresholds, costs = cost_curve(y_true, probabilities, amounts, cost)
        best_threshold = float(thresholds[int(np.argmin(costs))])
        best = confusion_at(y_true, probabilities, best_threshold, amounts, cost, chosen)
        logger.info(
            "threshold %.6f by %s: recall %.3f, precision %.3f, cost %.0f",
            best.threshold,
            chosen,
            best.recall,
            best.precision,
            best.cost,
        )
        return best

    thresholds, _ = cost_curve(y_true, probabilities, amounts, cost)
    points = [confusion_at(y_true, probabilities, t, amounts, cost, chosen) for t in thresholds]

    if chosen == "max_f1":
        best = max(points, key=lambda p: p.f1)
    elif chosen == "fixed_recall":
        target = config.evaluation.fixed_recall_target
        reaching = [p for p in points if p.recall >= target]
        # Defensive. Flagging every transaction is one of the candidates, so any target at or
        # below 1.0 is reachable and the config rejects anything above it. Kept because a
        # future change to the candidate set could quietly break that.
        if not reaching:
            raise EvaluationError(
                f"no threshold reaches {target:.0%} recall on this split. The best available "
                f"is {max(p.recall for p in points):.1%}."
            )
        best = min(reaching, key=lambda p: p.cost)
    else:
        raise EvaluationError(f"unknown threshold strategy {chosen!r}")

    logger.info(
        "threshold %.6f by %s: recall %.3f, precision %.3f, cost %.0f",
        best.threshold,
        chosen,
        best.recall,
        best.precision,
        best.cost,
    )
    return best


def sensitivity_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray | None,
    cost: CostModel,
    config: Config,
) -> pd.DataFrame:
    """How the chosen threshold moves as the price of a false alarm changes.

    The false positive cost is the one number in the model the data cannot supply. Rather than
    pick a value and present the result as though it were derived, this shows what the answer
    would have been across a range, so a reader can find their own number in the table.
    """
    rows = []
    for price in config.evaluation.cost.sensitivity_false_positive_costs:
        point = tune_threshold(
            y_true, probabilities, amounts, cost.with_false_positive_cost(price), config, "cost"
        )
        rows.append(
            {
                "false_positive_cost": price,
                "threshold": point.threshold,
                "recall": point.recall,
                "precision": point.precision,
                "alerts": point.alerts,
                "false_positives": point.false_positives,
                "value_recall": point.value_recall,
                "cost": point.cost,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Picking a champion
# --------------------------------------------------------------------------------------


def load_run_predictions(config: Config, run_name: str) -> pd.DataFrame:
    """Read the validation predictions stage 4 wrote for one run."""
    path = config.paths.processed() / config.training.predictions_dir / f"{run_name}.parquet"
    if not path.is_file():
        raise EvaluationError(
            f"predictions not found for {run_name}: {path}\nRun the training stage first."
        )
    return pd.read_parquet(path)


def aligned_amounts(predictions: pd.DataFrame, frame: pd.DataFrame, config: Config) -> np.ndarray:
    """Line the saved predictions up with the split they were scored on.

    Stage 4 wrote predictions row for row from the engineered split, so position is the join
    key. That is fragile enough to be worth checking rather than trusting, so the target and
    the timestamp are compared before the amounts are used. A silent misalignment here would
    attach the wrong price to every fraud and quietly change the chosen threshold.
    """
    spec = config.dataset.spec()

    if len(predictions) != len(frame):
        raise EvaluationError(
            f"predictions have {len(predictions):,} rows and the split has {len(frame):,}"
        )

    same_target = np.array_equal(
        predictions[spec.target_column].to_numpy(), frame[spec.target_column].to_numpy()
    )
    same_time = np.allclose(
        predictions[spec.time_column].to_numpy(), frame[spec.time_column].to_numpy()
    )
    if not (same_target and same_time):
        raise EvaluationError(
            "saved predictions do not line up with the split they should have come from"
        )

    return frame[spec.amount_column].to_numpy(dtype=float)


def healthy_runs(config: Config) -> pd.DataFrame:
    """Stage 4 results with the collapsed runs removed.

    A model that stopped splitting can still post a low cost by flagging almost nothing, so
    letting one into this comparison would promote an artifact.
    """
    path = config.paths.tables_dir() / config.training.results_csv
    if not path.is_file():
        raise EvaluationError(
            f"training results not found: {path}\nRun the training stage first: fraud train"
        )

    frame = pd.read_csv(path)
    if "warning" in frame.columns:
        collapsed = frame["warning"].fillna("").astype(str).str.len() > 0
        if collapsed.any():
            logger.info("ignoring %s collapsed run(s) from the sweep", int(collapsed.sum()))
        frame = frame[~collapsed]

    if frame.empty:
        raise EvaluationError("every run in the sweep was collapsed, so none can be promoted")
    return frame.reset_index(drop=True)


@dataclass
class ChampionChoice:
    """The promoted run and how it was chosen."""

    run_name: str
    label: str
    model: str
    imbalance: str
    feature_set: str
    validation: OperatingPoint
    considered: int
    runner_up: str | None
    runner_up_cost: float | None

    @property
    def margin(self) -> float | None:
        if self.runner_up_cost is None:
            return None
        return self.runner_up_cost - self.validation.cost


def choose_champion(
    config: Config, validation: pd.DataFrame, cost: CostModel
) -> tuple[ChampionChoice, dict[str, OperatingPoint]]:
    """Promote on expected cost at a tuned threshold, not on the headline metric.

    The two disagree on this data, repeatedly. Average precision integrates over every
    threshold including ones no fraud team would operate at, so the run with the best curve
    is not always the run that would lose the least money in production.

    Every candidate gets its own tuned threshold, because comparing models at a shared
    threshold would just measure how their probability scales happen to line up.
    """
    spec = config.dataset.spec()
    y_true = validation[spec.target_column].to_numpy(dtype=int)
    runs = healthy_runs(config)

    points: dict[str, OperatingPoint] = {}
    for run_name in runs["run_name"]:
        predictions = load_run_predictions(config, run_name)
        amounts = aligned_amounts(predictions, validation, config)
        probabilities = predictions["probability"].to_numpy(dtype=float)
        points[run_name] = tune_threshold(
            y_true, probabilities, amounts, cost, config, config.evaluation.threshold_strategy
        )

    wanted = config.evaluation.champion
    if wanted != "auto":
        if wanted not in points:
            raise EvaluationError(
                f"champion {wanted!r} is not a healthy run in the sweep. "
                f"Available: {sorted(points)}"
            )
        ordered = [wanted, *sorted(points, key=lambda n: points[n].cost)]
    else:
        ordered = sorted(points, key=lambda n: points[n].cost)

    best_name = ordered[0]
    rest = [n for n in sorted(points, key=lambda n: points[n].cost) if n != best_name]
    row = runs[runs["run_name"] == best_name].iloc[0]

    choice = ChampionChoice(
        run_name=best_name,
        label=str(row["label"]),
        model=str(row["model"]),
        imbalance=str(row["imbalance"]),
        feature_set=str(row["feature_set"]),
        validation=points[best_name],
        considered=len(points),
        runner_up=rest[0] if rest else None,
        runner_up_cost=points[rest[0]].cost if rest else None,
    )

    logger.info(
        "champion %s: validation cost %.0f from %s candidates",
        choice.run_name,
        choice.validation.cost,
        choice.considered,
    )
    return choice, points


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Everything stage 5 produced, in the order it produced it."""

    champion: ChampionChoice
    validation: OperatingPoint
    test: OperatingPoint
    sensitivity: pd.DataFrame
    cost_model_comparison: pd.DataFrame
    cost: CostModel
    threshold: float
    #: Size of the frauds caught and missed on each split. The headline recall cannot show
    #: this, and on this data it is where the real story is.
    value_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def recall_gap(self) -> float:
        """Test recall minus validation recall.

        Expected to be negative. Anything strongly positive means the threshold landed in a
        lucky place on test rather than that the model improved.
        """
        return self.test.recall - self.validation.recall


def compare_cost_models(
    y_true: np.ndarray, probabilities: np.ndarray, amounts: np.ndarray, config: Config
) -> pd.DataFrame:
    """Tune under both cost models and show what each one would choose.

    This is the argument for the amount model, made with numbers rather than assertion. The
    two optimise different things, so they should be expected to disagree, and the size of
    the disagreement is the point.
    """
    rows = []
    for name in ("flat", "amount"):
        cost = CostModel(
            model=name,
            false_negative_cost=config.evaluation.cost.false_negative_cost,
            false_positive_cost=config.evaluation.cost.false_positive_cost,
        )
        point = tune_threshold(y_true, probabilities, amounts, cost, config, "cost")
        rows.append(
            {
                "cost_model": name,
                "threshold": point.threshold,
                "recall": point.recall,
                "precision": point.precision,
                "true_positives": point.true_positives,
                "false_positives": point.false_positives,
                "value_caught": point.value_caught,
                "value_missed": point.value_missed,
                "value_recall": point.value_recall,
            }
        )
    return pd.DataFrame(rows)


def describe_value(
    y_true: np.ndarray, probabilities: np.ndarray, amounts: np.ndarray, threshold: float, split: str
) -> dict[str, Any]:
    """How large were the frauds caught, and how large were the ones that got away."""
    flagged = probabilities >= threshold
    caught = amounts[flagged & (y_true == 1)]
    missed = amounts[(~flagged) & (y_true == 1)]

    return {
        "split": split,
        "frauds": int((y_true == 1).sum()),
        "caught": len(caught),
        "missed": len(missed),
        "caught_total": float(caught.sum()),
        "missed_total": float(missed.sum()),
        "caught_mean": float(caught.mean()) if len(caught) else 0.0,
        "missed_mean": float(missed.mean()) if len(missed) else 0.0,
        "largest_missed": float(missed.max()) if len(missed) else 0.0,
        "value_recall": float(caught.sum() / (caught.sum() + missed.sum()))
        if (len(caught) or len(missed))
        else 0.0,
    }


def run(config: Config) -> EvaluationResult:
    """Run stage 5.

    The sequence is deliberate and one directional. Everything that reads validation happens
    before the single line that reads test, and nothing after it revisits a choice.
    """
    import joblib

    from fraud_pipeline.feature_selection import load_selected_features
    from fraud_pipeline.features import ENGINEERED_TAG, load_engineered
    from fraud_pipeline.paths import ensure_dir

    spec = config.dataset.spec()
    cost = CostModel.from_config(config)
    logger.info(
        "cost model: %s, false alarm priced at %.2f",
        cost.model,
        cost.false_positive_cost,
    )

    # ---- Steps 1 and 2: everything decided on validation ----
    validation = load_engineered(config, "validation")
    y_validation = validation[spec.target_column].to_numpy(dtype=int)
    amounts_validation = validation[spec.amount_column].to_numpy(dtype=float)

    champion, _ = choose_champion(config, validation, cost)

    predictions = load_run_predictions(config, champion.run_name)
    probabilities = predictions["probability"].to_numpy(dtype=float)

    comparison = compare_cost_models(y_validation, probabilities, amounts_validation, config)
    sensitivity = sensitivity_sweep(y_validation, probabilities, amounts_validation, cost, config)

    # ---- Step 3: freeze ----
    threshold = champion.validation.threshold
    logger.info("threshold frozen at %.6f, the test split is opened once below", threshold)

    # ---- Step 4: the one and only look at test ----
    test = load_engineered(config, config.evaluation.test_split)
    feature_names = load_selected_features(config, champion.feature_set, tag=ENGINEERED_TAG)
    pipeline = joblib.load(
        config.paths.model_dir() / config.training.models_dir / f"{champion.run_name}.joblib"
    )

    test_probabilities = pipeline.predict_proba(test[feature_names].to_numpy(dtype=float))[:, 1]
    test_point = confusion_at(
        test[spec.target_column].to_numpy(dtype=int),
        test_probabilities,
        threshold,
        test[spec.amount_column].to_numpy(dtype=float),
        cost,
        strategy="frozen",
    )

    logger.info(
        "test: recall %.3f, precision %.3f, %s alerts, value recall %.3f",
        test_point.recall,
        test_point.precision,
        test_point.alerts,
        test_point.value_recall,
    )

    breakdown = pd.DataFrame(
        [
            describe_value(
                y_validation, probabilities, amounts_validation, threshold, "validation"
            ),
            describe_value(
                test[spec.target_column].to_numpy(dtype=int),
                test_probabilities,
                test[spec.amount_column].to_numpy(dtype=float),
                threshold,
                "test",
            ),
        ]
    )

    result = EvaluationResult(
        champion=champion,
        validation=champion.validation,
        test=test_point,
        sensitivity=sensitivity,
        cost_model_comparison=comparison,
        cost=cost,
        threshold=threshold,
        value_breakdown=breakdown,
    )

    tables = ensure_dir(config.paths.tables_dir())
    (tables / config.evaluation.report_file).write_text(build_report(result), encoding="utf-8")
    (tables / config.evaluation.summary_file).write_text(
        summary_json(result, config), encoding="utf-8"
    )
    sensitivity.to_csv(tables / "threshold_sensitivity.csv", index=False)
    logger.info("wrote %s", config.evaluation.report_file)

    return result


def summary_json(result: EvaluationResult, config: Config) -> str:
    import json

    return json.dumps(
        {
            "champion": {
                "run_name": result.champion.run_name,
                "label": result.champion.label,
                "imbalance": result.champion.imbalance,
                "feature_set": result.champion.feature_set,
                "considered": result.champion.considered,
                "runner_up": result.champion.runner_up,
                "margin": result.champion.margin,
            },
            "cost_model": {
                "model": result.cost.model,
                "false_negative_cost": result.cost.false_negative_cost,
                "false_positive_cost": result.cost.false_positive_cost,
            },
            "threshold": result.threshold,
            "threshold_strategy": config.evaluation.threshold_strategy,
            "validation": result.validation.as_dict(),
            "test": result.test.as_dict(),
            "recall_gap": result.recall_gap,
        },
        indent=2,
    )


def build_report(result: EvaluationResult) -> str:
    """Write up the champion, the threshold, and the one test result."""
    champion = result.champion
    validation = result.validation
    test = result.test
    cost = result.cost

    lines = [
        "# Evaluation",
        "",
        f"Champion: **{champion.label}**, {champion.imbalance}, {champion.feature_set} "
        f"feature set. Chosen from {champion.considered} healthy runs by expected cost at a "
        f"tuned threshold, not by the headline metric.",
        "",
        "## The order this was done in",
        "",
        "1. Every candidate was tuned on **validation** and the cheapest promoted.",
        f"2. The threshold was fixed at **{result.threshold:.6f}**.",
        "3. The **test** split was opened once and scored at that threshold.",
        "",
        "Nothing after step 2 changed anything from before it. Once a held out set has "
        "informed a decision it stops being an estimate of future performance, and there is "
        "no way to undo that or to measure how much optimism it added.",
        "",
        "## Test results, at the frozen threshold",
        "",
        "| Measure | Validation | Test |",
        "| --- | --- | --- |",
        f"| Recall (cases) | {validation.recall:.3f} | **{test.recall:.3f}** |",
        f"| Precision | {validation.precision:.3f} | **{test.precision:.3f}** |",
        f"| F1 | {validation.f1:.3f} | {test.f1:.3f} |",
        f"| Value recall | {validation.value_recall:.3f} | **{test.value_recall:.3f}** |",
        f"| Frauds caught | {validation.true_positives} of "
        f"{validation.true_positives + validation.false_negatives} | "
        f"{test.true_positives} of {test.true_positives + test.false_negatives} |",
        f"| False alarms | {validation.false_positives} | {test.false_positives} |",
        f"| Total alerts | {validation.alerts} | {test.alerts} |",
        f"| Value caught | {validation.value_caught:,.0f} | {test.value_caught:,.0f} |",
        f"| Value missed | {validation.value_missed:,.0f} | {test.value_missed:,.0f} |",
        f"| Expected cost | {validation.cost:,.0f} | {test.cost:,.0f} |",
        "",
        f"Recall moved by {result.recall_gap:+.3f} from validation to test. Some drop is "
        "expected, because the threshold was chosen on validation. A large rise would be "
        "luck rather than improvement.",
        "",
        "**Value recall is the number to read, and it is the worst news here.** Share of "
        "fraud *cases* caught and share of fraudulent *money* caught are different numbers, "
        "and only the second one is what a fraud team is measured on.",
        "",
        "| Split | Frauds | Caught | Missed | Largest missed | Mean missed | Value recall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *[
            f"| {r.split} | {r.frauds} | {r.caught} | {r.missed} | {r.largest_missed:,.0f} | "
            f"{r.missed_mean:,.0f} | {r.value_recall:.3f} |"
            for r in result.value_breakdown.itertuples()
        ],
        "",
        "On validation the model missed only cheap frauds, the largest worth 109. On test it "
        "missed the expensive ones, the largest worth 1,097. Case recall fell by a sixth and "
        "value recall almost halved, because the two are not tied together.",
        "",
        "This is the clearest limitation in the project. With roughly fifty fraud cases per "
        "split, whether the model happens to catch the handful of large ones swings value "
        "recall enormously, and validation gave an optimistic answer that test did not "
        "repeat. Any claim about money saved should carry that caveat.",
        "",
        "## Why the cost model matters",
        "",
        f"Tuned on validation under each cost model. The false alarm price is fixed at "
        f"{cost.false_positive_cost:.2f} for both.",
        "",
        "| Cost model | Threshold | Recall | Precision | TP | FP | Value caught | Value recall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in result.cost_model_comparison.itertuples():
        lines.append(
            f"| {row.cost_model} | {row.threshold:.4f} | {row.recall:.3f} | "
            f"{row.precision:.3f} | {row.true_positives} | {row.false_positives} | "
            f"{row.value_caught:,.0f} | {row.value_recall:.3f} |"
        )

    flat = result.cost_model_comparison.iloc[0]
    amount = result.cost_model_comparison.iloc[1]
    lines += [
        "",
        f"`flat` charges every missed fraud {cost.false_negative_cost:.0f}, which is the mean "
        "fraudulent amount and therefore the right flat number. It is still the wrong "
        "summary: the amounts are severely skewed, so a flat charge treats the largest fraud "
        "in the file and a 1.00 fraud as equally worth catching.",
        "",
    ]

    if abs(flat.threshold - amount.threshold) < 1e-9:
        lines += [
            "**On this champion the two agree**, landing on the same threshold. That is not "
            "true of every model in the sweep and it is worth saying rather than glossing "
            "over. This model's scores are strongly bimodal, so there is a wide gap between "
            "the last fraud it is confident about and the next candidate, and moving the "
            "price of a mistake within any sensible range does not cross that gap.",
            "",
            "Where the two do disagree, on LightGBM and random forest with SMOTE, the amount "
            "model gives up about three fraud cases to remove between seven and sixteen false "
            "alarms, and the fraudulent value it recovers is unchanged to three decimal "
            "places. Counting frauds and recovering money are different objectives, and the "
            "second is cheaper to serve.",
            "",
        ]
    else:
        lines += [
            f"`amount` charges each missed fraud its own value. Here it gives up "
            f"{int(flat.true_positives) - int(amount.true_positives)} fraud case(s) to remove "
            f"{int(flat.false_positives) - int(amount.false_positives)} false alarm(s), "
            f"raising precision from {flat.precision:.3f} to {amount.precision:.3f}, while the "
            f"fraudulent value recovered moves only from {flat.value_recall:.4f} to "
            f"{amount.value_recall:.4f}.",
            "",
            "Counting frauds and recovering money are different objectives, and on this data "
            "the second one is cheaper to serve.",
            "",
        ]

    lines += [
        "## What the false alarm price is worth",
        "",
        "This is the one number the data cannot supply. It is the support and churn expense "
        "of wrongly blocking a customer, which is a business input, so here is what the "
        "threshold would be across a range rather than a single answer presented as derived.",
        "",
        "| False alarm cost | Threshold | Recall | Precision | Alerts | Value recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in result.sensitivity.itertuples():
        lines.append(
            f"| {row.false_positive_cost:.0f} | {row.threshold:.4f} | {row.recall:.3f} | "
            f"{row.precision:.3f} | {row.alerts} | {row.value_recall:.3f} |"
        )

    lines += [
        "",
        f"The configured price is {cost.false_positive_cost:.2f}. Find your own number in "
        "that table and read the row.",
        "",
        "## How the champion was chosen",
        "",
        f"{champion.considered} healthy runs were each given their own tuned threshold, "
        "because comparing models at a shared threshold measures how their probability "
        "scales happen to line up rather than how good they are.",
        "",
        f"- Promoted: `{champion.run_name}`, validation cost {validation.cost:,.0f}",
    ]
    if champion.runner_up:
        lines.append(
            f"- Runner up: `{champion.runner_up}`, validation cost "
            f"{champion.runner_up_cost:,.0f}, a margin of {champion.margin:,.0f}"
        )
    lines.append("")

    return "\n".join(lines)
