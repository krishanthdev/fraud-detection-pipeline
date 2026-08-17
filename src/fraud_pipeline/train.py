"""Stage 4. Fit every enabled model under every imbalance strategy and feature set.

The sweep is three dimensional: model, imbalance strategy, feature set. For the baselines
that is two models, three strategies and three feature sets, so eighteen fits.

Three rules shape this stage.

**Models are compared on validation. Test is not opened.**
    The config refuses to score on test here. Once a held out set has been used to pick
    between eighteen candidates, it is no longer held out, and reporting a number from it
    would be reporting the best of eighteen guesses rather than an estimate of future
    performance. Stage 5 opens it once, after the threshold is fixed.

**Nothing here picks a threshold.**
    Every metric logged is threshold free. Predicted probabilities are written to disk so
    stage 5 can tune a threshold against the cost model without refitting anything. Fitting
    a model and deciding when to block a card are separate jobs.

**Every score comes with an interval.**
    Validation holds 55 fraud cases. A single number to four decimals would read as solid
    when it is not, so the results table carries a bootstrap interval and refuses to rank
    models whose intervals overlap.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_pipeline import metrics, models
from fraud_pipeline.config import Config
from fraud_pipeline.features import ENGINEERED_TAG, load_engineered
from fraud_pipeline.paths import ensure_dir, repo_root

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Training cannot run or cannot produce a usable result."""


@dataclass
class RunResult:
    """Everything one fit produced."""

    model: str
    label: str
    imbalance: str
    feature_set: str
    #: Other configured set names that resolve to exactly this feature list. On the
    #: engineered features "safe" and "all" are the same 44 columns, because the only tier 1
    #: drop was Time and Time is not a model feature.
    feature_set_aliases: str
    n_features: int
    average_precision: float
    ap_low: float
    ap_high: float
    roc_auc: float
    brier_score: float
    precision_at_50_recall: float
    precision_at_80_recall: float
    train_average_precision: float
    fit_seconds: float
    eval_rows: int
    eval_positives: int
    run_name: str
    mlflow_run_id: str | None = None

    @property
    def interval(self) -> metrics.Interval:
        return metrics.Interval(low=self.ap_low, high=self.ap_high, confidence=0.95, samples=0)

    @property
    def overfit_gap(self) -> float:
        """How much better the model scores on data it was fitted on."""
        return self.train_average_precision - self.average_precision


@dataclass
class TrainingReport:
    """Every run from one sweep."""

    results: list[RunResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def best(self) -> RunResult:
        return max(self.results, key=lambda r: r.average_precision)

    def ranked(self) -> list[RunResult]:
        return sorted(self.results, key=lambda r: r.average_precision, reverse=True)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.results])

    def indistinguishable_from_best(self) -> list[RunResult]:
        """Runs whose interval overlaps the best run's interval.

        These cannot be separated from the winner on this validation set. Saying so is more
        useful than presenting a ranking that implies a precision the data does not support.
        """
        best = self.best()
        return [
            run
            for run in self.results
            if run.run_name != best.run_name
            and metrics.intervals_overlap(run.interval, best.interval)
        ]


# --------------------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------------------


def resolve_tracking_uri(config: Config) -> str:
    """Anchor a relative sqlite tracking URI at the repository root.

    ``sqlite:///mlflow.db`` is relative to the working directory, so running the pipeline
    from a different folder would silently start a second, empty experiment store. Anchoring
    it means one database wherever the command is run from.
    """
    uri = config.registry.tracking_uri
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri

    raw = uri[len(prefix) :]
    if Path(raw).is_absolute():
        return uri
    return f"{prefix}{(repo_root() / raw).as_posix()}"


class _NullTracker:
    """Stands in when MLflow is switched off, so the loop needs no branching."""

    def start(self, run_name: str) -> None:
        self.run_name = run_name

    def log(self, params: dict[str, Any], scores: dict[str, float]) -> None:
        pass

    def log_artifact(self, path: Path) -> None:
        pass

    def finish(self) -> str | None:
        return None


class _MlflowTracker:
    """Thin wrapper so the training loop never imports mlflow directly."""

    def __init__(self, config: Config) -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(resolve_tracking_uri(config))
        mlflow.set_experiment(config.registry.experiment_name)
        self._run = None

    def start(self, run_name: str) -> None:
        self._run = self._mlflow.start_run(run_name=run_name)

    def log(self, params: dict[str, Any], scores: dict[str, float]) -> None:
        self._mlflow.log_params(params)
        self._mlflow.log_metrics(scores)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish(self) -> str | None:
        run_id = self._run.info.run_id if self._run else None
        self._mlflow.end_run()
        self._run = None
        return run_id


def build_tracker(config: Config):
    """Return a tracker, falling back to a no op if MLflow cannot start.

    Tracking is valuable and it is not worth losing an eighteen fit sweep to. If the store
    cannot be opened the run continues and says so, because the results table on disk is the
    deliverable and MLflow is the convenience.
    """
    if not config.registry.enabled:
        logger.info("MLflow tracking is switched off in the config")
        return _NullTracker()

    try:
        return _MlflowTracker(config)
    except Exception as error:  # noqa: BLE001 - any failure here is non fatal by design
        logger.warning("MLflow could not start (%s), continuing without tracking", error)
        return _NullTracker()


# --------------------------------------------------------------------------------------
# One fit
# --------------------------------------------------------------------------------------


def resolve_feature_sets(config: Config) -> list[tuple[str, list[str], list[str]]]:
    """Load each configured feature set and merge any that are identical.

    Returns ``(name, aliases, features)`` for each distinct feature list.

    Merging matters more than it sounds. On the engineered features, ``safe`` and ``all`` are
    the same 44 columns, because the only tier 1 drop in the whole selection was ``Time`` and
    ``Time`` is not a model feature. Training both produced six duplicate fits and wasted
    about ten minutes of a thirty minute sweep, and then reported the identical numbers twice
    as though they were independent evidence.
    """
    from fraud_pipeline.feature_selection import load_selected_features

    seen: dict[tuple[str, ...], tuple[str, list[str], list[str]]] = {}
    order: list[tuple[str, ...]] = []

    for name in config.training.feature_sets:
        features = load_selected_features(config, name, tag=ENGINEERED_TAG)
        key = tuple(features)

        if key in seen:
            seen[key][1].append(name)
            logger.info("feature set %r is identical to %r, training it once", name, seen[key][0])
            continue

        seen[key] = (name, [], features)
        order.append(key)

    return [seen[key] for key in order]


def _feature_matrix(frame: pd.DataFrame, feature_names: list[str], target: str):
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise TrainingError(f"features missing from the engineered table: {missing}")
    return frame[feature_names].to_numpy(dtype=float), frame[target].to_numpy(dtype=int)


def fit_one(
    definition: models.ModelDefinition,
    strategy: str,
    feature_set: str,
    feature_names: list[str],
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    config: Config,
) -> tuple[RunResult, Any, pd.DataFrame]:
    """Fit one combination and score it. Returns the result, the pipeline and predictions."""
    target = config.dataset.spec().target_column
    x_train, y_train = _feature_matrix(train, feature_names, target)
    x_eval, y_eval = _feature_matrix(evaluation, feature_names, target)

    params = config.models[definition.name].params()
    pipeline = models.build_pipeline(definition, params, strategy, config)
    run_name = models.describe(definition, strategy, feature_set)

    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    eval_probabilities = pipeline.predict_proba(x_eval)[:, 1]
    train_probabilities = pipeline.predict_proba(x_train)[:, 1]

    scores = metrics.score(y_eval, eval_probabilities)
    interval = metrics.bootstrap_interval(
        y_eval,
        eval_probabilities,
        samples=config.training.bootstrap_samples,
        confidence=config.training.bootstrap_confidence,
        seed=config.project.seed,
    )
    train_scores = metrics.score(y_train, train_probabilities)

    predictions = pd.DataFrame(
        {
            config.dataset.spec().time_column: evaluation[config.dataset.spec().time_column],
            target: y_eval,
            "probability": eval_probabilities,
        }
    )

    result = RunResult(
        model=definition.name,
        label=definition.label,
        imbalance=strategy,
        feature_set=feature_set,
        feature_set_aliases="",
        n_features=len(feature_names),
        average_precision=scores.average_precision,
        ap_low=interval.low,
        ap_high=interval.high,
        roc_auc=scores.roc_auc,
        brier_score=scores.brier_score,
        precision_at_50_recall=scores.precision_at_50_recall,
        precision_at_80_recall=scores.precision_at_80_recall,
        train_average_precision=train_scores.average_precision,
        fit_seconds=fit_seconds,
        eval_rows=scores.rows,
        eval_positives=scores.positives,
        run_name=run_name,
    )

    return result, pipeline, predictions


# --------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------


def run(config: Config) -> TrainingReport:
    """Run the whole sweep and write the results table."""
    target = config.dataset.spec().target_column
    train = load_engineered(config, "train")
    evaluation = load_engineered(config, config.training.eval_split)

    logger.info(
        "training on %s rows (%s fraud), scoring on %s (%s fraud)",
        f"{len(train):,}",
        int(train[target].sum()),
        f"{len(evaluation):,}",
        int(evaluation[target].sum()),
    )

    definitions = models.available_models(config)
    if not definitions:
        raise TrainingError("no models are both enabled in the config and implemented")

    feature_sets = resolve_feature_sets(config)
    merged = sum(len(aliases) for _, aliases, _ in feature_sets)
    if merged:
        report_line = ", ".join(
            f"{name} = {'/'.join(aliases)}" for name, aliases, _ in feature_sets if aliases
        )
        logger.info("%s duplicate feature set(s) merged: %s", merged, report_line)

    report = TrainingReport()
    tracker = build_tracker(config)

    predictions_dir = ensure_dir(config.paths.processed() / config.training.predictions_dir)
    models_dir = ensure_dir(config.paths.model_dir() / config.training.models_dir)

    total = sum(
        len([s for s in config.imbalance.strategies if d.supports(s)]) for d in definitions.values()
    ) * len(feature_sets)
    logger.info("%s fits to run", total)

    done = 0
    for feature_set, aliases, feature_names in feature_sets:
        logger.info(
            "feature set %r%s: %s features",
            feature_set,
            f" (also {', '.join(aliases)})" if aliases else "",
            len(feature_names),
        )

        for definition in definitions.values():
            for strategy in config.imbalance.strategies:
                if not definition.supports(strategy):
                    message = f"{definition.name} does not support {strategy}"
                    logger.info("skipping: %s", message)
                    report.skipped.append(message)
                    continue

                done += 1
                logger.info(
                    "[%s/%s] %s | %s | %s",
                    done,
                    total,
                    definition.label,
                    strategy,
                    feature_set,
                )

                result, pipeline, predictions = fit_one(
                    definition, strategy, feature_set, feature_names, train, evaluation, config
                )
                result.feature_set_aliases = ", ".join(aliases)

                prediction_path = predictions_dir / f"{result.run_name}.parquet"
                predictions.to_parquet(prediction_path, index=False)

                if config.training.save_models:
                    joblib.dump(pipeline, models_dir / f"{result.run_name}.joblib")

                tracker.start(result.run_name)
                tracker.log(
                    params={
                        "model": definition.name,
                        "imbalance": strategy,
                        "feature_set": feature_set,
                        "n_features": len(feature_names),
                        "seed": config.project.seed,
                        "eval_split": config.training.eval_split,
                        **{
                            f"param_{k}": v
                            for k, v in config.models[definition.name].params().items()
                        },
                    },
                    scores={
                        "val_average_precision": result.average_precision,
                        "val_ap_low": result.ap_low,
                        "val_ap_high": result.ap_high,
                        "val_roc_auc": result.roc_auc,
                        "val_brier_score": result.brier_score,
                        "val_precision_at_50_recall": result.precision_at_50_recall,
                        "val_precision_at_80_recall": result.precision_at_80_recall,
                        "train_average_precision": result.train_average_precision,
                        "fit_seconds": result.fit_seconds,
                    },
                )
                tracker.log_artifact(prediction_path)
                result.mlflow_run_id = tracker.finish()

                logger.info(
                    "      PR AUC %.4f %s   ROC AUC %.4f   %.1fs",
                    result.average_precision,
                    metrics.Interval(result.ap_low, result.ap_high, 0.95, 0).format(),
                    result.roc_auc,
                    result.fit_seconds,
                )

                report.results.append(result)

    _write_results(report, config)
    return report


def _write_results(report: TrainingReport, config: Config) -> None:
    """Write the results table, as markdown for reading and csv for the next stage."""
    tables = ensure_dir(config.paths.tables_dir())

    frame = report.to_frame()
    frame.to_csv(tables / config.training.results_csv, index=False)

    report_path = tables / config.training.results_file
    report_path.write_text(_build_results_markdown(report, config), encoding="utf-8")
    logger.info("wrote %s", report_path.name)


def _build_results_markdown(report: TrainingReport, config: Config) -> str:
    best = report.best()
    tied = report.indistinguishable_from_best()
    ranked = report.ranked()

    lines = [
        "# Baseline model results",
        "",
        f"{len(report.results)} fits: "
        f"{len({r.model for r in report.results})} models x "
        f"{len({r.imbalance for r in report.results})} imbalance strategies x "
        f"{len({r.feature_set for r in report.results})} distinct feature sets.",
        "",
    ]

    aliased = [r for r in report.results if r.feature_set_aliases]
    if aliased:
        pairs = sorted({f"`{r.feature_set}` and `{r.feature_set_aliases}`" for r in aliased})
        lines += [
            f"Configured feature sets that turned out to be identical, so they were trained "
            f"once rather than twice: {', '.join(pairs)}. On the engineered features the only "
            f"tier 1 drop was `Time`, and `Time` is not a model feature, so the safe set has "
            f"nothing left to remove.",
            "",
        ]

    lines += [
        f"Scored on the **{config.training.eval_split}** split, "
        f"{best.eval_rows:,} rows and {best.eval_positives} fraud cases. "
        f"The test split has not been opened.",
        "",
        "## How to read this",
        "",
        f"The interval is a {config.training.bootstrap_confidence:.0%} bootstrap interval on "
        f"PR AUC, from {config.training.bootstrap_samples} resamples. With "
        f"{best.eval_positives} fraud cases in the scoring split, it is wide, and that width",
        "is the point: two models whose intervals overlap cannot be separated on this data,",
        "however different their headline numbers look.",
        "",
        "Accuracy is not reported. At this positive rate a model that never predicts fraud",
        "scores over 99.8 percent.",
        "",
        "## Results",
        "",
        "| Model | Imbalance | Features | PR AUC | Interval | ROC AUC | P@50% recall | P@80% recall | Brier | Fit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for run in ranked:
        set_label = run.feature_set
        if run.feature_set_aliases:
            set_label = f"{run.feature_set} = {run.feature_set_aliases}"
        lines.append(
            f"| {run.label} | {run.imbalance} | {set_label} ({run.n_features}) | "
            f"{run.average_precision:.4f} | "
            f"[{run.ap_low:.3f}, {run.ap_high:.3f}] | "
            f"{run.roc_auc:.4f} | {run.precision_at_50_recall:.3f} | "
            f"{run.precision_at_80_recall:.3f} | {run.brier_score:.5f} | "
            f"{run.fit_seconds:.1f}s |"
        )

    lines += [
        "",
        "## The best run",
        "",
        f"**{best.label}**, {best.imbalance}, {best.feature_set} feature set "
        f"({best.n_features} features).",
        "",
        f"- PR AUC {best.average_precision:.4f}, interval "
        f"[{best.ap_low:.3f}, {best.ap_high:.3f}]",
        f"- ROC AUC {best.roc_auc:.4f}",
        f"- Precision {best.precision_at_80_recall:.3f} at 80 percent recall",
        f"- On the training data it scores {best.train_average_precision:.4f}, "
        f"a gap of {best.overfit_gap:+.4f}",
        "",
    ]

    if tied:
        lines += [
            f"**{len(tied)} other runs cannot be told apart from it.** Their intervals all",
            "overlap the best run's, so this validation set does not have the resolution to",
            "rank them:",
            "",
        ]
        lines += [
            f"- {r.label}, {r.imbalance}, {r.feature_set}: {r.average_precision:.4f} "
            f"[{r.ap_low:.3f}, {r.ap_high:.3f}]"
            for r in tied[:8]
        ]
        lines.append("")
    else:
        lines += [
            "No other run's interval overlaps the best one, so the winner is separable on",
            "this data.",
            "",
        ]

    if report.skipped:
        lines += ["## Skipped combinations", ""]
        lines += [f"- {message}" for message in report.skipped]
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Reading results back
# --------------------------------------------------------------------------------------


def load_results(config: Config) -> pd.DataFrame:
    """Read the results table. Stage 5 uses this to pick which model to promote."""
    path = config.paths.tables_dir() / config.training.results_csv
    if not path.is_file():
        raise TrainingError(
            f"training results not found: {path}\nRun the train stage first: fraud train"
        )
    return pd.read_csv(path)


def load_predictions(config: Config, run_name: str) -> pd.DataFrame:
    """Read one run's validation predictions, so a threshold can be tuned without refitting."""
    path = config.paths.processed() / config.training.predictions_dir / f"{run_name}.parquet"
    if not path.is_file():
        raise TrainingError(f"predictions not found for {run_name}: {path}")
    return pd.read_parquet(path)


def load_model(config: Config, run_name: str):
    """Read one fitted pipeline back."""
    path = config.paths.model_dir() / config.training.models_dir / f"{run_name}.joblib"
    if not path.is_file():
        raise TrainingError(f"saved model not found for {run_name}: {path}")
    return joblib.load(path)


def summarise_by(report: TrainingReport, key: str) -> pd.DataFrame:
    """Average PR AUC grouped by one axis of the sweep.

    This is what answers the questions the sweep was built to answer: does feature selection
    help, and does imbalance handling help. Averaging over the other axes is crude, but it is
    the right first look before reading individual rows.
    """
    frame = report.to_frame()
    if key not in frame.columns:
        raise TrainingError(f"unknown grouping {key!r}")
    return (
        frame.groupby(key)["average_precision"]
        .agg(["mean", "min", "max", "count"])
        .sort_values("mean", ascending=False)
    )


def write_json_summary(report: TrainingReport, config: Config) -> Path:
    """A machine readable summary, for the serving stage and for quick checks."""
    path = ensure_dir(config.paths.tables_dir()) / "baseline_summary.json"
    best = report.best()
    payload = {
        "runs": len(report.results),
        "eval_split": config.training.eval_split,
        "eval_rows": best.eval_rows,
        "eval_positives": best.eval_positives,
        "best": asdict(best),
        "by_imbalance": summarise_by(report, "imbalance").to_dict(orient="index"),
        "by_feature_set": summarise_by(report, "feature_set").to_dict(orient="index"),
        "by_model": summarise_by(report, "model").to_dict(orient="index"),
    }
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
