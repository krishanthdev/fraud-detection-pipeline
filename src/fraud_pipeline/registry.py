"""Stage 6. Register the champion and decide whether it takes the title.

A registry is not a filing cabinet. Its job is to answer one question on every retrain: is
this new model better than the one currently in production, and by enough to justify swapping
it?

Two decisions shape this module.

**Promotion is judged on expected cost, not on the headline metric.** Stage 5 chose the
champion that way, so judging promotion any other way would let a model win the selection and
lose the promotion, or the reverse, for no reason a person could explain. The alternative is
kept as a config option because most registries work on a metric.

**A challenger has to win by a margin.** Without one, every rerun swaps the model on noise,
and with about fifty fraud cases in the scoring split the noise is large. A margin turns
"different" into "better", and the size of it is a decision worth arguing about rather than a
default worth ignoring.

MLflow is optional here for the same reason it is optional in training: the report on disk is
the deliverable and the tracking server is the convenience. If it cannot be reached the run
says so and carries on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)


class RegistryError(RuntimeError):
    """The registry cannot record or compare a model."""


@dataclass
class PromotionDecision:
    """Whether the challenger takes the title, and why."""

    promoted: bool
    reason: str
    basis: str
    challenger_run: str
    challenger_score: float
    champion_run: str | None
    champion_score: float | None
    margin_required: float
    margin_achieved: float | None
    model_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _better(basis: str, challenger: float, champion: float) -> float:
    """How much better the challenger is, in whatever direction counts as better.

    Cost is a thing to minimise and a metric is a thing to maximise, and getting the sign
    wrong here would silently promote the worse model every time.
    """
    return (champion - challenger) if basis == "cost" else (challenger - champion)


def decide(
    challenger_run: str,
    challenger_score: float,
    champion_run: str | None,
    champion_score: float | None,
    config: Config,
) -> PromotionDecision:
    """Compare a challenger against the incumbent.

    Pure, so the rule can be tested without a tracking server anywhere near it.
    """
    basis = config.registry.promotion_basis
    margin = config.registry.promotion_min_improvement

    if champion_run is None or champion_score is None:
        return PromotionDecision(
            promoted=True,
            reason="no champion is registered yet, so this one takes the title unopposed",
            basis=basis,
            challenger_run=challenger_run,
            challenger_score=challenger_score,
            champion_run=None,
            champion_score=None,
            margin_required=margin,
            margin_achieved=None,
        )

    achieved = _better(basis, challenger_score, champion_score)
    unit = "cost" if basis == "cost" else config.registry.promotion_metric

    if achieved >= margin:
        reason = (
            f"beats the champion by {achieved:.4f} on {unit}, which clears the "
            f"{margin:.4f} margin"
        )
    elif achieved > 0:
        reason = (
            f"is better by {achieved:.4f} on {unit}, but that is inside the {margin:.4f} "
            f"margin, so the difference is not worth a swap"
        )
    else:
        reason = f"does not beat the champion on {unit} ({achieved:+.4f})"

    return PromotionDecision(
        promoted=achieved >= margin,
        reason=reason,
        basis=basis,
        challenger_run=challenger_run,
        challenger_score=challenger_score,
        champion_run=champion_run,
        champion_score=champion_score,
        margin_required=margin,
        margin_achieved=achieved,
    )


# --------------------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------------------


def _client(config: Config):
    import mlflow
    from mlflow.tracking import MlflowClient

    from fraud_pipeline.train import resolve_tracking_uri

    mlflow.set_tracking_uri(resolve_tracking_uri(config))
    return MlflowClient()


def current_champion(config: Config) -> tuple[str | None, float | None, str | None]:
    """The registered model currently holding the title, if there is one.

    Returns the run name, its score, and the model version. Everything is read back from tags
    rather than recomputed, because the incumbent was scored on the data of its own day and
    rescoring it now would compare two different questions.
    """
    if not config.registry.enabled:
        # Switching tracking off has to mean not touching the store at all. Reading a
        # champion from a registry the run is not allowed to write to would compare against
        # a title nobody can defend, and it made a test silently consult the repository's own
        # database from inside a temporary directory.
        logger.info("MLflow is switched off, so no champion is read")
        return None, None, None

    try:
        client = _client(config)
        name = config.registry.registered_model_name
        stage = config.registry.model_stage

        versions = [
            v
            for v in client.search_model_versions(f"name='{name}'")
            if v.tags.get("stage") == stage
        ]
        if not versions:
            return None, None, None

        latest = max(versions, key=lambda v: int(v.version))
        score = latest.tags.get("promotion_score")
        return latest.tags.get("run_name"), float(score) if score else None, latest.version
    except Exception as error:  # noqa: BLE001 - a missing registry is not a failure
        logger.info("no champion could be read from the registry (%s)", error)
        return None, None, None


def build_signature(config: Config, run_name: str, feature_set: str):
    """Describe the inputs and outputs of the model being registered.

    Without one, MLflow warns and a served model accepts any shaped frame, failing somewhere
    inside the pipeline instead of at the door. With one, the registered model carries its own
    contract: these columns, these types, this output. Stage 7 gets that check for free.
    """
    import joblib
    from mlflow.models import infer_signature

    from fraud_pipeline.feature_selection import load_selected_features
    from fraud_pipeline.features import ENGINEERED_TAG, load_engineered

    names = load_selected_features(config, feature_set, tag=ENGINEERED_TAG)
    pipeline = joblib.load(
        config.paths.model_dir() / config.training.models_dir / f"{run_name}.joblib"
    )

    # A plain array, not a named frame, because that is how the model was fitted and how it
    # will be called. Declaring named columns would promise a contract the pipeline does not
    # have: its scaler was fitted without feature names, so passing a frame warns on every
    # call. The column *order* is the real contract and it lives in the selection file, which
    # is what the serving stage reads.
    sample = load_engineered(config, "train").head(50)[names].to_numpy(dtype=float)
    return infer_signature(sample, pipeline.predict_proba(sample)), sample


def register(
    config: Config,
    run_name: str,
    score: float,
    extra_tags: dict[str, str],
    feature_set: str = "selected",
) -> str | None:
    """Log the champion pipeline as a registered model version.

    Returns the version, or None when MLflow is unavailable. Unavailable is not an error: the
    evaluation report on disk is the deliverable.
    """
    import joblib

    try:
        import mlflow
        import mlflow.sklearn

        from fraud_pipeline.train import resolve_tracking_uri

        mlflow.set_tracking_uri(resolve_tracking_uri(config))
        mlflow.set_experiment(config.registry.experiment_name)

        path = config.paths.model_dir() / config.training.models_dir / f"{run_name}.joblib"
        if not path.is_file():
            raise RegistryError(f"no saved model to register: {path}")
        pipeline = joblib.load(path)

        with mlflow.start_run(run_name=f"register__{run_name}"):
            mlflow.log_params(
                {"champion_run": run_name, "promotion_basis": config.registry.promotion_basis}
            )
            mlflow.log_metric("promotion_score", score)

            signature, example = None, None
            try:
                signature, example = build_signature(config, run_name, feature_set)
            except Exception as error:  # noqa: BLE001 - a missing signature is not fatal
                logger.warning("could not infer a model signature (%s)", error)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=example,
                registered_model_name=config.registry.registered_model_name,
            )

        client = _client(config)
        versions = client.search_model_versions(f"name='{config.registry.registered_model_name}'")
        latest = max(versions, key=lambda v: int(v.version))

        for key, value in {
            "stage": config.registry.model_stage,
            "run_name": run_name,
            "promotion_score": str(score),
            **extra_tags,
        }.items():
            client.set_model_version_tag(
                config.registry.registered_model_name, latest.version, key, value
            )

        logger.info("registered version %s as %s", latest.version, config.registry.model_stage)
        return latest.version
    except RegistryError:
        raise
    except Exception as error:  # noqa: BLE001 - tracking is a convenience, not the deliverable
        logger.warning("could not register the model (%s), continuing", error)
        return None


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def run(config: Config, evaluation=None) -> PromotionDecision:
    """Run stage 6 against the champion stage 5 chose."""
    from fraud_pipeline import evaluate as evaluate_stage
    from fraud_pipeline.paths import ensure_dir

    summary_path = config.paths.tables_dir() / config.evaluation.summary_file
    if evaluation is None:
        if not summary_path.is_file():
            raise RegistryError(
                f"evaluation summary not found: {summary_path}\n"
                f"Run the evaluate stage first: fraud evaluate"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_name = summary["champion"]["run_name"]
        # The score the champion was chosen on, from validation. Test is reported, never used
        # to choose, and promoting on it would make the held out estimate part of the choice.
        score = (
            summary["validation"]["cost"]
            if config.registry.promotion_basis == "cost"
            else summary["validation"].get(config.registry.promotion_metric, 0.0)
        )
    else:
        run_name = evaluation.champion.run_name
        score = (
            evaluation.validation.cost
            if config.registry.promotion_basis == "cost"
            else getattr(evaluation.validation, config.registry.promotion_metric, 0.0)
        )
        summary = json.loads(evaluate_stage.summary_json(evaluation, config))

    champion_run, champion_score, _ = current_champion(config)
    decision = decide(run_name, score, champion_run, champion_score, config)

    logger.info("promotion: %s (%s)", "yes" if decision.promoted else "no", decision.reason)

    if decision.promoted and config.registry.enabled:
        decision.model_version = register(
            config,
            run_name,
            score,
            feature_set=summary["champion"]["feature_set"],
            extra_tags={
                "threshold": f"{summary['threshold']:.6f}",
                "test_recall": f"{summary['test']['recall']:.4f}",
                "test_value_recall": f"{summary['test']['value_recall']:.4f}",
            },
        )
    elif not config.registry.enabled:
        logger.info("MLflow is switched off, so nothing was written to a registry")

    report = ensure_dir(config.paths.tables_dir()) / config.registry.report_file
    report.write_text(build_report(decision, summary, config), encoding="utf-8")
    logger.info("wrote %s", config.registry.report_file)

    return decision


def build_report(decision: PromotionDecision, summary: dict[str, Any], config: Config) -> str:
    """Write up the promotion decision."""
    unit = "expected cost" if decision.basis == "cost" else config.registry.promotion_metric
    verdict = "Promoted" if decision.promoted else "Not promoted"

    lines = [
        "# Model registry",
        "",
        f"**{verdict}.** {decision.reason.capitalize()}.",
        "",
        "## The decision",
        "",
        "| | |",
        "| --- | --- |",
        f"| Challenger | `{decision.challenger_run}` |",
        f"| Challenger {unit} | {decision.challenger_score:,.4f} |",
        f"| Current champion | {f'`{decision.champion_run}`' if decision.champion_run else 'none'} |",
        f"| Champion {unit} | "
        f"{f'{decision.champion_score:,.4f}' if decision.champion_score is not None else 'n/a'} |",
        f"| Margin required | {decision.margin_required:,.4f} |",
        f"| Margin achieved | "
        f"{f'{decision.margin_achieved:,.4f}' if decision.margin_achieved is not None else 'n/a'} |",
        f"| Registered version | {decision.model_version or 'not written'} |",
        "",
        "## Why a margin",
        "",
        "Without one, every rerun swaps the model whenever the number moves at all, and with "
        "about fifty fraud cases in the scoring split it moves for no reason. A margin turns "
        "**different** into **better**. Its size is a decision to argue about rather than a "
        "default to ignore.",
        "",
        "## Why cost rather than the headline metric",
        "",
        "Stage 5 chose this champion on expected cost at a tuned threshold. Judging promotion "
        "on anything else would let a model win the selection and lose the promotion, which "
        "is not a difference anyone could explain to the team running it.",
        "",
        f"The score used is the **validation** cost, {decision.challenger_score:,.2f}. Test is "
        "reported and never used to choose. Promoting on the test number would fold the held "
        "out estimate back into the decision it is supposed to be independent of.",
        "",
        "## What the promoted model actually does",
        "",
        f"- Threshold {summary['threshold']:.6f}",
        f"- Test recall {summary['test']['recall']:.3f}, catching "
        f"{summary['test']['true_positives']} of "
        f"{summary['test']['true_positives'] + summary['test']['false_negatives']} frauds",
        f"- Test value recall **{summary['test']['value_recall']:.3f}**, which is the share of "
        f"fraudulent money recovered and the number worth watching",
        f"- {summary['test']['false_positives']} false alarms across "
        f"{summary['test']['rows']:,} transactions",
        "",
    ]
    return "\n".join(lines)
