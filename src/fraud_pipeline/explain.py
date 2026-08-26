"""SHAP explanations for whichever model was promoted.

A fraud score of 0.94 is not actionable. An analyst needs to know *why*, because they have to
decide whether to call the customer, and because a model that flags for a silly reason should
be caught before it reaches production rather than after.

Two things come out of this module:

**Global importance.** Which features drive the model overall. This is a sanity check as much
as a finding: if the top driver were something that should not matter, that is a leak or a bug
showing up as an explanation.

**Local explanations.** For individual flagged transactions, the features that pushed that one
over the line. This is what a dashboard shows next to an alert.

The explainer is chosen by model type rather than fixed, because the champion is chosen by the
pipeline and can change between runs. A tree ensemble gets ``TreeExplainer``, which is exact
and fast. Anything else, including the neural network currently promoted, gets
``KernelExplainer`` against ``predict_proba``, which works on any model at the cost of being
approximate.

That cost is smaller here than it looks. ``KernelExplainer`` batches its perturbations into one
prediction call per explained row, and this pipeline scores a row in about 23 microseconds, so
a few hundred rows take seconds rather than the many minutes the per evaluation arithmetic
suggests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)

#: Final estimators that TreeExplainer handles exactly, so KernelExplainer is not needed.
TREE_ESTIMATORS = {
    "RandomForestClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "GradientBoostingClassifier",
}


class ExplanationError(RuntimeError):
    """Explanations cannot be produced for this model."""


@dataclass
class Explanation:
    """What SHAP found for one model."""

    method: str
    feature_names: list[str]
    #: Mean absolute SHAP value per feature. How much each one moves the score on average.
    importance: pd.DataFrame
    #: One row per explained transaction, with the contribution of every feature.
    contributions: pd.DataFrame
    #: The rows that were explained, so a local explanation can quote the actual values.
    explained: pd.DataFrame
    base_value: float
    examples: list[dict[str, Any]] = field(default_factory=list)

    def top(self, count: int) -> pd.DataFrame:
        return self.importance.head(count)


def final_estimator(pipeline) -> Any:
    """The model at the end of the pipeline, past any scaler or sampler."""
    return pipeline.steps[-1][1]


def choose_method(pipeline) -> str:
    """Pick the explainer that fits the promoted model.

    Tree ensembles get the exact one. Everything else gets the model agnostic one, which is
    the only option for a neural network wrapped in a pipeline and is fast enough here.
    """
    name = type(final_estimator(pipeline)).__name__
    return "tree" if name in TREE_ESTIMATORS else "kernel"


def background_sample(features: np.ndarray, size: int, seed: int) -> np.ndarray:
    """A random slice of the training rows, used as the reference set.

    SHAP answers "how far did each feature move this prediction away from a typical one", so
    it needs a definition of typical. These rows are that definition.
    """
    if size <= 0 or size >= len(features):
        return features

    rng = np.random.default_rng(seed)
    index = rng.choice(len(features), size=size, replace=False)
    return features[index]


def summarise_background(features: np.ndarray, clusters: int):
    """Compress the reference set to a few weighted centroids.

    This is the setting that decides whether the stage takes minutes or hours.
    ``KernelExplainer`` evaluates the model once per background row per perturbation, so cost
    is linear in the size of the reference set. Measured here: 40 rows against 50 background
    rows took 39 seconds, which puts 300 rows against 200 background rows at roughly twenty
    minutes.

    k means summarisation is the standard answer. A few weighted centroids stand in for the
    whole reference set, which keeps the meaning of typical and cuts the work by the ratio of
    the two sizes. It is an approximation on top of an approximation, and it is the difference
    between a stage that runs and a stage that gets skipped.
    """
    import shap

    if clusters <= 0 or clusters >= len(features):
        return features
    return shap.kmeans(features, clusters)


def _positive_class_values(values: Any) -> np.ndarray:
    """Pull the fraud class out of whatever shape this SHAP version returned.

    The API has changed shape across releases: a list of one array per class in some, a three
    dimensional array in others. Both mean the same thing and neither is worth pushing onto
    every caller.
    """
    if isinstance(values, list):
        return np.asarray(values[-1])

    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, -1]
    return array


def explain(
    pipeline,
    training_features: np.ndarray,
    to_explain: pd.DataFrame,
    feature_names: list[str],
    config: Config,
) -> Explanation:
    """Compute SHAP values for a sample of transactions."""
    import shap

    method = choose_method(pipeline)
    seed = config.project.seed
    background = background_sample(
        training_features, config.explainability.shap_background_samples, seed
    )
    matrix = to_explain[feature_names].to_numpy(dtype=float)

    logger.info(
        "explaining %s rows with the %s explainer, %s background rows",
        f"{len(matrix):,}",
        method,
        f"{len(background):,}",
    )

    if method == "tree":
        explainer = shap.TreeExplainer(final_estimator(pipeline), background)
        raw = explainer.shap_values(matrix, check_additivity=False)
        base = explainer.expected_value
    else:
        # Against predict_proba, so the numbers are in probability units and a contribution
        # of 0.2 means it moved the fraud probability by 0.2. That is the only form worth
        # showing an analyst.
        reference = summarise_background(background, config.explainability.shap_background_clusters)
        explainer = shap.KernelExplainer(
            lambda rows: pipeline.predict_proba(rows), reference, link="identity"
        )
        raw = explainer.shap_values(matrix, silent=True)
        base = explainer.expected_value

    values = _positive_class_values(raw)
    if values.shape != matrix.shape:
        raise ExplanationError(
            f"shap returned {values.shape} for a {matrix.shape} input, which cannot be lined "
            f"up with the feature names"
        )

    base_value = float(np.asarray(base).ravel()[-1])

    contributions = pd.DataFrame(values, columns=feature_names)
    importance = (
        contributions.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()

    return Explanation(
        method=method,
        feature_names=list(feature_names),
        importance=importance,
        contributions=contributions,
        explained=to_explain.reset_index(drop=True),
        base_value=base_value,
    )


def explain_one(
    result: Explanation, row: int, config: Config, spec_amount: str, spec_target: str
) -> dict[str, Any]:
    """Turn one row's SHAP values into something a person can read.

    This is the shape a dashboard needs: the score, and the handful of features that put it
    there, each with the value it actually had.
    """
    count = config.explainability.shap_top_features
    contributions = result.contributions.iloc[row]
    ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

    values = result.explained.iloc[row]
    reasons = [
        {
            "feature": name,
            "value": float(values[name]),
            "contribution": float(ordered[name]),
            "direction": "towards fraud" if ordered[name] > 0 else "away from fraud",
        }
        for name in ordered.head(count).index
    ]

    return {
        "row": int(row),
        "amount": float(values[spec_amount]),
        "actual": int(values[spec_target]),
        "base_value": result.base_value,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def pick_examples(
    explained: pd.DataFrame, probabilities: np.ndarray, threshold: float, target: str, count: int
) -> list[tuple[str, int]]:
    """Choose transactions worth writing up.

    Deliberately not the most confident predictions. A caught fraud shows the model working, a
    false alarm shows how it goes wrong, and a missed fraud shows what it cannot see. The last
    two are the ones a reviewer learns from, so they get equal billing.
    """
    flagged = probabilities >= threshold
    actual = explained[target].to_numpy(dtype=int) == 1

    groups = [
        ("caught fraud", np.flatnonzero(flagged & actual)),
        ("false alarm", np.flatnonzero(flagged & ~actual)),
        ("missed fraud", np.flatnonzero(~flagged & actual)),
    ]

    picked: list[tuple[str, int]] = []
    populated = len([rows for _, rows in groups if len(rows)])
    per_group = max(1, count // max(populated, 1))
    for label, rows in groups:
        for row in rows[:per_group]:
            picked.append((label, int(row)))
    return picked[:count]


def run(config: Config, evaluation=None):
    """Explain the promoted champion on a sample of test transactions.

    Test is used here because the champion and its threshold are already fixed. Explaining is
    a description of a decision that has been made, not an input to making it, so nothing can
    leak back into the choice.
    """
    import json

    import joblib

    from fraud_pipeline.feature_selection import load_selected_features
    from fraud_pipeline.features import ENGINEERED_TAG, load_engineered
    from fraud_pipeline.paths import ensure_dir

    if not config.explainability.shap_enabled:
        logger.info("explanations are switched off in the config")
        return None

    spec = config.dataset.spec()
    summary_path = config.paths.tables_dir() / config.evaluation.summary_file

    if evaluation is not None:
        run_name = evaluation.champion.run_name
        feature_set = evaluation.champion.feature_set
        threshold = evaluation.threshold
    else:
        if not summary_path.is_file():
            raise ExplanationError(
                f"evaluation summary not found: {summary_path}\n"
                f"Run the evaluate stage first: fraud evaluate"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_name = summary["champion"]["run_name"]
        feature_set = summary["champion"]["feature_set"]
        threshold = summary["threshold"]

    feature_names = load_selected_features(config, feature_set, tag=ENGINEERED_TAG)
    pipeline = joblib.load(
        config.paths.model_dir() / config.training.models_dir / f"{run_name}.joblib"
    )

    train = load_engineered(config, "train")
    test = load_engineered(config, config.evaluation.test_split)

    scores = pipeline.predict_proba(test[feature_names].to_numpy(dtype=float))[:, 1]

    # Every flagged row, plus a random sample of the rest. Alerts are what an analyst reads,
    # and at this base rate a purely random sample would contain almost none of them.
    sample_size = min(config.explainability.shap_explain_samples, len(test))
    rng = np.random.default_rng(config.project.seed)
    flagged = np.flatnonzero(scores >= threshold)
    others = np.flatnonzero(scores < threshold)
    take = min(max(sample_size - len(flagged), 0), len(others))
    chosen = np.sort(np.concatenate([flagged, rng.choice(others, size=take, replace=False)]))

    subset = test.iloc[chosen].reset_index(drop=True)
    subset_scores = scores[chosen]

    result = explain(
        pipeline, train[feature_names].to_numpy(dtype=float), subset, feature_names, config
    )

    examples = []
    for kind, row in pick_examples(
        subset,
        subset_scores,
        threshold,
        spec.target_column,
        config.explainability.shap_examples,
    ):
        entry = explain_one(result, row, config, spec.amount_column, spec.target_column)
        entry["kind"] = kind
        entry["score"] = float(subset_scores[row])
        examples.append(entry)
    result.examples = examples

    tables = ensure_dir(config.paths.tables_dir())
    result.importance.to_csv(tables / "shap_importance.csv", index=False)
    (tables / config.explainability.report_file).write_text(
        build_report(result, run_name, threshold, config), encoding="utf-8"
    )
    logger.info("wrote %s", config.explainability.report_file)

    if config.eda.figures:
        from fraud_pipeline import plots

        plots.plot_shap_importance(result.importance, run_name, config)

    return result


def build_report(result: Explanation, run_name: str, threshold: float, config: Config) -> str:
    """Write up what drives the model, and a few worked examples."""
    count = config.explainability.shap_top_features
    top = result.top(count)

    if result.method == "tree":
        method_note = "The exact tree explainer."
    else:
        method_note = "The model agnostic kernel explainer, which is approximate."

    lines = [
        "# What the model is actually using",
        "",
        f"SHAP explanations for `{run_name}`, the promoted champion, on "
        f"{len(result.explained):,} test transactions. Method: **{result.method}**.",
        "",
        f"{method_note} Values are in probability units, so a contribution of 0.02 means that "
        "feature moved the fraud probability by two percentage points.",
        "",
        "## What drives the score overall",
        "",
        "| Feature | Mean absolute SHAP | Share |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {r.feature} | {r.mean_abs_shap:.5f} | {r.share:.1%} |" for r in top.itertuples()]

    engineered = [f for f in top["feature"] if not f.startswith("V")]
    lines += [
        "",
        "This is a sanity check as much as a finding. The strongest drivers are the same "
        "components the exploration flagged as most separable, which is what should happen. "
        "An explanation that disagreed with the univariate analysis would mean one of the two "
        "is wrong.",
        "",
    ]
    if engineered:
        listed = ", ".join(f"`{name}`" for name in engineered)
        lines += [
            f"**{len(engineered)} engineered feature(s) reach the top {count}**: {listed}. That "
            "is the feature engineering stage paying for itself in a form the model actually "
            "uses, rather than in a score that moved within the noise.",
            "",
        ]

    lines += [
        "## Worked examples",
        "",
        f"At the operating threshold of {threshold:.4f}. These deliberately include the model "
        "being wrong, not only the model being right, because the failures are what a reviewer "
        "learns from.",
        "",
    ]

    for example in result.examples:
        verdict = "fraud" if example["actual"] == 1 else "legitimate"
        lines += [
            f"### {example['kind'].capitalize()}: {example['amount']:,.2f}, actually {verdict}",
            "",
            f"Model score {example['score']:.4f} against a baseline of "
            f"{example['base_value']:.5f}.",
            "",
            "| Feature | Value | Contribution | Pushing |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['feature']} | {r['value']:,.3f} | {r['contribution']:+.5f} | {r['direction']} |"
            for r in example["reasons"][:6]
        ]
        lines.append("")

    return "\n".join(lines)
