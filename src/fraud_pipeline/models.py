"""Model definitions, and how each one is asked to handle the class imbalance.

Kept apart from the training loop so that the loop does not need to know anything about
particular estimators, and adding a model later means adding one entry here rather than an
``if`` branch in three places.

The awkward part this module hides is that the three imbalance strategies are not
interchangeable across models:

``none``
    Train on the data as it is. On a 0.18 percent positive rate this mostly predicts
    "normal" and is here as the honest baseline the others have to beat.
``class_weight``
    Tell the estimator to weight the rare class up. Scikit learn estimators take
    ``class_weight="balanced"``. The boosted trees do not, and use ``scale_pos_weight``
    instead, which is the same idea through a different door.
``smote``
    Synthesise new minority rows until the classes are closer. This one changes the training
    data rather than the loss, so it has to live inside the pipeline where it can be applied
    on fit and skipped on predict.

That last point is the one that goes wrong most often. Resampling before the split, or
resampling the data a model is scored on, produces excellent numbers that mean nothing.
Putting SMOTE inside an imblearn pipeline makes it structurally impossible: the pipeline
applies it during ``fit`` and ignores it during ``predict_proba``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)

IMBALANCE_STRATEGIES = ("none", "class_weight", "smote")


class ModelError(RuntimeError):
    """A model or imbalance strategy was asked for that cannot be built."""


@dataclass(frozen=True)
class ModelDefinition:
    """How to build one model, and what it can and cannot do about imbalance."""

    name: str
    label: str
    build: Callable[[dict[str, Any], int], Any]
    needs_scaling: bool
    #: How this estimator is told to care more about the rare class.
    #: "class_weight" for the scikit learn keyword, "scale_pos_weight" for the boosters,
    #: or None if it has no such option and SMOTE is the only route.
    weighting: str | None = "class_weight"

    def supports(self, strategy: str) -> bool:
        if strategy == "class_weight":
            return self.weighting is not None
        return strategy in IMBALANCE_STRATEGIES


def _logistic_regression(params: dict[str, Any], seed: int):
    return LogisticRegression(random_state=seed, **params)


def _random_forest(params: dict[str, Any], seed: int):
    return RandomForestClassifier(random_state=seed, **params)


def _xgboost(params: dict[str, Any], seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=seed,
        # Without this the classifier prints a deprecation banner on every one of the six
        # fits, which buries the pipeline's own logging.
        eval_metric="logloss",
        n_jobs=-1,
        **params,
    )


def _lightgbm(params: dict[str, Any], seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        random_state=seed,
        # LightGBM is chatty by default and says nothing useful during a sweep.
        verbose=-1,
        n_jobs=-1,
        **params,
    )


def _neural_net(params: dict[str, Any], seed: int):
    from fraud_pipeline.neural import TorchMLPClassifier

    return TorchMLPClassifier(random_state=seed, **params)


#: Every model this stage knows how to build. Models in the config that are absent from here
#: are skipped with a message naming the branch they arrive on, the same way an
#: unimplemented pipeline stage behaves.
REGISTRY: dict[str, ModelDefinition] = {
    "logistic_regression": ModelDefinition(
        name="logistic_regression",
        label="Logistic Regression",
        build=_logistic_regression,
        # Coefficients on unscaled columns are meaningless here. Amount runs to 25,000 while
        # the components sit near zero, so without scaling the solver spends its effort on
        # whichever column happens to be largest.
        needs_scaling=True,
    ),
    "random_forest": ModelDefinition(
        name="random_forest",
        label="Random Forest",
        build=_random_forest,
        # A tree splits on order, so the scale of a column changes nothing.
        needs_scaling=False,
    ),
    "xgboost": ModelDefinition(
        name="xgboost",
        label="XGBoost",
        build=_xgboost,
        needs_scaling=False,
        # The one model that does not take scikit learn's class_weight keyword. It expects a
        # single number saying how much more a positive counts, which has to be computed from
        # the training labels, so the training loop passes it in.
        weighting="scale_pos_weight",
    ),
    "lightgbm": ModelDefinition(
        name="lightgbm",
        label="LightGBM",
        build=_lightgbm,
        needs_scaling=False,
        # LightGBM's scikit learn wrapper does take class_weight, unlike XGBoost's.
        weighting="class_weight",
    ),
    "neural_net": ModelDefinition(
        name="neural_net",
        label="Neural Net",
        build=_neural_net,
        # The only model here that genuinely breaks without scaling. Gradient descent on
        # columns spanning six orders of magnitude spends every step on the largest one.
        needs_scaling=True,
        # The wrapper accepts class_weight="balanced" and turns it into a positive class
        # weight in the loss, so it looks the same as the others from out here.
        weighting="class_weight",
    ),
}

#: Which branch fills in each model that is configured but not yet implemented.
#: Empty now that the advanced models have landed. Kept because the mechanism is how a model
#: gets added without the sweep having to be restructured around it.
PLANNED: dict[str, str] = {}


def available_models(config: Config) -> dict[str, ModelDefinition]:
    """The models that are both switched on in the config and implemented here."""
    available = {}
    for name in config.enabled_models():
        if name in REGISTRY:
            available[name] = REGISTRY[name]
        elif name in PLANNED:
            logger.info("skipping %s, it arrives on %s", name, PLANNED[name])
        else:
            raise ModelError(f"model {name!r} is enabled in the config but unknown")
    return available


def positive_class_weight(y) -> float:
    """How many times more common the negative class is.

    This is what ``class_weight="balanced"`` amounts to for a two class problem, written out
    because XGBoost wants the number rather than the keyword. On this data it is around 540.
    """
    import numpy as np

    labels = np.asarray(y).ravel()
    positives = float((labels == 1).sum())
    negatives = float((labels == 0).sum())

    if positives == 0:
        raise ModelError("cannot weight the positive class when there are no positive rows")
    return negatives / positives


def build_pipeline(
    definition: ModelDefinition,
    params: dict[str, Any],
    strategy: str,
    config: Config,
    pos_weight: float | None = None,
):
    """Assemble the full estimator for one model and one imbalance strategy.

    The result is always a pipeline, even when it has a single step, so that every model is
    fitted and called the same way and nothing downstream has to remember which ones needed
    a scaler.

    Args:
        pos_weight: how much more a positive counts, needed only by models whose weighting is
            ``scale_pos_weight``. It depends on the training labels, so the caller computes it
            rather than this function guessing.
    """
    if strategy not in IMBALANCE_STRATEGIES:
        raise ModelError(
            f"unknown imbalance strategy {strategy!r}, expected one of {IMBALANCE_STRATEGIES}"
        )
    if not definition.supports(strategy):
        raise ModelError(f"{definition.name} cannot use the {strategy!r} strategy")

    seed = config.project.seed
    estimator_params = dict(params)

    if strategy == "class_weight":
        if definition.weighting == "class_weight":
            estimator_params["class_weight"] = "balanced"
        elif definition.weighting == "scale_pos_weight":
            if pos_weight is None:
                raise ModelError(
                    f"{definition.name} needs pos_weight to use the class_weight strategy, "
                    f"because it has no class_weight keyword of its own"
                )
            estimator_params["scale_pos_weight"] = pos_weight

    steps = []
    if definition.needs_scaling:
        steps.append(("scaler", StandardScaler()))

    if strategy == "smote":
        # Inside the pipeline on purpose. imblearn applies a sampler during fit and skips it
        # during predict, so validation and test are never resampled. Doing this by hand
        # outside a pipeline is how people accidentally score a model on synthetic rows.
        steps.append(
            (
                "smote",
                SMOTE(
                    sampling_strategy=config.imbalance.smote.sampling_strategy,
                    k_neighbors=config.imbalance.smote.k_neighbors,
                    random_state=seed,
                ),
            )
        )

    steps.append(("model", definition.build(estimator_params, seed)))

    # An imblearn pipeline is needed only when a sampler is present. A plain scikit learn one
    # is used otherwise, so nothing pays for machinery it is not using.
    factory = ImbalancedPipeline if strategy == "smote" else Pipeline
    return factory(steps)


def describe(definition: ModelDefinition, strategy: str, feature_set: str) -> str:
    """A stable name for one combination, used for run names and file names."""
    return f"{definition.name}__{strategy}__{feature_set}"
