"""Stage 3. Build the features the models actually train on.

Two rules govern everything here, and they pull in opposite directions. Getting the
difference between them right is the whole job of this module.

**A feature may look backwards. It may never look forwards.**
    A rolling average over the previous fifty transactions is fine, because a deployed
    system has exactly that information at the moment it scores a transaction. So these
    features are computed across the whole ordered stream, including across split
    boundaries, and that is not leakage. Computing them inside each split separately would
    be *worse*: the first rows of validation would get a baseline built from nothing, which
    is a situation production never encounters.

**A fitted statistic may only come from the training split.**
    A mean, a standard deviation, a quantile. These summarise the data they are fitted on,
    so fitting one on everything folds the future into every row. They are fitted on train
    and then applied unchanged to validation and test.

The short version: transforming test rows is required, letting test rows influence a
fitted number is not.

What gets built, given that no card identifier exists in this dataset:

===========================  ==========================================================
``amount_log``               log1p of the amount. Amounts are heavily skewed.
``hour``, ``hour_sin/cos``   time of day. Fraud is five times likelier in the small hours.
``seconds_since_prev``       gap since the previous transaction in the stream.
``amount_roll_mean_N``       average amount over the previous N transactions.
``amount_roll_std_N``        how variable those previous N amounts were.
``amount_dev_N``             how far this amount sits from that recent baseline.
``amount_ratio_N``           the same comparison as a ratio, which survives outliers better.
``txn_per_minute_N``         how busy the stream has been over the previous N transactions.
===========================  ==========================================================

The brief also asks for rolling aggregates per card and category encodings. Neither is
possible here: the ULB file contains ``Time``, ``V1`` to ``V28``, ``Amount`` and ``Class``,
and nothing identifies a cardholder. The same ideas are applied to the global transaction
stream instead. Per card versions arrive with IEEE CIS, which has real card and category
columns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_pipeline.config import Config
from fraud_pipeline.splits import SPLIT_NAMES, load_split

logger = logging.getLogger(__name__)

#: Floor added to every denominator in this module.
#:
#: A tiny epsilon would be the obvious choice and it is the wrong one. It stops the division
#: raising, which makes the failure silent rather than absent: dividing by 1e-9 turns a
#: perfectly ordinary row into a value of 1e12, and a finite check happily passes it through.
#:
#: Both denominators here are floored at one whole unit, and in both cases that unit means
#: something. ``Time`` is recorded to the second, so a gap below one second is finer than the
#: data can express. Amounts are in currency, so a standard deviation below one is a stretch
#: of transactions that were effectively all the same price. Below those points the ratio is
#: not informative, it is just noise amplification.
#:
#: This bit twice, on 54.8 percent of rows where the gap is exactly zero and on flat stretches
#: where the rolling standard deviation is exactly zero. See the build log.
_DENOMINATOR_FLOOR = 1.0

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0

#: Marks the selection pass this stage runs, so it does not overwrite the raw pass from the
#: eda stage. Stage 4 trains on the engineered set, so it reads this tag.
ENGINEERED_TAG = "engineered"


class FeatureError(RuntimeError):
    """The feature stage cannot build a usable table."""


# --------------------------------------------------------------------------------------
# Fitted state
# --------------------------------------------------------------------------------------


@dataclass
class FittedFeatures:
    """Everything learned from the training split, and nothing learned from anywhere else.

    Written to disk as json rather than pickled. It is a handful of numbers, and a format a
    human can open and check is worth more here than one that is quicker to load. The
    serving stage reads this same file, which is what stops the API and the training run
    from disagreeing about what a feature means.
    """

    amount_median: float
    seconds_since_prev_median: float
    fitted_on_split: str
    fitted_on_rows: int
    feature_names: list[str] = field(default_factory=list)
    passthrough: list[str] = field(default_factory=list)
    built: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FittedFeatures:
        return cls(**payload)


def fit(train: pd.DataFrame, config: Config) -> FittedFeatures:
    """Learn the few statistics the transform needs, from the training split only.

    There are deliberately very few of them. Most of the features here are either pointwise
    or built from a row's own recent history, and neither kind needs a fitted number. The
    two below exist to fill the gaps that history features leave at the very start of the
    stream, where there is no history yet.
    """
    spec = config.dataset.spec()
    amount = train[spec.amount_column]
    gaps = train[spec.time_column].diff().dropna()

    return FittedFeatures(
        amount_median=float(amount.median()),
        # A median, not a mean. The gap distribution has a long tail at night, and a mean
        # would be dragged around by it.
        seconds_since_prev_median=float(gaps.median()) if len(gaps) else 0.0,
        fitted_on_split=config.eda.analysis_split,
        fitted_on_rows=int(len(train)),
    )


# --------------------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------------------


def add_pointwise_features(frame: pd.DataFrame, config: Config) -> list[str]:
    """Features derived from a single row. No history and no fitted numbers involved."""
    spec = config.dataset.spec()
    built = []

    if config.features.amount_log:
        frame["amount_log"] = np.log1p(frame[spec.amount_column].clip(lower=0))
        built.append("amount_log")

    if config.features.hour_of_day:
        hour = (frame[spec.time_column] / SECONDS_PER_HOUR) % HOURS_PER_DAY
        frame["hour"] = hour
        # A sine and cosine pair so that 23:00 and 00:00 sit next to each other. On the raw
        # hour they are 23 apart, which is the largest possible distance, and any model
        # that treats the column as a number believes it.
        radians = 2 * np.pi * hour / HOURS_PER_DAY
        frame["hour_sin"] = np.sin(radians)
        frame["hour_cos"] = np.cos(radians)
        built += ["hour", "hour_sin", "hour_cos"]

    return built


def add_history_features(frame: pd.DataFrame, fitted: FittedFeatures, config: Config) -> list[str]:
    """Features built from earlier transactions in the stream.

    Every window here is shifted by one row before it is used, so a transaction is compared
    against the ones that came *before* it and never against itself. Without the shift, an
    unusually large amount quietly raises its own baseline and makes itself look normal,
    which is precisely backwards for fraud detection.

    The frame must already be sorted by time. :func:`build_features` guarantees that.
    """
    spec = config.dataset.spec()
    amount = frame[spec.amount_column]
    built = []

    if config.features.time_since_last:
        gap = frame[spec.time_column].diff()
        # The very first transaction has no predecessor. A fitted median is a more honest
        # filler than a zero, which would claim it arrived instantly after nothing.
        frame["seconds_since_prev"] = gap.fillna(fitted.seconds_since_prev_median)
        built.append("seconds_since_prev")

    prior = amount.shift(1)

    for window in config.features.rolling_windows:
        rolling = prior.rolling(window=window, min_periods=1)
        mean = rolling.mean()
        std = rolling.std()

        mean_name, std_name = f"amount_roll_mean_{window}", f"amount_roll_std_{window}"
        # At the start of the stream there is no history, so the fitted median stands in.
        frame[mean_name] = mean.fillna(fitted.amount_median)
        frame[std_name] = std.fillna(0.0)
        built += [mean_name, std_name]

        if config.features.amount_deviation:
            # How many recent standard deviations away this amount is. This is the local
            # version of a z score, and unlike a global one it is not just the amount in
            # different units, so it survives a tree model finding it useful.
            frame[f"amount_dev_{window}"] = (amount - frame[mean_name]) / (
                frame[std_name] + _DENOMINATOR_FLOOR
            )

            # The same comparison as a ratio. It says something different when the recent
            # variability is near zero, where the deviation above saturates.
            frame[f"amount_ratio_{window}"] = amount / (frame[mean_name] + _DENOMINATOR_FLOOR)
            built += [f"amount_dev_{window}", f"amount_ratio_{window}"]

        if config.features.transaction_rate and config.features.time_since_last:
            # How busy the stream has been. Volume collapses at night and that is exactly
            # when the fraud rate climbs, so this tracks the part of the daily cycle that
            # matters.
            #
            # The floor makes this a smoothed rate rather than a literal one. Over half of
            # the gaps in this data are exactly zero, because several transactions can share
            # a second, so the literal version divides by nothing and returns a number in the
            # billions. Smoothed, it runs from about 1.8 in the quietest stretches to 60 in
            # the busiest, and stays monotonic in how busy the stream is, which is the only
            # property a model needs from it.
            gaps = frame["seconds_since_prev"].shift(1).rolling(window=window, min_periods=1)
            mean_gap = gaps.mean().fillna(fitted.seconds_since_prev_median)
            frame[f"txn_rate_{window}"] = 60.0 / (mean_gap + _DENOMINATOR_FLOOR)
            built.append(f"txn_rate_{window}")

    return built


def build_features(
    ordered: pd.DataFrame, fitted: FittedFeatures, config: Config
) -> tuple[pd.DataFrame, list[str]]:
    """Build every feature on one time ordered frame.

    The frame is sorted first. Everything downstream assumes time order, and a history
    feature computed on unsorted rows is silently meaningless rather than loudly broken.

    The sort is stable, which matters more than it sounds. Timestamps repeat constantly in
    this data, because ``Time`` is whole seconds and busy periods carry several
    transactions per second. Rows sharing a timestamp have no true order, so a stable sort
    keeps the order they arrived in. That makes the pipeline reproducible for a given input
    file, which is the property actually needed. It does not make the output independent of
    the input ordering, and it cannot: which of two transactions in the same second came
    first is information the file does not contain.
    """
    spec = config.dataset.spec()
    frame = ordered.sort_values(spec.time_column, kind="stable").reset_index(drop=True)

    built = add_pointwise_features(frame, config)
    built += add_history_features(frame, fitted, config)

    check_feature_sanity(frame, built)
    return frame, built


#: Any engineered value larger than this is treated as a bug rather than an outlier.
#:
#: The largest genuine amount in the dataset is about 25,700, so a ratio or a deviation can
#: legitimately reach the tens of thousands. A million cannot happen from real data and only
#: ever comes from dividing by something that should have been floored.
MAX_PLAUSIBLE_MAGNITUDE = 1e6


def check_feature_sanity(frame: pd.DataFrame, built: list[str]) -> None:
    """Fail loudly on values that are technically valid and obviously wrong.

    Checking for infinity is not enough, and assuming otherwise is what let a real bug
    through here. A denominator floored at 1e-9 never produces an infinity. It produces
    1e12, which is finite, passes every null and finite check, and then quietly destroys any
    model that cares about scale while leaving the tree models looking fine.
    """
    missing = [name for name in built if frame[name].isna().any()]
    if missing:
        raise FeatureError(f"feature construction produced missing values in: {missing}")

    values = frame[built]
    infinite = [name for name in built if not np.isfinite(values[name]).all()]
    if infinite:
        raise FeatureError(f"feature construction produced infinite values in: {infinite}")

    absurd = {
        name: float(values[name].abs().max())
        for name in built
        if values[name].abs().max() > MAX_PLAUSIBLE_MAGNITUDE
    }
    if absurd:
        detail = ", ".join(f"{name} reaches {size:,.0f}" for name, size in absurd.items())
        raise FeatureError(
            f"feature construction produced implausibly large values: {detail}. "
            f"This is almost always a denominator that needed a floor rather than an epsilon."
        )


def passthrough_columns(config: Config) -> list[str]:
    """The original columns carried through unchanged.

    The anonymised components go through untouched. They are already principal components,
    so they are centred, orthogonal and on comparable scales, and there is nothing useful
    left to do to them here.
    """
    spec = config.dataset.spec()
    return [*spec.pca_columns(), spec.amount_column]


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def _combine_splits(config: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the three splits back into one time ordered stream, remembering which is which.

    Rejoining them is what lets a history feature at the start of validation see the end of
    training, which is the situation a deployed model is actually in. Splitting first and
    building features separately would hand validation and test a cold start that production
    never has.
    """
    frames, sizes = [], {}
    for name in SPLIT_NAMES:
        part = load_split(config, name)
        part = part.assign(_split=name)
        sizes[name] = len(part)
        frames.append(part)

    return pd.concat(frames, ignore_index=True), sizes


def run(config: Config):
    """Run stage 3 and write the engineered tables."""
    from fraud_pipeline import eda, feature_selection
    from fraud_pipeline.paths import ensure_dir

    spec = config.dataset.spec()
    combined, sizes = _combine_splits(config)
    logger.info(
        "building features over %s rows (train %s, validation %s, test %s)",
        f"{len(combined):,}",
        f"{sizes['train']:,}",
        f"{sizes['validation']:,}",
        f"{sizes['test']:,}",
    )

    train_rows = combined[combined["_split"] == "train"]
    fitted = fit(train_rows, config)
    logger.info(
        "fitted on %s training rows only: amount median %.2f, gap median %.2fs",
        f"{fitted.fitted_on_rows:,}",
        fitted.amount_median,
        fitted.seconds_since_prev_median,
    )

    engineered, built = build_features(combined, fitted, config)
    carried = passthrough_columns(config)

    fitted.built = built
    fitted.passthrough = carried
    fitted.feature_names = [*carried, *built]
    logger.info("built %s new features, carried %s through", len(built), len(carried))

    columns = [*fitted.feature_names, spec.target_column, spec.time_column, "_split"]
    engineered = engineered[columns]

    target_dir = ensure_dir(config.paths.processed() / config.features.output_dir)
    written = {}
    for name in SPLIT_NAMES:
        part = engineered[engineered["_split"] == name].drop(columns=["_split"])
        path = target_dir / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(path, index=False)
        written[name] = len(part)
        logger.info("wrote %-11s %8s rows  %s", name, f"{len(part):,}", path.name)

    for name in SPLIT_NAMES:
        if written[name] != sizes[name]:
            raise FeatureError(
                f"the {name} split changed size during feature building: "
                f"{sizes[name]} in, {written[name]} out"
            )

    stats_path = target_dir / config.features.stats_file
    stats_path.write_text(fitted.to_json(), encoding="utf-8")
    logger.info("wrote %s", stats_path.name)

    selection = None
    if config.features.reselect:
        selection = _reselect(engineered, fitted, config, eda, feature_selection)

    return fitted, selection


def _reselect(
    engineered: pd.DataFrame, fitted: FittedFeatures, config: Config, eda, feature_selection
):
    """Run the selection rules again, over the engineered features this time.

    The pass in the eda stage judged the columns the file arrived with. This one judges the
    columns the model will actually see, which is the set that matters. It is also the first
    real test of whether the new features earn their place, and it uses the same rules, the
    same noise ceiling method and the same evidence trail.
    """
    from fraud_pipeline.paths import ensure_dir

    target = config.dataset.spec().target_column
    columns = [*fitted.feature_names, target]

    train = engineered[engineered["_split"] == "train"][columns].reset_index(drop=True)
    drift = engineered[engineered["_split"] == config.eda.drift_split][columns].reset_index(
        drop=True
    )

    stats = eda.compute_feature_stats(train, drift, config)
    noise = eda.permutation_noise_ceiling(train, config)
    pairs = eda.correlated_pairs(train, config)
    duplicates = eda.exact_duplicate_columns(train, config)

    selection = feature_selection.select_features(stats, pairs, noise, config, duplicates)
    counts = selection.counts()
    logger.info(
        "engineered feature sets: all %s, safe %s, selected %s",
        counts["all"],
        counts["safe"],
        counts["selected"],
    )

    new_features = set(fitted.built)
    kept_new = [name for name in selection.features() if name in new_features]
    logger.info("%s of %s new features survived selection", len(kept_new), len(new_features))

    for decision in selection.decisions:
        if decision.dropped_by_tier is not None:
            marker = "new" if decision.feature in new_features else "raw"
            logger.info(
                "  drop %-22s %s tier %s  %s",
                decision.feature,
                marker,
                decision.dropped_by_tier,
                decision.reasons[0],
            )

    stats_path = ensure_dir(config.paths.tables_dir()) / "engineered_feature_stats.csv"
    stats.to_csv(stats_path, index=False)

    report_path = ensure_dir(config.paths.tables_dir()) / config.features.report_file
    report_path.write_text(_build_report(stats, selection, fitted), encoding="utf-8")
    logger.info("wrote %s", report_path.name)

    # Tagged, so this does not overwrite the raw pass the eda stage wrote.
    feature_selection.write_selection(selection, stats, config, tag=ENGINEERED_TAG)
    return selection


def _build_report(stats: pd.DataFrame, selection, fitted: FittedFeatures) -> str:
    """Write up which engineered features earned their place and which did not."""
    new_features = set(fitted.built)
    counts = selection.counts()
    kept = selection.features()

    lines = [
        "# Feature engineering",
        "",
        f"{len(fitted.passthrough)} original columns carried through, "
        f"{len(fitted.built)} new features built, "
        f"{counts['all']} in total.",
        "",
        f"After selection the active set keeps **{counts[selection.active_set]}**, "
        f"of which {len([n for n in kept if n in new_features])} are new.",
        "",
        "## What was fitted, and on what",
        "",
        f"Only two numbers are fitted, both on the {fitted.fitted_on_split} split "
        f"({fitted.fitted_on_rows:,} rows). They fill the gap at the very start of the",
        "stream where a history feature has no history yet.",
        "",
        f"- amount median: {fitted.amount_median:.2f}",
        f"- gap between transactions, median: {fitted.seconds_since_prev_median:.2f} seconds",
        "",
        "Everything else is either pointwise or built from a row's own earlier history, so",
        "there is nothing else that could carry information from one split into another.",
        "",
        "## The new features",
        "",
        "| Feature | AUC | KS | PSI | Range coverage | Kept |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    lookup = stats.set_index("feature")
    for name in fitted.built:
        if name not in lookup.index:
            continue
        row = lookup.loc[name]
        verdict = "yes" if name in kept else "no"
        lines.append(
            f"| {name} | {row['auc']:.4f} | {row['ks']:.4f} | {row['psi']:.3f} | "
            f"{row['range_coverage']:.4f} | {verdict} |"
        )

    dropped = [d for d in selection.decisions if d.dropped_by_tier is not None]
    lines += ["", "## Dropped", ""]
    if dropped:
        lines += ["| Feature | New | Tier | Why |", "| --- | --- | --- | --- |"]
        lines += [
            f"| {d.feature} | {'yes' if d.feature in new_features else 'no'} | "
            f"{d.dropped_by_tier} | {'; '.join(d.reasons)} |"
            for d in dropped
        ]
    else:
        lines.append("Nothing was dropped.")

    return "\n".join([*lines, ""])


# --------------------------------------------------------------------------------------
# Reading the output back
# --------------------------------------------------------------------------------------


def load_engineered(config: Config, name: str) -> pd.DataFrame:
    """Read one engineered split. Stage 4 onwards uses this."""
    if name not in SPLIT_NAMES:
        raise FeatureError(f"unknown split {name!r}, expected one of {SPLIT_NAMES}")

    path = config.paths.processed() / config.features.output_dir / f"{name}.parquet"
    if not path.is_file():
        raise FeatureError(
            f"engineered split not found: {path}\nRun the features stage first: fraud features"
        )
    return pd.read_parquet(path)


def load_fitted(config: Config) -> FittedFeatures:
    """Read the fitted statistics back. The serving stage needs these to match training."""
    path = config.paths.processed() / config.features.output_dir / config.features.stats_file
    if not path.is_file():
        raise FeatureError(
            f"fitted feature statistics not found: {path}\n"
            f"Run the features stage first: fraud features"
        )
    return FittedFeatures.from_dict(json.loads(path.read_text(encoding="utf-8")))
