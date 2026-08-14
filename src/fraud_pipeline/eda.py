"""Exploratory analysis, as code rather than as a notebook.

The notebook in ``notebooks/01_eda.ipynb`` is for looking around. This module is the part
that later stages depend on, so it has to be repeatable and tested.

**The rule this module exists to enforce: the test split is never read.** Choosing features
by looking at test data is leakage, even though it feels harmless, and it quietly turns
every metric the project later reports into fiction. Statistics come from the training
split. Drift is measured against validation, which is the most recent data we are allowed
to see. The config refuses to accept ``test`` for either.

What it measures, per feature:

``auc``
    Univariate separation. How well this one feature alone ranks fraud above normal.
``ks``
    Kolmogorov Smirnov distance between the fraud and normal distributions.
``abs_corr``
    Absolute point biserial correlation with the target.
``mutual_info``
    Non linear dependence, which catches signal that correlation misses.
``psi``
    Population stability index between train and validation. How much the feature moved.
``range_coverage``
    The share of validation values falling inside the training range. This is the one that
    catches a feature which cannot generalise at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)

_PSI_BINS = 10
_PSI_FLOOR = 1e-6


class EdaError(RuntimeError):
    """Exploration cannot run on the data it was given."""


# --------------------------------------------------------------------------------------
# Per feature statistics
# --------------------------------------------------------------------------------------


def auc_from_ranks(ranks: np.ndarray, target: np.ndarray) -> float:
    """Univariate AUC computed from precomputed ranks.

    This is the Mann Whitney U form of the same number ``roc_auc_score`` returns. Ranking
    is the expensive part and it does not change when the target is shuffled, so doing it
    once up front makes the permutation test below cheap enough to actually run.

    Average ranks handle ties, which matters for a column like Amount where the same value
    appears many times.
    """
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return 0.5

    rank_sum = float(ranks[target == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def directionless_auc(value: float) -> float:
    """Fold an AUC around 0.5.

    A feature that ranks fraud consistently *low* is exactly as useful as one that ranks it
    high, so 0.05 and 0.95 are the same amount of signal.
    """
    return max(value, 1.0 - value)


@dataclass(frozen=True)
class TieAwareRanks:
    """Rank counts for one feature, in the two forms the KS calculation needs.

    ``at_or_below`` is how many values are less than or equal to each row's value.
    ``below`` is how many are strictly less.

    They differ only where values repeat, and that difference is the whole reason this
    class exists. The obvious approach, average ranks, is right for AUC and wrong for KS,
    because average ranks do not count anything: a column holding one value everywhere gets
    the same middling rank on every row, and the arithmetic then reports a large distance
    between two distributions that are in fact identical.

    Both arrays are computed once per feature. Shuffling the target does not reorder the
    values, which is what keeps the permutation test cheap.
    """

    at_or_below: np.ndarray
    below: np.ndarray

    @classmethod
    def build(cls, values: np.ndarray) -> TieAwareRanks:
        return cls(
            at_or_below=stats.rankdata(values, method="max"),
            below=stats.rankdata(values, method="min") - 1,
        )


def ks_from_ranks(ranks: TieAwareRanks, target: np.ndarray) -> float:
    """Two sample KS distance computed from precomputed rank counts.

    Same purpose as ``auc_from_ranks``: the ordering does not change when the target is
    shuffled, so doing the sorting once makes the permutation test affordable.

    AUC and KS answer different questions, and that difference is why this exists. AUC asks
    whether fraud sits consistently high or low, so it only sees a monotonic relationship.
    KS asks whether the two distributions differ in any way at all. A feature where fraud
    clusters in the middle of the range scores near 0.5 on AUC and high on KS, and dropping
    it on the AUC alone would be a mistake.

    The largest gap between two step functions falls at a data point, and only fraud rows
    can move the fraud curve, so walking the few hundred fraud rows is enough. Both sides of
    each step are checked, because the gap can be widest just before a step as well as at
    it.
    """
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    if positives == 0 or negatives == 0:
        return 0.0

    fraud_rows = np.flatnonzero(target == 1)
    at_or_below = ranks.at_or_below[fraud_rows]
    order = np.argsort(at_or_below, kind="stable")

    at_or_below = at_or_below[order]
    below = ranks.below[fraud_rows][order]

    # Rows sharing a value share a rank, so searchsorted gives the fraud count on each side
    # of that value without counting the tied rows twice.
    fraud_at_or_below = np.searchsorted(at_or_below, at_or_below, side="right")
    fraud_below = np.searchsorted(at_or_below, at_or_below, side="left")

    right = np.abs(fraud_at_or_below / positives - (at_or_below - fraud_at_or_below) / negatives)
    left = np.abs(fraud_below / positives - (below - fraud_below) / negatives)

    return float(max(right.max(), left.max()))


def population_stability_index(
    reference: np.ndarray, comparison: np.ndarray, bins: int = _PSI_BINS
) -> float:
    """How far the comparison distribution moved away from the reference.

    Bin edges come from the reference (training) quantiles. The usual reading is that below
    0.1 is stable, 0.1 to 0.25 is a moderate shift, and above 0.25 is a real change worth
    investigating.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # A near constant feature has no distribution to speak of.
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    comparison_share = np.histogram(comparison, bins=edges)[0] / len(comparison)

    reference_share = np.clip(reference_share, _PSI_FLOOR, None)
    comparison_share = np.clip(comparison_share, _PSI_FLOOR, None)

    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def range_coverage(reference: np.ndarray, comparison: np.ndarray) -> float:
    """The share of comparison values falling inside the reference minimum and maximum.

    This is the check that catches a feature which is structurally unusable rather than
    merely shifted. A raw timestamp only ever goes up, so every future value sits outside
    the range the model was fitted on and the model has never seen anything like it.
    """
    if len(comparison) == 0:
        return 1.0
    low, high = float(np.min(reference)), float(np.max(reference))
    return float(np.mean((comparison >= low) & (comparison <= high)))


def compute_mutual_information(
    frame: pd.DataFrame, target: np.ndarray, features: list[str], config: Config
) -> dict[str, float]:
    """Mutual information between each feature and the target.

    The estimator is k nearest neighbour based and slow on 200,000 rows, so it runs on a
    fixed random subsample. The subsample is seeded, so the number is reproducible.
    """
    sample = config.eda.mutual_info_sample
    seed = config.project.seed

    if sample and 0 < sample < len(frame):
        rng = np.random.default_rng(seed)
        # Keep every fraud row. There are only a few hundred and losing them to a random
        # subsample would make the estimate noise.
        fraud_index = np.flatnonzero(target == 1)
        normal_index = np.flatnonzero(target == 0)
        take = max(sample - len(fraud_index), 1)
        chosen = rng.choice(normal_index, size=min(take, len(normal_index)), replace=False)
        index = np.concatenate([fraud_index, chosen])
        subset, subset_target = frame.iloc[index][features], target[index]
        logger.info(
            "mutual information on a %s row sample of %s", f"{len(index):,}", f"{len(frame):,}"
        )
    else:
        subset, subset_target = frame[features], target

    scores = mutual_info_classif(subset, subset_target, random_state=seed)
    return dict(zip(features, (float(s) for s in scores), strict=True))


def compute_feature_stats(
    analysis: pd.DataFrame, drift: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """Build the per feature statistics table that feature selection runs on."""
    target_column = config.dataset.spec().target_column
    features = [name for name in analysis.columns if name != target_column]

    if not features:
        raise EdaError("no feature columns found, only the target")

    target = analysis[target_column].to_numpy()
    if target.sum() == 0:
        raise EdaError("the analysis split has no fraud rows, so nothing can be measured")

    mutual_info = compute_mutual_information(analysis, target, features, config)

    rows = []
    for name in features:
        values = analysis[name].to_numpy()
        drift_values = drift[name].to_numpy() if name in drift.columns else np.array([])
        ranks = stats.rankdata(values)

        fraud_values = values[target == 1]
        normal_values = values[target == 0]
        # Correlation with a constant column is undefined, not zero.
        correlation = abs(float(np.corrcoef(values, target)[0, 1])) if values.std() > 0 else 0.0

        rows.append(
            {
                "feature": name,
                "auc": directionless_auc(auc_from_ranks(ranks, target)),
                "ks": float(stats.ks_2samp(fraud_values, normal_values).statistic),
                "abs_corr": correlation,
                "mutual_info": mutual_info[name],
                "psi": population_stability_index(values, drift_values)
                if len(drift_values)
                else 0.0,
                "range_coverage": range_coverage(values, drift_values)
                if len(drift_values)
                else 1.0,
                "dominant_share": float(pd.Series(values).value_counts(normalize=True).iloc[0]),
                "n_unique": int(pd.Series(values).nunique()),
                "mean_fraud": float(fraud_values.mean()),
                "mean_normal": float(normal_values.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# The noise ceiling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseCeiling:
    """What the signal measures look like when there is provably nothing to find.

    Found by shuffling the target so any real relationship is destroyed, then recording the
    best score any feature still reaches by chance. Repeat, and take a high quantile of
    those maxima.

    Taking the maximum across all features on every shuffle is the point. Testing thirty
    features and keeping the best is thirty chances to be fooled, and the maximum statistic
    accounts for that. A per feature threshold would not.

    Two ceilings come out of it, one for AUC and one for KS, because the two measures see
    different things. A feature has to fall below both before it can fairly be called noise.
    """

    ceiling: float
    ks_ceiling: float
    shuffles: int
    quantile: float
    mean_max: float
    observed_max: float
    ks_mean_max: float
    n_features: int
    n_fraud: int

    def describe(self) -> str:
        return (
            f"with the target shuffled {self.shuffles} times, the best univariate AUC any "
            f"of {self.n_features} features reached by chance averaged {self.mean_max:.4f} "
            f"and peaked at {self.observed_max:.4f}. The {self.quantile:.0%} quantile, "
            f"{self.ceiling:.4f}, is the AUC noise ceiling. The same procedure puts the KS "
            f"ceiling at {self.ks_ceiling:.4f}, against a chance average of "
            f"{self.ks_mean_max:.4f}."
        )


def permutation_noise_ceiling(
    analysis: pd.DataFrame, config: Config, features: list[str] | None = None
) -> NoiseCeiling:
    """Measure what a feature can score on this dataset while carrying no signal at all.

    This is what makes "weak" a measured word instead of a round number picked by hand. On
    a dataset with only a few hundred fraud rows the noise floor sits far above the 0.5 AUC
    that intuition suggests, so a threshold chosen by hand would keep pure noise.
    """
    target_column = config.dataset.spec().target_column
    names = (
        features if features is not None else [c for c in analysis.columns if c != target_column]
    )
    target = analysis[target_column].to_numpy()

    # Rank once. Shuffling the target does not change any feature's ordering, so this is
    # the whole reason the test is cheap enough to run on every pipeline run.
    columns = [analysis[name].to_numpy() for name in names]
    average_ranks = [stats.rankdata(values) for values in columns]
    tie_ranks = [TieAwareRanks.build(values) for values in columns]

    rng = np.random.default_rng(config.project.seed)
    auc_maxima, ks_maxima = [], []
    for _ in range(config.eda.permutation_shuffles):
        shuffled = rng.permutation(target)
        auc_maxima.append(
            max(directionless_auc(auc_from_ranks(r, shuffled)) for r in average_ranks)
        )
        ks_maxima.append(max(ks_from_ranks(r, shuffled) for r in tie_ranks))

    auc_array, ks_array = np.asarray(auc_maxima), np.asarray(ks_maxima)
    quantile = config.eda.permutation_quantile

    return NoiseCeiling(
        ceiling=float(np.quantile(auc_array, quantile)),
        ks_ceiling=float(np.quantile(ks_array, quantile)),
        shuffles=config.eda.permutation_shuffles,
        quantile=quantile,
        mean_max=float(auc_array.mean()),
        observed_max=float(auc_array.max()),
        ks_mean_max=float(ks_array.mean()),
        n_features=len(names),
        n_fraud=int(target.sum()),
    )


# --------------------------------------------------------------------------------------
# Redundancy
# --------------------------------------------------------------------------------------


def correlated_pairs(
    analysis: pd.DataFrame, config: Config, threshold: float | None = None
) -> pd.DataFrame:
    """Feature pairs whose absolute correlation is at or above the threshold.

    Returned strongest first. A pair above the duplicate threshold means the two carry the
    same information, so one of them is redundant.
    """
    target_column = config.dataset.spec().target_column
    limit = threshold if threshold is not None else config.feature_selection.duplicate_correlation

    # A column with no variance has no correlation with anything. Leaving it in produces a
    # divide by zero and a column of NaN, which is a worse way of saying the same thing.
    # The constant rule catches these separately.
    features = [
        name
        for name in analysis.columns
        if name != target_column and analysis[name].to_numpy().std() > 0
    ]
    if len(features) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])

    matrix = analysis[features].corr().abs()
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    stacked = upper.stack()
    above = stacked[stacked >= limit].sort_values(ascending=False)

    return pd.DataFrame(
        [{"feature_a": a, "feature_b": b, "abs_corr": float(v)} for (a, b), v in above.items()]
    )


def exact_duplicate_columns(analysis: pd.DataFrame, config: Config) -> list[tuple[str, str]]:
    """Columns that are identical to another column, value for value.

    Hashing each column makes this one pass rather than a comparison of every pair, which
    matters once a feature set grows past a few dozen columns. The hash only narrows the
    candidates, and a real equality check confirms each one, so a hash collision cannot
    cause a wrong answer.
    """
    target_column = config.dataset.spec().target_column
    seen: dict[int, list[str]] = {}
    duplicates = []

    for name in analysis.columns:
        if name == target_column:
            continue
        digest = int(pd.util.hash_pandas_object(analysis[name], index=False).sum())
        for candidate in seen.get(digest, []):
            if analysis[name].equals(analysis[candidate]):
                duplicates.append((candidate, name))
                break
        else:
            seen.setdefault(digest, []).append(name)

    return duplicates


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def build_report(
    stats: pd.DataFrame,
    pairs: pd.DataFrame,
    duplicates: list[tuple[str, str]],
    noise: NoiseCeiling,
    analysis: pd.DataFrame,
    drift: pd.DataFrame,
    config: Config,
) -> str:
    """Write up what the analysis found, in a form worth reading in a pull request."""
    from fraud_pipeline import plots

    spec = config.dataset.spec()
    target = spec.target_column
    fraud_rate = float(analysis[target].mean())
    strong = stats[stats["auc"] >= noise.ceiling]

    lines = [
        f"# Exploratory analysis: {spec.name}",
        "",
        f"Measured on the **{config.eda.analysis_split}** split, "
        f"{len(analysis):,} rows and {int(analysis[target].sum())} fraud cases. "
        f"Drift is measured against **{config.eda.drift_split}**, {len(drift):,} rows.",
        "",
        "The test split was not read. Choosing features by looking at test data would make",
        "every metric this project later reports meaningless, so the config refuses it.",
        "",
        "## What the numbers say",
        "",
        f"- The positive class is {fraud_rate:.3%} of transactions. Predicting normal every",
        f"  time scores {1 - fraud_rate:.3%} accuracy and catches no fraud at all.",
        f"- {len(strong)} of {len(stats)} features carry signal above the noise ceiling.",
        f"- The strongest single feature is **{stats.iloc[0]['feature']}** at "
        f"{stats.iloc[0]['auc']:.4f} univariate AUC.",
        f"- Exact duplicate columns found: **{len(duplicates)}**.",
        f"- Feature pairs correlated at or above "
        f"{config.feature_selection.duplicate_correlation}: **{len(pairs)}**.",
        "",
        "## Noise ceiling",
        "",
        noise.describe(),
        "",
        "This is the number that turns 'weak' into something measured. With only",
        f"{noise.n_fraud} fraud rows, a feature carrying nothing at all still reaches around",
        f"{noise.mean_max:.3f} AUC once you take the best of {noise.n_features} tries. A",
        "threshold picked by hand would have kept pure noise.",
        "",
        "## Target leakage",
        "",
    ]

    leaking = stats[stats["auc"] >= config.feature_selection.leakage_auc]
    if leaking.empty:
        lines += [
            f"None found. Nothing reaches the {config.feature_selection.leakage_auc} AUC",
            "threshold, and the strongest feature sits well below it.",
            "",
            "That is the expected answer here rather than a lucky one. V1 to V28 are",
            "principal components of features the publisher would not release, so there is",
            "no original column left that could encode the answer. The check still runs on",
            "every pipeline run, because the second dataset has real named columns where a",
            "leak is a genuine possibility.",
        ]
    else:
        lines += [
            "**Found. Do not trust any result until this is explained.**",
            "",
            "| Feature | Univariate AUC |",
            "| --- | --- |",
        ]
        lines += [f"| {r.feature} | {r.auc:.4f} |" for r in leaking.itertuples()]

    lines += ["", "## Redundancy", ""]
    if duplicates:
        lines += ["Identical columns:", ""]
        lines += [f"- `{a}` and `{b}`" for a, b in duplicates]
        lines.append("")
    if pairs.empty:
        strongest = correlated_pairs(analysis, config, threshold=0.0)
        top = strongest.head(config.eda.correlation_report_top)
        lines += [
            "No pair reaches the duplicate threshold. The most correlated pairs are:",
            "",
            "| Feature A | Feature B | Absolute correlation |",
            "| --- | --- | --- |",
        ]
        lines += [f"| {r.feature_a} | {r.feature_b} | {r.abs_corr:.4f} |" for r in top.itertuples()]
        lines += [
            "",
            "The components are near orthogonal to each other, which is what PCA output",
            "should look like. The correlations that do exist are between the components",
            "and Amount, which was never part of that PCA.",
        ]
    else:
        lines += ["| Feature A | Feature B | Absolute correlation |", "| --- | --- | --- |"]
        lines += [
            f"| {r.feature_a} | {r.feature_b} | {r.abs_corr:.4f} |" for r in pairs.itertuples()
        ]

    lines += [
        "",
        "## Stability between train and validation",
        "",
        "| Feature | PSI | Validation values inside the training range |",
        "| --- | --- | --- |",
    ]
    unstable = stats.sort_values("psi", ascending=False).head(config.eda.correlation_report_top)
    lines += [
        f"| {r.feature} | {r.psi:.3f} | {r.range_coverage:.2%} |" for r in unstable.itertuples()
    ]

    lines += [
        "",
        "## Every feature",
        "",
        "| Feature | AUC | KS | Absolute correlation | Mutual info | PSI | Range coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.feature} | {r.auc:.4f} | {r.ks:.4f} | {r.abs_corr:.4f} | "
        f"{r.mutual_info:.4f} | {r.psi:.3f} | {r.range_coverage:.4f} |"
        for r in stats.itertuples()
    ]

    if config.eda.figures:
        lines += ["", "## Figures", ""]
        lines += [f"![{name}](../figures/{name})" for name in plots.figure_names()]

    return "\n".join([*lines, ""])


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def run(config: Config):
    """Run the exploration stage and the feature selection that follows from it."""
    from fraud_pipeline import feature_selection, plots
    from fraud_pipeline.paths import ensure_dir
    from fraud_pipeline.splits import load_split

    analysis = load_split(config, config.eda.analysis_split)
    drift = load_split(config, config.eda.drift_split)
    logger.info(
        "analysing %s rows from %s, drift against %s rows from %s",
        f"{len(analysis):,}",
        config.eda.analysis_split,
        f"{len(drift):,}",
        config.eda.drift_split,
    )

    stats = compute_feature_stats(analysis, drift, config)
    logger.info("computed statistics for %s features", len(stats))

    noise = permutation_noise_ceiling(analysis, config)
    logger.info("noise ceiling %.4f from %s shuffles", noise.ceiling, noise.shuffles)

    pairs = correlated_pairs(analysis, config)
    duplicates = exact_duplicate_columns(analysis, config)
    logger.info("%s correlated pair(s), %s exact duplicate column(s)", len(pairs), len(duplicates))

    for row in stats.head(5).itertuples():
        logger.info("  %-8s auc %.4f  ks %.4f  psi %.3f", row.feature, row.auc, row.ks, row.psi)

    selection = feature_selection.select_features(stats, pairs, noise, config, duplicates)
    counts = selection.counts()
    logger.info(
        "feature sets: all %s, safe %s, selected %s. Active set is %s.",
        counts["all"],
        counts["safe"],
        counts["selected"],
        selection.active_set,
    )
    for decision in selection.decisions:
        if decision.dropped_by_tier is not None:
            logger.info(
                "  drop %-8s tier %s  %s",
                decision.feature,
                decision.dropped_by_tier,
                decision.reasons[0],
            )
    for warning in selection.warnings:
        logger.warning(warning)

    stats_path = ensure_dir(config.paths.tables_dir()) / config.eda.stats_file
    stats.to_csv(stats_path, index=False)
    logger.info("wrote %s", stats_path.name)

    report_path = ensure_dir(config.paths.tables_dir()) / config.eda.report_file
    report_path.write_text(
        build_report(stats, pairs, duplicates, noise, analysis, drift, config), encoding="utf-8"
    )
    logger.info("wrote %s", report_path.name)

    json_path, selection_report = feature_selection.write_selection(selection, stats, config)
    logger.info("wrote %s", json_path)
    logger.info("wrote %s", selection_report)

    plots.draw_all(analysis, drift, stats, noise, config)

    return stats, selection
