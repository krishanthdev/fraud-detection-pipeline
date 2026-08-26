"""Figures for the exploratory analysis.

Kept apart from ``eda.py`` so the statistics can be tested without a plotting library in
the way, and so a headless run can turn drawing off entirely.

Every figure answers one question. A plot that does not change a decision is decoration,
and decoration in a report is worse than nothing because it buries the plots that matter.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display on a CI runner or in a container.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud_pipeline.config import Config  # noqa: E402
from fraud_pipeline.paths import ensure_dir  # noqa: E402

logger = logging.getLogger(__name__)

FRAUD_COLOUR = "#c1442e"
NORMAL_COLOUR = "#3b6ea5"
ACCENT_COLOUR = "#7a5195"


def _save(fig: plt.Figure, name: str, config: Config) -> str:
    path = ensure_dir(config.paths.figures_dir()) / name
    fig.savefig(path, dpi=config.eda.figure_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", path.name)
    return str(path)


def plot_class_balance(analysis: pd.DataFrame, config: Config) -> str:
    """Why accuracy is not reported anywhere in this project, in one picture.

    Drawn on a log scale, because on a linear one the fraud bar is invisible, which is
    itself the point being made.
    """
    target = config.dataset.spec().target_column
    counts = analysis[target].value_counts().sort_index()
    fraud_rate = analysis[target].mean()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        ["Normal", "Fraud"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[NORMAL_COLOUR, FRAUD_COLOUR],
        width=0.55,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title(
        f"Class balance: {fraud_rate:.3%} of transactions are fraud\n"
        f"Predicting 'normal' every time would be {1 - fraud_rate:.2%} accurate",
        fontsize=11,
    )
    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:,}",
            ha="center",
            fontsize=10,
        )
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "01_class_balance.png", config)


def plot_fraud_rate_over_time(
    analysis: pd.DataFrame, drift: pd.DataFrame, config: Config, bins: int = 24
) -> str:
    """The drift that justifies splitting by time rather than at random.

    Only train and validation are drawn. The test split is deliberately absent, so nothing
    about it can influence a decision made here.
    """
    spec = config.dataset.spec()
    time_column, target = spec.time_column, spec.target_column

    combined = pd.concat(
        [analysis.assign(_part="train"), drift.assign(_part="validation")], ignore_index=True
    )
    hours = combined[time_column] / 3600.0
    edges = np.linspace(hours.min(), hours.max(), bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    grouped = combined.groupby(pd.cut(hours, edges, include_lowest=True), observed=False)[target]
    rate = grouped.mean().to_numpy() * 100
    counts = grouped.size().to_numpy()
    boundary = analysis[time_column].max() / 3600.0

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax.plot(centres, rate, color=FRAUD_COLOUR, marker="o", markersize=4, linewidth=1.6)
    ax.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
    ax.text(boundary, ax.get_ylim()[1] * 0.95, "  train ends", fontsize=9, va="top")
    ax.axhline(
        analysis[target].mean() * 100,
        color=NORMAL_COLOUR,
        linestyle=":",
        linewidth=1.2,
        label=f"train mean {analysis[target].mean() * 100:.3f}%",
    )
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title(
        "Fraud rate is not stable over time, which is why the split is chronological\n"
        "A random split would have mixed these periods together and hidden the drift",
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.bar(centres, counts, width=(edges[1] - edges[0]) * 0.9, color=NORMAL_COLOUR, alpha=0.65)
    ax2.axvline(boundary, color="black", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Transactions")
    ax2.set_xlabel("Hours since the first transaction in the file")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "02_fraud_rate_over_time.png", config)


def plot_fraud_by_hour(analysis: pd.DataFrame, drift: pd.DataFrame, config: Config) -> str:
    """Fraud rate against the hour of the day, which turned out to explain a lot.

    This plot exists because of what it corrected. The split level fraud rates looked like
    drift over the two days. Folding the timeline onto a 24 hour clock shows most of it is
    the daily cycle instead: fraud concentrates in the small hours, and the validation
    window happens to contain none of them.

    The shaded band is the clock window validation covers. It is the whole explanation in
    one picture, and it is also the argument for building an hour of day feature.
    """
    spec = config.dataset.spec()
    time_column, target = spec.time_column, spec.target_column

    clock = (analysis[time_column] / 3600.0) % 24
    grouped = analysis.assign(_hour=clock.astype(int)).groupby("_hour")[target]
    rate = grouped.mean() * 100
    volume = grouped.size()

    drift_clock = (drift[time_column] / 3600.0) % 24
    window = (float(drift_clock.min()), float(drift_clock.max()))

    night = analysis[clock.between(1, 5)][target].mean() * 100
    rest = analysis[~clock.between(1, 5)][target].mean() * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(*window, color="#f0c419", alpha=0.22, label="clock hours validation covers")
    ax.bar(rate.index, rate.to_numpy(), color=FRAUD_COLOUR, width=0.75, alpha=0.9)
    ax.axhline(
        analysis[target].mean() * 100,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"overall {analysis[target].mean() * 100:.3f}%",
    )

    ax2 = ax.twinx()
    ax2.plot(
        volume.index,
        volume.to_numpy(),
        color=NORMAL_COLOUR,
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    ax2.set_ylabel("Transactions per hour", color=NORMAL_COLOUR)
    ax2.tick_params(axis="y", labelcolor=NORMAL_COLOUR)
    ax2.spines[["top"]].set_visible(False)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)", color=FRAUD_COLOUR)
    ax.tick_params(axis="y", labelcolor=FRAUD_COLOUR)
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(
        f"Fraud concentrates in the small hours: {night:.3f}% between 01:00 and 05:00 "
        f"against {rest:.3f}% the rest of the day\n"
        "Validation covers none of that window, which explains most of the gap between "
        "the splits",
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines[["top"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "07_fraud_by_hour.png", config)


def plot_correlation_heatmap(analysis: pd.DataFrame, config: Config) -> str:
    """Whether any two features carry the same information.

    On this dataset the answer is no, and the reason is visible in the picture: V1 to V28
    are principal components, so they are orthogonal by construction.
    """
    spec = config.dataset.spec()
    features = [c for c in analysis.columns if c != spec.target_column]
    matrix = analysis[[*features, spec.target_column]].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(matrix)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(matrix.columns, fontsize=7)

    off_diagonal = matrix.to_numpy()[np.triu_indices(len(matrix), k=1)]
    ax.set_title(
        "Feature correlation\n"
        f"Strongest pair away from the diagonal is {np.abs(off_diagonal).max():.3f}, "
        "so nothing is a duplicate of anything else",
        fontsize=11,
    )
    fig.colorbar(image, ax=ax, shrink=0.75, label="Pearson correlation")
    return _save(fig, "03_correlation_heatmap.png", config)


def plot_univariate_ranking(stats: pd.DataFrame, noise, config: Config) -> str:
    """Which features carry signal, measured against what noise alone can reach.

    The vertical line is the honest part. Without it, a feature at 0.54 AUC looks like weak
    evidence. With it, that feature is visibly indistinguishable from a column of random
    numbers on a dataset with this few fraud cases.

    Three colours, because the AUC ranking alone would tell a misleading story. Some
    features below the AUC line still separate the classes, they just do it without fraud
    being consistently higher or lower, and only the KS test sees that.
    """
    ordered = stats.sort_values("auc")

    colours, labels = [], []
    for row in ordered.itertuples():
        if row.auc >= noise.ceiling:
            colours.append(FRAUD_COLOUR)
            labels.append("signal")
        elif row.ks >= noise.ks_ceiling:
            colours.append(ACCENT_COLOUR)
            labels.append("rescued")
        else:
            colours.append("#b0b0b0")
            labels.append("noise")

    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.barh(ordered["feature"], ordered["auc"], color=colours, height=0.7)
    ax.axvline(noise.ceiling, color="black", linestyle="--", linewidth=1.4)
    ax.text(
        noise.ceiling,
        len(ordered) - 0.4,
        f"  AUC ceiling {noise.ceiling:.3f}",
        fontsize=9,
        va="top",
    )
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FRAUD_COLOUR),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT_COLOUR),
        plt.Rectangle((0, 0), 1, 1, color="#b0b0b0"),
    ]
    ax.legend(
        handles,
        [
            "above the AUC ceiling",
            f"below it, but KS above {noise.ks_ceiling:.3f}, so kept",
            "below both, dropped as noise",
        ],
        fontsize=8.5,
        loc="lower right",
        frameon=False,
    )

    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Univariate AUC (direction ignored)")
    rescued = labels.count("rescued")
    dropped = labels.count("noise")
    ax.set_title(
        "How much each feature separates fraud on its own\n"
        f"{rescued + dropped} of {len(stats)} sit below the AUC ceiling, but only {dropped} "
        f"also fail the KS test and get dropped",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, "04_univariate_ranking.png", config)


def plot_top_feature_distributions(
    analysis: pd.DataFrame, stats: pd.DataFrame, config: Config
) -> str:
    """What the strongest features actually look like for the two classes.

    A ranking says a feature separates the classes. This says how, which is what makes a
    later SHAP explanation believable rather than something to take on trust.
    """
    spec = config.dataset.spec()
    count = config.eda.distribution_plot_features
    top = stats.nlargest(count, "auc")["feature"].tolist()

    fraud = analysis[analysis[spec.target_column] == 1]
    normal = analysis[analysis[spec.target_column] == 0]

    columns = 3
    rows = int(np.ceil(len(top) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, top, strict=False):
        low = float(min(normal[name].quantile(0.001), fraud[name].min()))
        high = float(max(normal[name].quantile(0.999), fraud[name].max()))
        bins = np.linspace(low, high, 60)

        ax.hist(
            normal[name], bins=bins, density=True, color=NORMAL_COLOUR, alpha=0.6, label="Normal"
        )
        ax.hist(fraud[name], bins=bins, density=True, color=FRAUD_COLOUR, alpha=0.65, label="Fraud")

        auc = float(stats.loc[stats["feature"] == name, "auc"].iloc[0])
        ax.set_title(f"{name}  (AUC {auc:.3f})", fontsize=10)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    for ax in axes[len(top) :]:
        ax.set_visible(False)

    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle(
        "The strongest features, fraud against normal, densities scaled to the same height",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    return _save(fig, "05_top_feature_distributions.png", config)


def plot_stability(stats: pd.DataFrame, config: Config) -> str:
    """Which features behave differently in the next time period.

    Two different problems are shown together on purpose. A high PSI means the distribution
    moved, which is survivable. Low range coverage means the new values are outside anything
    the model was fitted on, which is not.
    """
    ordered = stats.sort_values("psi", ascending=False).head(15)
    limit = config.feature_selection.psi_warn
    coverage_limit = config.feature_selection.min_range_coverage

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax.barh(ordered["feature"][::-1], ordered["psi"][::-1], color=ACCENT_COLOUR, height=0.7)
    ax.axvline(limit, color="black", linestyle="--", linewidth=1.2)
    ax.text(limit, -0.6, f" warn at {limit}", fontsize=9)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Population stability index, train against validation (log scale)")
    ax.set_title("How far each feature moved", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    worst = stats.nsmallest(15, "range_coverage")
    colours = [
        FRAUD_COLOUR if v < coverage_limit else NORMAL_COLOUR for v in worst["range_coverage"][::-1]
    ]
    ax2.barh(worst["feature"][::-1], worst["range_coverage"][::-1] * 100, color=colours, height=0.7)
    ax2.axvline(coverage_limit * 100, color="black", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Validation values inside the training range (%)")
    ax2.set_title("Whether the model has seen values like these before", fontsize=11)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _save(fig, "06_stability.png", config)


def plot_shap_importance(importance, run_name: str, config: Config) -> str:
    """What the promoted model leans on, in the order it leans on them.

    Engineered features are coloured differently from the anonymised components, because the
    question this figure answers for the project is whether the feature engineering earned
    its place in the model rather than only in a table.
    """
    top = importance.head(15).iloc[::-1]
    colours = [
        NORMAL_COLOUR if str(name).startswith("V") else ACCENT_COLOUR for name in top["feature"]
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colours, height=0.72)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NORMAL_COLOUR),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT_COLOUR),
    ]
    ax.legend(
        handles,
        ["anonymised component", "engineered feature"],
        fontsize=9,
        loc="lower right",
        frameon=False,
    )

    ax.set_xlabel("Mean absolute SHAP value (probability units)")
    built = sum(1 for name in importance.head(15)["feature"] if not str(name).startswith("V"))
    ax.set_title(
        f"What {run_name} actually uses"
        + chr(10)
        + f"{built} of the top 15 drivers are features this pipeline built",
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, config.explainability.figure_file, config)


def draw_all(
    analysis: pd.DataFrame,
    drift: pd.DataFrame,
    stats: pd.DataFrame,
    noise,
    config: Config,
) -> list[str]:
    """Draw every figure and return the paths written."""
    if not config.eda.figures:
        logger.info("figures are switched off in the config, skipping")
        return []

    ensure_dir(config.paths.figures_dir())
    return [
        plot_class_balance(analysis, config),
        plot_fraud_rate_over_time(analysis, drift, config),
        plot_correlation_heatmap(analysis, config),
        plot_univariate_ranking(stats, noise, config),
        plot_top_feature_distributions(analysis, stats, config),
        plot_stability(stats, config),
        plot_fraud_by_hour(analysis, drift, config),
    ]


def figure_names() -> list[str]:
    """The figure filenames, so the report can link to them without drawing anything."""
    return [
        "01_class_balance.png",
        "02_fraud_rate_over_time.png",
        "03_correlation_heatmap.png",
        "04_univariate_ranking.png",
        "05_top_feature_distributions.png",
        "06_stability.png",
        "07_fraud_by_hour.png",
    ]


def figures_exist(config: Config) -> bool:
    figures = config.paths.figures_dir()
    return all((Path(figures) / name).is_file() for name in figure_names())
