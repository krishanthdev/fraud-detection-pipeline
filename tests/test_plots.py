"""Tests for the figures.

These do not check that a plot looks good, which no test can do. They check that every
figure is actually produced, is a real image, and does not fall over on the awkward inputs
it will eventually meet: a feature with no variance, a single fraud case, a window too short
to cover a full day.

Without this, a broken figure is only discovered by a human running the pipeline and looking
at the output, which is exactly the kind of check that gets skipped when in a hurry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from fraud_pipeline import eda, plots
from fraud_pipeline.config import Config, load_config

from .synthetic import make_transactions

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def plot_config(config_path, tmp_path) -> Config:
    return load_config(
        config_path,
        overrides=[
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=1000",
            "eda.figure_dpi=60",
        ],
    )


@pytest.fixture
def drawn(plot_config):
    """Draw every figure once and hand the paths to the tests that check them."""
    frame = make_transactions(plot_config, rows=6000, fraud_rows=120)
    train, validation = frame.iloc[:4500], frame.iloc[4500:].reset_index(drop=True)

    stats = eda.compute_feature_stats(train, validation, plot_config)
    noise = eda.permutation_noise_ceiling(train, plot_config)
    paths = plots.draw_all(train, validation, stats, noise, plot_config)

    return paths, plot_config


def test_every_named_figure_is_drawn(drawn) -> None:
    paths, config = drawn
    assert len(paths) == len(plots.figure_names())

    for name in plots.figure_names():
        assert (config.paths.figures_dir() / name).is_file(), f"{name} was not drawn"


def test_the_figures_are_real_images(drawn) -> None:
    paths, config = drawn
    for name in plots.figure_names():
        data = (config.paths.figures_dir() / name).read_bytes()
        assert data.startswith(PNG_MAGIC), f"{name} is not a PNG"
        assert len(data) > 5000, f"{name} is suspiciously small, it may be blank"


def test_figures_exist_reports_correctly(drawn) -> None:
    _, config = drawn
    assert plots.figures_exist(config)


def test_figures_exist_is_false_before_anything_is_drawn(plot_config) -> None:
    assert not plots.figures_exist(plot_config)


def test_drawing_can_be_switched_off(config_path, tmp_path) -> None:
    config = load_config(
        config_path,
        overrides=[
            f"paths.figures={(tmp_path / 'figures').as_posix()}",
            "eda.figures=false",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=1000",
        ],
    )
    frame = make_transactions(config, rows=1500, fraud_rows=40)
    train, validation = frame.iloc[:1000], frame.iloc[1000:].reset_index(drop=True)
    stats = eda.compute_feature_stats(train, validation, config)
    noise = eda.permutation_noise_ceiling(train, config)

    assert plots.draw_all(train, validation, stats, noise, config) == []


def test_no_figure_is_left_open(drawn) -> None:
    """An unclosed figure leaks memory, and a long run draws a lot of them."""
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------------------
# Awkward inputs
# --------------------------------------------------------------------------------------


def test_drawing_survives_a_constant_feature(plot_config) -> None:
    frame = make_transactions(plot_config, rows=3000, fraud_rows=60)
    frame["V9"] = 1.0

    train, validation = frame.iloc[:2200], frame.iloc[2200:].reset_index(drop=True)
    stats = eda.compute_feature_stats(train, validation, plot_config)
    noise = eda.permutation_noise_ceiling(train, plot_config)

    assert len(plots.draw_all(train, validation, stats, noise, plot_config)) == len(
        plots.figure_names()
    )


def test_drawing_survives_a_single_fraud_case(plot_config) -> None:
    """A thin slice of a rare class is normal here, not a corner case."""
    frame = make_transactions(plot_config, rows=2000, fraud_rows=1)
    train, validation = frame.iloc[:1400], frame.iloc[1400:].reset_index(drop=True)

    # Guarantee the single fraud row lands in the training half.
    train = train.copy()
    train.loc[train.index[0], "Class"] = 1

    stats = eda.compute_feature_stats(train, validation, plot_config)
    noise = eda.permutation_noise_ceiling(train, plot_config)

    assert plots.draw_all(train, validation, stats, noise, plot_config)


def test_drawing_survives_a_window_shorter_than_a_day(plot_config) -> None:
    """The hour of day figure has to cope with a slice covering only a few hours."""
    frame = make_transactions(plot_config, rows=2000, fraud_rows=50)
    frame["Time"] = frame["Time"] / 100.0  # squeeze the whole file into a few minutes

    train, validation = frame.iloc[:1500], frame.iloc[1500:].reset_index(drop=True)
    stats = eda.compute_feature_stats(train, validation, plot_config)
    noise = eda.permutation_noise_ceiling(train, plot_config)

    assert plots.plot_fraud_by_hour(train, validation, plot_config)
    assert plots.plot_fraud_rate_over_time(train, validation, plot_config)
    assert plots.plot_univariate_ranking(stats, noise, plot_config)
