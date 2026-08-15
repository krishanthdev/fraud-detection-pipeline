"""Tests for stage 3, feature engineering.

The important test in this file is `test_features_never_look_into_the_future`. Every other
kind of bug here announces itself. A feature that peeks at later rows does not: it makes the
model look better, the metrics look plausible, and the whole project quietly worthless.

So that one is checked structurally rather than by inspection. Build features on the full
stream, build them again on a truncated copy, and assert the overlapping rows come out
identical. A feature that used even one later row cannot pass that.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraud_pipeline import features
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.features import FeatureError

from .synthetic import make_transactions, write_raw_csv


@pytest.fixture
def feature_config(config_path, tmp_path) -> Config:
    return load_config(
        config_path,
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.data_processed={(tmp_path / 'processed').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            "validation.min_rows=100",
            "validation.min_fraud_rows_per_split=1",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=1000",
            "eda.figures=false",
        ],
    )


@pytest.fixture
def ordered_frame(feature_config) -> pd.DataFrame:
    frame = make_transactions(feature_config, rows=3000, fraud_rows=22)
    # Uneven gaps, so the rate features have something to say. Not rounded, so every
    # timestamp is unique and the row order is unambiguous. Ties are covered separately.
    rng = np.random.default_rng(0)
    frame["Time"] = np.cumsum(rng.exponential(3.0, len(frame)))
    return frame


@pytest.fixture
def fitted(ordered_frame, feature_config) -> features.FittedFeatures:
    return features.fit(ordered_frame.iloc[:2000], feature_config)


@pytest.fixture
def prepared(feature_config) -> Config:
    """Run ingest and validate so the feature stage has splits to read.

    The fraud count is kept under the 1 percent ceiling the validation stage enforces, so
    the fixture exercises the real path rather than tripping a data quality rule.
    """
    from fraud_pipeline import ingest, validation

    write_raw_csv(feature_config, make_transactions(feature_config, rows=6000, fraud_rows=45))
    ingest.run(feature_config)
    validation.run(feature_config)
    return feature_config


# --------------------------------------------------------------------------------------
# The one that matters
# --------------------------------------------------------------------------------------


def test_features_never_look_into_the_future(ordered_frame, fitted, feature_config) -> None:
    """Truncating the stream must not change the features of the rows that remain.

    If any feature used a later row, the truncated copy would compute something different
    for the rows near the cut. This is the test that would catch a rolling window that
    forgot to shift, or a mean taken over the whole column.
    """
    full, built = features.build_features(ordered_frame, fitted, feature_config)

    cut = 1800
    truncated, _ = features.build_features(ordered_frame.iloc[:cut].copy(), fitted, feature_config)

    pd.testing.assert_frame_equal(
        full.iloc[:cut][built].reset_index(drop=True),
        truncated[built].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
    )


def test_a_deliberately_leaky_feature_would_be_caught(
    ordered_frame, fitted, feature_config
) -> None:
    """Prove the test above can actually fail, by planting a feature that peeks forward.

    A guard nobody has watched fail is not a guard. This builds the exact mistake the guard
    exists to catch, a window that looks ahead instead of behind, and confirms the
    truncation check notices.
    """
    full, _ = features.build_features(ordered_frame, fitted, feature_config)
    cut = 1800
    truncated, _ = features.build_features(ordered_frame.iloc[:cut].copy(), fitted, feature_config)

    # shift(-1) looks at the next row. This is the bug, written on purpose.
    full = full.assign(leaky=full["Amount"].shift(-1).fillna(0))
    truncated = truncated.assign(leaky=truncated["Amount"].shift(-1).fillna(0))

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            full.iloc[:cut][["leaky"]].reset_index(drop=True),
            truncated[["leaky"]].reset_index(drop=True),
        )


def test_rolling_baselines_exclude_the_current_row(fitted, feature_config) -> None:
    """An amount must not be part of the baseline it is compared against.

    Without the shift, a large amount raises its own baseline and looks less unusual than
    it is, which is backwards for fraud detection.
    """
    frame = pd.DataFrame(
        {
            "Time": np.arange(6, dtype="float64") * 10,
            "Amount": [10.0, 10.0, 10.0, 10.0, 1000.0, 10.0],
            "Class": np.zeros(6, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, _ = features.build_features(frame, fitted, feature_config)

    # The baseline at the spike is built from the four calm rows before it.
    assert built.loc[4, "amount_roll_mean_10"] == pytest.approx(10.0)
    # And the spike itself is flagged as far from that baseline. The denominator floor of
    # one unit makes this 1000 / (10 + 1) rather than 1000 / 10.
    assert built.loc[4, "amount_ratio_10"] == pytest.approx(1000.0 / 11.0, rel=1e-6)
    assert built.loc[4, "amount_ratio_10"] > built.loc[3, "amount_ratio_10"] * 50
    # The row after the spike sees a baseline that has moved.
    assert built.loc[5, "amount_roll_mean_10"] > 10.0


# --------------------------------------------------------------------------------------
# Fitting happens on train only
# --------------------------------------------------------------------------------------


def test_fitted_statistics_come_from_the_given_rows_only(ordered_frame, feature_config) -> None:
    train = ordered_frame.iloc[:2000]
    on_train = features.fit(train, feature_config)
    on_everything = features.fit(ordered_frame, feature_config)

    assert on_train.amount_median == pytest.approx(train["Amount"].median())
    assert on_train.fitted_on_rows == 2000
    # The two differ, which is the point: fitting on everything would be a different number.
    assert on_train.amount_median != on_everything.amount_median


def test_the_stage_fits_on_train_and_not_on_the_whole_stream(prepared) -> None:
    from fraud_pipeline.splits import load_split

    config = prepared
    fitted, _ = features.run(config)

    train = load_split(config, "train")
    assert fitted.fitted_on_rows == len(train)
    assert fitted.amount_median == pytest.approx(train["Amount"].median())
    assert fitted.fitted_on_split == "train"


def test_transforming_a_split_does_not_change_the_fitted_numbers(
    ordered_frame, feature_config
) -> None:
    fitted = features.fit(ordered_frame.iloc[:2000], feature_config)
    before = fitted.amount_median

    features.build_features(ordered_frame, fitted, feature_config)

    assert fitted.amount_median == before


# --------------------------------------------------------------------------------------
# What gets built
# --------------------------------------------------------------------------------------


def test_the_expected_features_are_built(ordered_frame, fitted, feature_config) -> None:
    _, built = features.build_features(ordered_frame, fitted, feature_config)

    expected = {"amount_log", "hour", "hour_sin", "hour_cos", "seconds_since_prev"}
    for window in feature_config.features.rolling_windows:
        expected |= {
            f"amount_roll_mean_{window}",
            f"amount_roll_std_{window}",
            f"amount_dev_{window}",
            f"amount_ratio_{window}",
            f"txn_rate_{window}",
        }

    assert set(built) == expected


def test_hour_is_cyclical_and_wraps_correctly(fitted, feature_config) -> None:
    """23:00 and 00:00 are one hour apart, and the encoding has to agree."""
    frame = pd.DataFrame(
        {
            "Time": np.array([0.0, 23 * 3600.0, 24 * 3600.0]),
            "Amount": [10.0, 10.0, 10.0],
            "Class": np.zeros(3, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, _ = features.build_features(frame, fitted, feature_config)

    midnight = built.loc[0, ["hour_sin", "hour_cos"]].to_numpy()
    eleven_pm = built.loc[1, ["hour_sin", "hour_cos"]].to_numpy()
    next_midnight = built.loc[2, ["hour_sin", "hour_cos"]].to_numpy()

    # Midnight on day two lands in the same place as midnight on day one.
    assert np.allclose(midnight, next_midnight, atol=1e-9)
    # And 23:00 is close to midnight, which the raw hour column cannot express.
    assert np.linalg.norm(eleven_pm - midnight) < 0.6


def test_hour_covers_the_whole_dial(ordered_frame, fitted, feature_config) -> None:
    built, _ = features.build_features(ordered_frame, fitted, feature_config)
    assert built["hour"].between(0, 24).all()


def test_amount_log_handles_a_zero_amount(fitted, feature_config) -> None:
    frame = pd.DataFrame(
        {
            "Time": np.arange(3, dtype="float64"),
            "Amount": [0.0, 1.0, 100.0],
            "Class": np.zeros(3, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, _ = features.build_features(frame, fitted, feature_config)

    assert built.loc[0, "amount_log"] == pytest.approx(0.0)
    assert built["amount_log"].is_monotonic_increasing


def test_no_feature_comes_out_missing_or_infinite(ordered_frame, fitted, feature_config) -> None:
    """A division by a flat baseline is the obvious way to produce an infinity here."""
    built, names = features.build_features(ordered_frame, fitted, feature_config)
    values = built[names].to_numpy()

    assert not np.isnan(values).any()
    assert np.isfinite(values).all()


def test_no_feature_reaches_an_absurd_magnitude(ordered_frame, fitted, feature_config) -> None:
    """The check the finite test was not doing.

    An epsilon in a denominator never produces an infinity, so a finite check passes it. It
    produces 1e12, which then wrecks anything that cares about scale. That bug shipped once
    and this is the guard that would have caught it.
    """
    built, names = features.build_features(ordered_frame, fitted, feature_config)

    for name in names:
        largest = built[name].abs().max()
        assert largest < features.MAX_PLAUSIBLE_MAGNITUDE, f"{name} reaches {largest:,.0f}"


def test_a_burst_of_same_second_transactions_stays_sane(fitted, feature_config) -> None:
    """The exact shape that broke it: many transactions inside one second.

    Over half the gaps in the real file are exactly zero, so the rolling mean gap is zero
    too, and an unfloored rate feature returns tens of billions.
    """
    rows = 300
    frame = pd.DataFrame(
        {
            # Every transaction in the same second, so every gap is zero.
            "Time": np.zeros(rows, dtype="float64"),
            "Amount": np.full(rows, 50.0),
            "Class": np.zeros(rows, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, names = features.build_features(frame, fitted, feature_config)

    # Row zero has no predecessor and gets the fitted median. Every other gap is zero.
    assert (built["seconds_since_prev"].iloc[1:] == 0).all()
    for name in names:
        assert built[name].abs().max() < features.MAX_PLAUSIBLE_MAGNITUDE, name

    # The rate feature saturates at its ceiling rather than exploding.
    for window in feature_config.features.rolling_windows:
        assert built[f"txn_rate_{window}"].max() <= 60.0


def test_the_sanity_check_rejects_an_exploded_feature(feature_config) -> None:
    """Prove the guard fires, using the value the real bug produced."""
    frame = pd.DataFrame({"good": [1.0, 2.0], "exploded": [1.0, 6e10]})

    with pytest.raises(FeatureError, match="implausibly large"):
        features.check_feature_sanity(frame, ["good", "exploded"])


def test_the_sanity_check_rejects_infinities(feature_config) -> None:
    frame = pd.DataFrame({"bad": [1.0, np.inf]})

    with pytest.raises(FeatureError, match="infinite"):
        features.check_feature_sanity(frame, ["bad"])


def test_the_sanity_check_rejects_missing_values(feature_config) -> None:
    frame = pd.DataFrame({"bad": [1.0, np.nan]})

    with pytest.raises(FeatureError, match="missing values"):
        features.check_feature_sanity(frame, ["bad"])


def test_a_completely_flat_stream_does_not_divide_by_zero(fitted, feature_config) -> None:
    """Every amount identical means a rolling standard deviation of exactly zero."""
    rows = 200
    frame = pd.DataFrame(
        {
            "Time": np.arange(rows, dtype="float64") * 5,
            "Amount": np.full(rows, 25.0),
            "Class": np.zeros(rows, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, names = features.build_features(frame, fitted, feature_config)

    assert np.isfinite(built[names].to_numpy()).all()


def test_an_unsorted_frame_is_sorted_before_anything_is_built(
    ordered_frame, fitted, feature_config
) -> None:
    """A history feature on unsorted rows is silently meaningless, so sorting is enforced.

    Timestamps are unique in this fixture, so sorting fully determines the order and the
    shuffled copy has to come out identical.
    """
    shuffled = ordered_frame.sample(frac=1.0, random_state=1).reset_index(drop=True)

    from_ordered, names = features.build_features(ordered_frame, fitted, feature_config)
    from_shuffled, _ = features.build_features(shuffled, fitted, feature_config)

    pd.testing.assert_frame_equal(
        from_ordered[names], from_shuffled[names], check_exact=False, rtol=1e-12
    )


def test_tied_timestamps_keep_the_order_they_arrived_in(fitted, feature_config) -> None:
    """Documented behaviour, not an accident.

    `Time` is whole seconds in the real file and busy periods carry several transactions
    per second, so ties are the normal case. Which of two transactions in the same second
    came first is information the file simply does not contain, so the sort is stable and
    input order decides. That makes a run reproducible for a given file, which is the
    property that actually matters, without pretending to an ordering that does not exist.
    """
    rows = 40
    frame = pd.DataFrame(
        {
            "Time": np.repeat(np.arange(rows // 4, dtype="float64"), 4),
            "Amount": np.arange(rows, dtype="float64") + 1.0,
            "Class": np.zeros(rows, dtype="int8"),
        }
    )
    for column in feature_config.dataset.spec().pca_columns():
        frame[column] = 0.0

    built, names = features.build_features(frame, fitted, feature_config)

    # Same input, same output, every time.
    again, _ = features.build_features(frame, fitted, feature_config)
    pd.testing.assert_frame_equal(built[names], again[names])

    # The amounts keep their arrival order within each tied second.
    assert built["Amount"].is_monotonic_increasing


def test_rolling_windows_must_be_usable(config_tree) -> None:
    config_tree["features"]["rolling_windows"] = [1, 50]
    with pytest.raises(ValueError, match="at least 2"):
        Config.model_validate(config_tree)


# --------------------------------------------------------------------------------------
# The stage end to end
# --------------------------------------------------------------------------------------


def test_the_stage_writes_all_three_splits(prepared) -> None:
    config = prepared
    fitted, _ = features.run(config)

    for name in ("train", "validation", "test"):
        part = features.load_engineered(config, name)
        assert not part.empty
        for column in fitted.feature_names:
            assert column in part.columns


def test_the_stage_preserves_every_row(prepared) -> None:
    from fraud_pipeline.splits import load_split

    config = prepared
    features.run(config)

    for name in ("train", "validation", "test"):
        assert len(features.load_engineered(config, name)) == len(load_split(config, name))


def test_the_stage_preserves_the_fraud_counts(prepared) -> None:
    from fraud_pipeline.splits import load_split

    config = prepared
    features.run(config)

    for name in ("train", "validation", "test"):
        engineered = features.load_engineered(config, name)["Class"].sum()
        original = load_split(config, name)["Class"].sum()
        assert engineered == original


def test_validation_features_continue_from_the_end_of_training(prepared) -> None:
    """The first validation row should have a real baseline, not a cold start.

    Building each split separately would give it one, and production never works that way.
    """
    config = prepared
    features.run(config)

    validation = features.load_engineered(config, "validation")
    first = validation.iloc[0]

    assert first["amount_roll_mean_50"] > 0
    # A cold start would have fallen back to the fitted median on the very first row.
    assert first["amount_roll_std_50"] > 0


def test_the_fitted_statistics_round_trip(prepared) -> None:
    config = prepared
    written, _ = features.run(config)
    loaded = features.load_fitted(config)

    assert loaded.amount_median == pytest.approx(written.amount_median)
    assert loaded.feature_names == written.feature_names
    assert loaded.built == written.built


def test_the_fitted_file_is_readable_json(prepared) -> None:
    """It is written as json rather than pickled so a human can open it and check."""
    config = prepared
    features.run(config)

    path = config.paths.processed() / config.features.output_dir / config.features.stats_file
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "amount_median" in payload
    assert "feature_names" in payload


def test_the_stage_is_reproducible(prepared) -> None:
    config = prepared
    first, _ = features.run(config)
    second, _ = features.run(config)

    assert first.feature_names == second.feature_names
    assert first.amount_median == pytest.approx(second.amount_median)


def test_reselection_runs_over_the_engineered_features(prepared) -> None:
    config = prepared
    fitted, selection = features.run(config)

    assert selection is not None
    assert selection.counts()["all"] == len(fitted.feature_names)
    report = (config.paths.tables_dir() / config.features.report_file).read_text(encoding="utf-8")
    assert "# Feature engineering" in report


def test_reselection_can_be_switched_off(prepared, config_path) -> None:
    without_reselection = load_config(
        config_path,
        overrides=[
            f"paths.data_interim={prepared.paths.interim().as_posix()}",
            f"paths.data_processed={prepared.paths.processed().as_posix()}",
            f"paths.tables={prepared.paths.tables_dir().as_posix()}",
            "features.reselect=false",
        ],
    )
    fitted, selection = features.run(without_reselection)

    assert selection is None
    # The tables are still written. Only the extra selection pass is skipped.
    assert features.load_engineered(without_reselection, "train").shape[0] > 0
    assert fitted.feature_names


def test_loading_before_the_stage_has_run_explains_itself(feature_config) -> None:
    with pytest.raises(FeatureError, match="fraud features"):
        features.load_engineered(feature_config, "train")


def test_loading_an_unknown_split_is_rejected(feature_config) -> None:
    with pytest.raises(FeatureError, match="unknown split"):
        features.load_engineered(feature_config, "holdout")


def test_loading_fitted_before_the_stage_has_run_explains_itself(feature_config) -> None:
    with pytest.raises(FeatureError, match="fraud features"):
        features.load_fitted(feature_config)


def test_the_stage_needs_the_splits_to_exist(feature_config) -> None:
    from fraud_pipeline.splits import SplitError

    with pytest.raises(SplitError, match="fraud validate"):
        features.run(feature_config)
