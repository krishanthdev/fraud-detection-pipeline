"""Tests for the exploratory analysis.

Two things are being checked. First, that the fast implementations agree with the reference
ones: the rank based AUC and KS exist only to make the permutation test affordable, and they
are worthless if they disagree with scipy and scikit learn. Second, that each measure
actually responds to the thing it is supposed to measure.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

from fraud_pipeline import eda
from fraud_pipeline.config import Config

from .synthetic import make_diagnostic_frame, make_transactions, split_for_drift


@pytest.fixture
def fast_eda_config(config_path, tmp_path) -> Config:
    """Sandbox config with the permutation test turned down so tests stay quick."""
    from fraud_pipeline.config import load_config

    return load_config(
        config_path,
        overrides=[
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            "eda.permutation_shuffles=8",
            "eda.mutual_info_sample=1500",
            "eda.figures=false",
        ],
    )


# --------------------------------------------------------------------------------------
# The fast statistics must agree with the reference implementations
# --------------------------------------------------------------------------------------


def test_rank_auc_matches_sklearn(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)
    target = frame["Class"].to_numpy()

    for column in ("strong", "noise", "humped", "counter"):
        values = frame[column].to_numpy()
        ranks = scipy_stats.rankdata(values)
        assert eda.auc_from_ranks(ranks, target) == pytest.approx(
            roc_auc_score(target, values), abs=1e-9
        )


def test_rank_ks_matches_scipy(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)
    target = frame["Class"].to_numpy()

    for column in ("strong", "noise", "humped", "counter", "constant", "almost_constant"):
        values = frame[column].to_numpy()
        ranks = eda.TieAwareRanks.build(values)
        reference = scipy_stats.ks_2samp(values[target == 1], values[target == 0]).statistic
        assert eda.ks_from_ranks(ranks, target) == pytest.approx(reference, abs=1e-9)


def test_ks_is_zero_for_a_constant_column(fast_eda_config) -> None:
    """The bug that broke the first version of this.

    Average ranks do not count anything, so a column holding one value everywhere gave every
    row the same middling rank and the arithmetic reported a distance of about 0.53 between
    two distributions that are identical. That inflated the noise ceiling and made real
    features look like noise.
    """
    values = np.full(1000, 5.0)
    target = np.zeros(1000, dtype="int8")
    target[:100] = 1

    assert eda.ks_from_ranks(eda.TieAwareRanks.build(values), target) == pytest.approx(0.0)


@pytest.mark.parametrize("rounding", [0, 1, 2])
def test_rank_statistics_survive_heavy_ties(rounding: int) -> None:
    """Amount repeats values constantly, so ties are the normal case, not a corner one."""
    rng = np.random.default_rng(3)
    target = np.zeros(1000, dtype="int8")
    target[rng.choice(1000, size=80, replace=False)] = 1
    values = np.round(rng.normal(0, 1, 1000) + target * 0.8, rounding)

    assert eda.auc_from_ranks(scipy_stats.rankdata(values), target) == pytest.approx(
        roc_auc_score(target, values), abs=1e-9
    )
    assert eda.ks_from_ranks(eda.TieAwareRanks.build(values), target) == pytest.approx(
        scipy_stats.ks_2samp(values[target == 1], values[target == 0]).statistic, abs=1e-9
    )


def test_auc_and_ks_are_defined_when_a_class_is_missing() -> None:
    values = np.arange(50.0)
    empty = np.zeros(50, dtype="int8")

    assert eda.auc_from_ranks(scipy_stats.rankdata(values), empty) == 0.5
    assert eda.ks_from_ranks(eda.TieAwareRanks.build(values), empty) == 0.0


def test_directionless_auc_folds_around_a_half() -> None:
    assert eda.directionless_auc(0.9) == pytest.approx(0.9)
    assert eda.directionless_auc(0.1) == pytest.approx(0.9)
    assert eda.directionless_auc(0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Each measure responds to what it measures
# --------------------------------------------------------------------------------------


def test_psi_is_near_zero_for_the_same_distribution() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert eda.population_stability_index(a, b) < 0.05


def test_psi_grows_when_the_distribution_moves() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    small_shift = eda.population_stability_index(a, rng.normal(0.3, 1, 20000))
    big_shift = eda.population_stability_index(a, rng.normal(3.0, 1, 20000))

    assert small_shift < big_shift
    assert big_shift > 0.25


def test_psi_is_zero_for_a_constant_feature() -> None:
    constant = np.full(500, 4.0)
    assert eda.population_stability_index(constant, constant) == 0.0


def test_range_coverage_is_total_when_the_ranges_match() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    assert eda.range_coverage(reference, reference) == pytest.approx(1.0)


def test_range_coverage_is_zero_for_a_counter() -> None:
    """The raw timestamp case: the next period lies entirely beyond the training range."""
    assert eda.range_coverage(np.arange(1000.0), np.arange(1000.0, 2000.0)) == 0.0


def test_range_coverage_is_partial_when_the_ranges_overlap() -> None:
    coverage = eda.range_coverage(np.arange(100.0), np.arange(50.0, 150.0))
    assert 0.4 < coverage < 0.6


# --------------------------------------------------------------------------------------
# The noise ceiling
# --------------------------------------------------------------------------------------


def test_noise_ceiling_sits_above_a_half(fast_eda_config) -> None:
    """The whole reason this exists. Chance alone beats 0.5 when fraud rows are scarce."""
    frame = make_diagnostic_frame(fast_eda_config, rows=3000, fraud_rows=60)
    noise = eda.permutation_noise_ceiling(frame, fast_eda_config)

    assert noise.ceiling > 0.5
    assert noise.ks_ceiling > 0.0
    assert noise.n_fraud == 60


def test_noise_ceiling_rises_as_fraud_rows_get_scarcer(fast_eda_config) -> None:
    """Fewer positives means more room for a meaningless feature to look impressive."""
    many = eda.permutation_noise_ceiling(
        make_diagnostic_frame(fast_eda_config, rows=4000, fraud_rows=400), fast_eda_config
    )
    few = eda.permutation_noise_ceiling(
        make_diagnostic_frame(fast_eda_config, rows=4000, fraud_rows=25), fast_eda_config
    )

    assert few.ceiling > many.ceiling


def test_noise_ceiling_is_reproducible(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=80)
    first = eda.permutation_noise_ceiling(frame, fast_eda_config)
    second = eda.permutation_noise_ceiling(frame, fast_eda_config)

    assert first.ceiling == second.ceiling
    assert first.ks_ceiling == second.ks_ceiling


def test_a_real_feature_clears_the_ceiling_and_noise_does_not(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=3000, fraud_rows=150)
    train, holdout = split_for_drift(frame)

    stats = eda.compute_feature_stats(train, holdout, fast_eda_config)
    noise = eda.permutation_noise_ceiling(train, fast_eda_config)
    lookup = stats.set_index("feature")["auc"]

    assert lookup["strong"] > noise.ceiling
    assert lookup["noise"] < noise.ceiling


def test_the_description_names_both_ceilings(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=1500, fraud_rows=70)
    text = eda.permutation_noise_ceiling(frame, fast_eda_config).describe()

    assert "AUC noise ceiling" in text
    assert "KS ceiling" in text


# --------------------------------------------------------------------------------------
# Redundancy detection
# --------------------------------------------------------------------------------------


def test_exact_duplicate_columns_are_found(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=1000, fraud_rows=50)
    duplicates = eda.exact_duplicate_columns(frame, fast_eda_config)

    assert ("strong", "strong_copy") in duplicates


def test_no_duplicates_reported_when_there_are_none(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=1000, fraud_rows=50).drop(
        columns=["strong_copy"]
    )
    assert eda.exact_duplicate_columns(frame, fast_eda_config) == []


def test_correlated_pairs_finds_the_near_duplicate(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)
    pairs = eda.correlated_pairs(frame, fast_eda_config)

    found = {frozenset({r.feature_a, r.feature_b}) for r in pairs.itertuples()}
    assert frozenset({"strong", "strong_noisy"}) in found


def test_correlated_pairs_is_empty_when_features_are_independent(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)[
        ["noise", "humped", "Class"]
    ]
    assert eda.correlated_pairs(frame, fast_eda_config).empty


def test_correlated_pairs_are_returned_strongest_first(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)
    pairs = eda.correlated_pairs(frame, fast_eda_config, threshold=0.0)

    assert list(pairs["abs_corr"]) == sorted(pairs["abs_corr"], reverse=True)


# --------------------------------------------------------------------------------------
# The statistics table
# --------------------------------------------------------------------------------------


def test_feature_stats_covers_every_feature_and_ranks_them(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=2000, fraud_rows=100)
    train, holdout = split_for_drift(frame)

    stats = eda.compute_feature_stats(train, holdout, fast_eda_config)

    assert len(stats) == len(frame.columns) - 1
    assert "Class" not in set(stats["feature"])
    assert list(stats["auc"]) == sorted(stats["auc"], reverse=True)
    for column in ("auc", "ks", "abs_corr", "mutual_info", "psi", "range_coverage"):
        assert column in stats.columns


def test_feature_stats_flags_the_planted_properties(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=3000, fraud_rows=150)
    train, holdout = split_for_drift(frame)

    stats = eda.compute_feature_stats(train, holdout, fast_eda_config).set_index("feature")

    assert stats.loc["leaky", "auc"] > 0.99
    assert stats.loc["counter", "range_coverage"] == 0.0
    assert stats.loc["constant", "n_unique"] == 1
    assert stats.loc["almost_constant", "dominant_share"] > 0.99


def test_the_humped_feature_hides_from_auc_but_not_from_ks(fast_eda_config) -> None:
    """The case that broke the first version of the low signal rule.

    Fraud clusters in the middle of the range, so neither class is consistently higher and
    AUC lands near chance. The distributions are obviously different, and KS says so. On the
    real data this is Amount.
    """
    frame = make_diagnostic_frame(fast_eda_config, rows=4000, fraud_rows=200)
    train, holdout = split_for_drift(frame)

    stats = eda.compute_feature_stats(train, holdout, fast_eda_config).set_index("feature")
    noise = eda.permutation_noise_ceiling(train, fast_eda_config)

    assert stats.loc["humped", "auc"] < noise.ceiling
    assert stats.loc["humped", "ks"] > noise.ks_ceiling


def test_feature_stats_rejects_a_split_with_no_fraud(fast_eda_config) -> None:
    frame = make_transactions(fast_eda_config, rows=500, fraud_rows=0)
    with pytest.raises(eda.EdaError, match="no fraud rows"):
        eda.compute_feature_stats(frame, frame, fast_eda_config)


def test_feature_stats_rejects_a_frame_with_only_the_target(fast_eda_config) -> None:
    frame = make_diagnostic_frame(fast_eda_config, rows=500, fraud_rows=50)[["Class"]]
    with pytest.raises(eda.EdaError, match="no feature columns"):
        eda.compute_feature_stats(frame, frame, fast_eda_config)


def test_mutual_information_keeps_every_fraud_row_when_subsampling(fast_eda_config) -> None:
    """Fraud rows are too scarce to lose to a random subsample."""
    frame = make_diagnostic_frame(fast_eda_config, rows=5000, fraud_rows=120)
    target = frame["Class"].to_numpy()
    features = [c for c in frame.columns if c != "Class"]

    scores = eda.compute_mutual_information(frame, target, features, fast_eda_config)

    assert set(scores) == set(features)
    assert scores["strong"] > scores["noise"]


# --------------------------------------------------------------------------------------
# The guard that matters most
# --------------------------------------------------------------------------------------


def test_the_config_refuses_to_point_exploration_at_the_test_split(config_tree) -> None:
    """Choosing features by looking at test data would invalidate every later metric."""
    config_tree["eda"]["analysis_split"] = "test"
    with pytest.raises(ValidationError, match="never read the test split"):
        Config.model_validate(config_tree)


def test_the_config_refuses_a_test_drift_split(config_tree) -> None:
    config_tree["eda"]["drift_split"] = "test"
    with pytest.raises(ValidationError, match="never read the test split"):
        Config.model_validate(config_tree)


def test_the_config_rejects_an_unknown_split(config_tree) -> None:
    config_tree["eda"]["analysis_split"] = "holdout"
    with pytest.raises(ValidationError):
        Config.model_validate(config_tree)


# --------------------------------------------------------------------------------------
# The stage, end to end
# --------------------------------------------------------------------------------------


@pytest.fixture
def prepared_pipeline(config_path, tmp_path) -> Config:
    """Run ingest and validate so the eda stage has splits to read."""
    from fraud_pipeline import ingest, validation
    from fraud_pipeline.config import load_config

    config = load_config(
        config_path,
        overrides=[
            f"paths.data_raw={(tmp_path / 'raw').as_posix()}",
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.figures={(tmp_path / 'reports' / 'figures').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            "validation.min_rows=100",
            "validation.min_fraud_rows_per_split=1",
            "eda.permutation_shuffles=5",
            "eda.mutual_info_sample=1000",
            "eda.figures=false",
        ],
    )
    from .synthetic import write_raw_csv

    write_raw_csv(config, make_transactions(config, rows=4000, fraud_rows=40))
    ingest.run(config)
    validation.run(config)
    return config


def test_the_stage_writes_everything_it_promises(prepared_pipeline) -> None:
    config = prepared_pipeline
    stats, selection = eda.run(config)

    tables = config.paths.tables_dir()
    assert (tables / config.eda.stats_file).is_file()
    assert (tables / config.eda.report_file).is_file()
    assert (tables / config.feature_selection.report_file).is_file()
    assert (config.paths.interim() / config.feature_selection.output_file).is_file()

    assert len(stats) == 30
    assert selection.features("selected")


def test_the_report_records_the_leakage_verdict(prepared_pipeline) -> None:
    config = prepared_pipeline
    eda.run(config)

    report = (config.paths.tables_dir() / config.eda.report_file).read_text(encoding="utf-8")
    assert "Target leakage" in report
    assert "Noise ceiling" in report
    assert "Every feature" in report


def test_the_stage_output_can_be_loaded_by_a_later_stage(prepared_pipeline) -> None:
    from fraud_pipeline import feature_selection

    config = prepared_pipeline
    _, selection = eda.run(config)

    assert feature_selection.load_selected_features(config) == selection.features("selected")


def test_the_stage_is_reproducible(prepared_pipeline) -> None:
    """Two runs on the same data must choose the same features."""
    config = prepared_pipeline
    _, first = eda.run(config)
    _, second = eda.run(config)

    assert first.features("selected") == second.features("selected")
    assert first.noise_ceiling == second.noise_ceiling


def test_the_stage_needs_the_splits_to_exist(config_path, tmp_path) -> None:
    from fraud_pipeline.config import load_config
    from fraud_pipeline.splits import SplitError

    config = load_config(
        config_path,
        overrides=[f"paths.data_interim={(tmp_path / 'empty').as_posix()}"],
    )
    with pytest.raises(SplitError, match="fraud validate"):
        eda.run(config)
