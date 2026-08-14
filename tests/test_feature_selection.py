"""Tests for the automatic feature selection.

Feature selection is a set of detectors. A detector that has never been shown the thing it
detects is not tested, it is decoration. Every rule here gets a column built specifically to
trip it, and a column built specifically not to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_pipeline import eda, feature_selection
from fraud_pipeline.config import Config, load_config
from fraud_pipeline.feature_selection import SelectionError

from .synthetic import make_diagnostic_frame, split_for_drift


@pytest.fixture
def selection_config(config_path, tmp_path) -> Config:
    from fraud_pipeline.config import load_config as _load

    return _load(
        config_path,
        overrides=[
            f"paths.data_interim={(tmp_path / 'interim').as_posix()}",
            f"paths.reports={(tmp_path / 'reports').as_posix()}",
            f"paths.tables={(tmp_path / 'reports' / 'tables').as_posix()}",
            "eda.permutation_shuffles=8",
            "eda.mutual_info_sample=1500",
            "eda.figures=false",
            "feature_selection.keep_as_input=[counter]",
        ],
    )


def _run_selection(config: Config, rows: int = 4000, fraud_rows: int = 200):
    """Build the diagnostic frame, measure it, and select from it."""
    frame = make_diagnostic_frame(config, rows=rows, fraud_rows=fraud_rows)
    train, holdout = split_for_drift(frame)

    stats = eda.compute_feature_stats(train, holdout, config)
    noise = eda.permutation_noise_ceiling(train, config)
    pairs = eda.correlated_pairs(train, config)
    duplicates = eda.exact_duplicate_columns(train, config)

    result = feature_selection.select_features(stats, pairs, noise, config, duplicates)
    return result, stats, noise


# --------------------------------------------------------------------------------------
# Tier 1, the correctness rules
# --------------------------------------------------------------------------------------


def test_target_leakage_is_caught(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    decision = result.by_name()["leaky"]

    assert decision.dropped_by_tier == 1
    assert any("leakage" in reason for reason in decision.reasons)


def test_target_leakage_produces_a_loud_warning(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    assert any("leaky" in warning for warning in result.warnings)


def test_an_exact_duplicate_column_is_dropped(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    decisions = result.by_name()

    # One of the identical pair goes, the other stays.
    dropped = [n for n in ("strong", "strong_copy") if decisions[n].dropped_by_tier == 1]
    assert len(dropped) == 1


def test_a_near_duplicate_is_dropped_and_the_stronger_one_survives(selection_config) -> None:
    result, stats, _ = _run_selection(selection_config)
    decisions = result.by_name()
    auc = stats.set_index("feature")["auc"]

    survivors = [n for n in ("strong", "strong_noisy") if decisions[n].dropped_by_tier is None]
    assert len(survivors) == 1
    # The survivor should be the one carrying more signal, not whichever came first.
    loser = "strong_noisy" if survivors[0] == "strong" else "strong"
    assert auc[survivors[0]] >= auc[loser]


def test_a_constant_column_is_dropped(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    decision = result.by_name()["constant"]

    assert decision.dropped_by_tier == 1
    assert any("constant" in reason for reason in decision.reasons)


def test_a_near_constant_column_is_dropped(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    assert result.by_name()["almost_constant"].dropped_by_tier == 1


def test_a_counter_fails_the_range_rule(selection_config) -> None:
    """The raw timestamp case, which is what this rule was written for."""
    result, _, _ = _run_selection(selection_config)
    decision = result.by_name()["counter"]

    assert decision.dropped_by_tier == 1
    assert any("cannot generalise" in reason for reason in decision.reasons)


def test_a_good_feature_survives_every_rule(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    assert "strong" in result.features("all")
    assert result.by_name()["humped"].dropped_by_tier is None


# --------------------------------------------------------------------------------------
# Tier 2, the judgement rule
# --------------------------------------------------------------------------------------


def test_pure_noise_is_dropped_at_tier_two(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    decision = result.by_name()["noise"]

    assert decision.dropped_by_tier == 2
    assert any("no measurable signal" in reason for reason in decision.reasons)


def test_a_non_monotonic_feature_is_not_mistaken_for_noise(selection_config) -> None:
    """The bug this rule had in its first version.

    `humped` has fraud clustered in the middle of its range, so AUC lands near chance while
    the distributions are plainly different. Dropping it on AUC alone was wrong, and on the
    real data the same mistake dropped Amount.
    """
    result, stats, noise = _run_selection(selection_config)
    lookup = stats.set_index("feature")

    assert lookup.loc["humped", "auc"] < noise.ceiling, "the setup should look weak on AUC"
    assert lookup.loc["humped", "ks"] > noise.ks_ceiling, "but obvious on KS"
    assert result.by_name()["humped"].dropped_by_tier is None


def test_low_signal_dropping_can_be_switched_off(config_path, tmp_path) -> None:
    config = load_config(
        config_path,
        overrides=[
            "eda.permutation_shuffles=8",
            "eda.mutual_info_sample=1500",
            "feature_selection.drop_low_signal=false",
        ],
    )
    result, _, _ = _run_selection(config)

    assert result.by_name()["noise"].dropped_by_tier is None
    assert result.counts()["safe"] == result.counts()["selected"]


# --------------------------------------------------------------------------------------
# The sets
# --------------------------------------------------------------------------------------


def test_the_sets_are_nested(selection_config) -> None:
    """selected is inside safe, and safe is inside all. Anything else is a bug."""
    result, _, _ = _run_selection(selection_config)

    everything = set(result.features("all"))
    safe = set(result.features("safe"))
    selected = set(result.features("selected"))

    assert selected <= safe <= everything


def test_tier_one_drops_are_absent_from_every_set_except_all(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)

    assert "leaky" in result.features("all")
    assert "leaky" not in result.features("safe")
    assert "leaky" not in result.features("selected")


def test_tier_two_drops_survive_in_safe(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)

    assert "noise" in result.features("safe")
    assert "noise" not in result.features("selected")


def test_counts_match_the_sets(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    counts = result.counts()

    for name in ("all", "safe", "selected"):
        assert counts[name] == len(result.features(name))
    assert counts["all"] >= counts["safe"] >= counts["selected"]


def test_an_unknown_set_is_rejected(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)
    with pytest.raises(SelectionError, match="unknown feature set"):
        result.features("everything")


def test_a_leak_outranks_a_weak_signal_in_the_recorded_reason(selection_config) -> None:
    """A feature that leaks should not be filed under 'weak'."""
    result, _, _ = _run_selection(selection_config)
    assert result.by_name()["leaky"].dropped_by_tier == 1


def test_selection_fails_loudly_if_it_would_drop_everything(config_path) -> None:
    config = load_config(
        config_path,
        overrides=[
            "eda.permutation_shuffles=8",
            "eda.mutual_info_sample=1500",
            "feature_selection.leakage_auc=0.0",  # calls every feature a leak
        ],
    )
    with pytest.raises(SelectionError, match="every feature was dropped"):
        _run_selection(config)


# --------------------------------------------------------------------------------------
# Pipeline inputs, evidence and round tripping
# --------------------------------------------------------------------------------------


def test_a_dropped_feature_can_still_be_a_pipeline_input(selection_config) -> None:
    """Time is dropped as a feature and kept as an input, so hour of day can be built."""
    result, _, _ = _run_selection(selection_config)

    assert "counter" in result.pipeline_inputs()
    assert "counter" not in result.features("selected")


def test_every_decision_carries_its_evidence(selection_config) -> None:
    result, _, _ = _run_selection(selection_config)

    for decision in result.decisions:
        assert set(decision.evidence) >= {"auc", "ks", "psi", "range_coverage"}
        if decision.dropped_by_tier is not None:
            assert decision.reasons, f"{decision.feature} was dropped with no reason given"


def test_the_json_round_trips(selection_config) -> None:
    result, stats, _ = _run_selection(selection_config)
    json_path, report_path = feature_selection.write_selection(result, stats, selection_config)

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["sets"]["selected"] == result.features("selected")
    assert payload["counts"] == result.counts()
    assert "ks_noise_ceiling" in payload

    report = Path(report_path).read_text(encoding="utf-8")
    assert "# Feature selection" in report
    assert "noise" in report


def test_loading_the_selection_back_gives_the_same_features(selection_config) -> None:
    result, stats, _ = _run_selection(selection_config)
    feature_selection.write_selection(result, stats, selection_config)

    assert feature_selection.load_selected_features(selection_config) == result.features("selected")
    assert feature_selection.load_selected_features(selection_config, "all") == result.features(
        "all"
    )


def test_loading_before_the_stage_has_run_explains_itself(selection_config) -> None:
    with pytest.raises(SelectionError, match="fraud eda"):
        feature_selection.load_selection(selection_config)


def test_loading_an_unknown_set_from_disk_is_rejected(selection_config) -> None:
    result, stats, _ = _run_selection(selection_config)
    feature_selection.write_selection(result, stats, selection_config)

    with pytest.raises(SelectionError, match="unknown feature set"):
        feature_selection.load_selected_features(selection_config, "everything")


def test_the_report_lists_every_drop_with_a_reason(selection_config) -> None:
    result, stats, _ = _run_selection(selection_config)
    report = result.to_markdown(stats)

    for decision in result.decisions:
        if decision.dropped_by_tier is not None:
            assert decision.feature in report
            assert decision.reasons[0][:30] in report


def test_the_active_set_comes_from_the_config(config_path) -> None:
    config = load_config(
        config_path,
        overrides=[
            "eda.permutation_shuffles=8",
            "eda.mutual_info_sample=1500",
            "feature_selection.active_set=safe",
        ],
    )
    result, _, _ = _run_selection(config)

    assert result.active_set == "safe"
    assert result.features() == result.features("safe")
