"""Tests for train, validation and test splitting.

The test that matters most here is the leakage one. A time split that does not actually
put every training row before every test row is worse than useless, because it reports a
score the deployed system would never reach.
"""

from __future__ import annotations

import pytest

from fraud_pipeline import splits
from fraud_pipeline.splits import SplitError

from .synthetic import make_transactions


def test_time_split_fractions_are_respected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=30)
    result = splits.make_splits(frame, sandbox_config, strategy="time")

    assert len(result.train) == 700
    assert len(result.validation) == 150
    assert len(result.test) == 150
    assert len(result.train) + len(result.validation) + len(result.test) == 1000


def test_time_split_puts_the_whole_training_set_in_the_past(sandbox_config) -> None:
    """No training transaction may happen after a test transaction."""
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=30)
    time_column = sandbox_config.dataset.spec().time_column

    result = splits.make_splits(frame, sandbox_config, strategy="time")

    assert result.train[time_column].max() <= result.validation[time_column].min()
    assert result.validation[time_column].max() <= result.test[time_column].min()


def test_time_split_sorts_an_unordered_file(sandbox_config) -> None:
    """The ordering promise has to hold even when the raw file arrives shuffled."""
    frame = make_transactions(sandbox_config, rows=600, fraud_rows=20, ordered_time=False)
    time_column = sandbox_config.dataset.spec().time_column

    result = splits.make_splits(frame, sandbox_config, strategy="time")

    assert result.train[time_column].max() <= result.test[time_column].min()
    assert result.train[time_column].is_monotonic_increasing


def test_no_row_appears_in_two_splits(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=800, fraud_rows=24)
    time_column = sandbox_config.dataset.spec().time_column

    result = splits.make_splits(frame, sandbox_config, strategy="time")
    seen = [set(part[time_column]) for _, part in result.items()]

    assert not seen[0] & seen[1]
    assert not seen[1] & seen[2]
    assert not seen[0] & seen[2]


def test_stratified_split_keeps_the_fraud_rate_even(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=4000, fraud_rows=200)
    target = sandbox_config.dataset.spec().target_column

    result = splits.make_splits(frame, sandbox_config, strategy="stratified")
    rates = [part[target].mean() for _, part in result.items()]

    assert max(rates) - min(rates) < 0.005


def test_stratified_split_is_reproducible(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=2000, fraud_rows=60)
    time_column = sandbox_config.dataset.spec().time_column

    first = splits.make_splits(frame, sandbox_config, strategy="stratified")
    second = splits.make_splits(frame, sandbox_config, strategy="stratified")

    assert list(first.train[time_column]) == list(second.train[time_column])
    assert list(first.test[time_column]) == list(second.test[time_column])


def test_stratified_split_uses_every_row_once(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=2000, fraud_rows=60)
    time_column = sandbox_config.dataset.spec().time_column

    result = splits.make_splits(frame, sandbox_config, strategy="stratified")
    total = sum(len(part) for _, part in result.items())
    seen = set().union(*(set(part[time_column]) for _, part in result.items()))

    assert total == 2000
    assert len(seen) == 2000


def test_the_configured_strategy_is_used_by_default(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=500, fraud_rows=15)
    assert splits.make_splits(frame, sandbox_config).strategy == sandbox_config.split.strategy


def test_too_few_rows_is_rejected(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=2, fraud_rows=1)
    with pytest.raises(SplitError, match="cannot split"):
        splits.make_splits(frame, sandbox_config)


def test_summary_counts_fraud_per_split(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=40)
    target = sandbox_config.dataset.spec().target_column

    result = splits.make_splits(frame, sandbox_config)
    summaries = result.summarise(target)

    assert [s.name for s in summaries] == ["train", "validation", "test"]
    assert sum(s.rows for s in summaries) == 1000
    assert sum(s.fraud_rows for s in summaries) == 40
    assert summaries[0].fraud_rate == summaries[0].fraud_rows / summaries[0].rows


def test_check_splits_flags_a_thin_split(config_path, tmp_path) -> None:
    from fraud_pipeline.config import load_config

    cfg = load_config(config_path, overrides=["validation.min_fraud_rows_per_split=1000"])
    frame = make_transactions(cfg, rows=1000, fraud_rows=30)

    problems = splits.check_splits(splits.make_splits(frame, cfg), cfg)

    assert len(problems) == 3
    assert "fraud rows" in problems[0]


def test_check_splits_is_quiet_when_coverage_is_fine(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=60)
    assert splits.check_splits(splits.make_splits(frame, sandbox_config), sandbox_config) == []


def test_write_and_load_round_trip(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=900, fraud_rows=27)
    result = splits.make_splits(frame, sandbox_config)

    written = splits.write_splits(result, sandbox_config)

    assert set(written) == {"train", "validation", "test"}
    for name, part in result.items():
        loaded = splits.load_split(sandbox_config, name)
        assert len(loaded) == len(part)
        assert list(loaded.columns) == list(part.columns)


def test_loading_an_unknown_split_is_rejected(sandbox_config) -> None:
    with pytest.raises(SplitError, match="unknown split"):
        splits.load_split(sandbox_config, "holdout")


def test_loading_before_the_split_exists_explains_itself(sandbox_config) -> None:
    with pytest.raises(SplitError, match="fraud validate"):
        splits.load_split(sandbox_config, "train")
