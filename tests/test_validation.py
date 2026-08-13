"""Tests for stage 2, validation.

Every check gets tested twice: once on data that should pass it, and once on data broken
in exactly the way the check exists to catch. A check that has never been seen to fail is
not a check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fraud_pipeline import ingest, validation
from fraud_pipeline.validation import ValidationError

from .synthetic import make_transactions, write_raw_csv


def _good_frame(config, **kwargs):
    return make_transactions(config, rows=1000, fraud_rows=5, **kwargs)


def _with_duplicates(frame: pd.DataFrame, copies: int) -> pd.DataFrame:
    """Append an exact copy of the first ``copies`` rows."""
    return pd.concat([frame, frame.iloc[:copies]], ignore_index=True)


def _result(report, name):
    matches = [r for r in report.results if r.name == name]
    assert matches, f"no check named {name} ran"
    return matches[0]


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def test_row_count_passes_on_enough_rows(sandbox_config) -> None:
    assert validation.check_row_count(_good_frame(sandbox_config), sandbox_config).passed


def test_row_count_fails_on_too_few(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=50, fraud_rows=1)
    result = validation.check_row_count(frame, sandbox_config)

    assert not result.passed
    assert result.is_blocking


def test_expected_row_count_warns_on_a_sample(sandbox_config) -> None:
    result = validation.check_expected_row_count(_good_frame(sandbox_config), sandbox_config)

    assert not result.passed
    assert result.severity == "warning"
    assert not result.is_blocking


def test_missing_values_pass_on_a_complete_frame(sandbox_config) -> None:
    assert validation.check_no_missing_values(_good_frame(sandbox_config), sandbox_config).passed


def test_missing_values_fail_and_name_the_worst_column(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame.loc[0:9, "V7"] = None

    result = validation.check_no_missing_values(frame, sandbox_config)

    assert result.is_blocking
    assert result.details["missing_cells"] == 10
    assert "V7" in result.details["worst_columns"]


def test_target_values_pass_on_zero_and_one(sandbox_config) -> None:
    assert validation.check_target_values(_good_frame(sandbox_config), sandbox_config).passed


def test_target_values_fail_on_a_stray_label(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame.loc[3, "Class"] = 7

    result = validation.check_target_values(frame, sandbox_config)

    assert result.is_blocking
    assert 7 in result.details["unexpected"]


def test_fraud_rate_passes_inside_the_band(sandbox_config) -> None:
    assert validation.check_fraud_rate(_good_frame(sandbox_config), sandbox_config).passed


def test_fraud_rate_fails_when_there_is_no_fraud(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=0)
    result = validation.check_fraud_rate(frame, sandbox_config)

    assert result.is_blocking
    assert result.details["fraud_rows"] == 0


def test_fraud_rate_fails_when_it_is_implausibly_high(sandbox_config) -> None:
    """A 40 percent fraud rate means the file is not what we think it is."""
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=400)
    assert validation.check_fraud_rate(frame, sandbox_config).is_blocking


def test_negative_amount_is_caught(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame.loc[2, "Amount"] = -1.0

    result = validation.check_amount_not_negative(frame, sandbox_config)

    assert result.is_blocking
    assert result.details["rows_below_floor"] == 1


def test_time_order_passes_on_an_ordered_file(sandbox_config) -> None:
    assert validation.check_time_is_ordered(_good_frame(sandbox_config), sandbox_config).passed


def test_time_order_warns_on_a_shuffled_file(sandbox_config) -> None:
    frame = _good_frame(sandbox_config, ordered_time=False)
    result = validation.check_time_is_ordered(frame, sandbox_config)

    assert not result.passed
    assert not result.is_blocking


def test_non_numeric_column_is_caught(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame["V4"] = "text"

    result = validation.check_dtypes_are_numeric(frame, sandbox_config)

    assert result.is_blocking
    assert "V4" in result.details["non_numeric"]


def test_duplicates_are_counted_and_warn_by_default(sandbox_config) -> None:
    frame = _with_duplicates(_good_frame(sandbox_config), 5)

    result = validation.check_duplicate_rows(frame, sandbox_config)

    assert result.details["duplicates"] == 5
    assert result.severity == "warning"
    assert not result.is_blocking


def test_duplicates_can_be_made_fatal(config_path) -> None:
    from fraud_pipeline.config import load_config

    cfg = load_config(
        config_path,
        overrides=["validation.fail_on_duplicate_rows=true", "validation.min_rows=100"],
    )
    frame = _with_duplicates(make_transactions(cfg, rows=200, fraud_rows=4), 3)

    assert validation.check_duplicate_rows(frame, cfg).is_blocking


def test_constant_column_is_flagged(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame["V9"] = 1.0

    result = validation.check_constant_columns(frame, sandbox_config)

    assert not result.passed
    assert result.details["constant_columns"] == ["V9"]


def test_a_clean_frame_has_no_constant_columns(sandbox_config) -> None:
    assert validation.check_constant_columns(_good_frame(sandbox_config), sandbox_config).passed


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def test_every_check_runs_even_when_one_fails(sandbox_config) -> None:
    """One broken column must not hide the other problems in the same file."""
    frame = _good_frame(sandbox_config)
    frame.loc[0, "Amount"] = -5.0
    frame.loc[1, "Class"] = 9

    report = validation.run_checks(frame, sandbox_config)

    assert len(report.results) == len(validation.CHECKS)
    assert len(report.failures) >= 2


def test_report_passes_when_only_warnings_fire(sandbox_config) -> None:
    report = validation.run_checks(_good_frame(sandbox_config), sandbox_config)

    assert report.passed
    assert report.warnings  # the row count warning always fires on a synthetic frame


def test_raise_if_failed_names_every_failure(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)
    frame.loc[0, "Amount"] = -5.0

    report = validation.run_checks(frame, sandbox_config)

    with pytest.raises(ValidationError, match="amount_not_negative"):
        report.raise_if_failed()


def test_markdown_report_lists_every_check(sandbox_config) -> None:
    report = validation.run_checks(_good_frame(sandbox_config), sandbox_config)
    markdown = report.to_markdown()

    assert markdown.startswith("# ")
    for check in validation.CHECKS:
        assert check.__name__.replace("check_", "") in markdown


def test_drop_duplicate_rows_keeps_the_first_copy(sandbox_config) -> None:
    frame = _good_frame(sandbox_config)

    cleaned, dropped = validation.drop_duplicate_rows(_with_duplicates(frame, 4))

    assert dropped == 4
    assert len(cleaned) == len(frame)
    pd.testing.assert_frame_equal(cleaned, frame)


# --------------------------------------------------------------------------------------
# The stage end to end
# --------------------------------------------------------------------------------------


def test_stage_writes_splits_and_a_report(sandbox_config) -> None:
    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=2000, fraud_rows=20))
    ingest.run(sandbox_config)

    report = validation.run(sandbox_config)

    split_dir = sandbox_config.paths.interim() / sandbox_config.split.output_dir
    assert (split_dir / "train.parquet").is_file()
    assert (split_dir / "validation.parquet").is_file()
    assert (split_dir / "test.parquet").is_file()

    report_file = sandbox_config.paths.tables_dir() / sandbox_config.validation.report_file
    assert report_file.is_file()
    assert "Data validation report" in report_file.read_text(encoding="utf-8")
    assert report.passed


def test_stage_drops_duplicates_before_splitting(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=10)
    write_raw_csv(sandbox_config, _with_duplicates(frame, 50))
    ingest.run(sandbox_config)

    report = validation.run(sandbox_config)

    assert _result(report, "duplicates_dropped").details["dropped"] == 50

    from fraud_pipeline.splits import load_split

    total = sum(len(load_split(sandbox_config, name)) for name in ("train", "validation", "test"))
    assert total == 1000


def test_stage_stops_before_splitting_when_the_data_is_broken(sandbox_config) -> None:
    frame = make_transactions(sandbox_config, rows=1000, fraud_rows=10)
    frame.loc[0, "Amount"] = -99.0
    write_raw_csv(sandbox_config, frame)
    ingest.run(sandbox_config)

    with pytest.raises(ValidationError):
        validation.run(sandbox_config)

    split_dir = sandbox_config.paths.interim() / sandbox_config.split.output_dir
    assert not (split_dir / "train.parquet").exists()


def test_stage_reports_fraud_counts_per_split(sandbox_config) -> None:
    write_raw_csv(sandbox_config, make_transactions(sandbox_config, rows=2000, fraud_rows=20))
    ingest.run(sandbox_config)

    report = validation.run(sandbox_config)

    counted = sum(
        _result(report, f"split_{name}").details["fraud_rows"]
        for name in ("train", "validation", "test")
    )
    assert counted == 20


@pytest.mark.needs_data
@pytest.mark.slow
def test_the_real_dataset_passes_validation(real_data_config) -> None:
    frame = ingest.read_raw(ingest.resolve_raw_file(real_data_config), real_data_config)
    report = validation.run_checks(frame, real_data_config)

    assert report.passed, [r.message for r in report.failures]
    assert _result(report, "expected_row_count").passed
    assert _result(report, "time_is_ordered").passed
    # The known 1081 duplicates should show up as a warning, not a failure.
    assert _result(report, "duplicate_rows").details["duplicates"] == 1081
