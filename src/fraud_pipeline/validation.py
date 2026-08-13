"""Stage 2. Data quality checks that run before a single model is trained.

The point of this stage is to fail early and say why. A model trained on bad data still
produces a number, and that number looks exactly like a real one. So every assumption the
later stages rely on is written down here as a named check.

Each check returns a :class:`CheckResult` rather than raising, so one run reports every
problem at once instead of stopping at the first. The stage as a whole fails if any check
with ``severity="error"`` fails.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)


class ValidationError(RuntimeError):
    """One or more error level checks failed. The pipeline must not continue."""


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a single named check."""

    name: str
    passed: bool
    severity: str  # "error" stops the pipeline, "warning" is reported and allowed
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return not self.passed and self.severity == "error"

    def status(self) -> str:
        if self.passed:
            return "pass"
        return "FAIL" if self.severity == "error" else "warn"


@dataclass
class ValidationReport:
    """Every check result from one run, plus the summary the pipeline acts on."""

    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def passed(self) -> bool:
        return not any(result.is_blocking for result in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if result.is_blocking]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        lines = [f"  {result.name}: {result.message}" for result in self.failures]
        raise ValidationError(
            f"{len(self.failures)} data quality check(s) failed:\n" + "\n".join(lines)
        )

    def to_markdown(self, title: str = "Data validation report") -> str:
        """Render the report so it can be committed and read in a pull request."""
        header = [
            f"# {title}",
            "",
            f"{len(self.results)} checks ran. "
            f"{sum(r.passed for r in self.results)} passed, "
            f"{len(self.failures)} failed, {len(self.warnings)} warned.",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
        ]
        rows = [f"| {r.name} | {r.status()} | {r.message} |" for r in self.results]
        return "\n".join([*header, *rows, ""])


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def check_row_count(frame: pd.DataFrame, config: Config) -> CheckResult:
    minimum = config.validation.min_rows
    rows = len(frame)
    return CheckResult(
        name="row_count",
        passed=rows >= minimum,
        severity="error",
        message=f"{rows:,} rows, minimum is {minimum:,}",
        details={"rows": rows, "minimum": minimum},
    )


def check_expected_row_count(frame: pd.DataFrame, config: Config) -> CheckResult:
    """The published row count for the dataset. A mismatch means a different file.

    This is a warning, not an error, because a truncated sample is a perfectly reasonable
    thing to develop against. It just should not pass unnoticed.
    """
    expected = config.dataset.spec().expected_rows
    rows = len(frame)
    return CheckResult(
        name="expected_row_count",
        passed=rows == expected,
        severity="warning",
        message=f"{rows:,} rows, the published dataset has {expected:,}",
        details={"rows": rows, "expected": expected},
    )


def check_no_missing_values(frame: pd.DataFrame, config: Config) -> CheckResult:
    missing = int(frame.isna().sum().sum())
    fraction = missing / frame.size if frame.size else 0.0
    limit = config.validation.max_missing_fraction
    worst = frame.isna().sum()
    worst_columns = worst[worst > 0].sort_values(ascending=False).head(5).to_dict()

    return CheckResult(
        name="no_missing_values",
        passed=fraction <= limit,
        severity="error",
        message=f"{missing:,} missing cells ({fraction:.6f} of the table), limit is {limit}",
        details={
            "missing_cells": missing,
            "worst_columns": {k: int(v) for k, v in worst_columns.items()},
        },
    )


def check_target_values(frame: pd.DataFrame, config: Config) -> CheckResult:
    target = config.dataset.spec().target_column
    allowed = set(config.validation.allowed_target_values)
    found = set(frame[target].unique().tolist())
    unexpected = sorted(found - allowed)

    return CheckResult(
        name="target_values",
        passed=not unexpected,
        severity="error",
        message=(
            f"{target} holds {sorted(found)}, allowed is {sorted(allowed)}"
            if unexpected
            else f"{target} holds only {sorted(found)}"
        ),
        details={"found": sorted(found), "unexpected": unexpected},
    )


def check_fraud_rate(frame: pd.DataFrame, config: Config) -> CheckResult:
    target = config.dataset.spec().target_column
    rate = float(frame[target].mean())
    low, high = config.validation.min_fraud_rate, config.validation.max_fraud_rate
    inside = low <= rate <= high

    return CheckResult(
        name="fraud_rate",
        passed=inside,
        severity="error",
        message=f"{rate * 100:.4f} percent fraud, expected between {low * 100:.4f} and {high * 100:.4f} percent",
        details={"rate": rate, "min": low, "max": high, "fraud_rows": int(frame[target].sum())},
    )


def check_amount_not_negative(frame: pd.DataFrame, config: Config) -> CheckResult:
    column = config.dataset.spec().amount_column
    floor = config.validation.amount_min
    below = int((frame[column] < floor).sum())

    return CheckResult(
        name="amount_not_negative",
        passed=below == 0,
        severity="error",
        message=f"{below:,} rows have {column} below {floor}",
        details={"rows_below_floor": below, "min_amount": float(frame[column].min())},
    )


def check_time_is_ordered(frame: pd.DataFrame, config: Config) -> CheckResult:
    """A time based split is only honest if the file is actually in time order.

    If this fails, the split has to sort first, otherwise "train on the past, test on the
    future" is a claim the code does not deliver.
    """
    column = config.dataset.spec().time_column
    ordered = bool(frame[column].is_monotonic_increasing)

    return CheckResult(
        name="time_is_ordered",
        passed=ordered,
        severity="warning",
        message=(
            f"{column} increases through the file, so a time split is safe"
            if ordered
            else f"{column} is not in order, the split stage has to sort first"
        ),
        details={"monotonic": ordered},
    )


def check_dtypes_are_numeric(frame: pd.DataFrame, config: Config) -> CheckResult:
    """Every modelling column has to be a number.

    A column that arrives as text usually means the csv had a stray value in it, and the
    error is far easier to understand here than inside an estimator later.
    """
    expected = [name for name in config.dataset.spec().expected_columns() if name in frame.columns]
    non_numeric = [name for name in expected if not pd.api.types.is_numeric_dtype(frame[name])]

    return CheckResult(
        name="dtypes_are_numeric",
        passed=not non_numeric,
        severity="error",
        message=(
            f"all {len(expected)} columns are numeric"
            if not non_numeric
            else f"non numeric columns found: {non_numeric}"
        ),
        details={"non_numeric": non_numeric, "checked": len(expected)},
    )


def check_duplicate_rows(frame: pd.DataFrame, config: Config) -> CheckResult:
    """Report exact duplicate rows.

    On the ULB file there are 1081 of them. With 28 continuous components matching to full
    precision, these are double entries rather than coincidence. They are dropped by
    default, which is why the severity here depends on the config rather than being fixed.
    """
    target = config.dataset.spec().target_column
    duplicated = frame.duplicated()
    count = int(duplicated.sum())
    fraud_among = int(frame.loc[duplicated, target].sum()) if count else 0

    return CheckResult(
        name="duplicate_rows",
        passed=count == 0,
        severity="error" if config.validation.fail_on_duplicate_rows else "warning",
        message=f"{count:,} exact duplicate rows, {fraud_among} of them fraud",
        details={"duplicates": count, "fraud_among_duplicates": fraud_among},
    )


def check_constant_columns(frame: pd.DataFrame, config: Config) -> CheckResult:
    """A column with one value everywhere carries no signal and usually means a bad read."""
    target = config.dataset.spec().target_column
    constant = [
        name for name in frame.columns if name != target and frame[name].nunique(dropna=False) <= 1
    ]
    return CheckResult(
        name="constant_columns",
        passed=not constant,
        severity="warning",
        message=(
            "no constant columns" if not constant else f"columns with a single value: {constant}"
        ),
        details={"constant_columns": constant},
    )


#: The checks that run, in order. Adding a check means adding it here and nowhere else.
CHECKS: list[Callable[[pd.DataFrame, Config], CheckResult]] = [
    check_row_count,
    check_expected_row_count,
    check_dtypes_are_numeric,
    check_no_missing_values,
    check_target_values,
    check_fraud_rate,
    check_amount_not_negative,
    check_time_is_ordered,
    check_duplicate_rows,
    check_constant_columns,
]


def run_checks(frame: pd.DataFrame, config: Config) -> ValidationReport:
    """Run every check and return the collected report."""
    report = ValidationReport()

    for check in CHECKS:
        result = report.add(check(frame, config))
        log = (
            logger.info
            if result.passed
            else (logger.error if result.severity == "error" else logger.warning)
        )
        log("%-22s %-4s %s", result.name, result.status(), result.message)

    return report


def drop_duplicate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows, keeping the first copy. Returns the frame and the count."""
    before = len(frame)
    deduplicated = frame.drop_duplicates(keep="first").reset_index(drop=True)
    return deduplicated, before - len(deduplicated)


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


def run(config: Config) -> ValidationReport:
    """Run stage 2: check the data, clean what the config says to clean, then split it.

    Splitting belongs here rather than in the feature stage. Once the split exists on disk
    before any feature code runs, it is structurally impossible for a scaler or a rolling
    aggregate to be fitted on rows it should never have seen.
    """
    from fraud_pipeline.ingest import load_interim, load_manifest
    from fraud_pipeline.paths import ensure_dir
    from fraud_pipeline.splits import check_splits, make_splits, write_splits

    frame = load_interim(config)
    manifest = load_manifest(config)
    spec = config.dataset.spec()
    logger.info("validating %s rows from %s", f"{len(frame):,}", manifest.source_file)

    report = run_checks(frame, config)

    if config.validation.drop_duplicates:
        frame, dropped = drop_duplicate_rows(frame)
        if dropped:
            logger.info("dropped %s duplicate rows, %s remain", f"{dropped:,}", f"{len(frame):,}")
        report.add(
            CheckResult(
                name="duplicates_dropped",
                passed=True,
                severity="warning",
                message=f"{dropped:,} duplicate rows removed, {len(frame):,} rows remain",
                details={"dropped": dropped, "remaining": len(frame)},
            )
        )

    # Stop before splitting if the data itself is unusable. There is no point splitting
    # a table that failed its schema checks.
    report.raise_if_failed()

    splits = make_splits(frame, config)
    problems = check_splits(splits, config)
    for problem in problems:
        logger.warning(problem)

    report.add(
        CheckResult(
            name="split_fraud_coverage",
            passed=not problems,
            severity="warning",
            message=(
                "; ".join(problems)
                if problems
                else f"every split has at least {config.validation.min_fraud_rows_per_split} fraud rows"
            ),
            details={"problems": problems},
        )
    )

    for summary in splits.summarise(spec.target_column):
        report.add(
            CheckResult(
                name=f"split_{summary.name}",
                passed=True,
                severity="warning",
                message=(
                    f"{summary.rows:,} rows, {summary.fraud_rows} fraud "
                    f"({summary.fraud_rate * 100:.4f} percent)"
                ),
                details={"rows": summary.rows, "fraud_rows": summary.fraud_rows},
            )
        )

    write_splits(splits, config)

    report_path = ensure_dir(config.paths.tables_dir()) / config.validation.report_file
    report_path.write_text(
        report.to_markdown(f"Data validation report: {spec.name}"), encoding="utf-8"
    )
    logger.info("wrote %s", report_path)

    return report
