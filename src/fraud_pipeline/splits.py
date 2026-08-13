"""Train, validation and test splitting.

This runs at the end of stage 2, before any feature is built. That order is deliberate.
Scalers, encoders and rolling aggregates all have to be fitted on the training rows only,
and the cleanest way to guarantee that is to have the split already exist before the
feature code is allowed to look at anything.

Two strategies:

``time``
    Cut the file in time order: the earliest 70 percent trains, the next 15 percent
    validates, the last 15 percent tests. This is the honest setup for fraud. A real
    system is always predicting transactions it has never seen, made after the ones it
    learned from. Fraud patterns drift, so a random split quietly lets the model learn
    from the future and reports a score the deployed system would never reach.

``stratified``
    A random split that keeps the fraud rate equal across the three parts. Kept as a
    comparison, because the gap between the two is itself a result worth reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_pipeline.config import Config

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "test")


class SplitError(RuntimeError):
    """The data cannot be split into usable parts."""


@dataclass(frozen=True)
class SplitSummary:
    """Row and fraud counts for one part of the split."""

    name: str
    rows: int
    fraud_rows: int

    @property
    def fraud_rate(self) -> float:
        return self.fraud_rows / self.rows if self.rows else 0.0


@dataclass(frozen=True)
class DataSplits:
    """The three parts, plus the strategy that produced them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("validation", self.validation), ("test", self.test)]

    def summarise(self, target_column: str) -> list[SplitSummary]:
        return [
            SplitSummary(name=name, rows=len(part), fraud_rows=int(part[target_column].sum()))
            for name, part in self.items()
        ]


def split_by_time(frame: pd.DataFrame, config: Config) -> DataSplits:
    """Cut the frame in time order, earliest rows first."""
    spec = config.dataset.spec()
    time_column = spec.time_column

    # Sorting is a no op on a file that is already ordered, and it is what makes the
    # promise true on one that is not.
    ordered = frame.sort_values(time_column, kind="stable").reset_index(drop=True)

    rows = len(ordered)
    train_end = int(rows * config.split.train_fraction)
    validation_end = train_end + int(rows * config.split.validation_fraction)

    return DataSplits(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
        test=ordered.iloc[validation_end:].reset_index(drop=True),
        strategy="time",
    )


def split_stratified(frame: pd.DataFrame, config: Config) -> DataSplits:
    """Random split that keeps the fraud rate the same in all three parts."""
    spec = config.dataset.spec()
    target = frame[spec.target_column]
    seed = config.project.seed

    holdout_fraction = config.split.validation_fraction + config.split.test_fraction
    train, holdout = train_test_split(
        frame,
        test_size=holdout_fraction,
        stratify=target,
        random_state=seed,
        shuffle=True,
    )

    # Divide the holdout between validation and test in the ratio the config asks for.
    test_share = config.split.test_fraction / holdout_fraction
    validation, test = train_test_split(
        holdout,
        test_size=test_share,
        stratify=holdout[spec.target_column],
        random_state=seed,
        shuffle=True,
    )

    return DataSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        strategy="stratified",
    )


def make_splits(frame: pd.DataFrame, config: Config, strategy: str | None = None) -> DataSplits:
    """Split the frame using the configured strategy."""
    chosen = strategy or config.split.strategy

    if len(frame) < 3:
        raise SplitError(f"cannot split {len(frame)} rows into three parts")

    splitter = {"time": split_by_time, "stratified": split_stratified}[chosen]
    splits = splitter(frame, config)

    empty = [name for name, part in splits.items() if part.empty]
    if empty:
        raise SplitError(f"the {chosen} split produced empty parts: {empty}")

    return splits


def check_splits(splits: DataSplits, config: Config) -> list[str]:
    """Return a list of problems with the split, empty when it is usable.

    The one that matters is fraud rows per part. With a 0.17 percent positive rate, a test
    set can easily end up with too few fraud cases for precision and recall to mean
    anything, and that is worth knowing before training rather than after.
    """
    spec = config.dataset.spec()
    minimum = config.validation.min_fraud_rows_per_split
    problems = []

    for summary in splits.summarise(spec.target_column):
        if summary.fraud_rows < minimum:
            problems.append(
                f"the {summary.name} split has only {summary.fraud_rows} fraud rows, "
                f"below the minimum of {minimum}"
            )

    return problems


def write_splits(splits: DataSplits, config: Config) -> dict[str, str]:
    """Write each part to parquet under data/interim and return the paths."""
    from fraud_pipeline.paths import ensure_dir

    target_dir = ensure_dir(config.paths.interim() / config.split.output_dir)
    written = {}

    for name, part in splits.items():
        path = target_dir / f"{name}.parquet"
        part.to_parquet(path, index=False)
        written[name] = str(path)
        logger.info("wrote %-11s %8s rows  %s", name, f"{len(part):,}", path.name)

    return written


def load_split(config: Config, name: str) -> pd.DataFrame:
    """Read one part back. Used by the feature, train and evaluate stages."""
    if name not in SPLIT_NAMES:
        raise SplitError(f"unknown split {name!r}, expected one of {SPLIT_NAMES}")

    path = config.paths.interim() / config.split.output_dir / f"{name}.parquet"
    if not path.is_file():
        raise SplitError(
            f"split file not found: {path}\nRun the validate stage first: fraud validate"
        )
    return pd.read_parquet(path)
