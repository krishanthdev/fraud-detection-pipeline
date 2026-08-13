"""A synthetic transaction generator for the tests.

The real dataset is 144 MB and is not in the repository, so CI has to be able to exercise
every code path without it. This builds a small frame with the same shape, the same column
names and the same kind of imbalance, so the tests are testing the real logic rather than a
mock of it.

Tests that need the genuine file are marked ``needs_data`` and are skipped when it is
absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_pipeline.config import Config


def make_transactions(
    config: Config,
    rows: int = 2000,
    fraud_rows: int = 20,
    seed: int = 0,
    ordered_time: bool = True,
) -> pd.DataFrame:
    """Build a frame with the same schema as the active dataset.

    Args:
        rows: total number of transactions.
        fraud_rows: how many of them are fraud. Kept small so the frame is imbalanced
            the way the real one is.
        ordered_time: when False the time column is shuffled, which is what the time
            ordering check is supposed to catch.
    """
    spec = config.dataset.spec()
    rng = np.random.default_rng(seed)

    time = np.arange(rows, dtype="float64") * 3.0
    if not ordered_time:
        time = rng.permutation(time)

    data: dict[str, np.ndarray] = {spec.time_column: time}
    for name in spec.pca_columns():
        data[name] = rng.normal(0.0, 1.0, rows)

    # Amounts are heavily skewed in the real data, so a lognormal is closer than a normal.
    data[spec.amount_column] = rng.lognormal(mean=3.0, sigma=1.2, size=rows).round(2)

    target = np.zeros(rows, dtype="int8")
    fraud_positions = rng.choice(rows, size=min(fraud_rows, rows), replace=False)
    target[fraud_positions] = 1
    data[spec.target_column] = target

    frame = pd.DataFrame(data)[spec.expected_columns()]

    # Give the fraud rows a real signal, otherwise a model trained in a test learns nothing
    # and the training tests become meaningless.
    fraud_mask = frame[spec.target_column] == 1
    frame.loc[fraud_mask, "V1"] += 4.0
    frame.loc[fraud_mask, "V2"] -= 3.5

    return frame


def write_raw_csv(config: Config, frame: pd.DataFrame) -> None:
    """Write a frame where the ingest stage expects to find the raw file."""
    from fraud_pipeline.paths import ensure_dir

    target = ensure_dir(config.paths.raw()) / config.dataset.spec().raw_file
    frame.to_csv(target, index=False)
