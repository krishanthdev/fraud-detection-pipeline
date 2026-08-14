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

    # Give the fraud rows a spread of signal strengths, roughly matching what the real file
    # looks like: a few strong features, several middling ones, and a lot of noise.
    #
    # The size of these shifts matters more than it looks. An earlier version used +4.0 and
    # -3.5, which separated the classes almost perfectly. That is not a convenience, it is a
    # trap: the leakage detector correctly flagged those columns as too good to be true, and
    # a model trained on them in a test would score near perfectly and prove nothing. The
    # shifts below land between 0.66 and 0.90 univariate AUC, under the 0.99 leakage
    # threshold and well clear of the noise ceiling.
    #
    # A shift of d on unit normal data gives AUC = Phi(d / sqrt(2)).
    fraud_mask = frame[spec.target_column] == 1
    for column, shift in (("V1", 1.8), ("V2", -1.5), ("V3", 1.1), ("V4", -0.8), ("V10", 0.6)):
        frame.loc[fraud_mask, column] += shift

    return frame


def make_diagnostic_frame(
    config: Config, rows: int = 3000, fraud_rows: int = 150, seed: int = 0
) -> pd.DataFrame:
    """A frame where every column has a deliberately planted, known property.

    Feature selection is a set of detectors, and a detector that has never been shown the
    thing it detects is not tested. Each column here exists to trip exactly one rule, so a
    test can assert that the right rule fired on the right column.

    | Column | What it is | Which rule should fire |
    | --- | --- | --- |
    | `strong` | clean separation | none, it should be kept |
    | `strong_copy` | identical to `strong` | exact duplicate |
    | `strong_noisy` | `strong` plus a little noise | near duplicate by correlation |
    | `leaky` | the target with a whisper of noise | target leakage |
    | `constant` | the same value everywhere | constant |
    | `almost_constant` | one value in 99.9 percent of rows | near constant |
    | `noise` | pure random, unrelated to anything | low signal |
    | `humped` | fraud sits in the middle of the range | none, AUC is blind but KS is not |
    | `counter` | rises forever, like a raw timestamp | range coverage |
    """
    spec = config.dataset.spec()
    rng = np.random.default_rng(seed)

    target = np.zeros(rows, dtype="int8")
    target[rng.choice(rows, size=fraud_rows, replace=False)] = 1
    fraud = target == 1

    strong = rng.normal(0, 1, rows) + fraud * 3.0
    frame = pd.DataFrame(
        {
            "strong": strong,
            "strong_copy": strong.copy(),
            "strong_noisy": strong + rng.normal(0, 0.05, rows),
            "leaky": target + rng.normal(0, 0.001, rows),
            "constant": np.full(rows, 7.0),
            "noise": rng.normal(0, 1, rows),
            "counter": np.arange(rows, dtype="float64"),
            spec.target_column: target,
        }
    )

    # Fraud clusters near zero while normal rows spread to both extremes. The two
    # distributions are clearly different, but neither is consistently higher, so AUC lands
    # near 0.5 and only KS notices. This is the Amount case from the real data.
    humped = rng.normal(0, 1, rows)
    humped[fraud] = rng.normal(0, 0.15, int(fraud.sum()))
    frame["humped"] = humped

    almost_constant = np.full(rows, 1.0)
    almost_constant[rng.choice(rows, size=max(rows // 1000, 1), replace=False)] = 2.0
    frame["almost_constant"] = almost_constant

    return frame


def split_for_drift(
    frame: pd.DataFrame, fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a diagnostic frame in two, in row order.

    Splitting in order is what makes ``counter`` fail range coverage, the same way a real
    timestamp does under a chronological split.
    """
    cut = int(len(frame) * fraction)
    return (
        frame.iloc[:cut].reset_index(drop=True),
        frame.iloc[cut:].reset_index(drop=True),
    )


def write_raw_csv(config: Config, frame: pd.DataFrame) -> None:
    """Write a frame where the ingest stage expects to find the raw file."""
    from fraud_pipeline.paths import ensure_dir

    target = ensure_dir(config.paths.raw()) / config.dataset.spec().raw_file
    frame.to_csv(target, index=False)
