"""Tests that seeding actually makes a run repeatable.

Reproducibility is part of the definition of done, so it gets a test rather than a promise
in the README.
"""

from __future__ import annotations

import os
import random

import numpy as np

from fraud_pipeline.seeding import set_global_seed


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())


def test_same_seed_gives_the_same_numbers() -> None:
    set_global_seed(42)
    first = _draw()

    set_global_seed(42)
    second = _draw()

    assert first == second


def test_different_seeds_give_different_numbers() -> None:
    set_global_seed(1)
    first = _draw()

    set_global_seed(2)
    second = _draw()

    assert first != second


def test_seed_is_exported_to_the_environment() -> None:
    set_global_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_set_global_seed_returns_the_seed() -> None:
    assert set_global_seed(7) == 7
