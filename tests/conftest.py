"""Shared fixtures: offline config, sample stores, mock LLM helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statagent.config import load_config
from statagent.data.store import DatasetStore


@pytest.fixture()
def cfg():
    return load_config({})


@pytest.fixture()
def store():
    s = DatasetStore()
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "x": np.round(rng.normal(10, 3, 50), 2),
            "y": np.round(2 * np.arange(50) + rng.normal(0, 1, 50), 2),
            "group": ["A"] * 25 + ["B"] * 25,
            "cat": rng.choice(["low", "mid", "high"], 50),
        }
    )
    s.add(df, "sample.csv")
    return s


@pytest.fixture()
def small_store():
    s = DatasetStore()
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],
            "g": ["x", "x", "y", "y", "y"],
        }
    )
    s.add(df, "small.csv")
    return s


@pytest.fixture()
def perfect_store():
    """Deterministic data with exact known statistics."""
    s = DatasetStore()
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [3.0, 5.0, 7.0, 9.0, 11.0],  # y = 2x + 1 exactly
        }
    )
    s.add(df, "perfect.csv")
    return s
