"""Tests for the statistical tools against known values and scipy."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.engine import execute_tool


def run(store, tool, params):
    return execute_tool(store, tool, params)


def test_describe_known_values(small_store):
    r = run(small_store, "describe_variable", {"dataset_id": "ds1", "column": "a"})
    assert r.ok
    assert r.summary["mean"] == pytest.approx(3.0)
    assert r.summary["median"] == pytest.approx(3.0)
    assert r.summary["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert r.summary["q1"] == pytest.approx(2.0)
    assert r.summary["q3"] == pytest.approx(4.0)
    assert r.summary["iqr"] == pytest.approx(2.0)
    assert r.n == 5


def test_describe_ci_contains_mean(small_store):
    r = run(small_store, "describe_variable", {"dataset_id": "ds1", "column": "a"})
    lo, hi = r.summary["ci95_mean"]
    assert lo <= r.summary["mean"] <= hi


def test_describe_missing_column(store):
    r = run(store, "describe_variable", {"dataset_id": "ds1", "column": "nope"})
    assert not r.ok
    assert "not found" in r.error


def test_describe_non_numeric_column(store):
    r = run(store, "describe_variable", {"dataset_id": "ds1", "column": "group"})
    assert not r.ok


def test_check_normality_normal_data():
    s = DatasetStore()
    rng = np.random.default_rng(3)
    import pandas as pd

    s.add(pd.DataFrame({"v": rng.normal(0, 1, 200)}), "n.csv")
    r = run(s, "check_normality", {"dataset_id": "ds1", "column": "v"})
    assert r.ok
    assert r.summary["shapiro_w"] is not None
    assert r.summary["normal_at_0.05"] is True


def test_check_normality_skewed_data():
    s = DatasetStore()
    rng = np.random.default_rng(4)
    import pandas as pd

    s.add(pd.DataFrame({"v": rng.exponential(1, 150)}), "e.csv")
    r = run(s, "check_normality", {"dataset_id": "ds1", "column": "v"})
    assert r.ok
    assert r.summary["normal_at_0.05"] is False


def test_correlation_perfect_linear(perfect_store):
    r = run(perfect_store, "correlation_test", {"dataset_id": "ds1", "x": "x", "y": "y", "method": "pearson"})
    assert r.ok
    assert r.summary["r"] == pytest.approx(1.0)
    assert r.summary["p_value"] < 0.001
    assert r.summary["ci95_r"][0] <= 1.0 <= r.summary["ci95_r"][1]


def test_correlation_matches_scipy(store):
    r = run(store, "correlation_test", {"dataset_id": "ds1", "x": "x", "y": "y", "method": "pearson"})
    df = store.get("ds1")
    ref = stats.pearsonr(df["x"], df["y"])
    assert r.summary["r"] == pytest.approx(ref.statistic, abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue, abs=1e-6)


def test_spearman_ranks(store):
    r = run(store, "correlation_test", {"dataset_id": "ds1", "x": "x", "y": "y", "method": "spearman"})
    df = store.get("ds1")
    ref = stats.spearmanr(df["x"], df["y"])
    assert r.summary["r"] == pytest.approx(ref.statistic, abs=1e-6)


def test_correlation_matrix(store):
    r = run(store, "correlation_matrix", {"dataset_id": "ds1", "columns": ["x", "y"], "method": "pearson"})
    assert r.ok
    corr = r.summary["correlations"]["x vs y"]
    df = store.get("ds1")
    assert corr == pytest.approx(stats.pearsonr(df["x"], df["y"]).statistic, abs=1e-6)
    assert len(r.figures) == 1


def test_correlation_requires_numeric(store):
    r = run(store, "correlation_test", {"dataset_id": "ds1", "x": "group", "y": "y"})
    assert not r.ok
