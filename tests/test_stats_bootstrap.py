"""Tests for bootstrap confidence intervals: reproducibility and coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from statagent.data.store import DatasetStore
from statagent.stats.engine import execute_tool


def run(store, tool, params):
    return execute_tool(store, tool, params)


def make_store(seed=31):
    s = DatasetStore()
    rng = np.random.default_rng(seed)
    s.add(pd.DataFrame({"v": rng.normal(50, 10, 100)}), "b.csv")
    return s


def test_bootstrap_reproducible_with_seed():
    s = make_store()
    r1 = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 1000, "seed": 42})
    r2 = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 1000, "seed": 42})
    assert r1.summary["ci95"] == r2.summary["ci95"]
    assert r1.summary["estimate"] == r2.summary["estimate"]


def test_bootstrap_different_seed_differs():
    s = make_store()
    r1 = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 1000, "seed": 1})
    r2 = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 1000, "seed": 2})
    assert r1.summary["ci95"] != r2.summary["ci95"]


def test_bootstrap_ci_contains_true_mean():
    s = make_store()
    r = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 2000, "seed": 7})
    lo, hi = r.summary["ci95"]
    assert lo <= 50.0 <= hi  # true mean of the generator is 50


def test_bootstrap_median():
    s = make_store()
    r = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "statistic": "median", "n_boot": 500, "seed": 3})
    assert r.ok
    lo, hi = r.summary["ci95"]
    assert lo <= r.summary["estimate"] <= hi


def test_bootstrap_correlation():
    s = make_store()
    s2 = DatasetStore()
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    s2.add(pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.5, 100)}), "c.csv")
    r = run(s2, "bootstrap_ci",
            {"dataset_id": "ds1", "column": "x", "x": "x", "y": "y", "statistic": "correlation", "method": "pearson", "n_boot": 500, "seed": 9})
    assert r.ok
    lo, hi = r.summary["ci95"]
    assert lo <= r.summary["estimate"] <= hi
    assert r.summary["estimate"] > 0.9


def test_bootstrap_param_validation():
    s = make_store()
    r = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 10, "seed": 1})
    assert not r.ok
    r = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "alpha": 0.9, "seed": 1})
    assert not r.ok


def test_bootstrap_tiny_sample_rejected():
    s = DatasetStore()
    s.add(pd.DataFrame({"v": [1.0, 2.0, 3.0]}), "t.csv")
    r = run(s, "bootstrap_ci", {"dataset_id": "ds1", "column": "v", "n_boot": 200, "seed": 1})
    assert not r.ok
