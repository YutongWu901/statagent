"""Tests for OLS regression: exact fits, scipy/statsmodels cross-checks,
diagnostics, and error handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statagent.data.store import DatasetStore
from statagent.stats.engine import execute_tool


def run(store, tool, params):
    return execute_tool(store, tool, params)


def test_exact_linear_fit(perfect_store):
    # y = 2x + 1 exactly.
    r = run(perfect_store, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"]})
    assert r.ok
    coefs = r.summary["coefficients"]
    assert coefs["x"]["coefficient"] == pytest.approx(2.0, abs=1e-6)
    assert coefs["Intercept"]["coefficient"] == pytest.approx(1.0, abs=1e-6)
    assert r.summary["r_squared"] == pytest.approx(1.0, abs=1e-6)
    assert coefs["x"]["p_value"] < 1e-9


def test_regression_matches_statsmodels(store):
    r = run(store, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"]})
    df = store.get("ds1")
    import statsmodels.api as sm

    X = sm.add_constant(df["x"])
    ref = sm.OLS(df["y"], X).fit()
    coefs = r.summary["coefficients"]
    assert coefs["x"]["coefficient"] == pytest.approx(ref.params["x"], abs=1e-6)
    assert coefs["x"]["se"] == pytest.approx(ref.bse["x"], abs=1e-6)
    assert coefs["x"]["p_value"] == pytest.approx(ref.pvalues["x"], abs=1e-6)
    assert r.summary["r_squared"] == pytest.approx(ref.rsquared, abs=1e-6)
    assert r.summary["adj_r_squared"] == pytest.approx(ref.rsquared_adj, abs=1e-6)


def test_multiple_regression_ci_brackets(store):
    r = run(store, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"]})
    coefs = r.summary["coefficients"]
    for name in ("x", "Intercept"):
        lo, hi = coefs[name]["ci95"]
        assert lo <= coefs[name]["coefficient"] <= hi


def test_categorical_predictor_one_hot():
    s = DatasetStore()
    rng = np.random.default_rng(21)
    n = 60
    df = pd.DataFrame(
        {
            "y": rng.normal(0, 1, n),
            "x1": rng.normal(0, 1, n),
            "region": rng.choice(["N", "S", "E"], n),
        }
    )
    s.add(df, "cat.csv")
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x1", "region"], "categorical": ["region"]})
    assert r.ok
    coefs = r.summary["coefficients"]
    # drop_first removes the alphabetically first level (E).
    assert "region_N" in coefs and "region_S" in coefs
    assert "region_E" not in coefs
    assert r.summary["k"] == 4  # intercept + x1 + 2 dummies


def test_diagnostics_present(store):
    r = run(store, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"]})
    d = r.details["diagnostics"]
    for key in ("durbin_watson", "jarque_bera", "breusch_pagan", "vif", "cook_max"):
        assert key in d
    assert d["durbin_watson"] > 0


def test_heteroscedasticity_detected():
    s = DatasetStore()
    rng = np.random.default_rng(22)
    n = 200
    x = rng.uniform(1, 30, n)
    y = 10 + 1.2 * x + rng.normal(0, 0.5 * x, n)
    s.add(pd.DataFrame({"x": x, "y": y}), "het.csv")
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"]})
    assert r.ok
    assert r.details["diagnostics"]["breusch_pagan_p"] < 0.05


def test_vif_flags_collinearity():
    s = DatasetStore()
    rng = np.random.default_rng(23)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.01, n)  # near-perfect collinearity
    y = x1 + rng.normal(0, 0.1, n)
    s.add(pd.DataFrame({"x1": x1, "x2": x2, "y": y}), "col.csv")
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x1", "x2"]})
    vif = r.details["diagnostics"]["vif"]
    assert any(v > 10 for v in vif.values())


def test_regression_errors():
    s = DatasetStore()
    s.add(pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]}), "e.csv")
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "nope", "x": ["x"]})
    assert not r.ok
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": []})
    assert not r.ok
    # A numeric column listed as categorical collapses into dummies with too
    # few observations -> rejected.
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x"], "categorical": ["x"]})
    assert not r.ok


def test_regression_too_few_observations():
    s = DatasetStore()
    df = pd.DataFrame({"y": [1.0, 2.0], "x": [1.0, 2.0], "z": [3.0, 4.0]})
    s.add(df, "small.csv")
    r = run(s, "linear_regression", {"dataset_id": "ds1", "y": "y", "x": ["x", "z"]})
    assert not r.ok
