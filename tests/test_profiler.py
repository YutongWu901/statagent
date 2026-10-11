"""Tests for automatic data profiling and variable type inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statagent.data.profiler import count_outliers, infer_variable_type, profile_frame
from statagent.models.dataset import VariableType


def _series(values, name="col"):
    return pd.Series(values, name=name)


def test_infer_continuous():
    assert infer_variable_type(_series(np.linspace(0, 1, 50)), "x")[0] == VariableType.NUMERIC_CONTINUOUS


def test_infer_binary():
    assert infer_variable_type(_series([0, 1, 0, 1, 0]), "flag")[0] == VariableType.BINARY


def test_infer_discrete_counts():
    assert infer_variable_type(_series([1, 2, 3, 2, 1, 4, 2]), "siblings")[0] == VariableType.NUMERIC_DISCRETE


def test_infer_ordinal_by_name():
    assert infer_variable_type(_series([1, 2, 3, 4, 5, 1, 2]), "satisfaction")[0] == VariableType.CATEGORICAL_ORDINAL


def test_infer_nominal_strings():
    assert infer_variable_type(_series(["a", "b", "a", "c", "b", "a"]), "cat")[0] == VariableType.CATEGORICAL_NOMINAL


def test_infer_identifier():
    vals = [f"id{i:03d}" for i in range(30)]
    assert infer_variable_type(_series(vals), "uuid")[0] == VariableType.IDENTIFIER


def test_infer_numeric_strings_coerced():
    vtype, coerced = infer_variable_type(_series(["1.5", "2.5", "3.5"]), "numstr")
    assert vtype == VariableType.NUMERIC_DISCRETE
    assert coerced is not None


def test_outlier_detection():
    vals = _series([1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 50, 60], "v")
    assert count_outliers(vals) == 2


def test_no_outliers_when_flat():
    assert count_outliers(_series([5, 5, 5, 5, 5, 5]), "v") == 0


def test_profile_reports_missing_and_duplicates():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan, 4.0], "b": ["x", "y", "x", "y"], "c": [1, 1, 1, 1]}
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    profile = profile_frame(df, "ds1", "t.csv")
    assert profile.n_rows == 5
    assert profile.n_duplicate_rows == 1
    a = profile.column("a")
    assert a.n_missing == 1
    assert a.missing_pct == pytest.approx(20.0)
    assert profile.column("c").is_constant
    assert profile.column("b").variable_type == VariableType.CATEGORICAL_NOMINAL


def test_profile_llm_context_contains_columns():
    df = pd.DataFrame({"hours": [1, 2, 3], "score": [50, 60, 70]})
    profile = profile_frame(df, "ds1", "t.csv")
    ctx = profile.llm_context()
    assert "hours" in ctx and "score" in ctx


# ---- Bug 1 regressions: type inference for hours/score ----------------------

def test_bug1_hours_column_is_continuous():
    """hours = 1..10 (10 distinct values) must NOT be classified discrete."""
    vtype, _ = infer_variable_type(_series(list(range(1, 11))), "hours")
    assert vtype == VariableType.NUMERIC_CONTINUOUS


def test_bug1_score_column_is_not_ordinal():
    """A column named 'score' with 10 distinct values is a continuous
    outcome, not a categorical ordinal."""
    vtype, _ = infer_variable_type(_series(list(range(10, 110, 10))), "score")
    assert vtype == VariableType.NUMERIC_CONTINUOUS


def test_likert_still_ordinal_by_name():
    vtype, _ = infer_variable_type(_series([1, 2, 3, 4, 5, 1, 2]), "rating")
    assert vtype == VariableType.CATEGORICAL_ORDINAL
