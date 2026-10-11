"""Tests for the tool registry and strict execution engine."""
from __future__ import annotations

import pytest

from statagent.data.store import DatasetStore
from statagent.stats.engine import Params, ToolRegistry, execute_step, execute_tool
from statagent.models.plan import AnalysisStep


def test_all_tools_registered():
    expected = {
        "describe_variable", "check_normality", "correlation_test",
        "correlation_matrix", "ttest_one_sample", "ttest_two_sample",
        "ttest_paired", "chi_square_test", "mann_whitney_u",
        "linear_regression", "bootstrap_ci",
    }
    assert expected <= set(ToolRegistry.names())


def test_unknown_tool_rejected(store):
    r = execute_tool(store, "not_a_tool", {})
    assert not r.ok
    assert "not_a_tool" in r.error


def test_extra_params_rejected(store):
    r = execute_tool(store, "describe_variable", {"dataset_id": "ds1", "column": "x", "bogus": 1})
    assert not r.ok
    assert "bogus" in r.error


def test_missing_params_rejected(store):
    r = execute_tool(store, "describe_variable", {"dataset_id": "ds1"})
    assert not r.ok


def test_execute_step(store):
    step = AnalysisStep(step_id=1, tool="describe_variable",
                        parameters={"dataset_id": "ds1", "column": "x"})
    result = execute_step(store, step)
    assert result.step_id == 1
    assert result.result.ok


def test_tool_llm_schema_shape():
    schema = ToolRegistry.llm_tool_list()
    assert len(schema) == len(ToolRegistry.names())
    for entry in schema:
        assert {"name", "description", "parameters"} <= set(entry)


def test_params_forbid_extra():
    with pytest.raises(Exception):
        Params(unknown_key=1)
