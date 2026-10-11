"""Tests for the validation engine: every rule is triggered by a crafted
preset error case."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statagent.data.store import DatasetStore
from statagent.models.plan import AnalysisPlan, AnalysisStep, StudyDesign
from statagent.validation.context import ValidationContext
from statagent.validation.engine import ValidationEngine

engine = ValidationEngine()


def make_ctx(store=None, dataset_id="ds1", design=None, plan=None, question="", interp="", steps=None):
    profile = store.profile(dataset_id) if store else None
    return ValidationContext(
        question=question,
        profile=profile,
        store=store,
        dataset_id=dataset_id,
        design=design or StudyDesign(),
        plan=plan,
        step_results=steps,
        interpretation_text=interp,
    )


def statuses(ctx, categories=("DATA", "DESIGN", "METHOD")):
    report = engine.validate_before_execution(ctx)
    return {f.rule_id: f.status for f in report.findings}


def store_with(df, name="t.csv"):
    s = DatasetStore()
    s.add(df, name)
    return s


# ---- DATA rules -------------------------------------------------------------

def test_data01_missing_values():
    df = pd.DataFrame({"a": [np.nan] * 60 + [1.0] * 40, "b": np.arange(100)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable",
                                      parameters={"dataset_id": "ds1", "column": "a"})]))
    s = statuses(ctx)
    assert s["DATA-01"] == "FAIL"


def test_data02_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3, 1, 2, 3], "b": [4, 5, 6, 4, 5, 6]})
    s = statuses(make_ctx(store_with(df)))
    assert s["DATA-02"] == "WARNING"


def test_data03_small_sample_inference():
    df = pd.DataFrame({"a": np.arange(8), "g": ["x"] * 4 + ["y"] * 4})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample")]))
    s = statuses(ctx)
    assert s["DATA-03"] == "FAIL"


def test_data04_constant_column():
    df = pd.DataFrame({"a": [5.0] * 20, "b": np.arange(20)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable",
                                      parameters={"dataset_id": "ds1", "column": "a"})]))
    assert statuses(ctx)["DATA-04"] == "FAIL"


def test_data05_outliers():
    vals = [float(i) for i in range(1, 21)] + [100.0, 120.0]
    df = pd.DataFrame({"a": vals, "b": np.arange(len(vals))})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_one_sample",
                                      parameters={"dataset_id": "ds1", "column": "a"})]))
    assert statuses(ctx)["DATA-05"] == "WARNING"


def test_data06_identifier_used():
    df = pd.DataFrame({"user_id": [f"u{i}" for i in range(30)], "a": np.arange(30)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable",
                                      parameters={"dataset_id": "ds1", "column": "user_id"})]))
    assert statuses(ctx)["DATA-06"] == "FAIL"


def test_data07_fully_missing():
    df = pd.DataFrame({"a": [np.nan] * 10, "b": np.arange(10)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable",
                                      parameters={"dataset_id": "ds1", "column": "a"})]))
    assert statuses(ctx)["DATA-07"] == "FAIL"


# ---- DESIGN rules -----------------------------------------------------------

def test_design01_unknown_sample_type_asks():
    df = pd.DataFrame({"a": np.arange(40), "g": ["x"] * 20 + ["y"] * 20})
    ctx = make_ctx(store_with(df), design=StudyDesign(sample_type="unknown"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample")]))
    s = statuses(ctx)
    assert s["DESIGN-01"] == "UNKNOWN"


def test_design02_unknown_study_type_causal_question():
    ctx = make_ctx(design=StudyDesign(study_design="unknown"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="chi_square_test")]),
                   question="Does smoking cause hypertension?")
    assert statuses(ctx)["DESIGN-02"] == "UNKNOWN"


def test_design03_paired_test_on_independent():
    ctx = make_ctx(design=StudyDesign(sample_type="independent"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_paired")]))
    assert statuses(ctx)["DESIGN-03"] == "FAIL"


def test_design04_independent_test_on_paired():
    ctx = make_ctx(design=StudyDesign(sample_type="paired"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample")]))
    assert statuses(ctx)["DESIGN-04"] == "FAIL"


def test_design05_observational_causal_warning():
    ctx = make_ctx(design=StudyDesign(study_design="observational", causal_question=True),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="linear_regression")]))
    assert statuses(ctx)["DESIGN-05"] == "WARNING"


def test_design06_confounding_unaddressed():
    ctx = make_ctx(design=StudyDesign(study_design="observational", causal_question=True,
                                      confounding_addressed=False),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="linear_regression")]))
    assert statuses(ctx)["DESIGN-06"] == "WARNING"


# ---- METHOD rules -----------------------------------------------------------

def test_method01_normality_warning():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.exponential(1, 40)})
    ctx = make_ctx(store_with(df), design=StudyDesign(sample_type="single"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_one_sample",
                                                                   parameters={"dataset_id": "ds1", "column": "a"})]))
    assert statuses(ctx)["METHOD-01"] == "WARNING"


def test_method03_chisq_expected_counts_fail():
    rng = np.random.default_rng(2)
    # Sparse 4x4 table: many expected counts < 1.
    a = rng.choice(list("abcd"), 40, p=[0.4, 0.3, 0.2, 0.1])
    b = rng.choice(list("wxyz"), 40, p=[0.7, 0.1, 0.1, 0.1])
    df = pd.DataFrame({"a": a, "b": b})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="chi_square_test",
                                      parameters={"dataset_id": "ds1", "column_a": "a", "column_b": "b"})]))
    s = statuses(ctx)
    assert s["METHOD-03"] in ("FAIL", "WARNING")


def test_method04_categorical_in_regression_unencoded():
    df = pd.DataFrame({"y": np.arange(30.0), "x": np.arange(30.0), "cat": ["a"] * 15 + ["b"] * 15})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="linear_regression",
                                      parameters={"dataset_id": "ds1", "y": "y", "x": ["x", "cat"]})]))
    assert statuses(ctx)["METHOD-04"] == "FAIL"


def test_method05_collinearity_warning():
    rng = np.random.default_rng(3)
    x1 = rng.normal(0, 1, 100)
    x2 = x1 + rng.normal(0, 0.01, 100)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": x1 + rng.normal(0, 0.1, 100)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="linear_regression",
                                      parameters={"dataset_id": "ds1", "y": "y", "x": ["x1", "x2"]})]))
    assert statuses(ctx)["METHOD-05"] == "WARNING"


def test_method06_type_mismatch_fail():
    df = pd.DataFrame({"a": np.arange(20), "cat": ["x"] * 10 + ["y"] * 10})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample",
                                      parameters={"dataset_id": "ds1", "column": "cat",
                                                  "group_column": "cat", "group_a": "x", "group_b": "y"})]))
    assert statuses(ctx)["METHOD-06"] == "FAIL"


def test_method06_pearson_on_ordinal_warning():
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5] * 4, "spend": np.arange(20.0)})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="correlation_test",
                                      parameters={"dataset_id": "ds1", "x": "rating", "y": "spend",
                                                  "method": "pearson"})]))
    assert statuses(ctx)["METHOD-06"] == "WARNING"


def test_method07_group_size_fail():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "g": ["x", "x", "x", "y"]})
    ctx = make_ctx(store_with(df), design=StudyDesign(sample_type="independent"), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample",
                                      parameters={"dataset_id": "ds1", "column": "a", "group_column": "g",
                                                  "group_a": "x", "group_b": "y"})]))
    assert statuses(ctx)["METHOD-07"] == "FAIL"


def test_method09_multiple_testing_warning():
    steps = [AnalysisStep(step_id=i, tool="correlation_test") for i in range(1, 5)]
    ctx = make_ctx(design=StudyDesign(sample_type="single"), plan=AnalysisPlan(goal="t", steps=steps))
    assert statuses(ctx)["METHOD-09"] == "WARNING"


# ---- INTERP rules -----------------------------------------------------------

def test_interp01_causal_language_observational():
    ctx = make_ctx(design=StudyDesign(study_design="observational"),
                   interp="More study hours causes higher exam scores.")
    report = engine.validate_interpretation(ctx)
    s = {f.rule_id: f.status for f in report.findings}
    assert s["INTERP-01"] == "FAIL"


def test_interp03_nonsignificant_claimed_no_effect():
    from statagent.models.results import StepResult, ToolResult

    step = StepResult(step_id=1, tool="correlation_test",
                      result=ToolResult(tool="correlation_test", ok=True,
                                        summary={"p_value": 0.42}))
    ctx = make_ctx(design=StudyDesign(study_design="observational"),
                   steps=[step],
                   interp="The p-value proves there is no effect whatsoever.")
    s = {f.rule_id: f.status for f in engine.validate_interpretation(ctx).findings}
    assert s["INTERP-03"] == "FAIL"


def test_interp04_ci_misinterpretation():
    ctx = make_ctx(interp="There is a 95% probability that the true mean lies in this interval.")
    s = {f.rule_id: f.status for f in engine.validate_interpretation(ctx).findings}
    assert s["INTERP-04"] == "FAIL"


def test_interp05_invented_number_fails():
    from statagent.models.results import StepResult, ToolResult

    step = StepResult(step_id=1, tool="correlation_test",
                      result=ToolResult(tool="correlation_test", ok=True,
                                        summary={"r": 0.2, "p_value": 0.3, "n": 100}))
    ctx = make_ctx(steps=[step], interp="The correlation is r = 0.98.")
    s = {f.rule_id: f.status for f in engine.validate_interpretation(ctx).findings}
    assert s["INTERP-05"] == "FAIL"


def test_interp05_real_numbers_pass():
    from statagent.models.results import StepResult, ToolResult

    step = StepResult(step_id=1, tool="correlation_test",
                      result=ToolResult(tool="correlation_test", ok=True,
                                        summary={"r": 0.42, "p_value": 0.002, "n": 100}))
    ctx = make_ctx(steps=[step], interp="The correlation is r = 0.42 (p = 0.002, n = 100).")
    s = {f.rule_id: f.status for f in engine.validate_interpretation(ctx).findings}
    assert s.get("INTERP-05") != "FAIL"


def test_interp06_caveat_passes():
    from statagent.models.results import StepResult, ToolResult

    step = StepResult(step_id=1, tool="correlation_test",
                      result=ToolResult(tool="correlation_test", ok=True,
                                        summary={"r": 0.5, "p_value": 0.001}))
    ctx = make_ctx(question="Does X cause Y?", steps=[step],
                   design=StudyDesign(study_design="observational", causal_question=True),
                   interp="There is an association, but correlation does not imply causation.")
    s = {f.rule_id: f.status for f in engine.validate_interpretation(ctx).findings}
    assert s["INTERP-06"] == "PASS"


# ---- gate -------------------------------------------------------------------

def test_gate_blocks_on_fail():
    df = pd.DataFrame({"a": [np.nan] * 60 + [1.0] * 40})
    ctx = make_ctx(store_with(df), plan=AnalysisPlan(
        goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable",
                                      parameters={"dataset_id": "ds1", "column": "a"})]))
    report = engine.validate_before_execution(ctx)
    assert report.blocked


def test_gate_allows_clean_analysis(store):
    ctx = make_ctx(store, design=StudyDesign(sample_type="independent"),
                   plan=AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="ttest_two_sample",
                                                                   parameters={
                                                                       "dataset_id": "ds1", "column": "x",
                                                                       "group_column": "group",
                                                                       "group_a": "A", "group_b": "B"})]))
    report = engine.validate_before_execution(ctx)
    assert not report.blocked
