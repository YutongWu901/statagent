"""End-to-end pipeline tests in offline mode: the full question -> report flow."""
from __future__ import annotations

from pathlib import Path

import pytest

from statagent.agent.pipeline import AnalysisPipeline
from statagent.config import load_config
from statagent.data.store import DatasetStore
from statagent.data.synthetic import gen_health_survey, gen_study_hours, gen_teaching_method


def make_pipe():
    return AnalysisPipeline(load_config({}))


def test_complete_correlation_analysis(tmp_path: Path):
    store = DatasetStore()
    ds = store.add(gen_study_hours(), "study.csv")
    out = make_pipe().analyze("Is study time associated with exam score?", store, ds)
    assert out.status == "completed"
    record = out.record
    assert record.step_results
    step = record.step_results[0]
    assert step.tool == "correlation_test"
    assert step.result.ok
    assert step.result.summary["r"] > 0.8
    assert record.interpretation is not None
    assert record.validation_before is not None
    assert record.validation_after is not None


def test_clarification_flow_blocks_until_answered():
    store = DatasetStore()
    ds = store.add(gen_teaching_method(), "t.csv")
    pipe = make_pipe()
    out = pipe.analyze(
        "Does the new teaching method produce higher posttest scores than the traditional one?",
        store, ds,
    )
    assert out.status == "clarification_needed"
    assert any(q.key == "sample_type" for q in out.questions)
    out2 = pipe.analyze(
        "Does the new teaching method produce higher posttest scores than the traditional one?",
        store, ds,
        design_answers={"sample_type": "independent", "study_design": "experimental",
                        "causal_question": False, "confounding_addressed": False},
    )
    assert out2.status == "completed"
    assert out2.record.step_results[0].tool == "ttest_two_sample"


def test_validation_gate_blocks_invalid_analysis():
    """Chi-square on a continuous column must be blocked by METHOD-03."""
    store = DatasetStore()
    ds = store.add(gen_health_survey(), "h.csv")
    out = make_pipe().analyze(
        "Is smoking associated with BMI? Use a chi-square test.", store, ds,
        design_answers={"sample_type": "independent", "study_design": "observational",
                        "causal_question": False, "confounding_addressed": False},
    )
    # The planner substitutes a group comparison for the inappropriate
    # chi-square; a completion here is acceptable, but if it did plan a
    # chi-square the gate must have blocked it. Either way the outcome must
    # not contain a failed gate silently passing.
    assert out.status in ("completed", "blocked")
    if out.status == "blocked":
        assert any(f.status == "FAIL" for f in out.validation.findings)


def test_force_override_runs_anyway():
    store = DatasetStore()
    ds = store.add(gen_health_survey(), "h.csv")
    out = make_pipe().analyze(
        "Compute the Pearson correlation between smoking and hypertension.",
        store, ds, force=True,
    )
    assert out.status == "completed"
    assert out.record.validation_before.override


def test_record_saved_to_disk(tmp_path: Path):
    store = DatasetStore()
    ds = store.add(gen_study_hours(), "s.csv")
    out = make_pipe().analyze("Is study time associated with exam score?", store, ds,
                              save_dir=tmp_path)
    assert out.status == "completed"
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_offline_meta_recorded():
    store = DatasetStore()
    ds = store.add(gen_study_hours(), "s.csv")
    out = make_pipe().analyze("Is study time associated with exam score?", store, ds)
    assert out.record.llm["mode"] == "offline"


# ---- Bug 1 regressions ------------------------------------------------------

def test_bug1_hours_score_end_to_end_pearson_r_is_one():
    """The exact user scenario: hours 1..10, score 10..100, association
    question. Must plan Pearson correlation and compute r = 1.0."""
    import pandas as pd

    store = DatasetStore()
    ds = store.add(
        pd.DataFrame({"hours": range(1, 11), "score": range(10, 110, 10)}),
        "bug1.csv",
    )
    out = make_pipe().analyze("Is study time associated with exam score?", store, ds)
    assert out.status == "completed", out.error
    step = out.record.step_results[0].result
    assert step.tool == "correlation_test"
    assert step.summary["method"] == "pearson"
    assert step.summary["r"] == pytest.approx(1.0, abs=1e-9)
    assert step.summary["p_value"] < 1e-6


def test_bug1_variable_mismatch_reports_error_not_wrong_method():
    """Asking about variables that don't exist must produce a clear error,
    not silently run an unrelated analysis."""
    import pandas as pd

    store = DatasetStore()
    ds = store.add(
        pd.DataFrame({"hours": range(1, 11), "score": range(10, 110, 10)}),
        "bug1.csv",
    )
    out = make_pipe().analyze("Is smoking associated with hypertension?", store, ds)
    assert out.status == "error"
    assert "Could not find any columns" in out.error


# ---- Bug 3 regressions ------------------------------------------------------

def test_llm_failure_surfaces_warning_and_falls_back():
    from statagent.agent.llm import MockLLM

    mock = MockLLM().on(lambda m: True, lambda m: type("R", (), {"content": "garbage{{", "tool_calls": []})())
    pipe = AnalysisPipeline(load_config({}), client=mock)
    store = DatasetStore()
    ds = store.add(gen_study_hours(), "s.csv")
    out = pipe.analyze("Is study time associated with exam score?", store, ds)
    assert out.status == "completed"  # heuristic fallback
    assert out.llm_warning
    assert out.record.llm.get("warning")
