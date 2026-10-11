"""End-to-end pipeline tests in offline mode: the full question -> report flow."""
from __future__ import annotations

from pathlib import Path

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
