"""Tests for the plan schema (Pydantic validation)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from statagent.models.plan import (
    AnalysisPlan,
    AnalysisStep,
    ClarificationQuestion,
    StudyDesign,
    design_answer_value,
)


def test_valid_plan_roundtrip():
    plan = AnalysisPlan(
        goal="test",
        steps=[AnalysisStep(step_id=1, tool="describe_variable",
                            parameters={"dataset_id": "ds1", "column": "x"})],
    )
    assert plan.steps[0].tool == "describe_variable"


def test_duplicate_step_ids_rejected():
    with pytest.raises(ValidationError):
        AnalysisPlan(
            goal="t",
            steps=[
                AnalysisStep(step_id=1, tool="a"),
                AnalysisStep(step_id=1, tool="b"),
            ],
        )


def test_study_design_literals():
    with pytest.raises(ValidationError):
        StudyDesign(sample_type="banana")


def test_design_fill_only_unknowns():
    d = StudyDesign(sample_type="paired")
    filled = d.fill({"sample_type": "independent", "study_design": "experimental"})
    assert filled.sample_type == "paired"  # already known, not overwritten
    assert filled.study_design == "experimental"


def test_needs_clarification_only_when_relevant():
    plan = AnalysisPlan(goal="t", steps=[AnalysisStep(step_id=1, tool="describe_variable")])
    assert not plan.needs_clarification()  # descriptive question: no design needed
    plan2 = AnalysisPlan(
        goal="t",
        steps=[AnalysisStep(step_id=1, tool="ttest_two_sample")],
        design=StudyDesign(sample_type="unknown"),
    )
    assert plan2.needs_clarification()
    plan3 = AnalysisPlan(
        goal="t",
        steps=[AnalysisStep(step_id=1, tool="ttest_two_sample")],
        design=StudyDesign(sample_type="independent"),
    )
    assert not plan3.needs_clarification()


def test_causal_question_requires_study_design():
    plan = AnalysisPlan(
        goal="t",
        steps=[AnalysisStep(step_id=1, tool="chi_square_test")],
        design=StudyDesign(causal_question=True, study_design="unknown"),
    )
    assert plan.needs_clarification()


def test_design_answer_normalization():
    assert design_answer_value("sample_type", "paired") == "paired"
    assert design_answer_value("sample_type", "independent") == "independent"
    assert design_answer_value("causal_question", "causal") is True
    assert design_answer_value("causal_question", "association") is False
    assert design_answer_value("confounding_addressed", "yes") is True
    assert design_answer_value("study_design", "experimental") == "experimental"
    assert design_answer_value("study_design", "observational") == "observational"


def test_clarification_question_model():
    q = ClarificationQuestion(question="paired?", key="sample_type",
                              options=["paired", "independent"], why="test")
    assert q.key == "sample_type"
