"""Tests for the planner: mock-LLM path (JSON validation, self-repair) and
the heuristic offline planner."""
from __future__ import annotations

import json

from statagent.agent.llm import LLMResponse, MockLLM
from statagent.agent.planner import HeuristicPlanner, PlannerAgent
from statagent.data.store import DatasetStore
from statagent.data.synthetic import (
    gen_health_survey,
    gen_likert_groups,
    gen_prepost,
    gen_study_hours,
    gen_teaching_method,
)


def profile_of(df, name="t.csv"):
    store = DatasetStore()
    ds = store.add(df, name)
    return store.profile(ds)


VALID_PLAN = {
    "goal": "correlate",
    "design": {"sample_type": "single", "study_design": "observational",
               "causal_question": False, "confounding_addressed": None,
               "repeated_measures": None, "notes": ""},
    "steps": [
        {
            "step_id": 1,
            "tool": "correlation_test",
            "parameters": {"dataset_id": "ds1", "x": "study_hours", "y": "exam_score", "method": "pearson"},
            "variables": ["study_hours", "exam_score"],
            "rationale": "assess association",
        }
    ],
    "clarifications": [],
    "method_rationale": "pearson fits continuous data",
    "limitations": [],
}


def test_llm_planner_valid_json():
    mock = MockLLM().on(
        lambda m: True,
        lambda m: LLMResponse(content=json.dumps(VALID_PLAN)),
    )
    agent = PlannerAgent(mock)
    plan = agent.plan("Is study time associated with exam score?",
                      profile_of(gen_study_hours()))
    assert plan.steps[0].tool == "correlation_test"
    assert plan.steps[0].parameters["x"] == "study_hours"


def test_llm_planner_repairs_invalid_json():
    """First reply is malformed JSON; the repair loop asks again."""
    calls = {"n": 0}

    def factory(messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(content="{not valid json")
        return LLMResponse(content=json.dumps(VALID_PLAN))

    mock = MockLLM().on(lambda m: True, factory)
    agent = PlannerAgent(mock)
    plan = agent.plan("q", profile_of(gen_study_hours()))
    assert plan.steps[0].tool == "correlation_test"
    assert calls["n"] == 2


def test_llm_planner_rejects_schema_violations():
    """Reply with duplicate step ids must trigger a repair."""
    bad = dict(VALID_PLAN)
    bad["steps"] = [dict(VALID_PLAN["steps"][0]), dict(VALID_PLAN["steps"][0])]
    calls = {"n": 0}

    def factory(messages):
        calls["n"] += 1
        return LLMResponse(content=json.dumps(VALID_PLAN if calls["n"] > 1 else bad))

    mock = MockLLM().on(lambda m: True, factory)
    agent = PlannerAgent(mock)
    plan = agent.plan("q", profile_of(gen_study_hours()))
    assert plan.steps[0].tool == "correlation_test"
    assert calls["n"] == 2


def test_planner_sanitizes_unknown_tools():
    bad = dict(VALID_PLAN)
    bad["steps"] = [
        {"step_id": 1, "tool": "rm_rf_everything", "parameters": {}, "variables": [], "rationale": ""}
    ]
    mock = MockLLM().on(lambda m: True, lambda m: LLMResponse(content=json.dumps(bad)))
    plan = PlannerAgent(mock).plan("q", profile_of(gen_study_hours()))
    assert plan.steps == []
    assert any("unknown tool" in lim for lim in plan.limitations)


def test_planner_sanitizes_invalid_params():
    bad = dict(VALID_PLAN)
    bad["steps"][0]["parameters"] = {"dataset_id": "ds1", "x": "study_hours", "y": "exam_score", "bogus": 3}
    mock = MockLLM().on(lambda m: True, lambda m: LLMResponse(content=json.dumps(bad)))
    plan = PlannerAgent(mock).plan("q", profile_of(gen_study_hours()))
    assert plan.steps == []


def test_heuristic_correlation_question():
    plan = HeuristicPlanner().plan(
        "Is study time associated with exam score?", profile_of(gen_study_hours())
    )
    assert plan.steps[0].tool == "correlation_test"
    assert plan.steps[0].parameters["x"] == "study_hours"


def test_heuristic_group_comparison_asks_clarification():
    plan = HeuristicPlanner().plan(
        "Does the new teaching method produce higher posttest scores than the traditional one?",
        profile_of(gen_teaching_method()),
    )
    assert any(c.key == "sample_type" for c in plan.clarifications)


def test_heuristic_group_comparison_with_answer():
    plan = HeuristicPlanner().plan(
        "Does the new teaching method produce higher posttest scores than the traditional one?",
        profile_of(gen_teaching_method()),
        design_answers={"sample_type": "independent", "study_design": "experimental",
                        "causal_question": False, "confounding_addressed": False},
    )
    assert plan.steps[0].tool == "ttest_two_sample"


def test_heuristic_paired_detection():
    plan = HeuristicPlanner().plan(
        "Do scores change from before to after the training program?",
        profile_of(gen_prepost()),
    )
    assert plan.steps[0].tool == "ttest_paired"


def test_heuristic_chi_square():
    plan = HeuristicPlanner().plan(
        "Is smoking associated with hypertension?", profile_of(gen_health_survey())
    )
    assert plan.steps[0].tool == "chi_square_test"


def test_heuristic_ordinal_uses_mann_whitney():
    plan = HeuristicPlanner().plan(
        "Compare the ratings of the treatment and control groups.",
        profile_of(gen_likert_groups()),
        design_answers={"sample_type": "independent", "study_design": "experimental",
                        "causal_question": False, "confounding_addressed": False},
    )
    assert plan.steps[0].tool == "mann_whitney_u"


def test_heuristic_causal_question_asks_design():
    plan = HeuristicPlanner().plan(
        "Does smoking cause hypertension?", profile_of(gen_health_survey())
    )
    assert plan.needs_clarification()


def test_heuristic_explicit_method_name_wins():
    plan = HeuristicPlanner().plan(
        "Run a chi-square test of independence between smoking and hypertension.",
        profile_of(gen_health_survey()),
    )
    assert plan.steps[0].tool == "chi_square_test"


# ---- Bug 1 regressions: "Is study time associated with exam score?" ---------

def bug1_dataframe():
    """The exact user-reported data: hours 1..10, score 10..100."""
    import pandas as pd

    return pd.DataFrame({"hours": range(1, 11), "score": range(10, 110, 10)})


def test_bug1_association_question_picks_pearson():
    plan = HeuristicPlanner().plan(
        "Is study time associated with exam score?", profile_of(bug1_dataframe())
    )
    assert plan.steps, plan.limitations
    step = plan.steps[0]
    assert step.tool == "correlation_test"
    assert step.parameters["method"] == "pearson"
    assert set(step.parameters) >= {"x", "y"}
    assert {step.parameters["x"], step.parameters["y"]} == {"hours", "score"}


def test_bug1_synonym_matching_time_to_hours():
    plan = HeuristicPlanner().plan(
        "Does the time spent studying relate to the score achieved?",
        profile_of(bug1_dataframe()),
    )
    assert plan.steps and plan.steps[0].tool == "correlation_test"


def test_heuristic_association_fallback_when_only_one_column_matched():
    """Only 'score' is name-matched; with exactly two numeric columns the
    planner must still associate both instead of describing one."""
    import pandas as pd

    df = pd.DataFrame({"workload": range(1, 11), "score": range(10, 110, 10)})
    plan = HeuristicPlanner().plan(
        "Is study time associated with exam score?", profile_of(df)
    )
    assert plan.steps and plan.steps[0].tool == "correlation_test"
    assert {plan.steps[0].parameters["x"], plan.steps[0].parameters["y"]} == {"workload", "score"}


def test_heuristic_relation_with_categorical_pair_not_correlation():
    """A relation question between two categoricals must stay chi-square even
    when the dataset also has exactly two numeric columns."""
    plan = HeuristicPlanner().plan(
        "Is smoking associated with hypertension?", profile_of(gen_health_survey())
    )
    assert plan.steps[0].tool == "chi_square_test"


def test_heuristic_no_matching_variables_reports_clearly():
    """Intent present but no column matches: fail loudly, don't summarize."""
    plan = HeuristicPlanner().plan(
        "Is smoking associated with hypertension?", profile_of(bug1_dataframe())
    )
    assert plan.steps == []
    assert any("Could not find any columns" in lim for lim in plan.limitations)
    assert plan.goal == "No matching variables found"


def test_planner_records_llm_error_and_falls_back():
    mock = MockLLM().on(lambda m: True, lambda m: LLMResponse(content="garbage{{"))
    agent = PlannerAgent(mock)
    plan = agent.plan("Is study time associated with exam score?",
                      profile_of(gen_study_hours()))
    assert plan.steps[0].tool == "correlation_test"  # heuristic fallback worked
    assert agent.last_error is not None
    assert any("heuristic fallback" in lim for lim in plan.limitations)
