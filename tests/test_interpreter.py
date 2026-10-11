"""Tests for the interpreter: LLM path, template fallback, number fidelity."""
from __future__ import annotations

import json

from statagent.agent.interpreter import InterpreterAgent
from statagent.agent.llm import LLMResponse, MockLLM
from statagent.models.plan import AnalysisPlan
from statagent.models.results import StepResult, ToolResult, ValidationReport


def test_llm_interpreter_parses_json():
    payload = {
        "summary": "s",
        "key_findings": ["k1"],
        "interpretation": "i",
        "limitations": ["l1"],
        "conclusion": "c",
    }
    mock = MockLLM().on(lambda m: True, lambda m: LLMResponse(content=json.dumps(payload)))
    agent = InterpreterAgent(mock)
    interp = agent.interpret("q", AnalysisPlan(goal="g"), [], None)
    assert interp.summary == "s"
    assert interp.conclusion == "c"
    assert "s" in interp.text


def test_llm_interpreter_falls_back_on_bad_json():
    mock = MockLLM().on(lambda m: True, lambda m: LLMResponse(content="not json"))
    agent = InterpreterAgent(mock)
    interp = agent.interpret("q", AnalysisPlan(goal="g"), [], None)
    assert "Offline mode" in interp.limitations[0]


def test_template_interpreter_reports_effect_size():
    step = StepResult(
        step_id=1,
        tool="ttest_two_sample",
        result=ToolResult(
            tool="ttest_two_sample", ok=True, n=80,
            summary={"test": "Welch two-sample t-test", "t": 3.1, "p_value": 0.003,
                     "mean_diff": 5.0, "ci95_mean_diff": [2.0, 8.0],
                     "cohens_d": 0.7, "n": 80, "group_a": "A", "group_b": "B"},
        ),
    )
    agent = InterpreterAgent(None)
    interp = agent.interpret("compare groups", AnalysisPlan(goal="g"), [step], None)
    assert "Cohen's d" in interp.text
    assert "5% level" in interp.text


def test_template_interpreter_nonsignificant_phrasing():
    step = StepResult(
        step_id=1,
        tool="correlation_test",
        result=ToolResult(
            tool="correlation_test", ok=True, n=50,
            summary={"method": "pearson", "r": 0.08, "p_value": 0.42,
                     "ci95_r": [-0.2, 0.35], "r_squared": 0.0064, "n": 50},
        ),
    )
    agent = InterpreterAgent(None)
    interp = agent.interpret("is x associated with y", AnalysisPlan(goal="g"), [step], None)
    assert "insufficient evidence" in interp.text
    assert "not statistically significant" in interp.text


def test_template_interpreter_small_sample_caveat():
    step = StepResult(
        step_id=1,
        tool="describe_variable",
        result=ToolResult(tool="describe_variable", ok=True, n=8,
                          summary={"count": 8, "mean": 3.0, "median": 3.0, "std": 1.0,
                                   "ci95_mean": [2.0, 4.0]}),
    )
    agent = InterpreterAgent(None)
    interp = agent.interpret("summarize", AnalysisPlan(goal="g"), [step], None)
    assert any("Small sample" in lim for lim in interp.limitations)


def test_template_interpreter_quotes_real_numbers():
    step = StepResult(
        step_id=1,
        tool="correlation_test",
        result=ToolResult(tool="correlation_test", ok=True, n=100,
                          summary={"method": "pearson", "r": 0.909, "p_value": 0.0001,
                                   "ci95_r": [0.88, 0.93], "r_squared": 0.826, "n": 100}),
    )
    interp = InterpreterAgent(None).interpret("q", AnalysisPlan(goal="g"), [step], None)
    assert "0.909" in interp.text


def test_llm_interpreter_receives_validation_context():
    """The prompt must include validation findings."""
    seen = {}

    def factory(messages):
        seen["user"] = messages[-1]["content"]
        return LLMResponse(content=json.dumps({
            "summary": "s", "key_findings": [], "interpretation": "i",
            "limitations": [], "conclusion": "c",
        }))

    mock = MockLLM().on(lambda m: True, factory)
    report = ValidationReport()
    from statagent.models.results import Finding

    report.findings.append(Finding(rule_id="DATA-05", category="DATA", status="WARNING",
                                   message="outliers present"))
    InterpreterAgent(mock).interpret("q", AnalysisPlan(goal="g"), [], report)
    assert "DATA-05" in seen["user"]


# ---- Bug 2 regression: the template's own disclaimer must pass INTERP-03 ----

def test_template_nonsignificant_interpretation_passes_interp03():
    """The offline template's 'not that no effect exists' phrasing must not
    be flagged by the interpretation validation engine."""
    from statagent.validation.context import ValidationContext
    from statagent.validation.engine import ValidationEngine

    step = StepResult(
        step_id=1,
        tool="correlation_test",
        result=ToolResult(
            tool="correlation_test", ok=True, n=50,
            summary={"method": "pearson", "r": 0.08, "p_value": 0.42,
                     "ci95_r": [-0.2, 0.35], "r_squared": 0.0064, "n": 50},
        ),
    )
    interp = InterpreterAgent(None).interpret(
        "Is x associated with y?", AnalysisPlan(goal="g"), [step], None
    )
    ctx = ValidationContext(step_results=[step], interpretation_text=interp.text)
    findings = ValidationEngine().validate_interpretation(ctx).findings
    statuses = {f.rule_id: f.status for f in findings}
    assert statuses.get("INTERP-03") != "FAIL"
