"""Tests for the benchmark: ground truth sanity, scoring, paired metrics,
and a small end-to-end runner smoke run."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from statagent.benchmark.metrics import (
    interpretation_score,
    mcnemar,
    numeric_correct,
    score_task,
    summarize,
    wilson_ci,
)
from statagent.benchmark.runner import BenchmarkRunner
from statagent.benchmark.systems import SystemB, SystemC
from statagent.benchmark.tasks import TASKS, TaskSpec, build_ground_truth
from statagent.config import load_config
from statagent.data.store import DatasetStore


def test_task_suite_size_and_categories():
    assert len(TASKS) >= 60
    categories = {t.category for t in TASKS}
    expected = {
        "method_selection", "wrong_method", "numerical_accuracy",
        "insufficient_design", "assumption_violation", "causal_misuse",
        "interpretation_error",
    }
    assert expected <= categories


def test_ground_truth_values_are_real():
    """Reference answers must agree with independent scipy computations."""
    for task in TASKS:
        df = task.build_dataframe()
        gt = task.ground_truth
        assert gt.answers, f"{task.id} has no reference answers"
        for key, value in gt.answers.items():
            assert isinstance(value, float) and math.isfinite(value)
        # Spot-check a few against direct recomputation.
        if task.id == "NA-01":
            d = df[["study_hours", "exam_score"]].dropna()
            ref = stats.pearsonr(d.study_hours, d.exam_score)
            assert math.isclose(gt.answers["r"], ref.statistic, rel_tol=1e-9)
            assert math.isclose(gt.answers["p"], ref.pvalue, rel_tol=1e-9)
        if task.id == "NA-02":
            num = df["posttest_score"]
            a = num[df["method"] == "New"]
            b = num[df["method"] == "Traditional"]
            ref = stats.ttest_ind(a, b, equal_var=False)
            assert abs(gt.answers["p"] - ref.pvalue) < 1e-9


def test_tasks_reproducible_with_seeds():
    d1 = TASKS[0].build_dataframe()
    d2 = TASKS[0].build_dataframe()
    assert d1.equals(d2)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(50, 100)
    assert 0 <= lo <= 0.5 <= hi <= 1
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(100, 100)
    assert hi == pytest.approx(1.0) and lo < 1.0
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and hi > 0.0


def test_mcnemar_symmetry():
    res = mcnemar(5, 20)
    assert res["b"] == 5 and res["c"] == 20
    assert res["p"] < 0.05
    assert mcnemar(0, 0)["p"] == 1.0


def test_numeric_correct_scoring():
    task = build_ground_truth([TASKS[0]])[0]
    outcome = {
        "system": "B_llm_tools",
        "step_results": [
            {"tool": "correlation_test", "ok": True,
             "summary": {"r": task.ground_truth.answers["r"], "p_value": task.ground_truth.answers["p"]}}
        ],
    }
    score, n = numeric_correct(task, outcome)
    assert score == 1.0 and n == 2
    wrong = {"system": "B_llm_tools", "step_results": [
        {"tool": "correlation_test", "ok": True, "summary": {"r": 0.0, "p_value": 0.5}}]}
    score, _ = numeric_correct(task, wrong)
    assert score == 0.0


def test_interpretation_rubric():
    task = TASKS[2]  # MS-03 paired — no required points
    outcome = {"interpretation": "anything"}
    assert interpretation_score(task, outcome) == 1.0
    cm = [t for t in TASKS if t.id == "CM-01"][0]
    bad = {"interpretation": "Smoking causes hypertension in this sample."}
    good = {"interpretation": "Smoking is associated with hypertension; this association does not imply causation."}
    assert interpretation_score(cm, bad) < 1.0
    assert interpretation_score(cm, good) == 1.0


def test_null_claim_marker_negation_aware_scoring():
    """Bug 2 regression in the benchmark rubric: a correct disclaimer must
    not be scored as the forbidden 'no effect' misinterpretation."""
    ie01 = build_ground_truth([t for t in TASKS if t.id == "IE-01"])[0]
    good = {"interpretation": (
        "The result is not statistically significant — the evidence is "
        "insufficient to detect an effect, not that no effect exists."
    )}
    bad = {"interpretation": "This proves there is no effect at all."}
    assert interpretation_score(ie01, good) == 1.0
    assert interpretation_score(ie01, bad) < 1.0


def test_score_task_fields():
    task = build_ground_truth([TASKS[0]])[0]
    outcome = {
        "task_id": task.id, "system": "B_llm_tools", "status": "completed",
        "clarifications_asked": [], "flagged": False,
        "method_selected": "correlation_test", "method_params": {"method": "pearson"},
        "step_results": [{"tool": "correlation_test", "ok": True,
                          "summary": {"r": task.ground_truth.answers["r"],
                                      "p_value": task.ground_truth.answers["p"]}}],
        "interpretation": "The variables are associated (r correlation, not causation).",
        "timing_s": 1.0, "usage": {},
    }
    row = score_task(task, outcome)
    assert row["completed"]
    assert row["method_correct"] is True
    assert row["numeric_accuracy"] == 1.0


def test_summarize_outputs_all_metrics():
    tasks = build_ground_truth([t for t in TASKS if t.id in {"MS-01", "MS-04"}])
    rows_b = []
    rows_c = []
    for task in tasks:
        summary = {}
        for k, v in task.ground_truth.answers.items():
            summary["p_value" if k == "p" else k] = v
        base = {
            "task_id": task.id, "status": "completed", "clarifications_asked": [],
            "flagged": False, "timing_s": 1.0, "usage": {"cost_usd": 0.01},
            "method_selected": task.ground_truth.method, "method_params": {},
            "step_results": [
                {"tool": task.ground_truth.method, "ok": True, "summary": summary}
            ],
            "interpretation": "associated, not causation",
        }
        rows_b.append(score_task(task, dict(base, system="B_llm_tools")))
        rows_c.append(score_task(task, dict(base, system="C_llm_tools_validation")))
    summary = summarize({"B_llm_tools": rows_b, "C_llm_tools_validation": rows_c}, tasks)
    for system in ("B_llm_tools", "C_llm_tools_validation"):
        m = summary[system]
        assert m["numerical_accuracy"] == 1.0
        assert m["task_completion_rate"] == 1.0
        assert "cost_per_correct_task_usd" in m
    assert "paired_comparison" in summary


def test_runner_smoke_run(tmp_path: Path):
    cfg = load_config({})
    systems = {
        "B_llm_tools": SystemB(cfg, None),
        "C_llm_tools_validation": SystemC(cfg, None),
    }
    tasks = [t for t in TASKS if t.id in {"MS-01", "MS-04", "ID-02", "CM-01", "WM-01"}]
    runner = BenchmarkRunner(cfg, systems, tmp_path / "run")
    run = runner.run(tasks, label="smoke")
    results_path = run.results_path
    assert results_path.exists()
    n_lines = sum(1 for _ in results_path.read_text(encoding="utf-8").splitlines())
    assert n_lines == len(tasks) * 2
    assert run.summary["B_llm_tools"]["n_tasks"] == len(tasks)
    assert run.summary["C_llm_tools_validation"]["invalid_analysis_rate"] == 0.0
    assert (tmp_path / "run" / "ground_truth.json").exists()
    assert (tmp_path / "run" / "summary.json").exists()


def test_benchmark_run_excludes_fabrication_guard():
    """Every task outcome must come from a real execution — no placeholders."""
    cfg = load_config({})
    systems = {"B_llm_tools": SystemB(cfg, None)}
    tasks = [t for t in TASKS if t.id == "MS-01"]
    runner = BenchmarkRunner(cfg, systems, Path("runs") / "benchmark" / "_test_guard")
    run = runner.run(tasks, label="guard")
    outcomes = []
    with run.results_path.open(encoding="utf-8") as fh:
        import json

        outcomes = [json.loads(line) for line in fh if line.strip()]
    assert outcomes
    assert outcomes[0]["task_id"] == "MS-01"
    assert outcomes[0]["system"] == "B_llm_tools"
