"""Benchmark scoring: per-task evaluation and aggregate metrics with
confidence intervals.

Scoring is deterministic and rule-based (no LLM judging): reference values
from independent scipy/numpy computations, regex-based interpretation
rubrics, and explicit invalidity matchers from the ground truth.
"""
from __future__ import annotations

import math
import re
from typing import Any

from statagent.benchmark.tasks import TaskSpec

SYSTEM_NAMES = ["A_llm_only", "B_llm_tools", "C_llm_tools_validation"]


def _get_path(d: dict[str, Any], path: Any) -> Any:
    if isinstance(path, str):
        return d.get(path)
    node: Any = d
    for key in path:
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and isinstance(key, int) and 0 <= key < len(node):
            node = node[key]
        else:
            return None
    return node


def extract_answer(task: TaskSpec, outcome: dict[str, Any], key: str) -> float | None:
    """Pull a metric value out of a system outcome for a ground-truth key."""
    if outcome.get("system") == "A_llm_only":
        answers = outcome.get("answers") or {}
        # Accept a few common aliases (p_value ~ p).
        aliases = {"p": ["p", "p_value", "pvalue"], key: [key, key + "_value"]}
        candidates = aliases.get(key, [key])
        for cand in candidates:
            if cand in answers and isinstance(answers[cand], (int, float)):
                return float(answers[cand])
        return None
    spec = task.ground_truth.extract.get(key)
    if spec is None:
        return None
    alternatives = spec if isinstance(spec, list) and isinstance(spec[0], (list, tuple)) else [spec]
    for tool, field in alternatives:
        for step in outcome.get("step_results") or []:
            if step.get("tool") != tool or not step.get("ok"):
                continue
            value = _get_path(step.get("summary") or {}, field)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
                return float(value[0])
    return None


def numeric_correct(task: TaskSpec, outcome: dict[str, Any]) -> tuple[float, int]:
    """Fraction of reference answer keys matched within tolerance."""
    truth = task.ground_truth.answers
    if not truth:
        return 1.0, 0
    ok = 0
    for key, expected in truth.items():
        got = extract_answer(task, outcome, key)
        if got is None:
            continue
        if math.isclose(got, expected, rel_tol=0.02, abs_tol=1e-3):
            ok += 1
    return ok / len(truth), len(truth)


def method_correct(task: TaskSpec, outcome: dict[str, Any]) -> bool | None:
    """Whether the system's primary tool is the expected one (B/C only)."""
    if outcome.get("system") == "A_llm_only":
        return None
    selected = outcome.get("method_selected")
    if not selected:
        return None
    if selected in task.ground_truth.allowed_methods:
        # Param-level traps: e.g. pearson where spearman is required.
        for tool, params in task.ground_truth.invalid_matchers:
            if selected == tool and all(
                outcome.get("method_params", {}).get(k) == v for k, v in params.items()
            ):
                return False
        return True
    return False


def invalid_matcher_hit(task: TaskSpec, outcome: dict[str, Any]) -> bool:
    """Did the system execute an analysis the ground truth marks invalid?"""
    if outcome.get("system") == "A_llm_only":
        from statagent.validation.rules_interp import has_causal_claim

        text = outcome.get("interpretation", "")
        for pattern in task.ground_truth.forbidden_patterns:
            if pattern == "CAUSAL_CLAIM":
                if has_causal_claim(text):
                    return True
            elif re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    selected = outcome.get("method_selected")
    if not selected:
        return False
    for tool, params in task.ground_truth.invalid_matchers:
        if selected == tool and all(
            outcome.get("method_params", {}).get(k) == v for k, v in params.items()
        ):
            return True
    return False


def _item_satisfied(item: Any, text: str) -> bool:
    """A rubric item matches if any of its alternative patterns match."""
    alternatives = item if isinstance(item, list) else [item]
    return any(re.search(p, text, re.IGNORECASE) for p in alternatives)


def interpretation_score(task: TaskSpec, outcome: dict[str, Any]) -> float:
    from statagent.validation.rules_interp import has_causal_claim

    text = outcome.get("interpretation") or ""
    items: list[Any] = list(task.ground_truth.required_points)
    items += [f"!{p}" if p != "CAUSAL_CLAIM" else "!CAUSAL_CLAIM" for p in task.ground_truth.forbidden_patterns]
    if not items:
        return 1.0
    ok = 0
    for item in items:
        if item == "!CAUSAL_CLAIM":
            if not has_causal_claim(text):
                ok += 1
        elif isinstance(item, str) and item.startswith("!"):
            if not re.search(item[1:], text, re.IGNORECASE):
                ok += 1
        elif _item_satisfied(item, text):
            ok += 1
    return ok / len(items)


def score_task(task: TaskSpec, outcome: dict[str, Any]) -> dict[str, Any]:
    """Per-task scores for one system outcome."""
    status = outcome.get("status")
    completed = status == "completed"
    clarified = bool(outcome.get("clarifications_asked"))
    invalid_hit = invalid_matcher_hit(task, outcome)
    flagged = bool(outcome.get("flagged"))
    numeric, n_keys = numeric_correct(task, outcome) if completed else (0.0, len(task.ground_truth.answers))
    return {
        "task_id": task.id,
        "category": task.category,
        "completed": completed,
        "status": status,
        "method_correct": method_correct(task, outcome),
        "method_selected": outcome.get("method_selected"),
        "numeric_accuracy": numeric,
        "n_answer_keys": n_keys,
        "interpretation_accuracy": interpretation_score(task, outcome) if completed else 0.0,
        "invalid_analysis": invalid_hit,
        "invalid_unflagged": invalid_hit and not flagged,
        "clarification_asked": clarified,
        "timing_s": outcome.get("timing_s"),
        "usage": outcome.get("usage") or {},
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(scores: dict[str, list[dict[str, Any]]], tasks: list[TaskSpec]) -> dict[str, Any]:
    """Aggregate metrics per system over the scored tasks."""
    summary: dict[str, Any] = {}
    for system, rows in scores.items():
        completed = [r for r in rows if r["completed"]]
        n = len(rows)
        # Method selection accuracy: tasks where a method is expected and the
        # system has a defined selection (B/C only).
        method_vals = [r["method_correct"] for r in rows if r["method_correct"] is not None]
        # Invalid analysis rate: only tasks that carry invalid-matcher traps.
        trap_rows = [
            r
            for r, t in zip(rows, tasks)
            if t.ground_truth.invalid_matchers or t.ground_truth.forbidden_patterns
        ]
        # Clarification rate: only tasks that require clarification.
        id_rows = [r for r, t in zip(rows, tasks) if t.ground_truth.must_clarify]
        numeric = [r["numeric_accuracy"] for r in completed]
        interp = [r["interpretation_accuracy"] for r in completed]

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        n_correct_tasks = sum(1 for r in rows if r["completed"] and r["numeric_accuracy"] == 1.0 and r["numeric_accuracy"] > 0)
        cost = sum((r.get("usage") or {}).get("cost_usd", 0.0) for r in rows if isinstance(r.get("usage"), dict))
        tokens_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
        tokens_out = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
        timing = [r["timing_s"] for r in rows if isinstance(r.get("timing_s"), (int, float))]

        k_completed = len(completed)
        k_method = sum(1 for v in method_vals if v)
        k_clarified = sum(1 for r in id_rows if r["clarification_asked"])
        k_invalid_unflagged = sum(1 for r in trap_rows if r["invalid_unflagged"])

        summary[system] = {
            "task_completion_rate": k_completed / max(n, 1),
            "task_completion_ci": wilson_ci(k_completed, n),
            "method_selection_accuracy": _mean(method_vals) if method_vals else None,
            "method_selection_ci": wilson_ci(k_method, len(method_vals)) if method_vals else None,
            "numerical_accuracy": _mean(numeric),
            "interpretation_accuracy": _mean(interp),
            "appropriate_clarification_rate": k_clarified / max(len(id_rows), 1),
            "clarification_ci": wilson_ci(k_clarified, len(id_rows)) if id_rows else None,
            "invalid_analysis_rate": k_invalid_unflagged / max(len(trap_rows), 1),
            "invalid_analysis_ci": wilson_ci(k_invalid_unflagged, len(trap_rows)) if trap_rows else None,
            "n_tasks": n,
            "n_completed": k_completed,
            "n_correct_tasks": n_correct_tasks,
            "total_time_s": round(sum(timing), 2),
            "mean_time_s": round(_mean(timing), 3),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "cost_per_correct_task_usd": round(cost / n_correct_tasks, 6) if n_correct_tasks else None,
        }
    # Paired comparison between B and C on shared binary outcomes.
    if "B_llm_tools" in scores and "C_llm_tools_validation" in scores:
        summary["paired_comparison"] = paired_comparisons(
            scores["B_llm_tools"], scores["C_llm_tools_validation"], tasks
        )
    return summary


def mcnemar(b: int, c: int) -> dict[str, Any]:
    """McNemar's test on discordant pairs (continuity-corrected)."""
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    from scipy import stats as sps

    p = float(sps.chi2.sf(stat, 1))
    return {"b": b, "c": c, "chi2": round(stat, 3), "p": round(p, 4)}


def paired_comparisons(
    rows_b: list[dict[str, Any]],
    rows_c: list[dict[str, Any]],
    tasks: list[TaskSpec],
) -> dict[str, Any]:
    """Per-task paired differences between B and C."""
    by_id_b = {r["task_id"]: r for r in rows_b}
    by_id_c = {r["task_id"]: r for r in rows_c}
    out: dict[str, Any] = {}
    for metric in ("completed", "method_correct", "invalid_unflagged", "clarification_asked"):
        b = c = 0
        for task in tasks:
            rb, rc = by_id_b.get(task.id), by_id_c.get(task.id)
            if rb is None or rc is None:
                continue
            vb = bool(rb[metric]) if metric != "method_correct" else bool(rb["method_correct"])
            vc = bool(rc[metric]) if metric != "method_correct" else bool(rc["method_correct"])
            if metric == "method_correct":
                if rb["method_correct"] is None or rc["method_correct"] is None:
                    continue
                vb, vc = bool(rb["method_correct"]), bool(rc["method_correct"])
            if vb and not vc:
                b += 1  # B yes, C no
            elif not vb and vc:
                c += 1  # B no, C yes
        out[metric] = mcnemar(b, c)
    return out


def category_breakdown(scores: dict[str, list[dict[str, Any]]], tasks: list[TaskSpec]) -> dict[str, Any]:
    """Per-category means for the headline metrics."""
    categories = sorted({t.category for t in tasks})
    out: dict[str, Any] = {}
    for cat in categories:
        idx = [i for i, t in enumerate(tasks) if t.category == cat]
        out[cat] = {}
        for system, rows in scores.items():
            sub = [rows[i] for i in idx]
            method_vals = [r["method_correct"] for r in sub if r["method_correct"] is not None]
            comp = [r for r in sub if r["completed"]]
            out[cat][system] = {
                "n": len(sub),
                "completion": sum(1 for r in sub if r["completed"]) / max(len(sub), 1),
                "method_accuracy": sum(method_vals) / len(method_vals) if method_vals else None,
                "numeric_accuracy": sum(r["numeric_accuracy"] for r in comp) / len(comp) if comp else 0.0,
                "interpretation_accuracy": sum(r["interpretation_accuracy"] for r in comp) / len(comp) if comp else 0.0,
                "invalid_unflagged_rate": sum(1 for r in sub if r["invalid_unflagged"]) / max(len(sub), 1),
            }
    return out
