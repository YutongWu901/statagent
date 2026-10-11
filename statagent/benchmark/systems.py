"""The three benchmark systems.

- System A (LLM only): the model answers with its own numbers, no tools.
- System B (LLM + tools): planner + strict execution engine, no validator.
- System C (LLM + tools + validation): B plus the independent rule-based
  validation engine, including the clarification loop.

All three run against the same tasks and store raw outcomes for audit.
"""
from __future__ import annotations

import time
from typing import Any

from statagent.agent.llm import DeepSeekClient, LLMError, MockLLM
from statagent.agent.pipeline import AnalysisPipeline
from statagent.benchmark.tasks import TaskSpec
from statagent.config import AppConfig
from statagent.data.store import DatasetStore
from statagent.models.dataset import DatasetProfile
from statagent.utils import to_jsonable

SYSTEM_A_PROMPT = """You are asked to answer a statistical question about a dataset
DIRECTLY, without any tools or computation. Answer from your own reasoning
only. Return a single JSON object:
{"answers": {"<metric_key>": <number>, ...}, "interpretation": "..."}
Metric keys should be short (e.g. r, p, t, mean_diff, mean, chi2, u,
r_squared, coef_x). The interpretation must be written in natural language
and must include your statistical reasoning and any caveats.
"""


def _assumed_design_answers(task: TaskSpec) -> dict[str, Any]:
    """System B's fallback assumptions when design facts are unknown.

    Deliberately naive defaults: B has no clarification mechanism, so it
    proceeds on common-but-unjustified assumptions. The ground truth is
    NOT leaked to B — the contrast with System C (which asks) is the point.
    """
    return {
        "sample_type": "independent",
        "study_design": "observational",
        "causal_question": False,
        "confounding_addressed": False,
    }


class SystemA:
    """LLM only: numbers come from the model's head."""

    name = "A_llm_only"
    description = "LLM answers directly, no tools, no validation."

    def __init__(self, client: DeepSeekClient | MockLLM) -> None:
        self.client = client

    def run(
        self,
        task: TaskSpec,
        store: DatasetStore,
        profile: DatasetProfile,
        dataset_id: str,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": SYSTEM_A_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Dataset summary:\n{profile.llm_context()}\n\n"
                    f"Question: {task.question}\n"
                    "Provide your best numeric answers and interpretation."
                ),
            },
        ]
        try:
            data = self.client.complete_json(messages)
        except LLMError as exc:
            return {
                "task_id": task.id,
                "system": self.name,
                "status": "error",
                "error": str(exc),
                "timing_s": round(time.perf_counter() - t0, 3),
                "clarifications_asked": [],
                "answers": {},
                "interpretation": "",
            }
        answers = data.get("answers") or {}
        if not isinstance(answers, dict):
            answers = {"_raw": answers}
        return {
            "task_id": task.id,
            "system": self.name,
            "status": "completed" if data.get("interpretation") else "error",
            "error": None if data.get("interpretation") else "no interpretation field",
            "timing_s": round(time.perf_counter() - t0, 3),
            "clarifications_asked": [],
            "answers": {str(k): (float(v) if isinstance(v, (int, float)) else None) for k, v in answers.items()},
            "interpretation": str(data.get("interpretation", "")),
            "raw": to_jsonable(data),
        }


class SystemB:
    """LLM + tools, no validation gate."""

    name = "B_llm_tools"
    description = "LLM plans, registered tools compute; no validator, no clarification loop."

    def __init__(self, config: AppConfig, client: DeepSeekClient | MockLLM | None) -> None:
        self.pipeline = AnalysisPipeline(config, client=client, use_validation=False)

    def run(
        self,
        task: TaskSpec,
        store: DatasetStore,
        profile: DatasetProfile,
        dataset_id: str,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        asked: list[str] = []
        out = self.pipeline.analyze(task.question, store, dataset_id)
        if out.status == "clarification_needed":
            asked = [q.key for q in out.questions]
            # No clarification mechanism: proceed with the design facts as
            # stated in the ground truth (best-case assumption for B).
            answers = _assumed_design_answers(task)
            out = self.pipeline.analyze(
                task.question, store, dataset_id, design_answers=answers
            )
        timing = round(time.perf_counter() - t0, 3)
        usage = self.pipeline.usage.to_dict()
        if out.status == "completed" and out.record is not None:
            record = out.record
            first = record.step_results[0] if record.step_results else None
            return {
                "task_id": task.id,
                "system": self.name,
                "status": "completed",
                "timing_s": timing,
                "usage": usage,
                "clarifications_asked": asked,
                "method_selected": first.tool if first else None,
                "method_params": first.result.parameters if first else {},
                "step_results": [
                    {
                        "tool": s.tool,
                        "ok": s.result.ok,
                        "summary": s.result.summary,
                        "notes": s.result.notes,
                        "error": s.result.error,
                    }
                    for s in record.step_results
                ],
                "interpretation": record.interpretation.text if record.interpretation else "",
                "flagged": False,
                "error": None,
            }
        return {
            "task_id": task.id,
            "system": self.name,
            "status": out.status,
            "timing_s": timing,
            "usage": usage,
            "clarifications_asked": asked,
            "error": out.error or (out.validation.llm_context() if out.validation else None),
            "interpretation": "",
        }


class SystemC:
    """LLM + tools + independent rule-based validation."""

    name = "C_llm_tools_validation"
    description = "System B plus the validation engine and the clarification loop."

    def __init__(self, config: AppConfig, client: DeepSeekClient | MockLLM | None) -> None:
        self.pipeline = AnalysisPipeline(config, client=client, use_validation=True)

    def run(
        self,
        task: TaskSpec,
        store: DatasetStore,
        profile: DatasetProfile,
        dataset_id: str,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        asked: list[str] = []
        out = self.pipeline.analyze(task.question, store, dataset_id)
        if out.status == "clarification_needed":
            asked = [q.key for q in out.questions]
            # Simulate a user who answers with the true design facts.
            answers = task.ground_truth.design
            out = self.pipeline.analyze(
                task.question, store, dataset_id, design_answers=answers
            )
        timing = round(time.perf_counter() - t0, 3)
        usage = self.pipeline.usage.to_dict()

        if out.status == "blocked":
            return {
                "task_id": task.id,
                "system": self.name,
                "status": "blocked",
                "timing_s": timing,
                "usage": usage,
                "clarifications_asked": asked,
                "flagged": True,
                "validation": [
                    {"rule_id": f.rule_id, "status": f.status, "message": f.message}
                    for f in (out.validation.findings if out.validation else [])
                ],
                "error": None,
                "interpretation": "",
            }
        if out.status == "completed" and out.record is not None:
            record = out.record
            first = record.step_results[0] if record.step_results else None
            findings_before = (
                [
                    {"rule_id": f.rule_id, "status": f.status}
                    for f in record.validation_before.findings
                ]
                if record.validation_before
                else []
            )
            findings_after = (
                [
                    {"rule_id": f.rule_id, "status": f.status}
                    for f in record.validation_after.findings
                ]
                if record.validation_after
                else []
            )
            all_findings = findings_before + findings_after
            flagged = any(f["status"] in ("FAIL", "WARNING") for f in all_findings)
            return {
                "task_id": task.id,
                "system": self.name,
                "status": "completed",
                "timing_s": timing,
                "usage": usage,
                "clarifications_asked": asked,
                "method_selected": first.tool if first else None,
                "method_params": first.result.parameters if first else {},
                "step_results": [
                    {
                        "tool": s.tool,
                        "ok": s.result.ok,
                        "summary": s.result.summary,
                        "notes": s.result.notes,
                        "error": s.result.error,
                    }
                    for s in record.step_results
                ],
                "interpretation": record.interpretation.text if record.interpretation else "",
                "flagged": flagged,
                "validation": all_findings,
                "error": None,
            }
        return {
            "task_id": task.id,
            "system": self.name,
            "status": out.status,
            "timing_s": timing,
            "usage": usage,
            "clarifications_asked": asked,
            "flagged": False,
            "error": out.error,
            "interpretation": "",
        }


def build_systems(
    config: AppConfig, client: DeepSeekClient | MockLLM | None
) -> list:
    """A needs an LLM; B/C fall back to the offline planner when client is None."""
    systems: list = []
    if client is not None:
        systems.append(SystemA(client))
    systems.append(SystemB(config, client))
    systems.append(SystemC(config, client))
    return systems
