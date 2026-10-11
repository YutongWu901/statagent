"""The analysis pipeline: the full question -> validated report flow.

Orchestrates planner, validation gate, executor, interpreter and the
post-hoc interpretation checks. This is what both the UI and the benchmark
(Systems B and C) run.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from statagent import __version__
from statagent.agent.interpreter import InterpreterAgent
from statagent.agent.llm import UsageTracker, create_client
from statagent.agent.planner import PlannerAgent
from statagent.config import AppConfig, get_config
from statagent.data.store import DatasetStore
from statagent.models.plan import (
    DESIGN_QUESTION_BANK,
    AnalysisPlan,
    ClarificationQuestion,
    design_answer_value,
)
from statagent.models.results import AnalysisRecord, ValidationReport
from statagent.stats.engine import execute_step
from statagent.utils import json_dumps
from statagent.validation.context import ValidationContext
from statagent.validation.engine import ValidationEngine

OutcomeStatus = Literal["completed", "clarification_needed", "blocked", "error"]


@dataclass
class AnalysisOutcome:
    status: OutcomeStatus
    record: AnalysisRecord | None = None
    questions: list[ClarificationQuestion] = field(default_factory=list)
    validation: ValidationReport | None = None
    error: str | None = None
    plan: AnalysisPlan | None = None
    #: Set when live-LLM planning failed and the offline fallback took over.
    llm_warning: str | None = None


class AnalysisPipeline:
    """Runs the end-to-end analysis flow with the validation gate."""

    def __init__(self, config: AppConfig | None = None, client=None, use_validation: bool = True) -> None:
        self.config = config or get_config()
        self.client = (
            client
            if client is not None
            else create_client(self.config.llm, self.config.pricing)
        )
        self.use_validation = use_validation
        self.planner = PlannerAgent(self.client, offline_fallback=self.config.offline_fallback)
        self.interpreter = InterpreterAgent(self.client)
        self.engine = ValidationEngine()

    @property
    def usage(self) -> UsageTracker:
        return self.client.usage if self.client is not None else UsageTracker()

    def llm_meta(self) -> dict[str, str]:
        if self.client is None:
            return {"mode": "offline", "model": "none"}
        return {
            "mode": "live" if self.config.llm.is_configured else "mock",
            "model": self.config.llm.model,
            "temperature": str(self.config.llm.temperature),
            "base_url": self.config.llm.base_url,
        }

    def analyze(
        self,
        question: str,
        store: DatasetStore,
        dataset_id: str,
        design_answers: dict[str, Any] | None = None,
        prior_plan: AnalysisPlan | None = None,
        force: bool = False,
        clarifications: list[dict[str, str]] | None = None,
        save_dir: Path | None = None,
    ) -> AnalysisOutcome:
        profile = store.profile(dataset_id)
        plan = self.planner.plan(question, profile, design_answers, prior_plan)
        llm_warning = getattr(self.planner, "last_error", None)

        if plan.needs_clarification():
            return AnalysisOutcome(
                status="clarification_needed",
                questions=plan.clarifications or self._synthesize_questions(plan),
                plan=plan,
                llm_warning=llm_warning,
            )

        if not plan.steps:
            message = "; ".join(plan.limitations) or (
                "No analysis steps could be planned from the question and the "
                "dataset — check that the question's variables exist in the data."
            )
            return AnalysisOutcome(
                status="error", error=message, plan=plan, llm_warning=llm_warning
            )

        ctx = ValidationContext(
            question=question,
            profile=profile,
            store=store,
            dataset_id=dataset_id,
            design=plan.design,
            plan=plan,
            clarifications_answered=bool(design_answers or clarifications),
        )
        validation_before = None
        if self.use_validation:
            validation_before = self.engine.validate_before_execution(ctx)
            if validation_before.blocked and not force:
                unknown_design = [
                    f
                    for f in validation_before.findings
                    if f.status == "UNKNOWN" and f.category == "DESIGN"
                ]
                if unknown_design and not (design_answers or clarifications):
                    return AnalysisOutcome(
                        status="clarification_needed",
                        questions=self._synthesize_questions(plan),
                        validation=validation_before,
                        plan=plan,
                        llm_warning=llm_warning,
                    )
                return AnalysisOutcome(
                    status="blocked", validation=validation_before, plan=plan,
                    llm_warning=llm_warning,
                )
            if force and validation_before.blocked:
                validation_before.override = True
                validation_before.override_reason = "User chose to proceed despite blocking findings."

        try:
            step_results = [execute_step(store, s) for s in plan.steps]
        except Exception as exc:  # noqa: BLE001 - execute_step already wraps; belt & braces
            return AnalysisOutcome(
                status="error", error=f"{type(exc).__name__}: {exc}",
                llm_warning=llm_warning,
            )

        ctx.step_results = step_results
        interpretation = self.interpreter.interpret(
            question, plan, step_results, validation_before
        )
        ctx.interpretation_text = interpretation.text
        validation_after = None
        if self.use_validation:
            validation_after = self.engine.validate_interpretation(ctx)

        exchanges = []
        if clarifications:
            for item in clarifications:
                exchanges.append(
                    {
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "key": item.get("key", ""),
                    }
                )
        llm_meta = self.llm_meta()
        if llm_warning:
            llm_meta["warning"] = llm_warning
        record = AnalysisRecord(
            record_id=f"rec_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
            question=question,
            dataset_id=dataset_id,
            dataset_source=store.source_name(dataset_id),
            dataset_sha256=profile.sha256,
            clarifications=exchanges,
            plan=plan,
            step_results=step_results,
            validation_before=validation_before,
            interpretation=interpretation,
            validation_after=validation_after,
            engine_version=__version__,
            llm=llm_meta,
            seed=self.config.seed,
        )
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{record.record_id}.json").write_text(
                json_dumps(record.model_dump(mode="json")), encoding="utf-8"
            )
        return AnalysisOutcome(
            status="completed", record=record, validation=validation_before,
            llm_warning=llm_warning,
        )

    @staticmethod
    def _synthesize_questions(plan: AnalysisPlan) -> list[ClarificationQuestion]:
        """Build clarification questions for missing design facts the planner
        did not ask about itself."""
        questions: list[ClarificationQuestion] = []
        for key in plan.design.unknowns():
            q = DESIGN_QUESTION_BANK.get(key)
            if q is not None:
                questions.append(q)
        return questions
