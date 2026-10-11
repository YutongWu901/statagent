"""Result schemas: tool outputs, validation findings, interpretation text,
and the full analysis record that reports are generated from."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from statagent.models.plan import AnalysisPlan

Status = Literal["PASS", "WARNING", "FAIL", "UNKNOWN"]
Severity = Literal["info", "warning", "critical"]

#: Finding categories whose FAIL status blocks execution unless overridden.
BLOCKING_CATEGORIES = ("DATA", "DESIGN", "METHOD")


class Finding(BaseModel):
    rule_id: str
    category: str
    status: Status
    message: str
    evidence: str = ""
    suggestion: str = ""
    severity: Severity = "warning"


class ValidationReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    gated: bool = False
    blocked: bool = False
    override: bool = False
    override_reason: str = ""

    def counts(self) -> dict[str, int]:
        out = {"PASS": 0, "WARNING": 0, "FAIL": 0, "UNKNOWN": 0}
        for f in self.findings:
            out[f.status] += 1
        return out

    def blocking_failures(self) -> list[Finding]:
        return [
            f
            for f in self.findings
            if f.status == "FAIL" and f.category in BLOCKING_CATEGORIES
        ]

    def llm_context(self) -> str:
        lines = []
        for f in self.findings:
            lines.append(
                f"[{f.status}] {f.rule_id} ({f.category}): {f.message}"
                + (f" Suggestion: {f.suggestion}" if f.suggestion else "")
            )
        return "\n".join(lines) or "(no findings)"


class ToolResult(BaseModel):
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    n: int | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    code: str = ""


class StepResult(BaseModel):
    step_id: int
    tool: str
    result: ToolResult


class Interpretation(BaseModel):
    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    conclusion: str = ""
    #: Combined text scanned by the INTERP validation rules.
    text: str = ""


class ClarificationExchange(BaseModel):
    question: str
    answer: str
    key: str = ""


class AnalysisRecord(BaseModel):
    record_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    question: str
    dataset_id: str
    dataset_source: str = ""
    dataset_sha256: str = ""
    clarifications: list[ClarificationExchange] = Field(default_factory=list)
    plan: AnalysisPlan
    step_results: list[StepResult] = Field(default_factory=list)
    validation_before: ValidationReport | None = None
    interpretation: Interpretation | None = None
    validation_after: ValidationReport | None = None
    engine_version: str = ""
    llm: dict[str, str] = Field(default_factory=dict)
    seed: int = 42

    def to_json(self) -> str:
        from statagent.utils import json_dumps

        return json_dumps(self.model_dump(mode="json"))
