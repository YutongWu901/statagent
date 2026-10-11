"""Validation context: everything a rule needs to inspect.

The validation engine is fully independent of the LLM: it inspects the
dataset profile, the study design facts, the plan, the executed results and
the interpretation text, and applies explicit, testable rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from statagent.data.store import DatasetStore
from statagent.models.dataset import DatasetProfile
from statagent.models.plan import AnalysisPlan, StudyDesign
from statagent.models.results import Finding, StepResult, Status


@dataclass
class ValidationContext:
    question: str = ""
    profile: DatasetProfile | None = None
    store: DatasetStore | None = None
    dataset_id: str | None = None
    design: StudyDesign = field(default_factory=StudyDesign)
    plan: AnalysisPlan | None = None
    step_results: list[StepResult] | None = None
    interpretation_text: str = ""
    clarifications_answered: bool = False

    def data(self) -> pd.DataFrame | None:
        if self.store is None or self.dataset_id is None:
            return None
        if not self.store.has(self.dataset_id):
            return None
        return self.store.get(self.dataset_id)

    def column_profile(self, name: str):
        if self.profile is None:
            return None
        return self.profile.column(name)

    def referenced_columns(self) -> list[str]:
        """All columns referenced by plan step parameters."""
        out: list[str] = []
        if self.plan is None:
            return out
        for step in self.plan.steps:
            p = step.parameters
            for key in ("column", "x", "y", "group_column", "column_a", "column_b"):
                if isinstance(p.get(key), str):
                    out.append(p[key])
            for key in ("columns", "categorical"):
                vals = p.get(key) or []
                if isinstance(vals, list):
                    out.extend(v for v in vals if isinstance(v, str))
        return list(dict.fromkeys(out))

    def finding(
        self,
        rule_id: str,
        category: str,
        status: Status,
        message: str,
        evidence: str = "",
        suggestion: str = "",
    ) -> Finding:
        return Finding(
            rule_id=rule_id,
            category=category,
            status=status,
            message=message,
            evidence=evidence,
            suggestion=suggestion,
        )
