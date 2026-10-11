"""The statistical validation engine.

An independent, rule-based system that judges whether an analysis is
reasonable. It is deliberately NOT another LLM pass: every check is an
explicit, testable rule over the data profile, study design, plan, computed
results and interpretation text. Findings are PASS / WARNING / FAIL /
UNKNOWN with a rule id, evidence and a suggested action.
"""
from __future__ import annotations

from statagent.models.results import ValidationReport
from statagent.validation.context import ValidationContext
from statagent.validation.rules_data import DATA_RULES, Rule
from statagent.validation.rules_design import DESIGN_RULES
from statagent.validation.rules_interp import INTERP_RULES
from statagent.validation.rules_method import METHOD_RULES

ALL_RULES: list[Rule] = DATA_RULES + DESIGN_RULES + METHOD_RULES + INTERP_RULES

PRE_EXECUTION_CATEGORIES = ("DATA", "DESIGN", "METHOD")
INTERPRETATION_CATEGORIES = ("INTERP",)


class ValidationEngine:
    """Runs rule checks and gates execution on FAIL / UNKNOWN findings."""

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else ALL_RULES

    def _run(self, ctx: ValidationContext, categories: tuple[str, ...]) -> ValidationReport:
        findings = []
        for rule in self.rules:
            if rule.category not in categories:
                continue
            try:
                finding = rule.check(ctx)
            except Exception:  # noqa: BLE001 - a broken rule must not break analysis
                finding = ctx.finding(
                    rule.rule_id, rule.category, "UNKNOWN",
                    f"Rule {rule.rule_id} failed to evaluate.",
                    suggestion="Report this as a bug in the validation engine.",
                )
            if finding is not None:
                findings.append(finding)
        report = ValidationReport(findings=findings, gated=True)
        blocking = report.blocking_failures()
        unknown_design = [
            f for f in findings
            if f.status == "UNKNOWN" and f.category == "DESIGN"
        ]
        report.blocked = bool(blocking) or bool(unknown_design)
        return report

    def validate_before_execution(self, ctx: ValidationContext) -> ValidationReport:
        """Data / design / method checks that run before computing anything."""
        return self._run(ctx, PRE_EXECUTION_CATEGORIES)

    def validate_interpretation(self, ctx: ValidationContext) -> ValidationReport:
        """Checks of the natural-language interpretation, after execution."""
        return self._run(ctx, INTERPRETATION_CATEGORIES)
