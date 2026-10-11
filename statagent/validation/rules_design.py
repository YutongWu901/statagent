"""Research-design validation rules (DESIGN category).

These rules guard against silent assumptions about the study design. When
critical design facts are unknown, the system asks the user instead of
guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from statagent.models.results import Finding
from statagent.validation.context import ValidationContext

COMPARISON_TOOLS = {"ttest_two_sample", "mann_whitney_u", "ttest_paired"}
PAIRED_TOOLS = {"ttest_paired"}
CAUSAL_VERBS = (
    "caus", "effect of", "leads to", "drives", "due to", "impact of", "increases",
    "decreases", "improves", "导致", "引起", "影响", "因果", "提升", "降低", "改善",
)


@dataclass
class Rule:
    rule_id: str
    category: str
    title: str
    description: str
    check: Callable[[ValidationContext], Finding | None]


def _plan_tools(ctx: ValidationContext) -> list[str]:
    if ctx.plan is None:
        return []
    return [s.tool for s in ctx.plan.steps]


def _uses_comparison(ctx: ValidationContext) -> bool:
    return any(t in COMPARISON_TOOLS for t in _plan_tools(ctx))


def _uses_paired(ctx: ValidationContext) -> bool:
    tools = _plan_tools(ctx)
    if "ttest_paired" in tools:
        return True
    if ctx.plan is not None:
        for s in ctx.plan.steps:
            if s.tool == "ttest_two_sample" and s.parameters.get("paired"):
                return True
    return False


def _rule_design_01_unknown_sample_type(ctx: ValidationContext) -> Finding | None:
    if ctx.clarifications_answered:
        return None
    if ctx.design.sample_type != "unknown":
        return None
    if _uses_comparison(ctx):
        return ctx.finding(
            "DESIGN-01", "DESIGN", "UNKNOWN",
            "The study design is unknown: are the compared samples independent, "
            "paired/matched, or repeated measures on the same units?",
            "sample_type is unknown and the plan includes group-comparison tests.",
            "Answer the clarification question about the sampling design before "
            "running tests; the correct test depends on it.",
        )
    return None


def _rule_design_02_unknown_study_type(ctx: ValidationContext) -> Finding | None:
    if ctx.clarifications_answered:
        return None
    if ctx.design.study_design != "unknown":
        return None
    q = ctx.question.lower()
    if any(v in q for v in CAUSAL_VERBS):
        return ctx.finding(
            "DESIGN-02", "DESIGN", "UNKNOWN",
            "The question suggests a causal claim but the study design "
            "(experiment vs observation) is unknown.",
            f"Question: {ctx.question!r}.",
            "State whether the data come from a randomized experiment or an "
            "observational study; causal conclusions require experimental "
            "design or strong identification assumptions.",
        )
    return None


def _rule_design_03_paired_mismatch(ctx: ValidationContext) -> Finding | None:
    if ctx.design.sample_type == "independent" and _uses_paired(ctx):
        return ctx.finding(
            "DESIGN-03", "DESIGN", "FAIL",
            "A paired test is planned but the design states independent samples.",
            "sample_type=independent with a paired tool selected.",
            "Use an independent-samples test (Welch t-test or Mann-Whitney U), "
            "or correct the design facts if the data are actually paired.",
        )
    return None


def _rule_design_04_independent_mismatch(ctx: ValidationContext) -> Finding | None:
    tools = _plan_tools(ctx)
    indep = [t for t in tools if t in {"ttest_two_sample", "mann_whitney_u"}]
    paired_flag = False
    if ctx.plan is not None:
        paired_flag = any(
            s.tool == "ttest_two_sample" and s.parameters.get("paired")
            for s in ctx.plan.steps
        )
    if ctx.design.sample_type == "paired" and indep and not paired_flag:
        return ctx.finding(
            "DESIGN-04", "DESIGN", "FAIL",
            "Independent-samples tests are planned but the design states paired "
            "or repeated-measures data.",
            f"sample_type=paired with tools {indep}.",
            "Use a paired t-test (or Wilcoxon signed-rank) on the matched pairs.",
        )
    return None


def _rule_design_05_observational_causal(ctx: ValidationContext) -> Finding | None:
    if (
        ctx.design.study_design == "observational"
        and ctx.design.causal_question is True
    ):
        return ctx.finding(
            "DESIGN-05", "DESIGN", "WARNING",
            "The question asks about a causal effect but the data are "
            "observational.",
            "study_design=observational, causal_question=true.",
            "Observational data can only support associations. Either reframe "
            "the question as an association, or control/adjust for measured "
            "confounders and still avoid causal language in the conclusion.",
        )
    return None


def _rule_design_06_confounding(ctx: ValidationContext) -> Finding | None:
    if (
        ctx.design.causal_question is True
        and ctx.design.confounding_addressed is False
    ):
        return ctx.finding(
            "DESIGN-06", "DESIGN", "WARNING",
            "A causal question is asked but confounding has not been addressed.",
            "confounding_addressed=false.",
            "Identify measured confounders and adjust for them (e.g., multiple "
            "regression), or downgrade the claim to an association.",
        )
    return None


DESIGN_RULES: list[Rule] = [
    Rule("DESIGN-01", "DESIGN", "Unknown sample structure", "Asks instead of assuming when the sample design is unknown.", _rule_design_01_unknown_sample_type),
    Rule("DESIGN-02", "DESIGN", "Unknown study type", "Asks whether the study is experimental or observational when the question is causal.", _rule_design_02_unknown_study_type),
    Rule("DESIGN-03", "DESIGN", "Paired/independent mismatch", "Rejects paired tests on independent samples.", _rule_design_03_paired_mismatch),
    Rule("DESIGN-04", "DESIGN", "Independent/paired mismatch", "Rejects independent tests on paired data.", _rule_design_04_independent_mismatch),
    Rule("DESIGN-05", "DESIGN", "Observational causal claim", "Warns against causal questions on observational data.", _rule_design_05_observational_causal),
    Rule("DESIGN-06", "DESIGN", "Confounding", "Warns when confounding is unaddressed in causal questions.", _rule_design_06_confounding),
]
