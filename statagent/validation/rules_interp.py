"""Result-interpretation validation rules (INTERP category).

These rules scan the natural-language interpretation produced after
execution and flag statistical misinterpretations: causal language on
observational data, confusing significance with importance, claiming a
non-significant result proves the null, CI misinterpretations, and numbers
in the text that do not match the computed results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from statagent.models.results import Finding
from statagent.validation.context import ValidationContext

_NUMBER_RE = re.compile(r"(?<![\w.])([-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+)(?![\w.])")

CAUSAL_PATTERNS = [
    r"\bcauses?\b", r"\bcaused\b", r"\bcausing\b", r"\b因果",
    r"\bleads? to\b", r"\bdrives\b", r"\b结果是.{0,6}(导致|造成)\b",
    r"\b导致\b", r"\b引起\b", r"\b使得\b.{0,10}(提高|降低|改善|恶化)\b",
    r"\bincreases\b.{0,20}\b(due to|because)\b",
]
CAUSAL_RE = re.compile("|".join(CAUSAL_PATTERNS), re.IGNORECASE)

NULL_CLAIM_PATTERNS = [
    r"\bproves? (that )?there is no (effect|difference|relationship)\b",
    r"\bno (effect|difference|association) exists\b",
    r"\bconfirms? the null\b",
    r"\b证明.{0,10}(没有|无)(效果|差异|关系|影响)\b",
    r"\b不存在(差异|效果|关系)\b",
]
NULL_CLAIM_RE = re.compile("|".join(NULL_CLAIM_PATTERNS), re.IGNORECASE)

CI_MISUSE_PATTERNS = [
    r"95% (probability|chance) (that|the).{0,40}(parameter|mean|true value)",
    r"probability.{0,30}lies? (in|within) the (interval|CI)",
    r"(参数|真值|均值)有95%的(概率|几率)(落|位于)",
    r"the (true mean|parameter) (is|lies) in this interval",
]
CI_MISUSE_RE = re.compile("|".join(CI_MISUSE_PATTERNS), re.IGNORECASE)

EFFECT_SIZE_TERMS = [
    "cohen", "effect size", "效应量", "rank-biserial", "cramer", "r²", "r2",
    "r-squared", "r squared", "eta", "glass",
]
EFFECT_SIZE_RE = re.compile("|".join(EFFECT_SIZE_TERMS), re.IGNORECASE)

CAVEAT_TERMS = [
    "does not imply causation", "correlation is not causation",
    "cannot establish caus", "no causal", "associat", "相关不等于因果",
    "不能推断因果", "关联", "观察性", "observational",
]
CAVEAT_RE = re.compile("|".join(CAVEAT_TERMS), re.IGNORECASE)

SIGNIFICANT_TERMS = [
    "significant", "statistically significant", "显著", "有统计学意义",
]
SIGNIFICANT_RE = re.compile("|".join(SIGNIFICANT_TERMS), re.IGNORECASE)


@dataclass
class Rule:
    rule_id: str
    category: str
    title: str
    description: str
    check: Callable[[ValidationContext], Finding | None]


def _all_result_numbers(ctx: ValidationContext) -> list[float]:
    """Every number present in the executed tool results."""
    out: list[float] = []

    def walk(obj):
        if isinstance(obj, (int, float, np.integer, np.floating)):
            out.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    for step in ctx.step_results or []:
        walk(step.result.summary)
        walk(step.result.details)
        if step.result.n is not None:
            out.append(float(step.result.n))
    return out


def _text_numbers(text: str) -> list[tuple[float, str]]:
    return [
        (float(m.group(0)), m.group(0))
        for m in _NUMBER_RE.finditer(text)
    ]


def _matches_result(value: float, results: list[float]) -> bool:
    if not results:
        return False
    best = min(abs(value - r) for r in results)
    return best <= 0.01 + 0.05 * abs(value)


def _rule_interp_01_causal(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text:
        return None
    if ctx.design.study_design != "observational":
        return None
    match = CAUSAL_RE.search(ctx.interpretation_text)
    if match:
        return ctx.finding(
            "INTERP-01", "INTERP", "FAIL",
            "The interpretation uses causal language although the design is "
            "observational.",
            f"Matched text: …{ctx.interpretation_text[max(0, match.start() - 40):match.end() + 40]}…",
            "Observational data cannot establish causation. Rephrase in terms "
            "of associations and note that confounding may explain them.",
        )
    return None


def _rule_interp_02_significance_importance(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text:
        return None
    any_significant = False
    for step in ctx.step_results or []:
        p = step.result.summary.get("p_value")
        if isinstance(p, (int, float)) and p < 0.05:
            any_significant = True
    if not any_significant:
        return None
    if SIGNIFICANT_RE.search(ctx.interpretation_text) and not EFFECT_SIZE_RE.search(
        ctx.interpretation_text
    ):
        return ctx.finding(
            "INTERP-02", "INTERP", "WARNING",
            "Statistical significance is reported without any effect-size "
            "measure.",
            "p < 0.05 was obtained but no effect size (Cohen's d, r, Cramer's V, "
            "R²) is mentioned.",
            "A small p-value only means the effect is unlikely to be zero, not "
            "that it is large or practically important. Report the effect size.",
        )
    return None


def _rule_interp_03_null_claim(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text:
        return None
    non_significant = True
    for step in ctx.step_results or []:
        p = step.result.summary.get("p_value")
        if isinstance(p, (int, float)) and p < 0.05:
            non_significant = False
    if not non_significant:
        return None
    match = NULL_CLAIM_RE.search(ctx.interpretation_text)
    if match:
        return ctx.finding(
            "INTERP-03", "INTERP", "FAIL",
            "The interpretation claims the null hypothesis (no effect / no "
            "difference) is proven, but a non-significant result cannot prove "
            "the null.",
            f"Matched text: …{ctx.interpretation_text[max(0, match.start() - 40):match.end() + 40]}…",
            "Say the evidence is insufficient to detect an effect, and report "
            "the confidence interval to show what effect sizes remain plausible.",
        )
    return None


def _rule_interp_04_ci_misuse(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text:
        return None
    match = CI_MISUSE_RE.search(ctx.interpretation_text)
    if match:
        return ctx.finding(
            "INTERP-04", "INTERP", "FAIL",
            "The interpretation misdescribes what a confidence interval means.",
            f"Matched text: …{ctx.interpretation_text[max(0, match.start() - 40):match.end() + 40]}…",
            "A 95% CI is a property of the procedure, not of the parameter: over "
            "repeated sampling, 95% of intervals would contain the true value. "
            "The parameter itself is fixed and either is or is not in the interval.",
        )
    return None


def _rule_interp_05_number_fidelity(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text or not ctx.step_results:
        return None
    results = _all_result_numbers(ctx)
    if not results:
        return None
    mismatches = [
        (text_num, snippet)
        for text_num, snippet in _text_numbers(ctx.interpretation_text)
        if not _matches_result(text_num, results)
    ]
    # Ignore obviously contextual numbers: small integers (n, df), percentages,
    # and standard statistical constants.
    COMMON_CONSTANTS = {0.05, 0.5, 0.95, 0.025, 0.975, 1.96, 0.01, 0.1, 1.0, 2.0}
    filtered = [
        (num, snip)
        for num, snip in mismatches
        if not (
            (num == int(num) and abs(num) <= 1000)
            or num in COMMON_CONSTANTS
        )
    ]
    if filtered:
        examples = ", ".join(snip for _, snip in filtered[:5])
        return ctx.finding(
            "INTERP-05", "INTERP", "FAIL",
            "The interpretation cites numbers that do not appear in the "
            f"computed results: {examples}.",
            "All statistical numbers must come from the Python computation "
            "engine; LLM-invented numbers are rejected.",
            "Regenerate the interpretation and require it to quote only values "
            "present in the provided results.",
        )
    return None


def _rule_interp_06_causation_caveat(ctx: ValidationContext) -> Finding | None:
    if not ctx.interpretation_text:
        return None
    tools = [s.tool for s in ctx.step_results or []]
    is_assoc = any(t in {"correlation_test", "correlation_matrix", "linear_regression"} for t in tools)
    if not is_assoc:
        return None
    question_causal = bool(CAUSAL_RE.search(ctx.question))
    if not question_causal and ctx.design.causal_question is not True:
        return None
    if CAVEAT_RE.search(ctx.interpretation_text):
        return ctx.finding(
            "INTERP-06", "INTERP", "PASS",
            "The interpretation correctly notes that the analysis measures "
            "association, not causation.",
            "Causal caveat found in the interpretation text.",
        )
    return ctx.finding(
        "INTERP-06", "INTERP", "WARNING",
        "The question has a causal flavour but the interpretation does not "
        "explicitly state that association does not imply causation.",
        "No causation caveat found for a correlation/regression analysis of a "
        "causal-sounding question.",
        "Add an explicit statement distinguishing association from causation.",
    )


INTERP_RULES: list[Rule] = [
    Rule("INTERP-01", "INTERP", "Causal language", "Rejects causal language for observational designs.", _rule_interp_01_causal),
    Rule("INTERP-02", "INTERP", "Significance vs importance", "Requires effect sizes alongside significance claims.", _rule_interp_02_significance_importance),
    Rule("INTERP-03", "INTERP", "Non-significance vs no effect", "Rejects claims that p > 0.05 proves the null.", _rule_interp_03_null_claim),
    Rule("INTERP-04", "INTERP", "CI interpretation", "Rejects probability statements about parameters in CIs.", _rule_interp_04_ci_misuse),
    Rule("INTERP-05", "INTERP", "Number fidelity", "Every cited number must match computed results.", _rule_interp_05_number_fidelity),
    Rule("INTERP-06", "INTERP", "Causation caveat", "Requires an association caveat for causal-sounding questions.", _rule_interp_06_causation_caveat),
]
