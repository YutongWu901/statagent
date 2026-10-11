"""The interpretation layer: computed results -> honest natural language.

The LLM is only allowed to paraphrase numbers that appear in the computed
results; the INTERP validation rules then verify this independently.
Offline mode uses a deterministic template that quotes the actual numbers.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from statagent.agent.llm import DeepSeekClient, LLMError, MockLLM
from statagent.models.plan import AnalysisPlan
from statagent.models.results import Interpretation, StepResult, ValidationReport

INTERP_SYSTEM_PROMPT = """You write the natural-language interpretation of a
completed statistical analysis. The analysis was computed by a Python engine;
you have NOT computed anything and must not add or change any number.

Hard rules:
1. Quote ONLY numbers that appear in the provided results. If a number is
   not in the results, do not write it. Invented numbers invalidate the report.
2. Association is not causation. For correlation/regression on observational
   data, state explicitly that the analysis measures association.
3. Statistical significance is not practical importance. Mention effect
   sizes (Cohen's d, r, Cramer's V, R²) when they are in the results.
4. A non-significant result (p > 0.05) is NOT evidence of no effect; say the
   evidence is insufficient, and report the confidence interval when given.
5. Describe confidence intervals correctly: they are a property of the
   procedure, not a probability statement about the parameter.
6. Distinguish statistical conclusions from study limitations. Incorporate
   the validation findings listed below.
7. Write in the same language as the research question.

Respond with a single JSON object matching:
{"summary": str, "key_findings": [str, ...], "interpretation": str,
 "limitations": [str, ...], "conclusion": str}
"""


def results_digest(step_results: list[StepResult]) -> str:
    lines = []
    for step in step_results:
        r = step.result
        if not r.ok:
            lines.append(f"Step {step.step_id} ({r.tool}) FAILED: {r.error}")
            continue
        lines.append(
            f"Step {step.step_id} ({r.tool}): summary={json.dumps(r.summary, ensure_ascii=False)}"
        )
        for note in r.notes:
            lines.append(f"  note: {note}")
    return "\n".join(lines) or "(no results)"


class InterpreterAgent:
    def __init__(self, client: DeepSeekClient | MockLLM | None = None) -> None:
        self.client = client

    def interpret(
        self,
        question: str,
        plan: AnalysisPlan,
        step_results: list[StepResult],
        validation_before: ValidationReport | None,
    ) -> Interpretation:
        if self.client is not None:
            try:
                return self._llm_interpret(question, plan, step_results, validation_before)
            except (LLMError, ValidationError):
                pass  # fall through to the honest template
        return self._template_interpret(question, plan, step_results, validation_before)

    def _llm_interpret(
        self,
        question: str,
        plan: AnalysisPlan,
        step_results: list[StepResult],
        validation_before: ValidationReport | None,
    ) -> Interpretation:
        user = "\n\n".join(
            [
                f"Research question: {question}",
                f"Analysis goal: {plan.goal}",
                f"Methods used: {', '.join(s.tool for s in plan.steps)}",
                f"Computed results:\n{results_digest(step_results)}",
                "Validation findings before execution:\n"
                + (validation_before.llm_context() if validation_before else "(none)"),
            ]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": INTERP_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        data = self.client.complete_json(messages)
        interp = Interpretation(**data)
        interp.text = "\n".join(
            filter(
                None,
                [
                    interp.summary,
                    *interp.key_findings,
                    interp.interpretation,
                    *interp.limitations,
                    interp.conclusion,
                ],
            )
        )
        return interp

    def _template_interpret(
        self,
        question: str,
        plan: AnalysisPlan,
        step_results: list[StepResult],
        validation_before: ValidationReport | None,
    ) -> Interpretation:
        """Deterministic offline interpretation built from real results."""
        key_findings: list[str] = []
        min_n = None
        for step in step_results:
            r = step.result
            if not r.ok:
                key_findings.append(f"Step {step.step_id} ({r.tool}) failed: {r.error}")
                continue
            if r.n is not None:
                min_n = r.n if min_n is None else min(min_n, r.n)
            s = r.summary
            line = self._finding_line(r.tool, s)
            if line:
                key_findings.append(line)
            for note in r.notes:
                key_findings.append(note)
        limitations = [
            "Offline mode: interpretation generated by a deterministic template "
            "from the computed results, not by a language model.",
        ]
        if min_n is not None and min_n < 30:
            limitations.append(
                f"Small sample (n = {min_n}): results are preliminary and "
                "should be interpreted with caution."
            )
        if validation_before is not None:
            for f in validation_before.findings:
                if f.status in ("WARNING", "FAIL"):
                    limitations.append(f"{f.rule_id}: {f.message}")
        summary = (
            f"Analysis of \"{question}\" ({plan.goal}) using "
            + (", ".join(dict.fromkeys(s.tool for s in plan.steps)) or "no tools")
            + "."
        )
        interpretation = (
            "The reported statistics were computed by the Python engine from the "
            "data. Association does not imply causation; observational data cannot "
            "establish causal effects. A non-significant result means the evidence "
            "is insufficient to detect an effect, not that no effect exists; "
            "confidence intervals show the range of effect sizes compatible with "
            "the data."
        )
        conclusion = "See the key findings and limitations above."
        text = "\n".join(
            [summary, *key_findings, interpretation, *limitations, conclusion]
        )
        return Interpretation(
            summary=summary,
            key_findings=key_findings,
            interpretation=interpretation,
            limitations=limitations,
            conclusion=conclusion,
            text=text,
        )

    @staticmethod
    def _finding_line(tool: str, s: dict) -> str | None:
        """One plain-language line per tool from its real summary numbers."""
        p = s.get("p_value")

        def p_text() -> str:
            if not isinstance(p, (int, float)):
                return ""
            if p < 0.05:
                return f"p = {p:.4g}, statistically significant at the 5% level"
            return (
                f"p = {p:.4g}: not statistically significant at the 5% level — "
                "insufficient evidence to detect an effect"
            )

        if tool == "correlation_test":
            ci = s.get("ci95_r")
            return (
                f"{s.get('method', 'correlation').capitalize()} correlation "
                f"r = {s['r']:.3f}"
                + (f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                + f", r² = {s.get('r_squared', s['r'] ** 2):.3f}, {p_text()} (n = {s.get('n')})."
            )
        if tool == "ttest_one_sample":
            ci = s.get("ci95_mean")
            return (
                f"Sample mean = {s['mean']:.3f}"
                + (f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                + f" vs mu0 = {s.get('mu0')}, t({s['df']}) = {s['t']:.3f}, "
                f"{p_text()}, Cohen's d = {s.get('cohens_d', float('nan')):.3f} (n = {s.get('n')})."
            )
        if tool in ("ttest_two_sample", "ttest_paired"):
            ci = s.get("ci95_mean_diff")
            d = s.get("cohens_d")
            effect = f", Cohen's d = {d:.3f}" if isinstance(d, (int, float)) else ""
            return (
                f"{s.get('test', 't-test')}: mean difference = {s['mean_diff']:.3f}"
                + (f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                + f", t = {s['t']:.3f}, {p_text()}{effect}"
                + f" ({s.get('group_a', '')} vs {s.get('group_b', '')}, n = {s.get('n')})."
            )
        if tool == "mann_whitney_u":
            return (
                f"Mann-Whitney U = {s['u']:.1f}, rank-biserial r = "
                f"{s.get('rank_biserial_r', float('nan')):.3f}, {p_text()} "
                f"(medians: {s.get('median_a')} vs {s.get('median_b')}; the test "
                "compares stochastic dominance, not specifically medians)."
            )
        if tool == "chi_square_test":
            return (
                f"Chi-square test of independence: χ²({s['df']}) = {s['chi2']:.3f}, "
                f"{p_text()}, Cramer's V = {s.get('cramers_v', float('nan')):.3f} (n = {s.get('n')})."
            )
        if tool == "linear_regression":
            coefs = ", ".join(
                f"{k} = {v['coefficient']:.3f} (p = {v['p_value']:.3g})"
                for k, v in (s.get("coefficients") or {}).items()
                if k != "Intercept"
            )
            return (
                f"OLS regression: R² = {s['r_squared']:.3f}, adjusted R² = "
                f"{s.get('adj_r_squared', float('nan')):.3f}, F({s.get('k', 0) - 1}) = "
                f"{s.get('f_stat', float('nan')):.3f} ({p_text()}); coefficients: {coefs}."
            )
        if tool == "describe_variable":
            return (
                f"{s.get('count')} valid observations; mean = {s['mean']:.3f}, "
                f"median = {s['median']:.3f}, SD = {s['std']:.3f}, "
                f"95% CI of the mean [{s['ci95_mean'][0]:.3f}, {s['ci95_mean'][1]:.3f}]."
            )
        if tool == "bootstrap_ci":
            ci = s.get("ci95")
            return (
                f"Bootstrap estimate {s.get('statistic')} = {s['estimate']:.3f}, "
                f"95% percentile CI [{ci[0]:.3f}, {ci[1]:.3f}] "
                f"({s.get('n_boot')} resamples, seed {s.get('seed')})."
            )
        return None
