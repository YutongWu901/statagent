"""The planning layer: question + data profile -> validated AnalysisPlan.

Live mode uses DeepSeek with JSON-mode output, a strict JSON Schema, a
Pydantic validation loop (with self-repair retries) and a deterministic
post-check that only registered tools with valid parameters survive.
Offline mode uses a rule-based heuristic planner instead of the LLM; the
statistical computation is identical in both modes.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from statagent.agent.llm import DeepSeekClient, LLMError, MockLLM
from statagent.models.dataset import DatasetProfile, VariableType
from statagent.models.plan import (
    AnalysisPlan,
    AnalysisStep,
    ClarificationQuestion,
    StudyDesign,
)
from statagent.stats.engine import ToolRegistry

PLAN_JSON_SCHEMA = json.dumps(AnalysisPlan.model_json_schema(), ensure_ascii=False)

SYSTEM_PROMPT = f"""You are a senior statistician designing an analysis plan for a
statistical agent. You plan; you never compute. The plan you return contains
NO numbers, NO results, and NO conclusions — only goals, design facts, tool
calls with parameters, and clarifications.

Hard rules:
1. Choose tools ONLY from the registered tool list below. Never invent a tool
   or write code. Every step must be one of the registered tools.
2. "dataset_id" in every step must be exactly the id given in the dataset
   description. Column names must match the profile exactly (case-sensitive).
3. Fill the "design" facts with ONLY what the question states or clearly
   implies. Anything else stays "unknown" or null — never silently assume.
   In particular: if the question compares two groups and the data might be
   paired or repeated (before/after, matched), set sample_type to "unknown"
   and ask a clarification instead of guessing "independent".
4. If the question asks about causation ("causes", "effect of", "影响",
   "导致") and the data are not described as a randomized experiment, set
   study_design to "observational" (or "unknown") and causal_question=true;
   add a limitation that observational data cannot establish causation.
5. Method-variable matching:
   - numeric continuous vs numeric -> Pearson correlation / t-test / regression
   - ordinal, skewed, or outlier-heavy -> Spearman / Mann-Whitney U
   - categorical x categorical -> chi_square_test
   - categorical predictors in regression: list their names in "categorical"
   - small samples or non-normal data -> nonparametric or bootstrap
6. Add a "clarifications" entry ONLY when the answer changes which method is
   correct. Each entry: key (one of sample_type, study_design,
   causal_question, confounding_addressed), a plain-language question, 2-4
   options, and "why" explaining what depends on the answer.
7. Prefer robust methods (Welch t-test, Spearman, Mann-Whitney) when
   assumptions are in doubt. Add diagnostics steps (check_normality) when
   relevant.
8. Multiple hypothesis tests on the same data should be flagged in
   "limitations" (multiple comparisons).

Registered tools (name, description, parameters JSON Schema):
{json.dumps(ToolRegistry.llm_tool_list(), ensure_ascii=False, indent=2)}

Respond with a single JSON object that validates against this JSON Schema:
{PLAN_JSON_SCHEMA}
Example step: {{"step_id": 1, "tool": "correlation_test",
"parameters": {{"dataset_id": "ds1", "x": "study_hours", "y": "exam_score",
"method": "pearson"}}, "variables": ["study_hours", "exam_score"],
"rationale": "Assess linear association between study time and score."}}
"""


class PlanGenerationError(RuntimeError):
    pass


def _sanitize_plan(plan: AnalysisPlan) -> AnalysisPlan:
    """Deterministic safety net: keep only registered tools with valid params."""
    kept: list[AnalysisStep] = []
    for step in plan.steps:
        tool = ToolRegistry.get(step.tool)
        if tool is None:
            plan.limitations.append(f"Step {step.step_id}: dropped — unknown tool '{step.tool}'.")
            continue
        try:
            tool.param_model(**step.parameters)
        except ValidationError as exc:
            plan.limitations.append(
                f"Step {step.step_id}: dropped — invalid parameters for "
                f"'{step.tool}' ({exc})."
            )
            continue
        kept.append(step)
    plan.steps = kept
    return plan


class PlannerAgent:
    def __init__(
        self,
        client: DeepSeekClient | MockLLM | None = None,
        offline_fallback: bool = True,
    ) -> None:
        self.client = client
        self.offline_fallback = offline_fallback
        self.heuristic = HeuristicPlanner()
        self.last_error: str | None = None

    def plan(
        self,
        question: str,
        profile: DatasetProfile,
        design_answers: dict[str, Any] | None = None,
        prior_plan: AnalysisPlan | None = None,
    ) -> AnalysisPlan:
        self.last_error = None
        if self.client is not None:
            try:
                return self._llm_plan(question, profile, design_answers, prior_plan)
            except (LLMError, PlanGenerationError) as exc:
                self.last_error = str(exc)
                if not self.offline_fallback:
                    raise
                plan = self.heuristic.plan(question, profile, design_answers)
                plan.limitations.append(f"LLM planning failed ({exc}); heuristic fallback used.")
                return plan
        if not self.offline_fallback:
            raise PlanGenerationError(
                "No API key configured and offline fallback disabled."
            )
        return self.heuristic.plan(question, profile, design_answers)

    def _llm_plan(
        self,
        question: str,
        profile: DatasetProfile,
        design_answers: dict[str, Any] | None,
        prior_plan: AnalysisPlan | None,
    ) -> AnalysisPlan:
        user_parts = [
            f"Dataset:\n{profile.llm_context()}",
            f"Research question: {question}",
        ]
        if design_answers:
            user_parts.append(
                "User-provided design facts (from clarifications): "
                + json.dumps(design_answers, ensure_ascii=False)
            )
        if prior_plan is not None:
            user_parts.append(
                "You previously asked clarifications. Incorporate the answers "
                "and return the full updated plan."
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        last_error: str | None = None
        for attempt in range(3):
            if last_error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON did not validate. Fix these errors "
                            f"and return only the corrected JSON:\n{last_error}"
                        ),
                    }
                )
            try:
                data = self.client.complete_json(messages)
            except LLMError as exc:
                if attempt == 2:
                    raise PlanGenerationError(f"LLM returned invalid JSON 3 times: {exc}") from exc
                last_error = str(exc)
                continue
            try:
                plan = AnalysisPlan(**data)
            except ValidationError as exc:
                last_error = str(exc)
                continue
            return _sanitize_plan(plan)
        raise PlanGenerationError(f"Plan validation failed repeatedly: {last_error}")


CAUSAL_WORDS = (
    "caus", "effect of", "leads to", "lead to", "drives", "due to", "impact of",
    "increase", "decrease", "improve", "affect", "导致", "引起", "影响", "因果", "提升", "降低", "改善",
)
EXPERIMENT_WORDS = ("randomiz", "experiment", "实验", "随机", "干预", "treatment group")
OBSERVATIONAL_WORDS = ("observational", "survey", "non-random", "观察", "调查")
COMPARE_WORDS = ("compare", "differ", "difference", "higher", "lower", "more than",
                 "better", "vs", "versus", "比较", "差异", "更高", "更低", "显著高于", "是否高于", "提高", "优于", "更好")
RELATION_WORDS = ("correlat", "relat", "associat", "相关", "关系", "关联", "联系")
REGRESSION_WORDS = ("predict", "regression", "regress", "controlling for", "adjust",
                    "after controlling", "预测", "回归", "控制", "调整")
DESCRIBE_WORDS = ("average", "mean", "median", "describe", "summarize", "summary",
                  "distribution", "均值", "中位数", "分布", "描述")
PAIRED_HINT_WORDS = ("before", "after", "pre-test", "post-test", "前测", "后测", "前后", "培训后", "治疗后", "improvement after")


#: Lightweight synonym expansion for column-name matching: a QUESTION word
#: maps to the column tokens it can refer to ("study time" -> hours).
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("hours", "hour"),
    "grade": ("score", "result"),
    "result": ("score", "grade"),
    "income": ("salary", "earnings"),
    "salary": ("income",),
    "earnings": ("income", "salary"),
}


def _column_tokens(name: str) -> list[str]:
    tokens = [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", name.lower()) if len(t) >= 2]
    if not tokens and len(name) == 1:
        tokens = [name]
    return tokens


def _question_number(question: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", question)
    return float(m.group(1)) if m else None


class HeuristicPlanner:
    """Deterministic rule-based planner for offline mode.

    Maps question patterns (explicit method names first, then keywords and
    variable types) to registered tools. It never invents numbers and never
    runs code; execution goes through the strict engine either way.
    """

    def plan(
        self,
        question: str,
        profile: DatasetProfile,
        design_answers: dict[str, Any] | None = None,
    ) -> AnalysisPlan:
        q = question.lower()
        design = StudyDesign(notes="Offline heuristic inference.")
        design = design.fill(design_answers or {})

        numeric = profile.numeric_columns
        categorical = profile.categorical_columns

        def mentioned(cols):
            out = []
            q_words = re.findall(r"[a-z\u4e00-\u9fff]+", q)
            for c in cols:
                name = c.name.lower()
                name_sp = name.replace("_", " ")
                if name_sp in q:
                    out.append(c)
                    continue
                tokens = _column_tokens(name)
                hit = any(t in q for t in tokens) or (
                    len(name) == 1 and re.search(rf"\b{re.escape(name)}\b", q)
                )
                if not hit:
                    # Synonym matching: "study time" ~ a column named hours.
                    hit = any(
                        t in _COLUMN_ALIASES.get(qw, ())
                        for qw in q_words
                        for t in tokens
                    )
                if not hit:
                    # Prefix matching: satisfied ~ satisfaction, ratings ~ rating.
                    for t in tokens:
                        if any(
                            (len(t) >= 5 and qw.startswith(t[:5]))
                            or (len(qw) >= 5 and t.startswith(qw[:5]))
                            for qw in q_words
                        ):
                            hit = True
                            break
                if hit:
                    out.append(c)
            return out

        mn = mentioned(numeric)
        mc = mentioned(categorical)
        # A categorical column is also "mentioned" when its level values appear.
        for c in categorical:
            if c in mc:
                continue
            for lv, _ in c.top_values:
                lv_s = str(lv).lower()
                if (len(lv_s) >= 2 and lv_s in q) or (
                    len(lv_s) == 1 and re.search(rf"\b{re.escape(lv_s)}\b", q)
                ):
                    mc.append(c)
                    break

        causal = any(w in q for w in CAUSAL_WORDS)
        if design.causal_question is None:
            design.causal_question = causal
        if design.study_design == "unknown":
            if any(w in q for w in EXPERIMENT_WORDS):
                design.study_design = "experimental"
            elif any(w in q for w in OBSERVATIONAL_WORDS):
                design.study_design = "observational"
        if design.causal_question and design.confounding_addressed is None:
            design.confounding_addressed = False

        steps: list[AnalysisStep] = []
        clarifications: list[ClarificationQuestion] = []
        sid = 0
        limitations: list[str] = []
        goal = "No analysis steps could be determined"

        def add(tool: str, params: dict, rationale: str, variables: list[str]) -> None:
            nonlocal sid
            sid += 1
            params = {"dataset_id": profile.dataset_id, **params}
            steps.append(
                AnalysisStep(step_id=sid, tool=tool, parameters=params,
                             rationale=rationale, variables=variables)
            )

        def ask(key: str, question_text: str, options: list[str], why: str) -> None:
            clarifications.append(
                ClarificationQuestion(question=question_text, key=key, options=options, why=why)
            )

        group_col = next((c for c in mc if c.n_unique >= 2), None)
        ordinal_cols = [c for c in mc if c.variable_type == VariableType.CATEGORICAL_ORDINAL]
        num_col = mn[0] if mn else (ordinal_cols[0] if ordinal_cols else (numeric[0] if numeric else None))
        skewed = bool(
            num_col and num_col.numeric and abs(num_col.numeric.skewness or 0) > 1
        )
        small_n = profile.n_rows < 30

        def group_pair(col) -> tuple[str, str] | None:
            """The two group levels to compare: levels mentioned in the
            question first, otherwise the two most frequent levels."""
            levels = [str(lv) for lv, _ in col.top_values]
            mentioned_levels = [lv for lv in levels if lv.lower() in q]
            if len(mentioned_levels) >= 2:
                return sorted(mentioned_levels)[0], sorted(mentioned_levels)[1]
            if len(levels) >= 2:
                return sorted(levels)[0], sorted(levels)[1]
            return None

        wants_compare = any(w in q for w in COMPARE_WORDS)
        wants_relation = any(w in q for w in RELATION_WORDS)
        wants_regression = any(w in q for w in REGRESSION_WORDS)
        wants_describe = any(w in q for w in DESCRIBE_WORDS)
        mu0 = _question_number(question)

        # ---- explicit method names ----------------------------------------
        explicit = None
        if re.search(r"chi[- ]square|卡方", q):
            explicit = "chi_square_test"
        elif re.search(r"mann[- ]?whitney|秩和", q):
            explicit = "mann_whitney_u"
        elif re.search(r"one[- ]sample|single[- ]sample|单样本", q) and re.search(r"t[- ]?test|t检验", q):
            explicit = "ttest_one_sample"
        elif re.search(r"paired|配对", q) and re.search(r"t[- ]?test|t检验", q):
            explicit = "ttest_paired"
        elif re.search(r"\bt[- ]?test|welch|\bt检验", q):
            explicit = "ttest_two_sample"
        elif re.search(r"\bpearson\b", q):
            explicit = "correlation_pearson"
        elif re.search(r"\bspearman\b", q):
            explicit = "correlation_spearman"
        elif re.search(r"regress|regression|OLS|最小二乘", q):
            explicit = "linear_regression"
        elif re.search(r"bootstrap|自助", q):
            explicit = "bootstrap_ci"

        if explicit == "chi_square_test":
            cand = [c for c in mc if c.n_unique >= 2]
            if len(cand) >= 2:
                a, b = cand[0].name, cand[1].name
                add("chi_square_test", {"column_a": a, "column_b": b},
                    f"Test independence of {a} and {b}.", [a, b])
                goal = f"Chi-square test of {a} × {b}"
            elif len(mc) >= 1 and len(mn) >= 1:
                # Chi-square requested on a continuous variable: inappropriate.
                # Compare the continuous variable across the categorical groups.
                g, v = mc[0], mn[0]
                ga, gb = sorted([g.top_values[0][0], g.top_values[1][0]])
                tool = "mann_whitney_u" if (v.variable_type == VariableType.CATEGORICAL_ORDINAL or skewed) else "ttest_two_sample"
                add(tool,
                    {"column": v.name, "group_column": g.name, "group_a": ga, "group_b": gb},
                    f"Chi-square is inappropriate for continuous '{v.name}'; compare "
                    f"it between groups of '{g.name}' instead.", [v.name, g.name])
                limitations.append(
                    f"A chi-square test was requested, but '{v.name}' is continuous; "
                    f"a group comparison ({tool}) is used instead."
                )
                goal = f"Group comparison of {v.name} by {g.name} (chi-square inappropriate)"
        elif explicit == "mann_whitney_u":
            if num_col and group_col:
                a, b = group_pair(group_col)
                add("mann_whitney_u",
                    {"column": num_col.name, "group_column": group_col.name, "group_a": a, "group_b": b},
                    f"Mann-Whitney U on {num_col.name} between {a} and {b}.",
                    [num_col.name, group_col.name])
                goal = f"Mann-Whitney U: {num_col.name} by {group_col.name}"
        elif explicit == "ttest_one_sample" and num_col and mu0 is not None:
            add("ttest_one_sample", {"column": num_col.name, "mu0": mu0},
                f"One-sample t-test of {num_col.name} against {mu0}.", [num_col.name])
            goal = f"One-sample t-test of {num_col.name} vs {mu0}"
        elif explicit == "ttest_paired" and len(mn) >= 2:
            a_col, b_col = mn[0].name, mn[1].name
            add("ttest_paired", {"column_a": a_col, "column_b": b_col},
                f"Paired t-test of {a_col} vs {b_col}.", [a_col, b_col])
            goal = f"Paired t-test: {a_col} vs {b_col}"
        elif explicit == "ttest_two_sample" and num_col and group_col:
            a, b = group_pair(group_col)
            if design.sample_type == "unknown":
                ask("sample_type",
                    f"Are the two groups in '{group_col.name}' independent samples, "
                    "or paired/repeated measurements?",
                    ["independent", "paired"],
                    "The correct test differs: Welch t-test for independent "
                    "samples vs paired t-test for matched data.")
            if num_col.variable_type == VariableType.CATEGORICAL_ORDINAL or skewed:
                add("mann_whitney_u",
                    {"column": num_col.name, "group_column": group_col.name, "group_a": a, "group_b": b},
                    f"{num_col.name} is ordinal/skewed, so a Mann-Whitney U test is "
                    f"used instead of the requested t-test ({a} vs {b}).",
                    [num_col.name, group_col.name])
                limitations.append(
                    "A t-test was requested but the outcome is ordinal or heavily "
                    "skewed; the nonparametric Mann-Whitney U test is used instead."
                )
                goal = f"Mann-Whitney U: {num_col.name} by {group_col.name} (instead of t-test)"
            else:
                add("ttest_two_sample",
                    {"column": num_col.name, "group_column": group_col.name, "group_a": a, "group_b": b},
                    f"Two-sample t-test of {num_col.name} between {a} and {b}.",
                    [num_col.name, group_col.name])
                goal = f"Two-sample t-test: {num_col.name} by {group_col.name}"
        elif explicit == "correlation_pearson":
            if len(mn) >= 2:
                pair = [mn[0].name, mn[1].name]
            elif len(mc) >= 2:
                pair = [mc[0].name, mc[1].name]
            else:
                pair = [c.name for c in numeric[:2]]
            if len(pair) == 2:
                add("correlation_test", {"x": pair[0], "y": pair[1], "method": "pearson"},
                    f"Pearson correlation of {pair[0]} and {pair[1]}.", pair)
                goal = f"Pearson correlation: {pair[0]} vs {pair[1]}"
        elif explicit == "correlation_spearman":
            pair = ([ordinal_cols[0].name if ordinal_cols else None]
                    if ordinal_cols else []) + [c.name for c in mn][:2]
            pair = [p for p in pair if p][:2]
            if len(pair) < 2 and len(numeric) >= 2:
                pair = [numeric[0].name, numeric[1].name]
            if len(pair) == 2:
                add("correlation_test", {"x": pair[0], "y": pair[1], "method": "spearman"},
                    f"Spearman correlation of {pair[0]} and {pair[1]}.", pair)
                goal = f"Spearman correlation: {pair[0]} vs {pair[1]}"
        elif explicit == "linear_regression":
            marker = next((m for m in (" on ", " from ", " using ") if m in q), None)
            if marker:
                before, after = q.split(marker)
                preds = [c for c in mn if any(t in after for t in _column_tokens(c.name))]
                outcome = next((c for c in mn if c not in preds), None)
            else:
                outcome, preds = (mn[0] if mn else None), mn[1:]
            if outcome and preds:
                add("linear_regression", {"y": outcome.name, "x": [c.name for c in preds]},
                    f"Regress {outcome.name} on {', '.join(c.name for c in preds)}.",
                    [outcome.name] + [c.name for c in preds])
                goal = f"Regression: {outcome.name} ~ {' + '.join(c.name for c in preds)}"
        elif explicit == "bootstrap_ci" and num_col:
            add("bootstrap_ci", {"column": num_col.name, "statistic": "mean", "seed": 42, "n_boot": 2000},
                f"Bootstrap CI for the mean of {num_col.name}.", [num_col.name])
            goal = f"Bootstrap CI for mean({num_col.name})"

        # ---- pattern branches ----------------------------------------------
        if not steps:
            if small_n and (wants_compare or explicit in ("ttest_two_sample", "mann_whitney_u", "ttest_paired", "ttest_one_sample")):
                if design.sample_type == "unknown" and group_col:
                    ask("sample_type",
                        f"Are the two groups in '{group_col.name}' independent "
                        "samples, or paired/repeated measurements?",
                        ["independent", "paired"],
                        "This determines the correct test — though with so few "
                        "rows, inference is unreliable either way.")
                if num_col:
                    add("describe_variable", {"column": num_col.name},
                        f"Summarize {num_col.name} (sample too small for inference).",
                        [num_col.name])
                    limitations.append(
                        f"Only {profile.n_rows} rows: inferential tests would be "
                        "unreliable; descriptive statistics reported instead."
                    )
                    goal = f"Descriptive summary of {num_col.name} (small sample)"
            elif group_col and group_col.variable_type != VariableType.CATEGORICAL_ORDINAL and (
                wants_compare or (causal and mn)
            ) and num_col:
                a, b = group_pair(group_col)
                if design.sample_type == "unknown":
                    ask("sample_type",
                        f"Are the two groups in '{group_col.name}' independent samples, "
                        "or paired/repeated measurements?",
                        ["independent", "paired"],
                        "The correct test differs: Welch t-test for independent "
                        "samples vs paired t-test for matched data.")
                if design.sample_type == "paired":
                    add("ttest_two_sample",
                        {"column": num_col.name, "group_column": group_col.name,
                         "group_a": a, "group_b": b, "paired": True},
                        f"Compare {num_col.name} between {a} and {b} (paired).",
                        [num_col.name, group_col.name])
                else:
                    tool = "mann_whitney_u" if (
                        num_col.variable_type == VariableType.CATEGORICAL_ORDINAL or skewed
                    ) else "ttest_two_sample"
                    add(tool,
                        {"column": num_col.name, "group_column": group_col.name,
                         "group_a": a, "group_b": b},
                        f"Compare {num_col.name} between {a} and {b}.",
                        [num_col.name, group_col.name])
                goal = f"Compare {num_col.name} between groups of {group_col.name}"
            elif (wants_regression or design.causal_question is True) and len(mn) >= 2:
                marker = next((m for m in (" from ", " using ", " with ", "根据", "用", " on ") if m in q), None)
                if marker:
                    preds = [c for c in mn if any(t in q.split(marker)[1] for t in _column_tokens(c.name))]
                    outcome = next((c for c in mn if c not in preds), None)
                elif design.causal_question is True:
                    verb_pos = min((q.find(w) for w in CAUSAL_WORDS if w in q), default=-1)
                    after = q[verb_pos + 1 :] if verb_pos >= 0 else ""
                    outcome = next((c for c in mn if any(t in after for t in _column_tokens(c.name))), None)
                    preds = [c for c in mn if c != outcome]
                else:
                    outcome, preds = mn[0], mn[1:]
                if outcome is None:
                    outcome, preds = num_col, [c for c in mn if c != num_col]
                if design.causal_question is None and not design_answers and "predict" in q:
                    ask("causal_question",
                        "Is this a causal question (does X cause Y) or an association / prediction question?",
                        ["association", "causal"],
                        "Causal claims require an experiment or confounder adjustment.")
                xs = [c.name for c in preds if c.name != outcome.name]
                add("linear_regression", {"y": outcome.name, "x": xs},
                    f"Model {outcome.name} as a function of {', '.join(xs)}.",
                    [outcome.name, *xs])
                goal = f"Regression of {outcome.name} on {', '.join(xs)}"
            elif len(mn) >= 2 and (
                design.sample_type == "paired"
                or any(w in q for w in PAIRED_HINT_WORDS)
                or any(w in question for w in ("improve after", "increase after", "提升后", "培训后"))
            ):
                a_col, b_col = mn[0].name, mn[1].name
                for c in mn:
                    lname = c.name.lower()
                    if "before" in lname or lname.startswith("pre") or "前" in lname:
                        a_col = c.name
                    if "after" in lname or lname.startswith("post") or "后" in lname:
                        b_col = c.name
                design.sample_type = "paired"
                add("ttest_paired", {"column_a": a_col, "column_b": b_col},
                    f"Paired comparison of {a_col} vs {b_col}.", [a_col, b_col])
                goal = f"Paired comparison of {a_col} and {b_col}"
            elif len(mn) < 2 and any(w in q for w in PAIRED_HINT_WORDS):
                # Before/after implied by the question even though both column
                # names are not literally mentioned.
                a_c = next((c for c in numeric if "before" in c.name.lower() or c.name.lower().startswith("pre")), None)
                b_c = next((c for c in numeric if "after" in c.name.lower() or c.name.lower().startswith("post")), None)
                if a_c and b_c:
                    if design.sample_type == "unknown":
                        ask("sample_type",
                            "Are the two measurements paired (same units measured "
                            "twice) or independent samples?",
                            ["paired", "independent"],
                            "Paired measurements need a paired t-test.")
                    if design.sample_type != "independent":
                        design.sample_type = "paired"
                        design.repeated_measures = True
                        add("ttest_paired", {"column_a": a_c.name, "column_b": b_c.name},
                            f"Paired comparison of {a_c.name} vs {b_c.name}.",
                            [a_c.name, b_c.name])
                        goal = f"Paired comparison of {a_c.name} and {b_c.name}"
            elif len(mn) >= 2 and wants_compare:
                a_col, b_col = mn[0].name, mn[1].name
                if design.sample_type == "unknown":
                    ask("sample_type",
                        f"Comparing '{a_col}' and '{b_col}': are they independent "
                        "samples, or paired/repeated measurements on the same units?",
                        ["independent", "paired"],
                        "Independent columns need a two-sample test; paired "
                        "measurements need a paired t-test.")
                if design.sample_type == "paired":
                    design.repeated_measures = True
                    add("ttest_paired", {"column_a": a_col, "column_b": b_col},
                        f"Paired comparison of {a_col} vs {b_col}.", [a_col, b_col])
                    goal = f"Paired comparison of {a_col} and {b_col}"
                else:
                    add("describe_variable", {"column": a_col}, f"Summarize {a_col}.", [a_col])
                    add("describe_variable", {"column": b_col}, f"Summarize {b_col}.", [b_col])
                    goal = (f"Descriptive comparison of {a_col} and {b_col} (no "
                            "grouping column for a two-sample test)")
            elif wants_compare and not mn and len(numeric) >= 2 and any(
                w in q for w in ("first", "second", "two column", "measurement", "第一", "第二", "两个")
            ):
                # Two columns compared without naming them (e.g. "first vs second").
                a_c, b_c = numeric[0], numeric[1]
                if design.sample_type == "unknown":
                    ask("sample_type",
                        f"Comparing '{a_c.name}' and '{b_c.name}': are they "
                        "independent samples or paired measurements?",
                        ["independent", "paired"],
                        "The correct test depends on whether the same units were "
                        "measured twice.")
                if design.sample_type == "paired":
                    design.repeated_measures = True
                    add("ttest_paired", {"column_a": a_c.name, "column_b": b_c.name},
                        f"Paired comparison of {a_c.name} vs {b_c.name}.", [a_c.name, b_c.name])
                    goal = f"Paired comparison of {a_c.name} and {b_c.name}"
                else:
                    add("describe_variable", {"column": a_c.name}, f"Summarize {a_c.name}.", [a_c.name])
                    add("describe_variable", {"column": b_c.name}, f"Summarize {b_c.name}.", [b_c.name])
                    goal = f"Descriptive comparison of {a_c.name} and {b_c.name}"
            elif wants_relation and len(mn) >= 2:
                x, y = mn[0].name, mn[1].name
                pair_skewed = any(
                    abs((c.numeric.skewness or 0) if c.numeric else 0) > 1 for c in mn[:2]
                )
                method = "spearman" if pair_skewed else "pearson"
                add("correlation_test", {"x": x, "y": y, "method": method},
                    f"Assess association between {x} and {y}.", [x, y])
                goal = f"Association between {x} and {y}"
            elif (wants_relation or wants_regression or design.causal_question is True) and ordinal_cols and len(mn) >= 1:
                x, y = ordinal_cols[0].name, mn[0].name
                if "predict" in q and design.causal_question is None and not design_answers:
                    ask("causal_question",
                        "Is this a causal question (does X cause Y) or an "
                        "association / prediction question?",
                        ["association", "causal"],
                        "Causal claims require an experiment or confounder adjustment.")
                add("correlation_test", {"x": x, "y": y, "method": "spearman"},
                    f"Rank association between ordinal {x} and {y}.", [x, y])
                goal = f"Rank association between {x} and {y}"
            elif (wants_relation or design.causal_question is True) and not ordinal_cols and len(mc) == 1 and len(mn) >= 1 and mc[0].n_unique >= 2:
                # Association between one categorical and one numeric variable:
                # compare the numeric across the categorical groups.
                g, v = mc[0], mn[0]
                ga, gb = group_pair(g)
                if design.sample_type == "unknown":
                    ask("sample_type",
                        f"Are the groups in '{g.name}' independent samples, or "
                        "paired/repeated measurements?",
                        ["independent", "paired"],
                        "The correct test differs: Welch t-test for independent "
                        "samples vs paired t-test for matched data.")
                tool = "mann_whitney_u" if (v.variable_type == VariableType.CATEGORICAL_ORDINAL or skewed) else "ttest_two_sample"
                add(tool,
                    {"column": v.name, "group_column": g.name, "group_a": ga, "group_b": gb},
                    f"Compare {v.name} between groups of {g.name}.", [v.name, g.name])
                goal = f"Association of {v.name} with {g.name}"
            elif (wants_relation or design.causal_question is True) and not ordinal_cols and len(mc) < 2 and len(numeric) == 2 and mn:
                # Association question with exactly one name-matched numeric
                # column (synonyms like "study time" ~ hours may hide the
                # other one). With exactly two numeric columns available,
                # associate them — but only when the question already matched
                # at least one column, so unrelated questions about variables
                # that don't exist still fail loudly.
                # Association question but fewer than two columns were matched
                # by name (synonyms like "study time" ~ hours). With exactly
                # two numeric columns available, associate them.
                x, y = numeric[0].name, numeric[1].name
                pair_skewed = any(
                    abs((c.numeric.skewness or 0) if c.numeric else 0) > 1 for c in numeric
                )
                method = "spearman" if pair_skewed else "pearson"
                add("correlation_test", {"x": x, "y": y, "method": method},
                    f"Assess association between {x} and {y} (the only two "
                    "numeric columns in the dataset).", [x, y])
                goal = f"Association between {x} and {y}"
            elif len(mc) >= 2 and (wants_relation or causal or "是否" in question):
                add("chi_square_test", {"column_a": mc[0].name, "column_b": mc[1].name},
                    f"Test independence of {mc[0].name} and {mc[1].name}.",
                    [mc[0].name, mc[1].name])
                goal = f"Independence of {mc[0].name} and {mc[1].name}"
            elif wants_describe and mu0 is not None and num_col and any(
                w in q for w in ("different from", "differs from", "compared to", "不同于", "相比")
            ):
                colp = num_col
                has_outliers = (colp.n_outliers or 0) > 0 or skewed
                if has_outliers:
                    add("bootstrap_ci", {"column": colp.name, "statistic": "mean", "seed": 42, "n_boot": 2000},
                        f"Bootstrap CI for the mean of {colp.name} (outliers make a "
                        "t-test unreliable).", [colp.name])
                    limitations.append(
                        f"'{colp.name}' contains outliers; a one-sample t-test would "
                        "be distorted, so a bootstrap CI is reported instead."
                    )
                    goal = f"Bootstrap CI for mean({colp.name})"
                else:
                    add("ttest_one_sample", {"column": colp.name, "mu0": mu0},
                        f"One-sample t-test of {colp.name} against {mu0}.", [colp.name])
                    goal = f"One-sample t-test of {colp.name} vs {mu0}"
            elif not mn and not mc and (
                wants_relation or wants_compare or wants_regression or causal or explicit
            ):
                # The question clearly asks for an analysis, but no column
                # could be matched. Fail loudly instead of silently
                # summarizing an unrelated column.
                limitations.append(
                    "Could not find any columns matching the variables in the "
                    f"question. Available columns: {', '.join(profile.names()) or '(none)'}."
                )
                goal = "No matching variables found"
            elif wants_describe and num_col:
                add("describe_variable", {"column": num_col.name},
                    f"Summarize {num_col.name}.", [num_col.name])
                goal = f"Summary of {num_col.name}"
            elif num_col:
                add("describe_variable", {"column": num_col.name},
                    f"Summarize {num_col.name}.", [num_col.name])
                goal = f"Exploratory summary of {num_col.name}"

        plan = AnalysisPlan(
            goal=goal,
            design=design,
            steps=steps,
            clarifications=clarifications,
            method_rationale=(
                "Rule-based offline planner: matched explicit method names, "
                "keywords and variable types; no LLM involved."
            ),
            limitations=limitations or (
                [] if clarifications else
                ["Offline heuristic planning is limited; supply an API key for full AI planning."]
            ),
        )
        return _sanitize_plan(plan)
