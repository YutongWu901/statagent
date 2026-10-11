"""Analysis-plan schemas: what the LLM planner produces and the executor
consumes. Plans reference *registered* statistical tools only; execution is
strictly validated against these models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SampleType = Literal["independent", "paired", "single", "unknown"]
DesignType = Literal["experimental", "observational", "unknown"]


class StudyDesign(BaseModel):
    """Research-design facts about the question under analysis.

    Values are only filled when inferable from the question or stated by the
    user; anything else stays ``unknown``/``None`` and triggers a
    clarification request instead of a silent assumption.
    """

    sample_type: SampleType = "unknown"
    study_design: DesignType = "unknown"
    causal_question: bool | None = None
    confounding_addressed: bool | None = None
    repeated_measures: bool | None = None
    notes: str = ""

    def unknowns(self) -> list[str]:
        """Design fields that block reliable analysis while unknown."""
        out: list[str] = []
        if self.sample_type == "unknown":
            out.append("sample_type")
        if self.study_design == "unknown":
            out.append("study_design")
        if self.causal_question is None:
            out.append("causal_question")
        if self.causal_question is True and self.confounding_addressed is None:
            out.append("confounding_addressed")
        return out

    def fill(self, answers: dict[str, Any]) -> "StudyDesign":
        """Fill unknown fields from clarification answers."""
        data = self.model_dump()
        for key, value in answers.items():
            if key in data and data[key] in ("unknown", None):
                data[key] = value
        return StudyDesign(**data)


class ClarificationQuestion(BaseModel):
    question: str
    key: str
    options: list[str] = Field(default_factory=list)
    why: str = ""


class AnalysisStep(BaseModel):
    step_id: int
    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    variables: list[str] = Field(default_factory=list)
    rationale: str = ""


class AnalysisPlan(BaseModel):
    goal: str
    design: StudyDesign = Field(default_factory=StudyDesign)
    steps: list[AnalysisStep] = Field(default_factory=list)
    clarifications: list[ClarificationQuestion] = Field(default_factory=list)
    method_rationale: str = ""
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "AnalysisPlan":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        return self

    def needs_clarification(self) -> bool:
        """True when missing design facts actually block a correct method choice.

        Only the facts that change the analysis count: sample structure for
        group comparisons, and study type / confounding for causal questions.
        """
        if self.clarifications:
            return True
        comparison_tools = {"ttest_two_sample", "ttest_paired", "mann_whitney_u"}
        has_comparison = any(s.tool in comparison_tools for s in self.steps)
        if has_comparison and self.design.sample_type == "unknown":
            return True
        if self.design.causal_question is True:
            if self.design.study_design == "unknown":
                return True
            if self.design.confounding_addressed is None:
                return True
        return False


#: Synthetic clarification questions for design facts the model did not ask about.
DESIGN_QUESTION_BANK: dict[str, ClarificationQuestion] = {
    "sample_type": ClarificationQuestion(
        question="Are the two compared groups independent samples, or paired / "
        "repeated measurements on the same units?",
        key="sample_type",
        options=["independent", "paired"],
        why="The correct test differs: Welch t-test for independent samples, "
        "paired t-test for matched data.",
    ),
    "study_design": ClarificationQuestion(
        question="Does the data come from a randomized experiment or from an "
        "observational study?",
        key="study_design",
        options=["experimental", "observational"],
        why="Causal conclusions require an experimental design (or strong "
        "identification assumptions).",
    ),
    "causal_question": ClarificationQuestion(
        question="Is your question about a causal effect, or about an "
        "association?",
        key="causal_question",
        options=["causal", "association"],
        why="Observational data can support associations but not causal claims.",
    ),
    "confounding_addressed": ClarificationQuestion(
        question="Have potential confounders been measured and controlled for?",
        key="confounding_addressed",
        options=["yes", "no"],
        why="Unadjusted confounders can produce spurious associations.",
    ),
}


def design_answer_value(key: str, answer: str):
    """Normalize a clarification answer (option text) to a design-field value."""
    a = str(answer).strip().lower()
    if key == "causal_question":
        return a in ("causal", "true", "yes", "是", "因果")
    if key == "confounding_addressed":
        return a in ("yes", "true", "是", "已控制", "controlled")
    if key == "sample_type":
        return "paired" if a in ("paired", "配对", "配对样本", "repeated") else "independent"
    if key == "study_design":
        return "experimental" if a in ("experimental", "实验", "随机") else "observational"
    return answer
