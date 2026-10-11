"""Benchmark task suite: 60 statistical tasks with ground truth.

Every task couples a seeded synthetic dataset with a natural-language
question, the correct method (with acceptable alternatives), reference
answers computed by *independent* code (direct scipy/numpy formulas, not the
analysis engine), interpretation requirements, and trap conditions that make
an analysis invalid. Categories follow the study design:

- method_selection      correct method is unambiguous
- wrong_method          the obvious method is wrong (traps)
- numerical_accuracy    computation must match reference values
- insufficient_design   critical design facts are missing -> must ask
- assumption_violation  naive methods break assumptions
- causal_misuse         causal claims on observational data
- interpretation_error  correct numbers, tricky interpretation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from statagent.data.synthetic import GENERATORS


@dataclass
class GroundTruth:
    method: str = ""                      # expected primary tool
    allowed_methods: list[str] = field(default_factory=list)
    design: dict[str, Any] = field(default_factory=dict)  # true design facts
    answers: dict[str, float] = field(default_factory=dict)  # reference values
    # key -> (tool, summary_field) or list of alternative (tool, field) paths
    extract: dict[str, Any] = field(default_factory=dict)
    required_points: list[str] = field(default_factory=list)  # regexes that must appear
    forbidden_patterns: list[str] = field(default_factory=list)  # regexes that must not appear
    invalid_matchers: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    must_clarify: bool = False


@dataclass
class TaskSpec:
    id: str
    category: str
    title: str
    dataset: str
    question: str
    ground_truth: GroundTruth
    design_hint: str = ""

    def build_dataframe(self) -> pd.DataFrame:
        return GENERATORS[self.dataset]()


# ---------------------------------------------------------------- reference
# Reference computations: independent of the analysis engine.

def ref_corr(df: pd.DataFrame, x: str, y: str, method: str) -> dict[str, float]:
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    res = stats.pearsonr(d[x], d[y]) if method == "pearson" else stats.spearmanr(d[x], d[y])
    return {"r": float(res.statistic), "p": float(res.pvalue)}


def ref_ttest(df: pd.DataFrame, col: str, group: str, a: str, b: str) -> dict[str, float]:
    num = pd.to_numeric(df[col], errors="coerce")
    x = num[df[group].astype(str) == a].dropna().to_numpy(float)
    y = num[df[group].astype(str) == b].dropna().to_numpy(float)
    res = stats.ttest_ind(x, y, equal_var=False)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df_w = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    crit = stats.t.ppf(0.975, df_w)
    diff = float(x.mean() - y.mean())
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "mean_diff": diff,
        "ci_lo": diff - crit * se,
        "ci_hi": diff + crit * se,
    }


def ref_paired(df: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    d = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    res = stats.ttest_rel(d[a], d[b])
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "mean_diff": float((d[a] - d[b]).mean()),
    }


def ref_chi2(df: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    table = pd.crosstab(df[a].astype(str), df[b].astype(str))
    chi2, p, _, _ = stats.chi2_contingency(table.values, correction=False)
    return {"chi2": float(chi2), "p": float(p)}


def ref_mw(df: pd.DataFrame, col: str, group: str, a: str, b: str) -> dict[str, float]:
    num = pd.to_numeric(df[col], errors="coerce")
    x = num[df[group].astype(str) == a].dropna().to_numpy(float)
    y = num[df[group].astype(str) == b].dropna().to_numpy(float)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"u": float(res.statistic), "p": float(res.pvalue)}


def ref_reg(df: pd.DataFrame, y: str, xs: list[str]) -> dict[str, float]:
    d = df[[y, *xs]].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in xs])
    yv = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    r2 = 1 - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum())
    out = {"r_squared": float(r2)}
    for i, c in enumerate(xs, start=1):
        out[f"coef_{c}"] = float(beta[i])
    return out


def ref_desc(df: pd.DataFrame, col: str) -> dict[str, float]:
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return {"mean": float(v.mean()), "median": float(v.median())}


def ref_one(df: pd.DataFrame, col: str, mu0: float) -> dict[str, float]:
    v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    res = stats.ttest_1samp(v, mu0)
    return {"t": float(res.statistic), "p": float(res.pvalue), "mean": float(v.mean())}


# ---------------------------------------------------------------- helpers

PEARSON_TRAP = [("correlation_test", {"method": "pearson"})]
TTEST = [("ttest_two_sample", {})]
#: Forbidden-pattern marker: a causal *claim* anywhere in the interpretation
#: (caveats like "does not imply causation" are correctly allowed).
CAUSAL_FORBIDDEN = ["CAUSAL_CLAIM"]
ASSOC_CAVEAT = [[
    "associat", "does not imply causation",
    r"correlation.{0,40}(not|no).{0,10}caus", "相关不等于因果", "关联",
]]


def T(
    task_id: str,
    category: str,
    dataset: str,
    question: str,
    method: str = "",
    allowed: list[str] | None = None,
    design: dict[str, Any] | None = None,
    invalid: list[tuple[str, dict[str, Any]]] | None = None,
    must_clarify: bool = False,
    required: list[str] | None = None,
    forbidden: list[str] | None = None,
    title: str = "",
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        category=category,
        title=title or question[:60],
        dataset=dataset,
        question=question,
        ground_truth=GroundTruth(
            method=method,
            allowed_methods=allowed or ([method] if method else []),
            design=design or {},
            invalid_matchers=invalid or [],
            must_clarify=must_clarify,
            required_points=required or [],
            forbidden_patterns=forbidden or [],
        ),
    )


# ---------------------------------------------------------------- the suite

TASKS: list[TaskSpec] = []

# ---- method_selection (12) ------------------------------------------------
TASKS += [
    T("MS-01", "method_selection", "study_hours",
      "Is study time associated with exam score?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=ASSOC_CAVEAT, forbidden=CAUSAL_FORBIDDEN,
      title="Association between study time and score (Pearson)"),
    T("MS-02", "method_selection", "teaching_method",
      "Does the new teaching method produce higher posttest scores than the traditional method?",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      required=[r"significan|显著"], forbidden=CAUSAL_FORBIDDEN,
      title="Two-group comparison (independent)"),
    T("MS-03", "method_selection", "prepost",
      "Do scores change from before to after the training program?",
      method="ttest_paired", allowed=["ttest_paired"],
      design={"sample_type": "paired", "study_design": "experimental", "causal_question": False},
      title="Paired before/after comparison"),
    T("MS-04", "method_selection", "health_survey",
      "Is smoking associated with hypertension?",
      method="chi_square_test", allowed=["chi_square_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      forbidden=CAUSAL_FORBIDDEN,
      title="Categorical association (chi-square)"),
    T("MS-05", "method_selection", "ordinal_survey",
      "Is customer satisfaction associated with monthly spending?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      forbidden=CAUSAL_FORBIDDEN,
      title="Ordinal association (Spearman)"),
    T("MS-06", "method_selection", "confounded",
      "Predict monthly spending from income and education years.",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Multiple regression prediction"),
    T("MS-07", "method_selection", "study_hours",
      "What is the average exam score of the students?",
      method="describe_variable", allowed=["describe_variable"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Descriptive summary"),
    T("MS-08", "method_selection", "likert_groups",
      "Do the treatment group ratings differ from the control group ratings?",
      method="mann_whitney_u", allowed=["mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Ordinal group comparison (Mann-Whitney)"),
    T("MS-09", "method_selection", "study_hours",
      "Is the mean exam score different from 80?",
      method="ttest_one_sample", allowed=["ttest_one_sample"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="One-sample t-test"),
    T("MS-10", "method_selection", "salaries",
      "Compare salaries between R&D and Sales employees.",
      method="mann_whitney_u", allowed=["mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Skewed group comparison (Mann-Whitney)"),
    T("MS-11", "method_selection", "study_hours",
      "Is class attendance associated with exam score?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      forbidden=CAUSAL_FORBIDDEN,
      title="Association attendance-score (Pearson)"),
    T("MS-12", "method_selection", "heteroscedastic",
      "Predict y from x using regression.",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Simple regression"),
]

# ---- wrong_method (10) -----------------------------------------------------
TASKS += [
    T("WM-01", "wrong_method", "salaries",
      "Is there a significant difference in salary between R&D and Sales? Use the standard test for comparing means.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Trap: t-test on heavily skewed salaries"),
    T("WM-02", "wrong_method", "ordinal_survey",
      "Compute the Pearson correlation between satisfaction and monthly spend.",
      method="correlation_test", allowed=["correlation_test"], invalid=PEARSON_TRAP,
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Trap: Pearson on ordinal satisfaction"),
    T("WM-03", "wrong_method", "health_survey",
      "Is smoking associated with BMI? Use a chi-square test.",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      invalid=[("chi_square_test", {})],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Trap: chi-square on continuous BMI"),
    T("WM-04", "wrong_method", "teaching_method",
      "Is teaching method associated with posttest score? Use a chi-square test.",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      invalid=[("chi_square_test", {})],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Trap: chi-square on continuous score"),
    T("WM-05", "wrong_method", "likert_groups",
      "Compare the mean ratings of the treatment and control groups with a t-test.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Trap: t-test on Likert ratings"),
    T("WM-06", "wrong_method", "study_hours",
      "Is gender associated with exam score? Use a chi-square test.",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      invalid=[("chi_square_test", {})],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Trap: chi-square for binary x continuous"),
    T("WM-07", "wrong_method", "health_survey",
      "Is age associated with hypertension? Use a chi-square test.",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      invalid=[("chi_square_test", {})],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Trap: chi-square on continuous age"),
    T("WM-08", "wrong_method", "small_sample",
      "Test whether group A and group B differ significantly.",
      method="describe_variable", allowed=["describe_variable"], invalid=TTEST + [("mann_whitney_u", {})],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Trap: inferential test on n=8"),
    T("WM-09", "wrong_method", "outliers",
      "Compare the mean values of group C and group D with a t-test.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Trap: t-test with extreme outliers"),
    T("WM-10", "wrong_method", "health_survey",
      "Compute the Pearson correlation between smoking and hypertension.",
      method="chi_square_test", allowed=["chi_square_test"],
      invalid=[("correlation_test", {})],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Trap: correlation on two binary variables"),
]

# ---- numerical_accuracy (10) -----------------------------------------------
TASKS += [
    T("NA-01", "numerical_accuracy", "study_hours",
      "What is the Pearson correlation between study hours and exam score?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: Pearson r and p"),
    T("NA-02", "numerical_accuracy", "teaching_method",
      "Compare posttest scores between the two methods with Welch's t-test.",
      method="ttest_two_sample", allowed=["ttest_two_sample"],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Accuracy: Welch t-test"),
    T("NA-03", "numerical_accuracy", "prepost",
      "Run a paired t-test on the before and after scores.",
      method="ttest_paired", allowed=["ttest_paired"],
      design={"sample_type": "paired", "study_design": "experimental", "causal_question": False},
      title="Accuracy: paired t-test"),
    T("NA-04", "numerical_accuracy", "health_survey",
      "Run a chi-square test of independence between smoking and hypertension.",
      method="chi_square_test", allowed=["chi_square_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: chi-square"),
    T("NA-05", "numerical_accuracy", "confounded",
      "Regress monthly spending on income and education years and report the coefficients.",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: regression coefficients"),
    T("NA-06", "numerical_accuracy", "study_hours",
      "Test whether the mean exam score differs from 70 with a one-sample t-test.",
      method="ttest_one_sample", allowed=["ttest_one_sample"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: one-sample t-test"),
    T("NA-07", "numerical_accuracy", "likert_groups",
      "Run a Mann-Whitney U test on the ratings of the two groups.",
      method="mann_whitney_u", allowed=["mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Accuracy: Mann-Whitney U"),
    T("NA-08", "numerical_accuracy", "ordinal_survey",
      "What is the Spearman correlation between satisfaction and monthly spend?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: Spearman r and p"),
    T("NA-09", "numerical_accuracy", "salaries",
      "Report the mean and median salary.",
      method="describe_variable", allowed=["describe_variable"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: skewed mean/median"),
    T("NA-10", "numerical_accuracy", "heteroscedastic",
      "Regress y on x and report R-squared and the slope.",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Accuracy: regression on heteroscedastic data"),
]

# ---- insufficient_design (10) ----------------------------------------------
TASKS += [
    T("ID-01", "insufficient_design", "prepost",
      "Do the scores in the two measurement columns differ?",
      method="ttest_paired", allowed=["ttest_paired"], must_clarify=True,
      design={"sample_type": "paired", "study_design": "experimental", "causal_question": False},
      title="Design missing: paired vs independent"),
    T("ID-02", "insufficient_design", "teaching_method",
      "Is the new teaching method better than the traditional one?",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"], must_clarify=True,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Design missing: sample structure"),
    T("ID-03", "insufficient_design", "health_survey",
      "Does smoking cause hypertension?",
      method="chi_square_test", allowed=["chi_square_test"], must_clarify=True,
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Design missing: causal claim on unknown design"),
    T("ID-04", "insufficient_design", "likert_groups",
      "Do the two groups differ in their ratings?",
      method="mann_whitney_u", allowed=["mann_whitney_u"], must_clarify=True,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Design missing: group comparison"),
    T("ID-05", "insufficient_design", "confounded",
      "Does education affect monthly spending?",
      method="linear_regression", allowed=["linear_regression"], must_clarify=True,
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": True},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Design missing: confounding control"),
    T("ID-06", "insufficient_design", "prepost",
      "Is there an improvement after the training?",
      method="ttest_paired", allowed=["ttest_paired"], must_clarify=True,
      design={"sample_type": "paired", "study_design": "experimental", "causal_question": False},
      title="Design missing: repeated measures"),
    T("ID-07", "insufficient_design", "small_sample",
      "Compare group A and group B.",
      method="describe_variable", allowed=["describe_variable"], must_clarify=True,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Design missing: ambiguous comparison"),
    T("ID-08", "insufficient_design", "prepost",
      "Compare the first measurement with the second measurement.",
      method="ttest_paired", allowed=["ttest_paired"], must_clarify=True,
      design={"sample_type": "paired", "study_design": "experimental", "causal_question": False},
      title="Design missing: measurement pairing"),
    T("ID-09", "insufficient_design", "health_survey",
      "Does smoking lead to hypertension?",
      method="chi_square_test", allowed=["chi_square_test"], must_clarify=True,
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Design missing: causal wording"),
    T("ID-10", "insufficient_design", "ordinal_survey",
      "Does satisfaction predict monthly spending?",
      method="correlation_test", allowed=["correlation_test"], must_clarify=True,
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      forbidden=CAUSAL_FORBIDDEN,
      title="Design missing: prediction vs causation"),
]

# ---- assumption_violation (8) ----------------------------------------------
TASKS += [
    T("AV-01", "assumption_violation", "salaries",
      "Compare salaries between R&D and Sales.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      title="Assumption: severe skew in salaries"),
    T("AV-02", "assumption_violation", "outliers",
      "Compare the values of group C and group D.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Assumption: extreme outliers"),
    T("AV-03", "assumption_violation", "heteroscedastic",
      "Regress y on x and check the residuals.",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=[r"heteroscedastic|non-constant variance|方差非齐|异方差"],
      title="Assumption: heteroscedastic residuals"),
    T("AV-04", "assumption_violation", "small_sample",
      "Test for a difference between group A and group B.",
      method="describe_variable", allowed=["describe_variable"],
      invalid=TTEST + [("mann_whitney_u", {})],
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Assumption: tiny sample (n=8)"),
    T("AV-05", "assumption_violation", "likert_groups",
      "Compare the ratings of the treatment and control groups.",
      method="mann_whitney_u", allowed=["mann_whitney_u"], invalid=TTEST,
      design={"sample_type": "independent", "study_design": "experimental", "causal_question": False},
      title="Assumption: ordinal scale (t-test inappropriate)"),
    T("AV-06", "assumption_violation", "sparse_cat",
      "Is region associated with satisfaction?",
      method="chi_square_test", allowed=["chi_square_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=[r"small expected|expected.{0,30}below 5|fisher|expected counts|期望频数"],
      title="Assumption: sparse contingency cells"),
    T("AV-07", "assumption_violation", "outliers",
      "Is the mean of the value column different from 50?",
      method="bootstrap_ci", allowed=["bootstrap_ci", "describe_variable"],
      invalid=[("ttest_one_sample", {})],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Assumption: one-sample t-test with outliers"),
    T("AV-08", "assumption_violation", "ordinal_survey",
      "Is satisfaction correlated with monthly spending?",
      method="correlation_test", allowed=["correlation_test"], invalid=PEARSON_TRAP,
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      title="Assumption: ordinal scale (Pearson inappropriate)"),
]

# ---- causal_misuse (5) ------------------------------------------------------
TASKS += [
    T("CM-01", "causal_misuse", "health_survey",
      "Does smoking cause hypertension? (observational survey data)",
      method="chi_square_test", allowed=["chi_square_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Causal misuse: smoking -> hypertension"),
    T("CM-02", "causal_misuse", "confounded",
      "Does more education cause higher monthly spending? (observational data)",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Causal misuse: education -> spending"),
    T("CM-03", "causal_misuse", "study_hours",
      "Does studying more hours cause higher exam scores? (survey data)",
      method="linear_regression", allowed=["linear_regression"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Causal misuse: hours -> scores"),
    T("CM-04", "causal_misuse", "teaching_method",
      "Does the new method cause the improvement in scores? (non-randomized groups)",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Causal misuse: method -> scores"),
    T("CM-05", "causal_misuse", "ordinal_survey",
      "Does satisfaction cause higher spending? (survey data)",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": True, "confounding_addressed": False},
      forbidden=CAUSAL_FORBIDDEN, required=ASSOC_CAVEAT,
      title="Causal misuse: satisfaction -> spending"),
]

# ---- interpretation_error (5) ----------------------------------------------
TASKS += [
    T("IE-01", "interpretation_error", "noise",
      "Is x associated with y?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=[r"insufficient|no (significant )?evidence|not significant|不显著|证据不足"],
      forbidden=[r"no (effect|association|relationship)|不存在(关系|关联|差异)|证明.{0,10}(没有|无)"],
      title="Interpretation: p>0.05 is not proof of no effect"),
    T("IE-02", "interpretation_error", "study_hours",
      "Is study time associated with exam score?",
      method="correlation_test", allowed=["correlation_test"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=[r"r\s*=\s*0\.9|r²|r2|r-squared|strong", r"associat|相关|关联"],
      forbidden=CAUSAL_FORBIDDEN,
      title="Interpretation: report effect size, no causality"),
    T("IE-03", "interpretation_error", "salaries",
      "Compare salaries between R&D and Sales.",
      method="mann_whitney_u", allowed=["mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      required=[r"rank-biserial|effect size|效应量|stochastic|随机优势"],
      forbidden=[r"median(s)? (are|differ|不同)"],
      title="Interpretation: MW is not a median test"),
    T("IE-04", "interpretation_error", "teaching_method",
      "Does the new method improve scores? (non-randomized)",
      method="ttest_two_sample", allowed=["ttest_two_sample", "mann_whitney_u"],
      design={"sample_type": "independent", "study_design": "observational", "causal_question": False},
      required=[r"cohen|effect size|效应量", r"not random|non-random|观察性|observational|associat"],
      forbidden=CAUSAL_FORBIDDEN,
      title="Interpretation: significance + effect size + design caveat"),
    T("IE-05", "interpretation_error", "small_sample",
      "Summarize the value column.",
      method="describe_variable", allowed=["describe_variable"],
      design={"sample_type": "single", "study_design": "observational", "causal_question": False},
      required=[r"small sample|limited|preliminar|cautio|样本量小|初步"],
      title="Interpretation: small-sample caveat"),
]


def build_ground_truth(tasks: list[TaskSpec] | None = None) -> list[TaskSpec]:
    """Fill reference answers and extraction maps for every task."""
    for task in tasks if tasks is not None else TASKS:
        df = task.build_dataframe()
        gt = task.ground_truth
        m = gt.method
        if m == "correlation_test":
            if task.dataset == "ordinal_survey":
                gt.answers = ref_corr(df, "satisfaction", "monthly_spend", "spearman")
            elif task.dataset == "noise":
                gt.answers = ref_corr(df, "x", "y", "pearson")
            elif task.dataset == "study_hours" and "attendance" in task.question:
                gt.answers = ref_corr(df, "attendance_pct", "exam_score", "pearson")
            else:
                gt.answers = ref_corr(df, "study_hours", "exam_score", "pearson")
            gt.extract = {"r": ("correlation_test", "r"), "p": ("correlation_test", "p_value")}
        elif m == "ttest_two_sample":
            if "gender" in task.question:
                a, b, col, gcol = "Female", "Male", "exam_score", "gender"
            elif "age" in task.question:
                a, b, col, gcol = "No", "Yes", "age", "hypertension"
            elif "BMI" in task.question:
                a, b, col, gcol = "No", "Yes", "bmi", "smoking"
            else:
                a, b, col, gcol = "New", "Traditional", "posttest_score", "method"
            gt.answers = ref_ttest(df, col, gcol, a, b)
            gt.extract = {
                "t": ("ttest_two_sample", "t"),
                "p": ("ttest_two_sample", "p_value"),
                "mean_diff": ("ttest_two_sample", "mean_diff"),
                "ci_lo": ("ttest_two_sample", ("ci95_mean_diff", 0)),
                "ci_hi": ("ttest_two_sample", ("ci95_mean_diff", 1)),
            }
        elif m == "ttest_paired":
            gt.answers = ref_paired(df, "before_score", "after_score")
            gt.extract = {
                "t": ("ttest_paired", "t"),
                "p": ("ttest_paired", "p_value"),
                "mean_diff": ("ttest_paired", "mean_diff"),
            }
        elif m == "chi_square_test":
            if task.dataset == "sparse_cat":
                gt.answers = ref_chi2(df, "region", "satisfied")
            else:
                gt.answers = ref_chi2(df, "smoking", "hypertension")
            gt.extract = {
                "chi2": ("chi_square_test", "chi2"),
                "p": ("chi_square_test", "p_value"),
            }
        elif m == "mann_whitney_u":
            if task.dataset == "likert_groups":
                a, b, col, gcol = "Control", "Treatment", "rating", "group"
            elif task.dataset == "salaries":
                a, b, col, gcol = "R&D", "Sales", "salary", "department"
            else:
                a, b, col, gcol = "C", "D", "value", "group"
            gt.answers = ref_mw(df, col, gcol, a, b)
            gt.extract = {
                "u": ("mann_whitney_u", "u"),
                "p": ("mann_whitney_u", "p_value"),
            }
        elif m == "linear_regression":
            if task.dataset == "confounded":
                if "income" in task.question:
                    gt.answers = ref_reg(df, "monthly_spending", ["income", "education_years"])
                    gt.extract = {
                        "r_squared": ("linear_regression", "r_squared"),
                        "coef_income": ("linear_regression", ["coefficients", "income", "coefficient"]),
                        "coef_education_years": ("linear_regression", ["coefficients", "education_years", "coefficient"]),
                    }
                else:
                    gt.answers = ref_reg(df, "monthly_spending", ["education_years"])
                    gt.extract = {
                        "r_squared": ("linear_regression", "r_squared"),
                        "coef_education_years": ("linear_regression", ["coefficients", "education_years", "coefficient"]),
                    }
            elif task.dataset == "study_hours":
                gt.answers = ref_reg(df, "exam_score", ["study_hours"])
                gt.extract = {
                    "r_squared": ("linear_regression", "r_squared"),
                    "coef_study_hours": ("linear_regression", ["coefficients", "study_hours", "coefficient"]),
                }
            else:
                gt.answers = ref_reg(df, "y", ["x"])
                gt.extract = {
                    "r_squared": ("linear_regression", "r_squared"),
                    "coef_x": ("linear_regression", ["coefficients", "x", "coefficient"]),
                }
        elif m == "ttest_one_sample":
            mu0 = 70.0 if "70" in task.question else 80.0
            gt.answers = ref_one(df, "exam_score", mu0)
            gt.extract = {
                "t": ("ttest_one_sample", "t"),
                "p": ("ttest_one_sample", "p_value"),
                "mean": ("ttest_one_sample", "mean"),
            }
        elif m == "describe_variable":
            if task.dataset == "small_sample" or task.dataset == "outliers":
                col = "value"
            elif "salary" in task.question.lower():
                col = "salary"
            else:
                col = "exam_score"
            gt.answers = ref_desc(df, col)
            gt.extract = {
                "mean": ("describe_variable", "mean"),
                "median": ("describe_variable", "median"),
            }
        elif m == "bootstrap_ci":
            v = pd.to_numeric(df["value"], errors="coerce").dropna()
            gt.answers = {"mean": float(v.mean())}
            gt.extract = {
                "mean": [("describe_variable", "mean"), ("bootstrap_ci", "estimate")],
            }
    return tasks


#: Fill reference values once at import time.
TASKS = build_ground_truth(TASKS)
