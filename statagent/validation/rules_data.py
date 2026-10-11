"""Data-quality validation rules (DATA category)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from statagent.models.dataset import VariableType
from statagent.models.results import Finding
from statagent.validation.context import ValidationContext


@dataclass
class Rule:
    rule_id: str
    category: str
    title: str
    description: str
    check: Callable[[ValidationContext], Finding | None]


def _rule_data_01_missing(ctx: ValidationContext) -> Finding | None:
    if ctx.profile is None:
        return None
    used = [c for c in ctx.referenced_columns() if ctx.profile.column(c)]
    bad = [c for c in used if ctx.profile.column(c).missing_pct >= 50]
    if bad:
        details = ", ".join(f"{c} ({ctx.profile.column(c).missing_pct:.0f}%)" for c in bad)
        return ctx.finding(
            "DATA-01", "DATA", "FAIL",
            f"More than half the values are missing in: {details}.",
            f"Missing fractions: {details}.",
            "Imputation is not automatic: consider dropping these columns, using "
            "complete-case analysis with a clear statement, or collecting more data.",
        )
    moderate = [c for c in used if ctx.profile.column(c).missing_pct >= 20]
    if moderate:
        details = ", ".join(f"{c} ({ctx.profile.column(c).missing_pct:.0f}%)" for c in moderate)
        return ctx.finding(
            "DATA-01", "DATA", "WARNING",
            f"Substantial missingness in: {details}.",
            details,
            "Results may be biased if missingness is not completely at random. "
            "State how missing values were handled in the report.",
        )
    return None


def _rule_data_02_duplicates(ctx: ValidationContext) -> Finding | None:
    if ctx.profile is None:
        return None
    n = ctx.profile.n_duplicate_rows
    if n > 0:
        return ctx.finding(
            "DATA-02", "DATA", "WARNING",
            f"The dataset contains {n} duplicated row(s).",
            f"n_duplicate_rows = {n} of {ctx.profile.n_rows}.",
            "Check whether duplicates are legitimate repeated measurements or "
            "accidental double-entry; consider de-duplicating.",
        )
    return None


def _rule_data_03_sample_size(ctx: ValidationContext) -> Finding | None:
    if ctx.profile is None:
        return None
    n = ctx.profile.n_rows
    inferential = bool(ctx.plan and ctx.plan.steps) and any(
        s.tool
        in {
            "ttest_one_sample", "ttest_two_sample", "ttest_paired",
            "chi_square_test", "mann_whitney_u", "correlation_test",
            "linear_regression",
        }
        for s in ctx.plan.steps
    )
    if inferential and n < 10:
        return ctx.finding(
            "DATA-03", "DATA", "FAIL",
            f"Only {n} rows: far too few for inferential statistics.",
            f"n_rows = {n}.",
            "Inferential results are not trustworthy below ~10 observations; "
            "use descriptive statistics only, or collect more data.",
        )
    if inferential and n < 30:
        return ctx.finding(
            "DATA-03", "DATA", "WARNING",
            f"Small sample ({n} rows): estimates are imprecise and normality "
            "checks are unreliable.",
            f"n_rows = {n}.",
            "Interpret results with caution; report confidence intervals and "
            "consider nonparametric methods or bootstrapping.",
        )
    return None


def _rule_data_04_constant(ctx: ValidationContext) -> Finding | None:
    used = [c for c in ctx.referenced_columns() if ctx.column_profile(c)]
    const = [c for c in used if ctx.column_profile(c).is_constant]
    if const:
        return ctx.finding(
            "DATA-04", "DATA", "FAIL",
            f"Constant column(s) included in the analysis: {', '.join(const)}.",
            "A constant has zero variance; no statistic is defined.",
            "Remove constant columns from the analysis steps.",
        )
    return None


def _rule_data_05_outliers(ctx: ValidationContext) -> Finding | None:
    if ctx.profile is None:
        return None
    used = [c for c in ctx.referenced_columns() if ctx.column_profile(c)]
    out_cols = [c for c in used if (ctx.column_profile(c).n_outliers or 0) > 0]
    if out_cols:
        details = ", ".join(
            f"{c} ({ctx.column_profile(c).n_outliers} outliers)" for c in out_cols
        )
        return ctx.finding(
            "DATA-05", "DATA", "WARNING",
            f"IQR-rule outliers detected in: {details}.",
            "Outliers were detected with the 1.5×IQR rule.",
            "Outliers can distort means and t-tests. Verify they are real data "
            "(not entry errors); consider robust methods (median, Mann-Whitney U, "
            "Spearman) or a sensitivity analysis with and without them.",
        )
    return None


def _rule_data_06_identifier(ctx: ValidationContext) -> Finding | None:
    used = [c for c in ctx.referenced_columns() if ctx.column_profile(c)]
    ids = [
        c for c in used
        if ctx.column_profile(c).variable_type == VariableType.IDENTIFIER
    ]
    if ids:
        return ctx.finding(
            "DATA-06", "DATA", "FAIL",
            f"Identifier column(s) used in statistical analysis: {', '.join(ids)}.",
            "Identifiers carry no analysable information.",
            "Remove identifier columns from the analysis steps.",
        )
    return None


def _rule_data_07_fully_missing(ctx: ValidationContext) -> Finding | None:
    used = [c for c in ctx.referenced_columns() if ctx.column_profile(c)]
    empty = [c for c in used if ctx.column_profile(c).missing_pct >= 100]
    if empty:
        return ctx.finding(
            "DATA-07", "DATA", "FAIL",
            f"Column(s) with no observed values: {', '.join(empty)}.",
            "missing_pct = 100%.",
            "Remove these columns or collect data for them.",
        )
    return None


DATA_RULES: list[Rule] = [
    Rule("DATA-01", "DATA", "Missing values", "Flags columns with heavy missingness.", _rule_data_01_missing),
    Rule("DATA-02", "DATA", "Duplicate rows", "Flags duplicated rows.", _rule_data_02_duplicates),
    Rule("DATA-03", "DATA", "Sample size", "Checks the sample size is adequate for inference.", _rule_data_03_sample_size),
    Rule("DATA-04", "DATA", "Constant columns", "Rejects constant columns in analysis steps.", _rule_data_04_constant),
    Rule("DATA-05", "DATA", "Outliers", "Flags IQR outliers in analysed columns.", _rule_data_05_outliers),
    Rule("DATA-06", "DATA", "Identifier misuse", "Rejects identifier columns in analysis steps.", _rule_data_06_identifier),
    Rule("DATA-07", "DATA", "Fully missing columns", "Rejects columns with no data.", _rule_data_07_fully_missing),
]
