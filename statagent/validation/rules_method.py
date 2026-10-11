"""Statistical-method validation rules (METHOD category).

These rules verify that the selected methods match the variable types and
the data, and that key assumptions are checked — all computed directly from
the data, independently of the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from statagent.models.dataset import NUMERIC_TYPES, VariableType
from statagent.models.results import Finding
from statagent.validation.context import ValidationContext

NUMERIC_TOOLS = {
    "describe_variable", "check_normality", "ttest_one_sample", "ttest_two_sample",
    "ttest_paired", "mann_whitney_u", "correlation_test", "bootstrap_ci",
}
GROUP_TOOLS = {"ttest_two_sample", "mann_whitney_u"}


@dataclass
class Rule:
    rule_id: str
    category: str
    title: str
    description: str
    check: Callable[[ValidationContext], Finding | None]


def _steps(ctx: ValidationContext):
    return ctx.plan.steps if ctx.plan else []


def _numeric_series(ctx: ValidationContext, column: str) -> np.ndarray | None:
    df = ctx.data()
    if df is None or column not in df.columns:
        return None
    arr = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(float)
    return arr if len(arr) >= 3 else None


def _shapiro_p(ctx: ValidationContext, column: str) -> float | None:
    arr = _numeric_series(ctx, column)
    if arr is None or len(arr) < 3 or len(arr) > 5000 or np.std(arr) == 0:
        return None
    return float(stats.shapiro(arr).pvalue)


def _rule_method_01_normality(ctx: ValidationContext) -> Finding | None:
    """t-test / Pearson columns far from normal at small-to-moderate n."""
    findings: list[str] = []
    for step in _steps(ctx):
        cols = []
        if step.tool in {"ttest_one_sample", "ttest_two_sample", "ttest_paired"}:
            p = step.parameters
            cols += [p.get("column"), p.get("column_a"), p.get("column_b")]
        elif step.tool == "correlation_test" and step.parameters.get("method") == "pearson":
            cols += [step.parameters.get("x"), step.parameters.get("y")]
        for c in [x for x in cols if isinstance(x, str)]:
            arr = _numeric_series(ctx, c)
            if arr is None:
                continue
            p_val = _shapiro_p(ctx, c)
            if p_val is None:
                continue
            if p_val < 0.05 and len(arr) < 100:
                findings.append(f"{c} (Shapiro p={p_val:.4f}, n={len(arr)})")
    if findings:
        return ctx.finding(
            "METHOD-01", "METHOD", "WARNING",
            "Columns used in t-tests / Pearson correlation deviate from "
            f"normality: {', '.join(findings)}.",
            "Shapiro-Wilk p < 0.05 at n < 100.",
            "Use robust alternatives: Mann-Whitney U instead of t-tests, "
            "Spearman instead of Pearson, or bootstrap CIs.",
        )
    return None


def _rule_method_02_variance(ctx: ValidationContext) -> Finding | None:
    """Equal-variance check for planned two-sample t-tests."""
    df = ctx.data()
    for step in _steps(ctx):
        if step.tool != "ttest_two_sample" or step.parameters.get("paired"):
            continue
        p = step.parameters
        if df is None:
            continue
        try:
            num = pd.to_numeric(df[p["column"]], errors="coerce")
            x = num[df[p["group_column"]].astype(str) == p["group_a"]].dropna()
            y = num[df[p["group_column"]].astype(str) == p["group_b"]].dropna()
        except (KeyError, TypeError):
            continue
        if len(x) < 2 or len(y) < 2:
            continue
        lev_p = float(stats.levene(x, y).pvalue)
        if lev_p < 0.05:
            return ctx.finding(
                "METHOD-02", "METHOD", "PASS",
                "Group variances differ (Levene p < 0.05); the engine uses "
                "Welch's t-test, which does not assume equal variances.",
                f"Levene p = {lev_p:.4f}.",
                "No action needed: Welch's t-test is the engine default.",
            )
    return None


def _rule_method_03_chisq_expected(ctx: ValidationContext) -> Finding | None:
    df = ctx.data()
    for step in _steps(ctx):
        if step.tool != "chi_square_test":
            continue
        p = step.parameters
        if df is None:
            continue
        try:
            table = pd.crosstab(df[p["column_a"]].astype(str), df[p["column_b"]].astype(str))
        except (KeyError, TypeError):
            continue
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        _, _, _, expected = stats.chi2_contingency(table.values, correction=False)
        min_exp = float(expected.min())
        if min_exp < 1:
            return ctx.finding(
                "METHOD-03", "METHOD", "FAIL",
                f"Minimum expected cell count is {min_exp:.2f}: the chi-square "
                "approximation is invalid.",
                f"min expected count = {min_exp:.2f}.",
                "Collapse categories to raise expected counts, or use Fisher's "
                "exact test (automatic for 2x2 tables in the engine).",
            )
        if min_exp < 5:
            return ctx.finding(
                "METHOD-03", "METHOD", "WARNING",
                f"Some expected cell counts are below 5 (minimum {min_exp:.2f}).",
                f"min expected count = {min_exp:.2f}.",
                "Interpret the chi-square p-value cautiously; collapse "
                "categories or use an exact test.",
            )
    return None


def _rule_method_04_categorical_in_regression(ctx: ValidationContext) -> Finding | None:
    for step in _steps(ctx):
        if step.tool != "linear_regression":
            continue
        p = step.parameters
        encoded = set(p.get("categorical") or [])
        for c in p.get("x") or []:
            colp = ctx.column_profile(c)
            if colp is None:
                continue
            if (
                colp.variable_type in CATEGORICAL_FOR_REGRESSION
                and c not in encoded
            ):
                return ctx.finding(
                    "METHOD-04", "METHOD", "FAIL",
                    f"Categorical predictor '{c}' is used in regression without "
                    "dummy encoding.",
                    f"variable_type={colp.variable_type.value}.",
                    f"Add '{c}' to the 'categorical' parameter so the engine "
                    "one-hot encodes it, or drop it.",
                )
    return None


CATEGORICAL_FOR_REGRESSION = {
    VariableType.CATEGORICAL_NOMINAL,
    VariableType.CATEGORICAL_ORDINAL,
    VariableType.BINARY,
}


def _rule_method_05_collinearity(ctx: ValidationContext) -> Finding | None:
    df = ctx.data()
    for step in _steps(ctx):
        if step.tool != "linear_regression":
            continue
        p = step.parameters
        num_x = [c for c in (p.get("x") or []) if c not in (p.get("categorical") or [])]
        if len(num_x) < 2 or df is None:
            continue
        try:
            sub = df[num_x].apply(pd.to_numeric, errors="coerce").dropna()
        except KeyError:
            continue
        if len(sub) < 10:
            continue
        X = np.column_stack([np.ones(len(sub)), sub.to_numpy(float)])
        from statagent.stats.regression import _vif

        vif = _vif(X, ["Intercept"] + num_x)
        high = {k: v for k, v in vif.items() if v > 10}
        if high:
            return ctx.finding(
                "METHOD-05", "METHOD", "WARNING",
                f"Predictor collinearity detected (VIF > 10): {high}.",
                f"VIFs: {high}.",
                "Collinear predictors make coefficients unstable. Consider "
                "dropping or combining them, or use regularisation.",
            )
    return None


def _rule_method_06_type_match(ctx: ValidationContext) -> Finding | None:
    for step in _steps(ctx):
        p = step.parameters
        if step.tool in NUMERIC_TOOLS:
            for key in ("column", "column_a", "column_b", "x", "y"):
                c = p.get(key)
                colp = ctx.column_profile(c) if isinstance(c, str) else None
                if colp is None:
                    continue
                if step.tool == "correlation_test":
                    if colp.variable_type == VariableType.CATEGORICAL_ORDINAL:
                        if p.get("method") == "pearson":
                            return ctx.finding(
                                "METHOD-06", "METHOD", "WARNING",
                                f"Pearson correlation used on ordinal variable '{c}'.",
                                f"variable_type={colp.variable_type.value}.",
                                "Pearson assumes interval data; use Spearman for ordinals.",
                            )
                    if colp.variable_type not in NUMERIC_TYPES and colp.variable_type != VariableType.CATEGORICAL_ORDINAL:
                        return ctx.finding(
                            "METHOD-06", "METHOD", "FAIL",
                            f"Correlation used on non-numeric variable '{c}'.",
                            f"variable_type={colp.variable_type.value}.",
                            "Correlation requires numeric (or ordinal via Spearman) variables.",
                        )
                elif colp.variable_type not in NUMERIC_TYPES and colp.variable_type != VariableType.CATEGORICAL_ORDINAL:
                    return ctx.finding(
                        "METHOD-06", "METHOD", "FAIL",
                        f"Tool '{step.tool}' requires a numeric variable but '{c}' "
                        f"is {colp.variable_type.value}.",
                        f"variable_type={colp.variable_type.value}.",
                        "Pick a numeric column, or a categorical method (chi-square).",
                    )
    return None


def _rule_method_07_group_sizes(ctx: ValidationContext) -> Finding | None:
    df = ctx.data()
    for step in _steps(ctx):
        if step.tool not in GROUP_TOOLS:
            continue
        p = step.parameters
        if df is None:
            continue
        try:
            sizes = df[df[p["group_column"]].astype(str).isin([p["group_a"], p["group_b"]])].groupby(
                df[p["group_column"]].astype(str), dropna=False
            ).size()
            n_a = int(sizes.get(p["group_a"], 0))
            n_b = int(sizes.get(p["group_b"], 0))
        except (KeyError, TypeError):
            continue
        if min(n_a, n_b) < 3:
            return ctx.finding(
                "METHOD-07", "METHOD", "FAIL",
                f"Group sizes too small for '{step.tool}' "
                f"({p['group_a']}: {n_a}, {p['group_b']}: {n_b}).",
                f"n_a={n_a}, n_b={n_b}.",
                "Need at least 3 observations per group; interpret small-group "
                "results with extreme caution.",
            )
        if min(n_a, n_b) < 30:
            return ctx.finding(
                "METHOD-07", "METHOD", "WARNING",
                f"Small group sizes ({p['group_a']}: {n_a}, {p['group_b']}: {n_b}).",
                f"n_a={n_a}, n_b={n_b}.",
                "Small groups limit power; report confidence intervals and "
                "consider nonparametric tests.",
            )
    return None


def _rule_method_08_paired_sizes(ctx: ValidationContext) -> Finding | None:
    df = ctx.data()
    for step in _steps(ctx):
        if step.tool == "ttest_two_sample" and step.parameters.get("paired"):
            p = step.parameters
            if df is None:
                continue
            try:
                n_a = int((df[p["group_column"]].astype(str) == p["group_a"]).sum())
                n_b = int((df[p["group_column"]].astype(str) == p["group_b"]).sum())
            except (KeyError, TypeError):
                continue
            if n_a != n_b:
                return ctx.finding(
                    "METHOD-08", "METHOD", "FAIL",
                    f"Paired test requested but group sizes differ ({n_a} vs {n_b}).",
                    "Paired tests need equal-size matched groups.",
                    "Use an independent-samples test, or provide true matched pairs.",
                )
    return None


def _rule_method_09_multiple_testing(ctx: ValidationContext) -> Finding | None:
    n_tests = sum(
        1
        for s in _steps(ctx)
        if s.tool
        in {
            "ttest_one_sample", "ttest_two_sample", "ttest_paired",
            "chi_square_test", "mann_whitney_u", "correlation_test",
        }
    )
    if n_tests > 3:
        return ctx.finding(
            "METHOD-09", "METHOD", "WARNING",
            f"The plan runs {n_tests} hypothesis tests on the same data.",
            f"{n_tests} tests planned.",
            "Multiple testing inflates the family-wise error rate. Consider a "
            "Bonferroni correction or pre-register a single primary test.",
        )
    return None


def _rule_method_10_bootstrap_n(ctx: ValidationContext) -> Finding | None:
    for step in _steps(ctx):
        if step.tool != "bootstrap_ci":
            continue
        c = step.parameters.get("column") or step.parameters.get("x")
        arr = _numeric_series(ctx, c) if isinstance(c, str) else None
        if arr is not None and len(arr) < 20:
            return ctx.finding(
                "METHOD-10", "METHOD", "WARNING",
                f"Bootstrap on a very small sample (n={len(arr)}).",
                f"n={len(arr)}.",
                "Bootstrap CIs from samples under ~20 observations are "
                "unstable; treat as indicative only.",
            )
    return None


def _rule_method_11_regression_n_p(ctx: ValidationContext) -> Finding | None:
    for step in _steps(ctx):
        if step.tool != "linear_regression":
            continue
        p = step.parameters
        n_pred = len(p.get("x") or []) + len(p.get("categorical") or []) + 1
        if ctx.profile is not None and ctx.profile.n_rows < 10 * n_pred:
            return ctx.finding(
                "METHOD-11", "METHOD", "WARNING",
                f"Few observations per regression parameter "
                f"(n={ctx.profile.n_rows}, ~{n_pred} parameters).",
                "Rule of thumb: at least ~10 observations per parameter.",
                "Results may overfit; prefer fewer predictors or more data.",
            )
    return None


METHOD_RULES: list[Rule] = [
    Rule("METHOD-01", "METHOD", "Normality", "Checks normality for t-tests and Pearson.", _rule_method_01_normality),
    Rule("METHOD-02", "METHOD", "Equal variances", "Levene check for two-sample t-tests.", _rule_method_02_variance),
    Rule("METHOD-03", "METHOD", "Chi-square expected counts", "Checks expected cell counts.", _rule_method_03_chisq_expected),
    Rule("METHOD-04", "METHOD", "Categorical in regression", "Rejects unencoded categorical predictors.", _rule_method_04_categorical_in_regression),
    Rule("METHOD-05", "METHOD", "Collinearity", "Flags VIF > 10.", _rule_method_05_collinearity),
    Rule("METHOD-06", "METHOD", "Method/variable type match", "Method must fit the variable types.", _rule_method_06_type_match),
    Rule("METHOD-07", "METHOD", "Group sizes", "Checks per-group sample sizes.", _rule_method_07_group_sizes),
    Rule("METHOD-08", "METHOD", "Paired group sizes", "Paired tests need equal group sizes.", _rule_method_08_paired_sizes),
    Rule("METHOD-09", "METHOD", "Multiple testing", "Flags many tests on the same data.", _rule_method_09_multiple_testing),
    Rule("METHOD-10", "METHOD", "Bootstrap sample size", "Flags bootstrap on tiny samples.", _rule_method_10_bootstrap_n),
    Rule("METHOD-11", "METHOD", "Regression n/p ratio", "Flags few observations per parameter.", _rule_method_11_regression_n_p),
]
