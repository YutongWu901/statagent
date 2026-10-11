"""Hypothesis tests: t-tests (one-sample, Welch two-sample, paired),
chi-square test of independence, and Mann-Whitney U."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.common import (
    COLOR,
    COLOR_2,
    base_figure,
    code_snippet,
    cohens_d,
    fmt_p,
    get_series,
    shapiro_safe,
    to_numeric,
    welch_ci,
)
from statagent.stats.engine import Params, ToolRegistry


class TTestOneParams(Params):
    dataset_id: str
    column: str
    mu0: float = 0.0


class TTestTwoParams(Params):
    dataset_id: str
    column: str
    group_column: str
    group_a: str
    group_b: str
    paired: bool = False


class TTestPairedParams(Params):
    dataset_id: str
    column_a: str
    column_b: str


class ChiSquareParams(Params):
    dataset_id: str
    column_a: str
    column_b: str


class MannWhitneyParams(Params):
    dataset_id: str
    column: str
    group_column: str
    group_a: str
    group_b: str


def _group_split(df: pd.DataFrame, column: str, group_column: str, a: str, b: str):
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found. Available: {list(df.columns)}")
    if column not in df.columns:
        raise ValueError(f"Value column '{column}' not found. Available: {list(df.columns)}")
    levels = [str(v) for v in df[group_column].dropna().unique()]
    if a not in levels or b not in levels:
        raise ValueError(
            f"Group levels {a!r}/{b!r} not found in '{group_column}'. Present levels: {levels}"
        )
    num = pd.to_numeric(df[column], errors="coerce")
    x = num[df[group_column].astype(str) == a].dropna().to_numpy(float)
    y = num[df[group_column].astype(str) == b].dropna().to_numpy(float)
    return x, y, levels


def _box_compare(x, y, a, b, value_col: str) -> go.Figure:
    fig = base_figure(f"{value_col} by group")
    fig.add_trace(go.Box(y=x, name=a, marker_color=COLOR, boxmean="sd"))
    fig.add_trace(go.Box(y=y, name=b, marker_color=COLOR_2, boxmean="sd"))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title=value_col)
    return fig


@ToolRegistry.register(
    "ttest_one_sample",
    "One-sample t-test comparing the mean of a numeric column against a "
    "hypothesized value mu0. Reports the t statistic, p-value, 95% CI of the "
    "mean and Cohen's d against mu0.",
    TTestOneParams,
    "Returns t, df, p-value, sample mean, 95% CI of the mean, Cohen's d.",
)
def ttest_one_sample(store: DatasetStore, params: TTestOneParams) -> dict:
    s = get_series(store, params.dataset_id, params.column)
    num = to_numeric(s, params.column).dropna().to_numpy(float)
    n = len(num)
    if n < 2:
        raise ValueError("One-sample t-test requires at least 2 values.")
    res = stats.ttest_1samp(num, params.mu0)
    mean = float(num.mean())
    sd = float(num.std(ddof=1))
    ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
    d = (mean - params.mu0) / sd if sd > 0 else float("nan")
    notes = [
        f"One-sample t-test: mean({params.column}) vs {params.mu0}: "
        f"t({n - 1}) = {res.statistic:.3f}, p = {fmt_p(res.pvalue)}."
    ]
    return {
        "summary": {
            "t": round(float(res.statistic), 6),
            "df": n - 1,
            "p_value": round(float(res.pvalue), 6),
            "mean": round(mean, 6),
            "ci95_mean": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
            "cohens_d": round(d, 6),
            "n": n,
            "mu0": params.mu0,
        },
        "details": {"p_formatted": fmt_p(res.pvalue)},
        "figures": [],
        "notes": notes,
        "code": code_snippet(
            "statagent.stats.hypothesis.ttest_one_sample",
            {"dataset_id": params.dataset_id, "column": params.column, "mu0": params.mu0},
        ),
        "n": n,
    }


@ToolRegistry.register(
    "ttest_two_sample",
    "Two-sample t-test comparing a numeric column between two groups of a "
    "categorical column. Uses Welch's t-test by default (does not assume "
    "equal variances); also reports Levene's test, Student's reference "
    "result, 95% CI of the mean difference and Cohen's d. Set paired=true "
    "only when the two groups are matched pairs.",
    TTestTwoParams,
    "Returns Welch t, df, p, mean difference with 95% CI, Cohen's d, group "
    "sizes/means, Levene's test, and a box plot.",
)
def ttest_two_sample(store: DatasetStore, params: TTestTwoParams) -> dict:
    df = store.get(params.dataset_id)
    x, y, levels = _group_split(
        df, params.column, params.group_column, params.group_a, params.group_b
    )
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Each group needs >= 2 observations (got {n1} and {n2}).")
    notes: list[str] = []

    if params.paired:
        if n1 != n2:
            raise ValueError(
                f"Paired test requires equal group sizes (got {n1} vs {n2})."
            )
        res = stats.ttest_rel(x, y)
        diff = float(np.mean(x - y))
        sd_diff = float(np.std(x - y, ddof=1))
        ci_lo, ci_hi = stats.t.interval(0.95, n1 - 1, loc=diff, scale=sd_diff / np.sqrt(n1))
        d = diff / sd_diff if sd_diff > 0 else float("nan")
        kind = "Paired t-test"
        test_name = "ttest_rel"
    else:
        welch = stats.ttest_ind(x, y, equal_var=False)
        student = stats.ttest_ind(x, y, equal_var=True)
        diff, ci_lo, ci_hi, df_w = welch_ci(x, y)
        d = cohens_d(x, y)
        lev_p = float(stats.levene(x, y).pvalue)
        notes.append(
            f"Levene's test for equal variances: p = {fmt_p(lev_p)} "
            + ("(variances differ; Welch's t-test used, which does not assume equal variances)"
               if lev_p < 0.05
               else "(variances not detectably different)")
        )
        res, kind, test_name = welch, "Welch two-sample t-test", "ttest_ind(equal_var=False)"
        details_extra = {
            "student_t": round(float(student.statistic), 6),
            "student_p": round(float(student.pvalue), 6),
            "levene_p": round(lev_p, 6),
        }
    summary = {
        "test": kind,
        "t": round(float(res.statistic), 6),
        "df": round(float(getattr(res, "df", df_w if not params.paired else n1 - 1)), 3),
        "p_value": round(float(res.pvalue), 6),
        "mean_diff": round(float(diff), 6),
        "ci95_mean_diff": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
        "cohens_d": round(float(d), 6),
        "n": n1 + n2,
        "n_a": n1,
        "n_b": n2,
        "mean_a": round(float(x.mean()), 6),
        "mean_b": round(float(y.mean()), 6),
        "group_a": params.group_a,
        "group_b": params.group_b,
    }
    details = dict(summary)
    details.update(details_extra if not params.paired else {})
    details["p_formatted"] = fmt_p(res.pvalue)
    for name, arr in ((params.group_a, x), (params.group_b, y)):
        sh = shapiro_safe(arr)
        if sh is not None:
            details[f"shapiro_{name}"] = {"w": round(sh[0], 6), "p": round(sh[1], 6)}
    notes.append(
        f"{kind}: mean difference {diff:.4g} ({ci_lo:.4g}, {ci_hi:.4g}), "
        f"p = {fmt_p(res.pvalue)}."
    )
    fig = _box_compare(x, y, params.group_a, params.group_b, params.column)
    return {
        "summary": summary,
        "details": details,
        "figures": [fig],
        "notes": notes,
        "code": code_snippet(
            "statagent.stats.hypothesis.ttest_two_sample",
            {
                "dataset_id": params.dataset_id,
                "column": params.column,
                "group_column": params.group_column,
                "group_a": params.group_a,
                "group_b": params.group_b,
                "paired": params.paired,
            },
        ),
        "n": n1 + n2,
    }


@ToolRegistry.register(
    "ttest_paired",
    "Paired t-test between two matched numeric columns (same rows are "
    "paired). Use when each row holds a before/after or matched pair.",
    TTestPairedParams,
    "Returns t, df, p, mean of differences with 95% CI, Cohen's d_z.",
)
def ttest_paired(store: DatasetStore, params: TTestPairedParams) -> dict:
    df = store.get(params.dataset_id)
    for c in (params.column_a, params.column_b):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    pair = df[[params.column_a, params.column_b]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(pair)
    if n < 3:
        raise ValueError(f"Paired t-test requires at least 3 complete pairs (got {n}).")
    a = pair[params.column_a].to_numpy(float)
    b = pair[params.column_b].to_numpy(float)
    res = stats.ttest_rel(a, b)
    diff = float(np.mean(a - b))
    sd_diff = float(np.std(a - b, ddof=1))
    ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=diff, scale=sd_diff / np.sqrt(n))
    d = diff / sd_diff if sd_diff > 0 else float("nan")
    fig = base_figure(f"Paired values: {params.column_a} vs {params.column_b}")
    fig.add_trace(go.Scatter(x=a, y=b, mode="markers", marker=dict(color=COLOR, size=7, opacity=0.6)))
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color=COLOR_2), name="no change"))
    fig.update_xaxes(title=params.column_a)
    fig.update_yaxes(title=params.column_b)
    return {
        "summary": {
            "t": round(float(res.statistic), 6),
            "df": n - 1,
            "p_value": round(float(res.pvalue), 6),
            "mean_diff": round(diff, 6),
            "ci95_mean_diff": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
            "cohens_d": round(d, 6),
            "n_pairs": n,
        },
        "details": {"p_formatted": fmt_p(res.pvalue)},
        "figures": [fig],
        "notes": [
            f"Paired t-test: mean({params.column_a} - {params.column_b}) = {diff:.4g}, "
            f"p = {fmt_p(res.pvalue)}, {n} pairs."
        ],
        "code": code_snippet(
            "statagent.stats.hypothesis.ttest_paired",
            {"dataset_id": params.dataset_id, "column_a": params.column_a, "column_b": params.column_b},
        ),
        "n": n,
    }


@ToolRegistry.register(
    "chi_square_test",
    "Chi-square test of independence between two categorical columns. "
    "Reports expected counts and Cramer's V; switches to Fisher's exact "
    "test for 2x2 tables with small expected counts.",
    ChiSquareParams,
    "Returns chi2, df, p, expected counts, Cramer's V, and a grouped bar "
    "chart. Includes Fisher's exact p for 2x2 tables.",
)
def chi_square_test(store: DatasetStore, params: ChiSquareParams) -> dict:
    df = store.get(params.dataset_id)
    for c in (params.column_a, params.column_b):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    table = pd.crosstab(
        df[params.column_a].astype(str), df[params.column_b].astype(str)
    )
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Chi-square test needs at least 2 categories per variable.")
    chi2, p, dof, expected = stats.chi2_contingency(table.values, correction=False)
    n = int(table.values.sum())
    min_dim = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    notes: list[str] = []
    details: dict = {
        "observed": {
            str(i): {str(j): int(v) for j, v in enumerate(row)}
            for i, row in zip(table.index, table.values)
        },
        "expected": {
            str(i): {str(j): float(v) for j, v in enumerate(row)}
            for i, row in zip(table.index, expected)
        },
        "min_expected": round(float(expected.min()), 3),
        "p_formatted": fmt_p(p),
    }
    if expected.min() < 5:
        notes.append(
            "Some expected counts are below 5; the chi-square approximation "
            "is unreliable. For 2x2 tables Fisher's exact test is reported."
        )
    if table.shape == (2, 2):
        or_val, p_fisher = stats.fisher_exact(table.values)
        details["fisher_odds_ratio"] = round(float(or_val), 6)
        details["fisher_p"] = round(float(p_fisher), 6)
    fig = base_figure(f"{params.column_a} × {params.column_b}")
    for col in table.columns:
        fig.add_trace(go.Bar(x=[str(i) for i in table.index], y=table[col].values, name=str(col)))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title=params.column_a)
    fig.update_yaxes(title="count")
    return {
        "summary": {
            "chi2": round(float(chi2), 6),
            "df": int(dof),
            "p_value": round(float(p), 6),
            "cramers_v": round(float(cramers_v), 6),
            "n": n,
            "min_expected": round(float(expected.min()), 3),
        },
        "details": details,
        "figures": [fig],
        "notes": notes + [
            f"Chi-square test of independence: chi2({dof}) = {chi2:.3f}, p = {fmt_p(p)}."
        ],
        "code": code_snippet(
            "statagent.stats.hypothesis.chi_square_test",
            {"dataset_id": params.dataset_id, "column_a": params.column_a, "column_b": params.column_b},
        ),
        "n": n,
    }


@ToolRegistry.register(
    "mann_whitney_u",
    "Mann-Whitney U test comparing a numeric (or ordinal) column between "
    "two groups. Nonparametric: does not assume normality; tests whether "
    "one group tends to be stochastically larger. Use when data are skewed, "
    "ordinal, or contain outliers.",
    MannWhitneyParams,
    "Returns U, p, rank-biserial effect size r, group medians, and a box "
    "plot.",
)
def mann_whitney_u(store: DatasetStore, params: MannWhitneyParams) -> dict:
    df = store.get(params.dataset_id)
    x, y, levels = _group_split(
        df, params.column, params.group_column, params.group_a, params.group_b
    )
    if len(x) < 3 or len(y) < 3:
        raise ValueError("Mann-Whitney U requires at least 3 observations per group.")
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    u = float(res.statistic)
    n1, n2 = len(x), len(y)
    rank_biserial = 1.0 - 2.0 * u / (n1 * n2)
    fig = _box_compare(x, y, params.group_a, params.group_b, params.column)
    return {
        "summary": {
            "u": round(u, 6),
            "p_value": round(float(res.pvalue), 6),
            "rank_biserial_r": round(rank_biserial, 6),
            "median_a": round(float(np.median(x)), 6),
            "median_b": round(float(np.median(y)), 6),
            "n": n1 + n2,
            "n_a": n1,
            "n_b": n2,
        },
        "details": {"p_formatted": fmt_p(res.pvalue)},
        "figures": [fig],
        "notes": [
            f"Mann-Whitney U = {u:.1f}, p = {fmt_p(res.pvalue)}: {params.group_a} vs {params.group_b}.",
            "Interpretation: tests stochastic dominance of one group over the "
            "other, not specifically a difference of medians.",
        ],
        "code": code_snippet(
            "statagent.stats.hypothesis.mann_whitney_u",
            {
                "dataset_id": params.dataset_id,
                "column": params.column,
                "group_column": params.group_column,
                "group_a": params.group_a,
                "group_b": params.group_b,
            },
        ),
        "n": n1 + n2,
    }
