"""Descriptive statistics and distribution diagnostics."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.common import COLOR, base_figure, code_snippet, get_series, to_numeric
from statagent.stats.engine import Params, ToolRegistry


class DescribeParams(Params):
    dataset_id: str
    column: str


class NormalityParams(Params):
    dataset_id: str
    column: str


def _histogram_box(values: np.ndarray, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Histogram(x=values, nbinsx=30, marker_color=COLOR, name="frequency"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Box(x=values, marker_color=COLOR, name=""), row=2, col=1)
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=430, template="plotly_white")
    return fig


@ToolRegistry.register(
    "describe_variable",
    "Compute descriptive statistics for one numeric column: count, missing "
    "values, mean, median, standard deviation, variance, min, max, quartiles, "
    "IQR, skewness, kurtosis, standard error of the mean and a 95% CI for the "
    "mean. Use for 'summarize / describe' questions.",
    DescribeParams,
    "Returns the statistics above plus a histogram/box plot.",
)
def describe_variable(store: DatasetStore, params: DescribeParams) -> dict:
    s = get_series(store, params.dataset_id, params.column)
    num = to_numeric(s, params.column)
    valid = num.dropna()
    n = int(valid.shape[0])
    if n == 0:
        raise ValueError(f"Column '{params.column}' has no non-missing values.")
    if n == 1:
        raise ValueError(f"Column '{params.column}' has only one non-missing value.")

    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    sem = std / np.sqrt(n)
    ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)

    def _round(v):
        return round(float(v), 6)

    summary = {
        "count": n,
        "n_missing": int(s.isna().sum()),
        "missing_pct": round(100.0 * s.isna().sum() / max(len(s), 1), 2),
        "mean": _round(mean),
        "median": _round(valid.median()),
        "std": _round(std),
        "variance": _round(valid.var(ddof=1)),
        "min": _round(valid.min()),
        "max": _round(valid.max()),
        "q1": _round(valid.quantile(0.25)),
        "q3": _round(valid.quantile(0.75)),
        "iqr": _round(valid.quantile(0.75) - valid.quantile(0.25)),
        "sem": _round(sem),
        "ci95_mean": [_round(ci_lo), _round(ci_hi)],
    }
    details = dict(summary)
    details["skewness"] = _round(stats.skew(valid)) if n > 2 else None
    details["kurtosis"] = _round(stats.kurtosis(valid)) if n > 3 else None
    if valid.nunique() <= 10:
        details["mode"] = int(valid.mode().iloc[0])
    fig = _histogram_box(valid.to_numpy(), f"Distribution of {params.column}")
    return {
        "summary": summary,
        "details": details,
        "figures": [fig],
        "notes": [f"Missing values excluded: {int(s.isna().sum())} rows dropped for this analysis."]
        if s.isna().any()
        else [],
        "code": code_snippet(
            "statagent.stats.descriptive.describe_variable",
            {"dataset_id": params.dataset_id, "column": params.column},
        ),
        "n": n,
    }


@ToolRegistry.register(
    "check_normality",
    "Test whether a numeric column is consistent with a normal distribution "
    "(Shapiro-Wilk test, valid for 3 <= n <= 5000) and return skewness / "
    "kurtosis. Use as a diagnostic before t-tests or Pearson correlation.",
    NormalityParams,
    "Returns the Shapiro-Wilk statistic and p-value, skewness, kurtosis, and "
    "a normal Q-Q plot.",
)
def check_normality(store: DatasetStore, params: NormalityParams) -> dict:
    s = get_series(store, params.dataset_id, params.column)
    num = to_numeric(s, params.column)
    valid = num.dropna().to_numpy(float)
    n = len(valid)
    if n < 3:
        raise ValueError("Normality check requires at least 3 values.")

    result = shapiro = stats.shapiro(valid) if n <= 5000 else None
    summary: dict = {
        "n": n,
        "skewness": round(float(stats.skew(valid)), 6) if n > 2 else None,
        "kurtosis": round(float(stats.kurtosis(valid)), 6) if n > 3 else None,
    }
    notes: list[str] = []
    if shapiro is not None:
        summary["shapiro_w"] = round(float(shapiro.statistic), 6)
        summary["shapiro_p"] = round(float(shapiro.pvalue), 6)
        summary["normal_at_0.05"] = bool(shapiro.pvalue >= 0.05)
        if shapiro.pvalue < 0.05:
            notes.append(
                "Shapiro-Wilk rejects normality at alpha=0.05; consider "
                "nonparametric alternatives (Mann-Whitney U, Spearman)."
            )
    else:
        summary["shapiro_w"] = None
        summary["shapiro_p"] = None
        notes.append("n > 5000: Shapiro-Wilk not computed (CLT applies to means).")

    # Q-Q plot
    observed = np.sort(valid)
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    fig = base_figure(f"Normal Q-Q plot of {params.column}")
    fig.add_trace(go.Scatter(x=expected, y=observed, mode="markers", marker=dict(color=COLOR, size=6)))
    lo, hi = expected.min(), expected.max()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo * observed.std(ddof=1) + observed.mean(), hi * observed.std(ddof=1) + observed.mean()],
            mode="lines",
            line=dict(color=COLOR_2, dash="dash"),
            name="normal line",
        )
    )
    fig.update_xaxes(title="Theoretical normal quantiles")
    fig.update_yaxes(title="Observed quantiles")
    return {
        "summary": summary,
        "details": {"shapiro_raw": [shapiro.statistic, shapiro.pvalue] if shapiro else None},
        "figures": [fig],
        "notes": notes,
        "code": code_snippet(
            "statagent.stats.descriptive.check_normality",
            {"dataset_id": params.dataset_id, "column": params.column},
        ),
        "n": n,
    }
