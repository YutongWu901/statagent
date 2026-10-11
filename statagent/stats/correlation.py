"""Correlation analysis: Pearson, Spearman, correlation matrices."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from statagent.data.store import DatasetStore
from statagent.stats.common import (
    COLOR,
    COLOR_2,
    base_figure,
    clean_pair,
    code_snippet,
    fisher_z_ci,
    fmt_p,
    get_series,
    to_numeric,
)
from statagent.stats.engine import Params, ToolRegistry


class CorrTestParams(Params):
    dataset_id: str
    x: str
    y: str
    method: Literal["pearson", "spearman"] = "pearson"


class CorrMatrixParams(Params):
    dataset_id: str
    columns: list[str]
    method: Literal["pearson", "spearman"] = "pearson"


def _corr_pvalue(r: float, n: int) -> float:
    if abs(r) >= 1.0:
        return 0.0
    from scipy import stats

    t = r * np.sqrt((n - 2) / (1 - r**2))
    return float(2 * stats.t.sf(abs(t), n - 2))


@ToolRegistry.register(
    "correlation_test",
    "Correlation between two numeric columns with a p-value and a 95% CI "
    "for the coefficient. Use 'pearson' for linear association between "
    "continuous variables, 'spearman' when either variable is ordinal, "
    "skewed, or has outliers.",
    CorrTestParams,
    "Returns r, p-value, 95% CI of r, n (complete pairs), and a scatter plot "
    "with an OLS fit line.",
)
def correlation_test(store: DatasetStore, params: CorrTestParams) -> dict:
    df = store.get(params.dataset_id)
    x, y, n = clean_pair(df, params.x, params.y)

    from scipy import stats

    res = stats.pearsonr(x, y) if params.method == "pearson" else stats.spearmanr(x, y)
    r, p = float(res.statistic), float(res.pvalue)
    ci_lo, ci_hi = fisher_z_ci(r, n)
    r2 = r**2

    fig = base_figure(f"{params.method.capitalize()} correlation: {params.x} vs {params.y}")
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", marker=dict(color=COLOR, size=7, opacity=0.6), name="observations"
        )
    )
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(
        go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                   line=dict(color=COLOR_2, width=2), name="OLS fit")
    )
    fig.update_xaxes(title=params.x)
    fig.update_yaxes(title=params.y)

    def _r(v):
        return round(float(v), 6)

    strength = (
        "negligible" if abs(r) < 0.1 else
        "weak" if abs(r) < 0.3 else
        "moderate" if abs(r) < 0.5 else
        "strong" if abs(r) < 0.7 else "very strong"
    )
    return {
        "summary": {
            "method": params.method,
            "r": _r(r),
            "p_value": _r(p),
            "ci95_r": [_r(ci_lo), _r(ci_hi)],
            "r_squared": _r(r2),
            "n": n,
        },
        "details": {
            "r": _r(r),
            "p_value": _r(p),
            "ci95_r": [_r(ci_lo), _r(ci_hi)],
            "r_squared": _r(r2),
            "n": n,
            "p_formatted": fmt_p(p),
            "association_strength": f"{strength} {'positive' if r > 0 else 'negative'} association",
        },
        "figures": [fig],
        "notes": [
            f"{params.method.capitalize()} correlation between '{params.x}' and "
            f"'{params.y}': r = {r:.3f}, p = {fmt_p(p)}, n = {n}.",
            "Correlation measures association, NOT causation.",
        ],
        "code": code_snippet(
            "statagent.stats.correlation.correlation_test",
            {"dataset_id": params.dataset_id, "x": params.x, "y": params.y, "method": params.method},
        ),
        "n": n,
    }


@ToolRegistry.register(
    "correlation_matrix",
    "Pairwise correlation matrix for a set of numeric columns. Use to "
    "overview associations among several variables.",
    CorrMatrixParams,
    "Returns the r matrix, the p-value matrix, and a heatmap.",
)
def correlation_matrix(store: DatasetStore, params: CorrMatrixParams) -> dict:
    df = store.get(params.dataset_id)
    if len(params.columns) < 2:
        raise ValueError("correlation_matrix needs at least 2 columns.")
    for c in params.columns:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(df.columns)}")
    sub = df[params.columns].apply(pd.to_numeric, errors="coerce")
    n_complete = int(sub.dropna().shape[0])

    from scipy import stats

    k = len(params.columns)
    r_mat = np.eye(k)
    p_mat = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            pair = sub[[params.columns[i], params.columns[j]]].dropna()
            if len(pair) < 3:
                continue
            if params.method == "pearson":
                res = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            else:
                res = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            r_mat[i, j] = r_mat[j, i] = float(res.statistic)
            p_mat[i, j] = p_mat[j, i] = float(res.pvalue)

    fig = base_figure(f"{params.method.capitalize()} correlation matrix")
    fig.add_trace(
        go.Heatmap(
            z=r_mat,
            x=params.columns,
            y=params.columns,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=np.round(r_mat, 2),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        )
    )
    return {
        "summary": {
            "method": params.method,
            "columns": params.columns,
            "n_complete_pairs": n_complete,
            "correlations": {
                f"{params.columns[i]} vs {params.columns[j]}": round(float(r_mat[i, j]), 6)
                for i in range(k)
                for j in range(i + 1, k)
            },
        },
        "details": {
            "r_matrix": {c: {d: round(float(r_mat[i, j]), 6) for j, d in enumerate(params.columns)} for i, c in enumerate(params.columns)},
            "p_matrix": {c: {d: round(float(p_mat[i, j]), 6) for j, d in enumerate(params.columns)} for i, c in enumerate(params.columns)},
        },
        "figures": [fig],
        "notes": [
            f"Pairwise {params.method} correlations over {len(params.columns)} columns; "
            f"{n_complete} rows with complete cases."
        ],
        "code": code_snippet(
            "statagent.stats.correlation.correlation_matrix",
            {"dataset_id": params.dataset_id, "columns": params.columns, "method": params.method},
        ),
        "n": n_complete,
    }
