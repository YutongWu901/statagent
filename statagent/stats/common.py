"""Shared helpers for statistical tools: data extraction, effect sizes,
confidence intervals, figure styling and code snippets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from statagent.data.store import DatasetStore

COLOR = "#3B5BDB"
COLOR_2 = "#E8590C"
COLOR_GRID = "#E4E7F0"

FIG_KWARGS = dict(height=380, template="plotly_white", margin=dict(l=50, r=20, t=55, b=45))


def get_series(store: DatasetStore, dataset_id: str, column: str) -> pd.Series:
    if not store.has(dataset_id):
        raise ValueError(f"Unknown dataset '{dataset_id}'.")
    df = store.get(dataset_id)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")
    return df[column]


def to_numeric(series: pd.Series, what: str = "column") -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().sum() == 0:
        raise ValueError(f"The {what} contains no numeric values.")
    return num


def clean_pair(df: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray, int]:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise ValueError(f"Need at least 3 complete (x, y) pairs, got {len(data)}.")
    return data[x].to_numpy(float), data[y].to_numpy(float), len(data)


def shapiro_safe(x: np.ndarray) -> tuple[float, float] | None:
    """Shapiro-Wilk with the validity window the test supports."""
    n = len(x)
    if n < 3 or n > 5000:
        return None
    if np.std(x) == 0:
        return None
    w, p = stats.shapiro(x)
    return float(w), float(p)


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for Pearson/Spearman r via the Fisher z-transform."""
    if n <= 3:
        raise ValueError("Correlation CI requires n > 3.")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(
        ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    )
    if sp == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / sp)


def welch_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Mean difference (x - y) with Welch-Satterthwaite CI. Returns diff, lo, hi, df."""
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    diff = x.mean() - y.mean()
    return float(diff), float(diff - crit * se), float(diff + crit * se), float(df)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "nan"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def base_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(size=15)), **FIG_KWARGS)
    fig.update_xaxes(gridcolor=COLOR_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=COLOR_GRID, zeroline=False)
    return fig


def code_snippet(fn_path: str, params: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items()))
    return f"{fn_path}(store, params={{{args}}})"
