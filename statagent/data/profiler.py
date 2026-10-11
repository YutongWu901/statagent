"""Automatic data profiling: variable typing, missing values, outliers,
and the LLM-facing dataset description."""
from __future__ import annotations

import numpy as np
import pandas as pd

from statagent.models.dataset import (
    NUMERIC_TYPES,
    _ID_NAME_HINTS,
    _ORDINAL_NAME_HINTS,
    DatasetProfile,
    NumericSummary,
    VariableProfile,
    VariableType,
    frame_sha256,
)

#: Thresholds for type inference on numeric columns.
_BINARY_MAX_UNIQUE = 2
_DISCRETE_MAX_UNIQUE = 10
#: Fraction of values that must parse for a string column to be treated as numeric.
_NUMERIC_COERCION_MIN = 0.95


def _is_integer_like(series: pd.Series) -> bool:
    try:
        return bool(np.all(np.equal(np.mod(series.dropna(), 1), 0)))
    except TypeError:
        return False


def infer_variable_type(series: pd.Series, name: str) -> tuple[VariableType, pd.Series | None]:
    """Infer the statistical type of a column.

    Returns the type and, for object columns coerced to numeric, the coerced
    series (so summaries are computed on real numbers).
    """
    if pd.api.types.is_bool_dtype(series):
        return VariableType.BINARY, None
    if pd.api.types.is_datetime64_any_dtype(series):
        return VariableType.DATETIME, None

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()
        n_unique = valid.nunique()
        if n_unique <= _BINARY_MAX_UNIQUE:
            return VariableType.BINARY, None
        if _ORDINAL_NAME_HINTS.search(name) and n_unique <= _DISCRETE_MAX_UNIQUE:
            return VariableType.CATEGORICAL_ORDINAL, None
        if n_unique <= _DISCRETE_MAX_UNIQUE and not _ID_NAME_HINTS.search(name):
            return VariableType.NUMERIC_DISCRETE, None
        return VariableType.NUMERIC_CONTINUOUS, None

    # Object / category columns.
    non_null = series.dropna().astype(str)
    n = series.notna().sum()
    n_unique = series.nunique()

    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.notna().sum() >= _NUMERIC_COERCION_MIN * max(n, 1):
        return infer_variable_type(coerced, name)[0], coerced

    if n_unique == 0:
        return VariableType.UNSUPPORTED, None
    if _ID_NAME_HINTS.search(name):
        return VariableType.IDENTIFIER, None
    if n_unique == n and n > 20:
        return VariableType.IDENTIFIER, None
    if non_null.str.len().mean() > 40 and n_unique == n:
        return VariableType.TEXT, None
    return VariableType.CATEGORICAL_NOMINAL, None


def numeric_summary(series: pd.Series) -> NumericSummary | None:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) == 0:
        return None
    from scipy.stats import kurtosis, skew

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    return NumericSummary(
        mean=float(valid.mean()),
        median=float(valid.median()),
        std=float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        variance=float(valid.var(ddof=1)) if len(valid) > 1 else 0.0,
        min=float(valid.min()),
        max=float(valid.max()),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        skewness=float(skew(valid)) if len(valid) > 2 else None,
        kurtosis=float(kurtosis(valid)) if len(valid) > 3 else None,
    )


def count_outliers(series: pd.Series, k: float = 1.5) -> int:
    """Count IQR-rule outliers in a numeric series."""
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < 4:
        return 0
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return int(((valid < lo) | (valid > hi)).sum())


def profile_frame(
    df: pd.DataFrame, dataset_id: str, source_name: str, notes: list[str] | None = None
) -> DatasetProfile:
    columns: list[VariableProfile] = []
    for name in df.columns:
        series = df[name]
        vtype, coerced = infer_variable_type(series, str(name))
        if vtype in (VariableType.DATETIME, VariableType.TEXT, VariableType.UNSUPPORTED):
            num = None
            outliers = None
        else:
            base = coerced if coerced is not None else series
            num = numeric_summary(base)
            outliers = count_outliers(base) if vtype in NUMERIC_TYPES else None
        n_missing = int(series.isna().sum()) + int(
            (series.astype(str).str.strip() == "").sum()
        )
        top_values: list[tuple[str, int]] = []
        if vtype in (VariableType.CATEGORICAL_NOMINAL, VariableType.CATEGORICAL_ORDINAL):
            counts = series.astype(str).value_counts().head(5)
            top_values = [(str(v), int(c)) for v, c in counts.items()]
        columns.append(
            VariableProfile(
                name=str(name),
                dtype=str(series.dtype),
                variable_type=vtype,
                n=int(series.notna().sum()),
                n_missing=n_missing,
                missing_pct=round(100.0 * n_missing / max(len(df), 1), 2),
                n_unique=int(series.nunique(dropna=True)),
                numeric=num,
                top_values=top_values,
                n_outliers=outliers,
                is_constant=bool(series.nunique(dropna=True) <= 1),
            )
        )
    return DatasetProfile(
        dataset_id=dataset_id,
        source_name=source_name,
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        n_duplicate_rows=int(df.duplicated().sum()),
        columns=columns,
        sha256=frame_sha256(df),
        notes=notes or [],
    )
