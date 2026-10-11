"""Dataset schemas: variable typing, profiling, and the dataset profile
fed to the LLM planner and the validation engine."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class VariableType(str, Enum):
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    BINARY = "binary"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    UNSUPPORTED = "unsupported"


#: Types that can be fed into numeric statistical procedures.
NUMERIC_TYPES = {
    VariableType.NUMERIC_CONTINUOUS,
    VariableType.NUMERIC_DISCRETE,
}
#: Types that can serve as grouping variables.
CATEGORICAL_TYPES = {
    VariableType.CATEGORICAL_NOMINAL,
    VariableType.CATEGORICAL_ORDINAL,
    VariableType.BINARY,
}


class NumericSummary(BaseModel):
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    variance: float | None = None
    min: float | None = None
    max: float | None = None
    q1: float | None = None
    q3: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None


class VariableProfile(BaseModel):
    name: str
    dtype: str
    variable_type: VariableType
    n: int
    n_missing: int
    missing_pct: float
    n_unique: int
    numeric: NumericSummary | None = None
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    n_outliers: int | None = None
    is_constant: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.variable_type in NUMERIC_TYPES

    @property
    def is_categorical(self) -> bool:
        return self.variable_type in CATEGORICAL_TYPES

    def llm_line(self) -> str:
        """One compact line describing the variable for LLM context."""
        parts = [
            f"- `{self.name}`: {self.variable_type.value}",
            f"n={self.n}",
            f"missing={self.n_missing} ({self.missing_pct:.1f}%)",
            f"unique={self.n_unique}",
        ]
        if self.numeric is not None:
            ns = self.numeric
            parts.append(
                f"min={ns.min:.4g} max={ns.max:.4g} mean={ns.mean:.4g} "
                f"median={ns.median:.4g} sd={ns.std:.4g}"
            )
        if self.top_values:
            top = ", ".join(f"{v!r}×{c}" for v, c in self.top_values[:5])
            parts.append(f"top={{{top}}}")
        if self.n_outliers:
            parts.append(f"outliers≈{self.n_outliers}")
        if self.is_constant:
            parts.append("CONSTANT")
        return "".join(parts)


class DatasetProfile(BaseModel):
    dataset_id: str
    source_name: str
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    columns: list[VariableProfile]
    sha256: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    notes: list[str] = Field(default_factory=list)

    def column(self, name: str) -> VariableProfile | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> list[VariableProfile]:
        return [c for c in self.columns if c.is_numeric]

    @property
    def categorical_columns(self) -> list[VariableProfile]:
        return [c for c in self.columns if c.is_categorical]

    def llm_context(self) -> str:
        """Text block describing the dataset for planner/interpreter prompts."""
        lines = [
            f"Dataset `{self.dataset_id}` ({self.source_name}):",
            f"{self.n_rows} rows, {self.n_cols} columns, "
            f"{self.n_duplicate_rows} duplicate rows.",
        ]
        lines += [c.llm_line() for c in self.columns]
        if self.notes:
            lines.append("Profiler notes: " + "; ".join(self.notes))
        return "\n".join(lines)


def frame_sha256(df) -> str:  # noqa: ANN001 - pragmatic signature kept loose
    """Deterministic content hash of a DataFrame (CSV text of the values)."""
    return hashlib.sha256(
        df.to_csv(index=False, header=True).encode("utf-8", errors="replace")
    ).hexdigest()


#: Column-name hints suggesting ordinal variables even when stored as numbers.
_ORDINAL_NAME_HINTS = re.compile(
    r"(satisf|rating|rank|level|scale|score|grade|likert|agree|"
    r"满意|评分|等级|级别|排行|量表)", re.IGNORECASE
)

#: Column-name hints suggesting identifier-like variables.
_ID_NAME_HINTS = re.compile(r"(^id$|_id$|^uuid$|编号|序号)", re.IGNORECASE)
