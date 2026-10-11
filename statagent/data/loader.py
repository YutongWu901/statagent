"""Data loading for CSV and XLSX files with robust encoding handling."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Raised when an uploaded file cannot be read or is empty."""


#: Encoding candidates tried in order for text-like files.
CSV_ENCODINGS = ("utf-8-sig", "gbk", "cp1252", "latin-1")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Drop fully empty rows / columns.
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:  # noqa: PERF203
            last_err = exc
    raise DataLoadError(f"Could not parse file as CSV ({last_err}).")


def load_dataframe(data: bytes | str | Path, filename: str) -> pd.DataFrame:
    """Load a DataFrame from file bytes/path, dispatching on extension."""
    name = Path(filename).name if isinstance(filename, str) else "data"
    suffix = Path(name).suffix.lower()

    if suffix in (".csv", ".txt", ".tsv", ".data"):
        raw = data.encode("utf-8") if isinstance(data, str) else data
        try:
            df = _read_csv_bytes(raw)
        except DataLoadError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise DataLoadError(f"Could not read CSV: {exc}") from exc
    elif suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(io.BytesIO(data) if isinstance(data, bytes) else data)
        except Exception as exc:
            raise DataLoadError(
                f"Could not read Excel file: {exc}. "
                "Legacy .xls files may need conversion to .xlsx."
            ) from exc
    else:
        raise DataLoadError(
            f"Unsupported file type '{suffix}'. Please upload a CSV or XLSX file."
        )

    if df is None or df.shape[1] == 0:
        raise DataLoadError("The file contains no columns.")
    if df.shape[0] == 0:
        raise DataLoadError("The file contains no data rows.")
    return _clean_frame(df)
