"""Tests for CSV/XLSX loading with encoding fallback and error handling."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from statagent.data.loader import DataLoadError, load_dataframe


def test_load_csv_utf8():
    raw = b"a,b\n1,2\n3,4\n"
    df = load_dataframe(raw, "t.csv")
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_csv_gbk_encoding():
    raw = "姓名,分数\n张三,90\n李四,80\n".encode("gbk")
    df = load_dataframe(raw, "t.csv")
    assert list(df.columns) == ["姓名", "分数"]
    assert df.shape == (2, 2)


def test_load_csv_whitespace_columns_trimmed():
    raw = b" a , b \n1,2\n"
    df = load_dataframe(raw, "t.csv")
    assert list(df.columns) == ["a", "b"]


def test_load_xlsx():
    buf = io.BytesIO()
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(buf, index=False)
    df = load_dataframe(buf.getvalue(), "t.xlsx")
    assert df.shape == (2, 2)


def test_load_empty_csv_raises():
    with pytest.raises(DataLoadError):
        load_dataframe(b"", "t.csv")


def test_load_unsupported_extension_raises():
    with pytest.raises(DataLoadError):
        load_dataframe(b"abc", "t.parquet")


def test_load_empty_rows_dropped():
    raw = b"a,b\n1,2\n,\n"
    df = load_dataframe(raw, "t.csv")
    assert df.shape == (1, 2)
