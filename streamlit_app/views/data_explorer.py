"""Data Explorer: upload, profiling, variable inspection, relationships."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from statagent.data.loader import DataLoadError, load_dataframe
from statagent.stats.engine import ToolRegistry, execute_tool
from streamlit_app.state import get_cfg, get_store
from streamlit_app.ui import fig_from_json

st.title("📊 Data Explorer")
store = get_store()
cfg = get_cfg()

upload_col, demo_col = st.columns(2)
with upload_col:
    uploaded = st.file_uploader(
        "Upload a dataset", type=["csv", "xlsx", "xls", "txt"],
        help="CSV or Excel files. Column types, missing values, duplicates and outliers are detected automatically.",
    )
    if uploaded is not None:
        try:
            df = load_dataframe(uploaded.getvalue(), uploaded.name)
            dataset_id = store.add(df, uploaded.name)
            st.success(f"Loaded **{uploaded.name}** ({df.shape[0]} rows × {df.shape[1]} columns) as `{dataset_id}`.")
        except DataLoadError as exc:
            st.error(str(exc))

with demo_col:
    demo_files = sorted(p.name for p in cfg.demo_data_dir.glob("*")) if cfg.demo_data_dir.exists() else []
    if demo_files:
        choice = st.selectbox("…or load demo data", ["—"] + demo_files)
        if choice != "—":
            path = cfg.demo_data_dir / choice
            try:
                df = load_dataframe(path.read_bytes(), choice)
                dataset_id = store.add(df, choice)
                st.success(f"Loaded demo dataset **{choice}** as `{dataset_id}`.")
            except DataLoadError as exc:
                st.error(str(exc))
    else:
        st.caption("Run `python scripts/generate_demo_data.py` to create demo datasets.")

if len(store) == 0:
    st.info("Upload a CSV/XLSX file or load a demo dataset to begin.")
    st.stop()

dataset_id = st.selectbox("Dataset", store.ids(), format_func=lambda i: f"{i} — {store.source_name(i)}")
profile = store.profile(dataset_id)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
n_missing_cells = int(sum(c.n_missing for c in profile.columns))
kpi1.metric("Rows", profile.n_rows)
kpi2.metric("Columns", profile.n_cols)
kpi3.metric("Duplicate rows", profile.n_duplicate_rows)
kpi4.metric("Missing cells", n_missing_cells)
kpi5.metric("Numeric / categorical", f"{len(profile.numeric_columns)} / {len(profile.categorical_columns)}")

tab_overview, tab_vars, tab_rel, tab_quality = st.tabs(
    ["Overview", "Variable explorer", "Relationships", "Quality checks"]
)

with tab_overview:
    rows = [
        {
            "column": c.name,
            "type": c.variable_type.value,
            "n": c.n,
            "missing": f"{c.n_missing} ({c.missing_pct:.1f}%)",
            "unique": c.n_unique,
            "min": c.numeric.min if c.numeric else "",
            "max": c.numeric.max if c.numeric else "",
            "mean": round(c.numeric.mean, 2) if c.numeric and c.numeric.mean is not None else "",
            "median": round(c.numeric.median, 2) if c.numeric and c.numeric.median is not None else "",
            "outliers": c.n_outliers if c.n_outliers is not None else "",
        }
        for c in profile.columns
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.dataframe(store.get(dataset_id).head(10), use_container_width=True)

with tab_vars:
    col = st.selectbox("Column", profile.names())
    cp = profile.column(col)
    if cp is None:
        st.stop()
    st.markdown(
        f"**{col}** — `{cp.variable_type.value}` · n = {cp.n} · missing = "
        f"{cp.n_missing} ({cp.missing_pct:.1f}%) · unique = {cp.n_unique}"
    )
    if cp.is_numeric:
        result = execute_tool(store, "describe_variable", {"dataset_id": dataset_id, "column": col})
        for fig_dict in result.figures:
            st.plotly_chart(fig_from_json(fig_dict), use_container_width=True)
    elif cp.top_values:
        counts = pd.DataFrame([{"value": v, "count": c} for v, c in cp.top_values])
        fig = go.Figure(go.Bar(x=counts["value"], y=counts["count"], marker_color="#3B5BDB"))
        fig.update_layout(title=f"Top categories of {col}", template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

with tab_rel:
    numeric_names = [c.name for c in profile.numeric_columns]
    if len(numeric_names) >= 2:
        method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
        result = execute_tool(
            store, "correlation_matrix",
            {"dataset_id": dataset_id, "columns": numeric_names, "method": method},
        )
        for fig_dict in result.figures:
            st.plotly_chart(fig_from_json(fig_dict), use_container_width=True)
        x_col = st.selectbox("X", numeric_names, index=0)
        y_col = st.selectbox("Y", numeric_names, index=min(1, len(numeric_names) - 1))
        result = execute_tool(
            store, "correlation_test",
            {"dataset_id": dataset_id, "x": x_col, "y": y_col, "method": method},
        )
        for fig_dict in result.figures:
            st.plotly_chart(fig_from_json(fig_dict), use_container_width=True)
    else:
        st.caption("Need at least two numeric columns for relationship plots.")

with tab_quality:
    issues = []
    if profile.n_duplicate_rows:
        issues.append(f"⚠️ {profile.n_duplicate_rows} duplicated row(s).")
    for c in profile.columns:
        if c.missing_pct >= 20:
            issues.append(f"⚠️ `{c.name}` has {c.missing_pct:.0f}% missing values.")
        if c.is_constant:
            issues.append(f"⚠️ `{c.name}` is constant (no variance).")
        if (c.n_outliers or 0) > 0:
            issues.append(f"⚠️ `{c.name}` has {c.n_outliers} IQR outliers.")
        if c.variable_type.value == "identifier":
            issues.append(f"ℹ️ `{c.name}` looks like an identifier.")
    if issues:
        for i in issues:
            st.markdown(i)
    else:
        st.success("No quality issues detected at the dataset level.")
