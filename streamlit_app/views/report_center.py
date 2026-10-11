"""Report Center: browse, view and export saved analysis reports."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from statagent.models.results import AnalysisRecord
from statagent.report.generator import build_html, build_markdown
from streamlit_app.state import get_cfg, get_records, get_store

st.title("📄 Report Center")
cfg = get_cfg()

# Session records plus records persisted to disk.
records: dict[str, AnalysisRecord] = dict(get_records())
if cfg.reports_dir.exists():
    for path in sorted(cfg.reports_dir.glob("*.json")):
        try:
            record = AnalysisRecord(**json.loads(path.read_text(encoding="utf-8")))
            records[record.record_id] = record
        except Exception:  # noqa: BLE001 - skip corrupt files silently
            continue

if not records:
    st.info("No reports yet. Complete an analysis in the **AI Analysis** page.")
    st.stop()

pick = st.selectbox(
    "Report",
    list(records),
    format_func=lambda rid: f"{rid} — {records[rid].question[:70]}",
)
record = records[pick]

st.markdown(f"**{record.question}**")
st.caption(
    f"{record.created_at} · dataset `{record.dataset_source}` · engine {record.engine_version} · "
    f"llm mode {record.llm.get('mode', 'offline')}"
)

dl1, dl2, dl3 = st.columns(3)
dl1.download_button(
    "Download Markdown", build_markdown(record), file_name=f"{record.record_id}.md",
    use_container_width=True,
)
dl2.download_button(
    "Download HTML", build_html(record), file_name=f"{record.record_id}.html",
    use_container_width=True,
)
dl3.download_button(
    "Download record (JSON)", record.to_json(), file_name=f"{record.record_id}.json",
    use_container_width=True,
)

with st.expander("Full report (Markdown)", expanded=True):
    st.markdown(build_markdown(record))

st.caption(
    "HTML reports embed interactive Plotly charts and require internet access "
    "for the Plotly CDN when opened locally."
)
