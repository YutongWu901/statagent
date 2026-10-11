"""Shared UI components: badges, findings, step results, figures, CSS."""
from __future__ import annotations

import json

import plotly.io as pio
import streamlit as st

from statagent.models.results import Finding, StepResult, ValidationReport
from statagent.utils import json_dumps

STATUS_COLORS = {"PASS": "#2B8A3E", "WARNING": "#E67700", "FAIL": "#C92A2A", "UNKNOWN": "#495057"}
STATUS_ICONS = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "⛔", "UNKNOWN": "❓"}

_CSS = """
<style>
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600;
        font-size:0.78rem; color:#fff; margin-right:6px;}
.card {border:1px solid #e3e7f1; border-radius:10px; padding:14px 16px;
       background:#fbfcfe; margin-bottom:10px;}
.finding {border-left:4px solid #3B5BDB; padding:6px 12px; margin:6px 0;
          background:#f7f8fc; border-radius:0 8px 8px 0;}
.finding.WARNING {border-color:#E67700;} .finding.FAIL {border-color:#C92A2A;}
.finding.PASS {border-color:#2B8A3E;} .finding.UNKNOWN {border-color:#868e96;}
.kpi {text-align:center; padding:10px 6px; border-radius:10px; background:#f2f4fa;}
.kpi .v {font-size:1.5rem; font-weight:700; color:#1f2430;}
.kpi .l {font-size:0.78rem; color:#5c6470;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#495057")
    return f"<span class='badge' style='background:{color}'>{status}</span>"


def render_findings(report: ValidationReport | list[Finding], prefix: str = "") -> None:
    findings = report.findings if isinstance(report, ValidationReport) else report
    if not findings:
        st.caption("No validation findings.")
        return
    counts = {"PASS": 0, "WARNING": 0, "FAIL": 0, "UNKNOWN": 0}
    for f in findings:
        counts[f.status] += 1
    st.markdown(
        " ".join(badge(s) + f"×{counts[s]}" if counts[s] else "" for s in ("PASS", "WARNING", "FAIL", "UNKNOWN")),
        unsafe_allow_html=True,
    )
    for i, f in enumerate(findings):
        with st.expander(
            f"{STATUS_ICONS[f.status]} {f.rule_id} — {f.message[:90]}",
            expanded=f.status in ("FAIL", "UNKNOWN"),
        ):
            st.markdown(f"**Category:** {f.category} · **Severity:** {f.severity}")
            if f.evidence:
                st.markdown(f"**Evidence:** {f.evidence}")
            if f.suggestion:
                st.markdown(f"**Suggestion:** {f.suggestion}")


def fig_from_json(fig_dict: dict):
    return pio.from_json(json.dumps(fig_dict, ensure_ascii=False))


def _flatten(d: dict, prefix: str = ""):
    out = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.append((key, json_dumps(v)))
        else:
            out.append((key, v))
    return out


def render_step_result(step: StepResult, key: str = "") -> None:
    r = step.result
    with st.container(border=True):
        cols = st.columns([3, 1])
        cols[0].markdown(f"**Step {step.step_id} — `{r.tool}`**")
        if r.ok:
            cols[1].markdown(badge("OK") + f" n = {r.n if r.n is not None else '—'}", unsafe_allow_html=True)
        else:
            cols[1].markdown(badge("ERROR"), unsafe_allow_html=True)
        if r.code:
            st.code(r.code, language="python")
        if not r.ok:
            st.error(r.error or "unknown error")
            return
        if r.summary:
            rows = _flatten(r.summary)
            st.dataframe(
                [{"metric": k, "value": v} for k, v in rows],
                use_container_width=True,
                hide_index=True,
            )
        for i, fig_dict in enumerate(r.figures):
            try:
                st.plotly_chart(
                    fig_from_json(fig_dict), use_container_width=True, key=f"{key}_{r.tool}_{i}"
                )
            except Exception:  # noqa: BLE001
                continue
        for note in r.notes:
            st.markdown(f"> {note}")
