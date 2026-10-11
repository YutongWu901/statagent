"""Report generation: AnalysisRecord -> Markdown + self-contained HTML.

Reports include the question, clarifications, the analysis plan, every
computed result with figures, both validation reports, the interpretation,
and a reproducibility appendix (versions, dataset hash, seeds, exact tool
calls). Nothing is embellished: the report reflects exactly what ran.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import plotly.io as pio

from statagent import __version__
from statagent.models.results import AnalysisRecord, Finding, StepResult
from statagent.utils import json_dumps

STATUS_COLORS = {"PASS": "#2B8A3E", "WARNING": "#E67700", "FAIL": "#C92A2A", "UNKNOWN": "#495057"}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1e6 or (0 < abs(v) < 1e-4):
            return f"{v:.4g}"
        return f"{v:.4f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _flatten(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten(v, key))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.append((key, json_dumps(v)))
        else:
            out.append((key, v))
    return out


def _md_table(rows: list[tuple[str, Any]], headers: tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in rows]
    return "\n".join(lines)


def finding_md(f: Finding) -> str:
    return (
        f"- **{f.status}** `{f.rule_id}` ({f.category}) — {f.message}"
        + (f"  \n  *Evidence:* {f.evidence}" if f.evidence else "")
        + (f"  \n  *Suggestion:* {f.suggestion}" if f.suggestion else "")
    )


def findings_md(findings: list[Finding], title: str) -> str:
    lines = [f"### {title}", ""]
    if not findings:
        lines.append("_(no findings)_")
    else:
        lines += [finding_md(f) for f in findings]
    return "\n".join(lines) + "\n"


def step_md(step: StepResult, figure_links: dict[int, list[str]] | None = None) -> str:
    r = step.result
    lines = [f"#### Step {step.step_id}: `{r.tool}`", ""]
    if r.code:
        lines.append(f"```python\n{r.code}\n```\n")
    if not r.ok:
        lines.append(f"**Failed:** {r.error}\n")
        return "\n".join(lines)
    if r.n is not None:
        lines.append(f"_n = {r.n}_\n")
    if r.summary:
        lines.append("**Key results:**\n")
        lines.append(_md_table(_flatten(r.summary), ("Metric", "Value")))
        lines.append("")
    if r.details:
        lines.append("**Full details:**\n")
        lines.append(_md_table(_flatten(r.details), ("Metric", "Value")))
        lines.append("")
    for note in r.notes:
        lines.append(f"> {note}")
    if r.notes:
        lines.append("")
    links = (figure_links or {}).get(step.step_id, [])
    for i, link in enumerate(links):
        lines.append(f"**Figure {i + 1}:** [open interactive chart]({link})")
    if links:
        lines.append("")
    return "\n".join(lines)


def plan_md(record: AnalysisRecord) -> str:
    p = record.plan
    lines = [f"### Analysis plan", "", f"**Goal:** {p.goal}", ""]
    lines.append(
        "**Design facts:** "
        + "; ".join(
            f"{k} = {v}" for k, v in p.design.model_dump().items() if v not in ("unknown", None)
        )
        + (" (unknown fields omitted)" if p.design.unknowns() else "")
        + "\n"
    )
    if p.method_rationale:
        lines.append(f"**Method choice rationale:** {p.method_rationale}\n")
    if p.steps:
        lines.append("| # | Tool | Variables | Rationale |")
        lines.append("|---|---|---|---|")
        for s in p.steps:
            lines.append(
                f"| {s.step_id} | `{s.tool}` | {', '.join(s.variables)} | {s.rationale} |"
            )
        lines.append("")
    if p.limitations:
        lines.append("**Plan-level limitations:**")
        lines += [f"- {x}" for x in p.limitations]
        lines.append("")
    return "\n".join(lines)


def build_markdown(record: AnalysisRecord, figure_links: dict[int, list[str]] | None = None) -> str:
    parts: list[str] = []
    parts.append(f"# StatAgent Analysis Report — {record.record_id}")
    parts.append("")
    parts.append(f"- **Created:** {record.created_at}")
    parts.append(f"- **Dataset:** {record.dataset_source} (`{record.dataset_id}`), sha256 `{record.dataset_sha256[:16]}…`")
    parts.append(f"- **Engine version:** {record.engine_version}")
    parts.append("")
    parts.append("## Research question")
    parts.append("")
    parts.append(f"> {record.question}")
    parts.append("")
    if record.clarifications:
        parts.append("## Clarifications")
        parts.append("")
        for ex in record.clarifications:
            parts.append(f"- **Q:** {ex.question}  ")
            parts.append(f"  **A:** {ex.answer}")
        parts.append("")
    parts.append("## Analysis plan")
    parts.append("")
    parts.append(plan_md(record))
    if record.validation_before is not None:
        parts.append(findings_md(
            record.validation_before.findings,
            f"Validation before execution (gated={record.validation_before.gated}, "
            f"blocked={record.validation_before.blocked}, override={record.validation_before.override})",
        ))
    parts.append("## Results")
    parts.append("")
    if not record.step_results:
        parts.append("_(no steps executed)_\n")
    for step in record.step_results:
        parts.append(step_md(step, figure_links))
    if record.interpretation is not None:
        interp = record.interpretation
        parts.append("## Interpretation")
        parts.append("")
        parts.append(f"**Summary:** {interp.summary}")
        parts.append("")
        if interp.key_findings:
            parts.append("**Key findings:**")
            parts += [f"- {k}" for k in interp.key_findings]
            parts.append("")
        parts.append(interp.interpretation)
        parts.append("")
        if interp.limitations:
            parts.append("**Limitations:**")
            parts += [f"- {x}" for x in interp.limitations]
            parts.append("")
        parts.append(f"**Conclusion:** {interp.conclusion}")
        parts.append("")
    if record.validation_after is not None:
        parts.append(findings_md(record.validation_after.findings, "Validation of interpretation"))
    parts.append("## Reproducibility")
    parts.append("")
    parts.append(f"- Engine: StatAgent {record.engine_version} (Python statistics via scipy/statsmodels-grade implementations)")
    parts.append(f"- LLM: mode={record.llm.get('mode', 'offline')}, model={record.llm.get('model', 'none')}, temperature={record.llm.get('temperature', '0')}")
    parts.append(f"- Seed: {record.seed}")
    parts.append(f"- Dataset hash: `{record.dataset_sha256}`")
    parts.append("- All statistical numbers were computed by the Python engine from the data; the LLM only planned and paraphrased.")
    parts.append("")
    return "\n".join(parts)


_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       color: #1f2430; margin: 2rem auto; max-width: 980px; padding: 0 1.2rem; line-height: 1.55; }
h1 { font-size: 1.7rem; border-bottom: 3px solid #3B5BDB; padding-bottom: .4rem; }
h2 { font-size: 1.3rem; margin-top: 2.2rem; color: #2b3450; }
h3 { font-size: 1.1rem; margin-top: 1.6rem; }
h4 { font-size: 1rem; margin-top: 1.2rem; color: #3B5BDB; }
table { border-collapse: collapse; margin: .6rem 0 1.2rem; font-size: .92rem; }
th, td { border: 1px solid #dde1ec; padding: .35rem .7rem; text-align: left; }
th { background: #f2f4fa; }
blockquote { border-left: 4px solid #3B5BDB; margin: .4rem 0; padding: .2rem 1rem; background: #f7f8fc; }
code { background: #f2f4fa; padding: .1rem .3rem; border-radius: 4px; }
pre { background: #f2f4fa; padding: .8rem; border-radius: 6px; overflow-x: auto; }
.meta { color: #5c6470; font-size: .9rem; }
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 10px; font-weight: 600;
         font-size: .78rem; color: #fff; }
"""


def findings_html(findings: list[Finding], title: str) -> str:
    rows = []
    for f in findings:
        color = STATUS_COLORS[f.status]
        rows.append(
            f"<tr><td><span class='badge' style='background:{color}'>{f.status}</span></td>"
            f"<td><code>{f.rule_id}</code></td><td>{f.category}</td>"
            f"<td>{html.escape(f.message)}"
            + (f"<br><em>Evidence:</em> {html.escape(f.evidence)}" if f.evidence else "")
            + (f"<br><em>Suggestion:</em> {html.escape(f.suggestion)}" if f.suggestion else "")
            + "</td></tr>"
        )
    if not rows:
        rows = ["<tr><td colspan='4'><em>no findings</em></td></tr>"]
    return (
        f"<h3>{html.escape(title)}</h3><table><tr><th>Status</th><th>Rule</th><th>Category</th>"
        f"<th>Finding</th></tr>{''.join(rows)}</table>"
    )


def step_html(step: StepResult) -> str:
    r = step.result
    parts = [f"<h4>Step {step.step_id}: <code>{html.escape(r.tool)}</code></h4>"]
    if r.code:
        parts.append(f"<pre>{html.escape(r.code)}</pre>")
    if not r.ok:
        parts.append(f"<p><strong>Failed:</strong> {html.escape(r.error or '')}</p>")
        return "".join(parts)
    if r.n is not None:
        parts.append(f"<p class='meta'>n = {r.n}</p>")
    for label, data in (("Key results", r.summary), ("Full details", r.details)):
        if not data:
            continue
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt(v))}</td></tr>"
            for k, v in _flatten(data)
        )
        parts.append(f"<p><strong>{label}</strong></p><table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>")
    if r.notes:
        parts.append("".join(f"<blockquote>{html.escape(n)}</blockquote>" for n in r.notes))
    for fig_dict in r.figures:
        try:
            fig = pio.from_json(json_dumps(fig_dict))
            parts.append(
                fig.to_html(full_html=False, include_plotlyjs=False)
                .replace("<!-- Plotly charts -->", "")
            )
        except Exception:  # noqa: BLE001 - a broken figure must not break the report
            continue
    return "".join(parts)


def build_html(record: AnalysisRecord) -> str:
    body = [f"<h1>StatAgent Analysis Report — {html.escape(record.record_id)}</h1>"]
    body.append(
        f"<p class='meta'>Created {html.escape(record.created_at)} · Dataset "
        f"{html.escape(record.dataset_source)} (<code>{html.escape(record.dataset_id)}</code>) "
        f"· sha256 <code>{html.escape(record.dataset_sha256[:16])}…</code> · engine {html.escape(record.engine_version)}</p>"
    )
    body.append(f"<h2>Research question</h2><blockquote>{html.escape(record.question)}</blockquote>")
    if record.clarifications:
        body.append("<h2>Clarifications</h2><ul>")
        for ex in record.clarifications:
            body.append(
                f"<li><strong>Q:</strong> {html.escape(ex.question)}<br>"
                f"<strong>A:</strong> {html.escape(ex.answer)}</li>"
            )
        body.append("</ul>")
    body.append("<h2>Analysis plan</h2>")
    body.append(f"<p><strong>Goal:</strong> {html.escape(record.plan.goal)}</p>")
    if record.plan.method_rationale:
        body.append(f"<p>{html.escape(record.plan.method_rationale)}</p>")
    if record.plan.steps:
        rows = "".join(
            f"<tr><td>{s.step_id}</td><td><code>{html.escape(s.tool)}</code></td>"
            f"<td>{html.escape(', '.join(s.variables))}</td><td>{html.escape(s.rationale)}</td></tr>"
            for s in record.plan.steps
        )
        body.append(
            f"<table><tr><th>#</th><th>Tool</th><th>Variables</th><th>Rationale</th></tr>{rows}</table>"
        )
    if record.plan.limitations:
        body.append("<p><strong>Limitations:</strong></p><ul>")
        body += [f"<li>{html.escape(x)}</li>" for x in record.plan.limitations]
        body.append("</ul>")
    if record.validation_before is not None:
        body.append(
            findings_html(
                record.validation_before.findings,
                f"Validation before execution (blocked={record.validation_before.blocked}, "
                f"override={record.validation_before.override})",
            )
        )
    body.append("<h2>Results</h2>")
    if not record.step_results:
        body.append("<p><em>no steps executed</em></p>")
    for step in record.step_results:
        body.append(step_html(step))
    if record.interpretation is not None:
        interp = record.interpretation
        body.append("<h2>Interpretation</h2>")
        body.append(f"<p><strong>Summary:</strong> {html.escape(interp.summary)}</p>")
        if interp.key_findings:
            body.append("<p><strong>Key findings:</strong></p><ul>")
            body += [f"<li>{html.escape(k)}</li>" for k in interp.key_findings]
            body.append("</ul>")
        body.append(f"<p>{html.escape(interp.interpretation)}</p>")
        if interp.limitations:
            body.append("<p><strong>Limitations:</strong></p><ul>")
            body += [f"<li>{html.escape(x)}</li>" for x in interp.limitations]
            body.append("</ul>")
        body.append(f"<p><strong>Conclusion:</strong> {html.escape(interp.conclusion)}</p>")
    if record.validation_after is not None:
        body.append(findings_html(record.validation_after.findings, "Validation of interpretation"))
    body.append("<h2>Reproducibility</h2><ul>")
    body.append(f"<li>Engine: StatAgent {html.escape(record.engine_version)}</li>")
    body.append(
        f"<li>LLM: mode={html.escape(record.llm.get('mode', 'offline'))}, "
        f"model={html.escape(record.llm.get('model', 'none'))}</li>"
    )
    body.append(f"<li>Seed: {record.seed}</li>")
    body.append(f"<li>Dataset hash: <code>{html.escape(record.dataset_sha256)}</code></li>")
    body.append(
        "<li>All statistical numbers were computed by the Python engine from the data.</li></ul>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>StatAgent report {html.escape(record.record_id)}</title>"
        "<script src='https://cdn.plot.ly/plotly-3.1.0.min.js' charset='utf-8'></script>"
        f"<style>{_CSS}</style></head><body>{''.join(body)}</body></html>"
    )


def save_report(record: AnalysisRecord, out_dir: Path) -> dict[str, Path]:
    """Persist the record JSON plus Markdown and HTML reports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / record.record_id
    (out_dir / f"{record.record_id}.json").write_text(
        json_dumps(record.model_dump(mode="json")), encoding="utf-8"
    )
    md_path = Path(f"{base}.md")
    md_path.write_text(build_markdown(record), encoding="utf-8")
    html_path = Path(f"{base}.html")
    html_path.write_text(build_html(record), encoding="utf-8")
    return {"json": out_dir / f"{record.record_id}.json", "md": md_path, "html": html_path}
