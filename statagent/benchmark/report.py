"""Benchmark report generation: audit-friendly Markdown and HTML summaries
of a run, including per-system metrics with Wilson CIs and paired
comparisons between System B and System C."""
from __future__ import annotations

import html
from typing import Any

METRIC_LABELS = {
    "task_completion_rate": "Task completion rate",
    "method_selection_accuracy": "Method selection accuracy",
    "numerical_accuracy": "Numerical accuracy",
    "interpretation_accuracy": "Interpretation accuracy",
    "appropriate_clarification_rate": "Appropriate clarification rate",
    "invalid_analysis_rate": "Invalid analysis rate (unflagged)",
    "cost_usd": "Total cost (USD)",
    "mean_time_s": "Mean time per task (s)",
    "cost_per_correct_task_usd": "Cost per correct task (USD)",
}

SYSTEM_LABELS = {
    "A_llm_only": "System A — LLM only",
    "B_llm_tools": "System B — LLM + tools",
    "C_llm_tools_validation": "System C — LLM + tools + validation",
}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)) and v <= 1.0:
        return f"{100 * v:.1f}%"
    return str(v)


def _metric_rows(payload: dict[str, Any]) -> list[tuple[str, list[str]]]:
    summary = payload["summary"]
    systems = [s for s in payload["systems"] if s in summary]
    rows: list[tuple[str, list[str]]] = []
    for key, label in METRIC_LABELS.items():
        cells = []
        for s in systems:
            val = summary[s].get(key)
            cells.append(_pct(val) if "rate" in key or "accuracy" in key else _fmt(val))
        rows.append((label, cells))
    return rows, systems


def build_report_md(run, payload: dict[str, Any]) -> str:
    lines = [f"# StatAgent Benchmark Report — {run.label}", ""]
    lines.append(
        f"- **Run:** {run.label} · started {run.started_at} · finished {run.finished_at}"
    )
    lines.append(
        f"- **LLM mode:** {payload['llm_mode']} · model: {payload['model']} · "
        f"temperature: {payload['temperature']} · base URL: {payload['config']['base_url']}"
    )
    lines.append(f"- **Tasks:** {payload['n_tasks']} (ids: {', '.join(run.tasks_run[:8])}{'…' if len(run.tasks_run) > 8 else ''})")
    lines.append(f"- **Raw results:** `results.jsonl` · **Ground truth:** `ground_truth.json`")
    lines.append("")
    rows, systems = _metric_rows(payload)
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | " + " | ".join(SYSTEM_LABELS.get(s, s) for s in systems) + " |")
    lines.append("|" + "---|" * (len(systems) + 1))
    for label, cells in rows:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    pc = payload["summary"].get("paired_comparison")
    if pc:
        lines.append("## Paired comparison: System B vs System C (McNemar)")
        lines.append("")
        lines.append("| Metric | B only | C only | chi² | p |")
        lines.append("|---|---|---|---|---|")
        for metric, res in pc.items():
            lines.append(
                f"| {metric} | {res['b']} | {res['c']} | {res['chi2']} | {res['p']} |"
            )
        lines.append("")
        lines.append("_Only tasks where the two systems disagree are counted._")
        lines.append("")
    lines.append("## Per-category breakdown")
    lines.append("")
    breakdown = payload["category_breakdown"]
    for cat, systems_dict in breakdown.items():
        lines.append(f"### {cat}")
        lines.append("")
        header = ["System", "n", "Completion", "Method acc.", "Numeric acc.", "Interp. acc.", "Invalid unflagged"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header) + "|")
        for s, m in systems_dict.items():
            lines.append(
                f"| {SYSTEM_LABELS.get(s, s)} | {m['n']} | {_pct(m['completion'])} | "
                f"{_pct(m['method_accuracy'])} | {_pct(m['numeric_accuracy'])} | "
                f"{_pct(m['interpretation_accuracy'])} | {_pct(m['invalid_unflagged_rate'])} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Method selection accuracy is defined only for Systems B and C (System A "
        "computes no methods)."
    )
    lines.append(
        "- Appropriate clarification rate is measured on the insufficient-design "
        "tasks only (System A has no clarification mechanism)."
    )
    lines.append(
        "- Invalid analysis rate counts tasks where the ground truth marks the "
        "executed analysis invalid and the system did not flag or block it."
    )
    lines.append(
        "- Numerical accuracy compares each system's reported values against "
        "reference values computed with independent scipy/numpy code (2% relative "
        "tolerance)."
    )
    lines.append(
        "- If llm_mode is offline, this run validated the pipeline with the "
        "deterministic heuristic planner; it is NOT evidence about the LLM."
    )
    lines.append("")
    return "\n".join(lines)


_HTML_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color: #1f2430;
       margin: 2rem auto; max-width: 1040px; padding: 0 1.2rem; line-height: 1.5; }
h1 { border-bottom: 3px solid #3B5BDB; padding-bottom: .4rem; }
table { border-collapse: collapse; margin: .6rem 0 1.2rem; font-size: .92rem; width: 100%; }
th, td { border: 1px solid #dde1ec; padding: .35rem .7rem; text-align: left; }
th { background: #f2f4fa; }
.meta { color: #5c6470; font-size: .9rem; }
"""


def build_report_html(run, payload: dict[str, Any]) -> str:
    rows, systems = _metric_rows(payload)
    body = [f"<h1>StatAgent Benchmark Report — {html.escape(run.label)}</h1>"]
    body.append(
        f"<p class='meta'>Started {html.escape(run.started_at)} · finished "
        f"{html.escape(run.finished_at)} · LLM mode {html.escape(payload['llm_mode'])} · "
        f"model {html.escape(payload['model'])} · {payload['n_tasks']} tasks</p>"
    )
    body.append("<h2>Headline metrics</h2><table><tr><th>Metric</th>")
    for s in systems:
        body.append(f"<th>{html.escape(SYSTEM_LABELS.get(s, s))}</th>")
    body.append("</tr>")
    for label, cells in rows:
        body.append(f"<tr><td>{html.escape(label)}</td>")
        for cell in cells:
            body.append(f"<td>{html.escape(cell)}</td>")
        body.append("</tr>")
    body.append("</table>")
    pc = payload["summary"].get("paired_comparison")
    if pc:
        body.append("<h2>Paired comparison: System B vs System C (McNemar)</h2><table>")
        body.append("<tr><th>Metric</th><th>B only</th><th>C only</th><th>chi²</th><th>p</th></tr>")
        for metric, res in pc.items():
            body.append(
                f"<tr><td>{html.escape(metric)}</td><td>{res['b']}</td><td>{res['c']}</td>"
                f"<td>{res['chi2']}</td><td>{res['p']}</td></tr>"
            )
        body.append("</table>")
    breakdown = payload["category_breakdown"]
    body.append("<h2>Per-category breakdown</h2>")
    for cat, systems_dict in breakdown.items():
        body.append(f"<h3>{html.escape(cat)}</h3><table>")
        body.append("<tr><th>System</th><th>n</th><th>Completion</th><th>Method acc.</th>"
                    "<th>Numeric acc.</th><th>Interp. acc.</th><th>Invalid unflagged</th></tr>")
        for s, m in systems_dict.items():
            body.append(
                f"<tr><td>{html.escape(SYSTEM_LABELS.get(s, s))}</td><td>{m['n']}</td>"
                f"<td>{html.escape(_pct(m['completion']))}</td>"
                f"<td>{html.escape(_pct(m['method_accuracy']))}</td>"
                f"<td>{html.escape(_pct(m['numeric_accuracy']))}</td>"
                f"<td>{html.escape(_pct(m['interpretation_accuracy']))}</td>"
                f"<td>{html.escape(_pct(m['invalid_unflagged_rate']))}</td></tr>"
            )
        body.append("</table>")
    body.append(
        "<h2>Notes</h2><ul>"
        "<li>Method selection accuracy is defined only for Systems B and C.</li>"
        "<li>Appropriate clarification rate is measured on insufficient-design tasks.</li>"
        "<li>Invalid analysis rate counts ground-truth-invalid analyses the system did not flag or block.</li>"
        "<li>If llm_mode is offline, this run validated the pipeline with the heuristic "
        "planner; it is not evidence about the LLM itself.</li></ul>"
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>StatAgent benchmark {html.escape(run.label)}</title>"
        f"<style>{_HTML_CSS}</style></head><body>{''.join(body)}</body></html>"
    )
