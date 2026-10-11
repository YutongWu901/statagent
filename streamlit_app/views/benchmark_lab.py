"""Benchmark Lab: run and inspect the three-system evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from statagent.benchmark.runner import BenchmarkRunner
from statagent.benchmark.systems import SystemA, SystemB, SystemC
from statagent.benchmark.tasks import TASKS
from statagent.agent.llm import MockLLM
from streamlit_app.state import get_cfg

st.title("🧪 Benchmark Lab")
st.markdown(
    "Evaluates three systems on the same 60-task statistical task suite: "
    "**A** — LLM only · **B** — LLM + tools · **C** — LLM + tools + validation. "
    "Every raw outcome is saved for audit."
)
cfg = get_cfg()


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float) and v <= 1.0:
        return f"{v:.0%}"
    return f"{v:.3f}" if isinstance(v, float) else str(v)

# ---- past runs --------------------------------------------------------------
runs = []
if cfg.benchmark_dir.exists():
    for summary_path in sorted(cfg.benchmark_dir.glob("*/summary.json"), reverse=True):
        try:
            runs.append((summary_path.parent, json.loads(summary_path.read_text(encoding="utf-8"))))
        except Exception:  # noqa: BLE001
            continue

if runs:
    st.markdown("### Past runs")
    rows = []
    for directory, payload in runs:
        s = payload.get("summary", {})
        row = {
            "run": payload.get("label", directory.name),
            "mode": payload.get("llm_mode", "?"),
            "tasks": payload.get("n_tasks", "?"),
            "A numeric": f"{s.get('A_llm_only', {}).get('numerical_accuracy', 0):.0%}" if "A_llm_only" in s else "—",
            "B numeric": f"{s.get('B_llm_tools', {}).get('numerical_accuracy', 0):.0%}" if "B_llm_tools" in s else "—",
            "C numeric": f"{s.get('C_llm_tools_validation', {}).get('numerical_accuracy', 0):.0%}" if "C_llm_tools_validation" in s else "—",
            "C invalid rate": f"{s.get('C_llm_tools_validation', {}).get('invalid_analysis_rate', 0):.0%}" if "C_llm_tools_validation" in s else "—",
            "B invalid rate": f"{s.get('B_llm_tools', {}).get('invalid_analysis_rate', 0):.0%}" if "B_llm_tools" in s else "—",
        }
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    selected = st.selectbox("Inspect run", [p.get("label", d.name) for d, p in runs])
    for directory, payload in runs:
        if payload.get("label", directory.name) != selected:
            continue
        s = payload.get("summary", {})
        metrics = pd.DataFrame(
            [
                {
                    "metric": k,
                    "A": _fmt(s.get("A_llm_only", {}).get(k)),
                    "B": _fmt(s.get("B_llm_tools", {}).get(k)),
                    "C": _fmt(s.get("C_llm_tools_validation", {}).get(k)),
                }
                for k in (
                    "task_completion_rate", "method_selection_accuracy",
                    "numerical_accuracy", "interpretation_accuracy",
                    "appropriate_clarification_rate", "invalid_analysis_rate",
                    "cost_usd", "mean_time_s", "cost_per_correct_task_usd",
                )
            ]
        )
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        with st.expander("Category breakdown"):
            breakdown = payload.get("category_breakdown", {})
            rows_cat = []
            for cat, systems in breakdown.items():
                for sys_name, m in systems.items():
                    rows_cat.append(
                        {
                            "category": cat,
                            "system": sys_name,
                            "n": m["n"],
                            "completion": f"{m['completion']:.0%}",
                            "method_acc": _fmt(m["method_accuracy"]),
                            "numeric": f"{m['numeric_accuracy']:.0%}",
                            "interp": f"{m['interpretation_accuracy']:.0%}",
                            "invalid_unflagged": f"{m['invalid_unflagged_rate']:.0%}",
                        }
                    )
            st.dataframe(pd.DataFrame(rows_cat), use_container_width=True, hide_index=True)
        files = list(directory.glob("*"))
        if files:
            st.caption("Artifacts: " + ", ".join(f"`{f.name}`" for f in sorted(files)))
    st.markdown("---")


# ---- run --------------------------------------------------------------------
st.markdown("### Run a new benchmark")
st.caption(
    "With an API key configured, systems use the live DeepSeek model (System A "
    "included). Without one, B and C run the offline heuristic planner — a "
    "pipeline validation, clearly labelled `offline` in the results."
)
col1, col2 = st.columns(2)
n_tasks = col1.slider("Number of tasks (pilot subset)", min_value=8, max_value=len(TASKS), value=60, step=4)
smoke = col2.checkbox("Include System A (mock LLM in offline mode)", value=True)
run_label = st.text_input("Run label", value="")

if st.button("Run benchmark", type="primary"):
    client = None
    if cfg.llm.is_configured:
        from statagent.agent.llm import create_client

        client = create_client(cfg.llm, cfg.pricing)
    systems = {}
    if client is not None:
        systems["A_llm_only"] = SystemA(client)
    elif smoke:
        mock = MockLLM()
        mock.on(
            lambda m: any("DIRECTLY" in str(x.get("content", "")) for x in m),
            lambda m: type(
                "R", (), {
                    "content": '{"answers": {"r": 0.30, "p": 0.05}, "interpretation": "These variables are strongly causally related (r = 0.30, p = 0.05)."}',
                    "tool_calls": [],
                },
            )(),
        )
        systems["A_llm_only"] = SystemA(mock)
    systems["B_llm_tools"] = SystemB(cfg, client)
    systems["C_llm_tools_validation"] = SystemC(cfg, client)

    label = run_label.strip() or f"run_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    out_dir = cfg.benchmark_dir / label
    runner = BenchmarkRunner(cfg, systems, out_dir)
    progress_bar = st.progress(0.0, text="Running…")

    def progress(done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"Task {done}/{total}")

    tasks = TASKS[:n_tasks]
    run = runner.run(tasks, label=label, progress=progress)
    progress_bar.empty()
    st.success(f"Run `{run.label}` finished — artifacts in `{run.out_dir}`.")
    st.rerun()
