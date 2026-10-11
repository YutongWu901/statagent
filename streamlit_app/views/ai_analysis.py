"""AI Analysis: question -> plan -> validation gate -> execution -> report."""
from __future__ import annotations

import streamlit as st

from statagent.models.plan import design_answer_value
from statagent.report.generator import build_html, build_markdown
from streamlit_app.state import add_record, get_pipeline, get_records, get_store
from streamlit_app.ui import badge, render_findings, render_step_result

st.title("🤖 AI Analysis")
store = get_store()

if len(store) == 0:
    st.info("No dataset loaded yet — upload one in the **Data Explorer** first.")
    st.stop()

pipe = get_pipeline()


def _handle_outcome(out, store, question: str | None = None, dataset: str | None = None) -> None:
    if out.status == "clarification_needed":
        st.session_state.pending = {
            "status": "clarification",
            "question": question or st.session_state.ai_question,
            "dataset": dataset or st.session_state.ai_dataset,
            "questions": out.questions,
            "plan": out.plan,
        }
    elif out.status == "blocked":
        st.session_state.blocked = {
            "question": question or st.session_state.ai_question,
            "dataset": dataset or st.session_state.ai_dataset,
            "validation": out.validation,
        }
    elif out.status == "completed" and out.record is not None:
        add_record(out.record)
        st.session_state.pending = None
        st.session_state.blocked = None
    else:
        st.session_state.error = out.error or "analysis failed"

left, right = st.columns([2, 1], vertical_alignment="bottom")
with left:
    dataset_id = st.selectbox(
        "Dataset", store.ids(), format_func=lambda i: f"{i} — {store.source_name(i)}"
    )
with right:
    if "pipeline" in st.session_state:
        mode = st.session_state.pipeline.llm_meta()["mode"]
        st.markdown({"live": badge("LLM: live"), "offline": badge("LLM: offline fallback")}.get(mode, ""), unsafe_allow_html=True)

question = st.text_area(
    "Research question",
    placeholder=(
        "e.g. “Is study time associated with exam score?” · "
        "“Does the new teaching method produce higher scores than the traditional one?” · "
        "“Is smoking associated with hypertension?”"
    ),
    height=90,
)

examples = st.columns(4)
example_questions = [
    "Is study time associated with exam score?",
    "Does the new teaching method produce higher posttest scores than the traditional method?",
    "Is smoking associated with hypertension?",
    "Predict monthly spending from income and education years.",
]
for col, ex in zip(examples, example_questions):
    if col.button(ex[:28] + "…", key=f"ex_{ex[:12]}", use_container_width=True):
        st.session_state.ai_question = ex
        st.session_state.ai_dataset = dataset_id
        st.session_state.pending = None
        st.session_state.blocked = None

if st.session_state.ai_question:
    question = st.session_state.ai_question

run_col, clear_col = st.columns([1, 4])
run = run_col.button("Run analysis", type="primary", use_container_width=True)
if clear_col.button("Clear", use_container_width=False):
    st.session_state.ai_question = ""
    st.session_state.pending = None
    st.session_state.blocked = None
    st.rerun()

# ---- clarification loop ----------------------------------------------------
pending = st.session_state.pending
if pending is not None and pending.get("status") == "clarification":
    st.markdown("### ❓ A few questions before we proceed")
    st.caption(
        "StatAgent does not silently assume research-design facts that change "
        "the analysis. Please answer:"
    )
    answers: dict[str, str] = {}
    with st.form("clarifications"):
        for q in pending["questions"]:
            st.markdown(f"**{q.question}**")
            st.caption(q.why or "")
            options = q.options or ["yes", "no"]
            answers[q.key] = st.radio(
                f"answer_{q.key}", options, key=f"clar_{q.key}", label_visibility="collapsed"
            )
        submitted = st.form_submit_button("Submit answers and continue", type="primary")
    if submitted:
        design_answers = {
            key: design_answer_value(key, value) for key, value in answers.items()
        }
        out = pipe.analyze(
            pending["question"], store, pending["dataset"],
            design_answers=design_answers, prior_plan=pending.get("plan"),
            clarifications=[
                {"question": pending["questions"][i].question, "answer": answers[q.key], "key": q.key}
                for i, q in enumerate(pending["questions"])
            ],
        )
        _handle_outcome(out, store)
        st.rerun()
    st.stop()

# ---- blocked state ----------------------------------------------------------
blocked = st.session_state.blocked
if blocked is not None:
    st.markdown("### ⛔ Analysis blocked by validation")
    render_findings(blocked["validation"])
    force = st.checkbox(
        "I understand the issues and want to run this analysis anyway",
        value=False,
    )
    if st.button("Force run", type="primary") and force:
        out = pipe.analyze(
            blocked["question"], store, blocked["dataset"], force=True
        )
        _handle_outcome(out, store)
        st.rerun()
    if st.button("Discard"):
        st.session_state.blocked = None
        st.rerun()
    st.stop()

# ---- run -------------------------------------------------------------------
if run and question.strip():
    st.session_state.pending = None
    st.session_state.blocked = None
    out = pipe.analyze(question, store, dataset_id)
    _handle_outcome(out, store, question=question, dataset=dataset_id)
    st.rerun()


# ---- render completed analyses --------------------------------------------
records = get_records()
if records:
    st.markdown("---")
    st.markdown("### Completed analyses")
    ids = list(records)
    pick = st.selectbox("Analysis", ids, format_func=lambda r: records[r].question[:80])
    record = records[pick]
    st.markdown(f"**{record.question}** · {record.created_at}")
    if record.clarifications:
        with st.expander("Clarifications"):
            for ex in record.clarifications:
                st.markdown(f"**Q:** {ex.question}  \n**A:** {ex.answer}")
    with st.expander("Analysis plan", expanded=False):
        st.markdown(f"**Goal:** {record.plan.goal}")
        st.markdown(
            "**Design:** "
            + "; ".join(
                f"{k} = {v}" for k, v in record.plan.design.model_dump().items() if v not in ("unknown", None)
            )
        )
        if record.plan.method_rationale:
            st.markdown(record.plan.method_rationale)
        st.dataframe(
            [
                {"#": s.step_id, "tool": s.tool, "variables": ", ".join(s.variables), "rationale": s.rationale}
                for s in record.plan.steps
            ],
            use_container_width=True, hide_index=True,
        )
    if record.validation_before is not None:
        with st.expander("Validation (before execution)", expanded=record.validation_before.blocked):
            render_findings(record.validation_before)
    for step in record.step_results:
        render_step_result(step, key=f"rec_{record.record_id}")
    if record.interpretation is not None:
        interp = record.interpretation
        st.markdown("### Interpretation")
        st.markdown(f"**Summary:** {interp.summary}")
        if interp.key_findings:
            for k in interp.key_findings:
                st.markdown(f"- {k}")
        st.markdown(interp.interpretation)
        if interp.limitations:
            with st.expander("Limitations"):
                for x in interp.limitations:
                    st.markdown(f"- {x}")
        st.markdown(f"**Conclusion:** {interp.conclusion}")
    if record.validation_after is not None:
        with st.expander("Validation of the interpretation"):
            render_findings(record.validation_after)
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download Markdown report",
        build_markdown(record),
        file_name=f"{record.record_id}.md",
        use_container_width=True,
    )
    dl2.download_button(
        "Download HTML report",
        build_html(record),
        file_name=f"{record.record_id}.html",
        use_container_width=True,
    )
