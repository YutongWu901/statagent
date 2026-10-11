"""Dashboard: dataset summary, recent analyses, feature overview."""
import streamlit as st

from streamlit_app.state import get_records, get_store

st.title("📈 StatAgent")
st.markdown(
    "**A verifiable AI agent for statistical analysis.** Ask questions in "
    "natural language; a Python engine computes every number, and an "
    "independent rule-based validator checks the analysis."
)

store = get_store()
records = get_records()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Datasets loaded", len(store))
kpi2.metric("Analyses this session", len(records))
kpi3.metric("Registered tools", 9)
kpi4.metric("Validation rules", 26)

st.markdown("### How it works")
steps = st.columns(4)
steps[0].markdown("**1. Upload data**\n\nCSV or XLSX in the Data Explorer. Types, missing values and outliers are profiled automatically.")
steps[1].markdown("**2. Ask a question**\n\nIn plain language, e.g. *“Is study time associated with exam score?”*")
steps[2].markdown("**3. AI plans & engine computes**\n\nThe LLM plans the analysis; registered Python tools compute real statistics. Design gaps trigger clarification questions, never silent assumptions.")
steps[3].markdown("**4. Validation & report**\n\n26 independent rules check data, design, method and interpretation. Download a reproducible report.")

st.markdown("### Recent analyses")
if records:
    for rid, rec in list(records.items())[-6:][::-1]:
        with st.container(border=True):
            st.markdown(f"**{rec.question}**")
            c1, c2 = st.columns([3, 1])
            c1.caption(
                f"{rec.created_at} · dataset `{rec.dataset_source}` · "
                + ", ".join(dict.fromkeys(s.tool for s in rec.step_results))
            )
            if rec.validation_before is not None and rec.validation_before.blocked:
                c2.markdown("⛔ blocked")
            else:
                c2.markdown("✅ completed")
else:
    st.info(
        "No analyses yet. Upload a dataset in the **Data Explorer**, then ask a "
        "question in the **AI Analysis** page — or load the demo data there."
    )
