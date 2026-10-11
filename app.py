"""StatAgent entry point: `streamlit run app.py`."""
import streamlit as st

st.set_page_config(
    page_title="StatAgent — Verifiable AI Statistical Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from statagent.config import get_config  # noqa: E402
from streamlit_app.state import init_state  # noqa: E402

init_state()
config = get_config()

pages = [
    st.Page("streamlit_app/views/dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("streamlit_app/views/data_explorer.py", title="Data Explorer", icon="📊"),
    st.Page("streamlit_app/views/ai_analysis.py", title="AI Analysis", icon="🤖"),
    st.Page("streamlit_app/views/report_center.py", title="Report Center", icon="📄"),
    st.Page("streamlit_app/views/benchmark_lab.py", title="Benchmark Lab", icon="🧪"),
]

with st.sidebar:
    mode = config.llm_mode
    badge = "🟢 live" if mode == "live" else "🟡 offline (rule-based planner)"
    st.markdown(f"**LLM mode:** {badge}")
    st.markdown(
        f"**Model:** `{config.llm.model}` · engine v0.1.0"
        if mode == "live"
        else "Set `DEEPSEEK_API_KEY` in `.env` to enable AI planning."
    )
    st.caption(
        "All statistics are computed by the Python engine — never by the LLM."
    )

pg = st.navigation(pages)
pg.run()
