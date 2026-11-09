"""NeuroBridge Streamlit entry point. Run: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NeuroBridge", page_icon=":material/hub:", layout="wide", initial_sidebar_state="expanded")

pages = {
    "": [st.Page("views/home.py", title="Overview", icon=":material/home:", default=True)],
    "Research": [
        st.Page("views/copilot.py", title="Research Copilot", icon=":material/psychology:"),
        st.Page("views/datasets.py", title="Dataset Explorer", icon=":material/database:"),
        st.Page("views/biology.py", title="Biological Explorer", icon=":material/biotech:"),
        st.Page("views/validation.py", title="Validation Lab", icon=":material/fact_check:"),
        st.Page("views/report.py", title="Evidence Report", icon=":material/description:"),
    ],
    "System": [st.Page("views/architecture.py", title="Architecture & tools", icon=":material/account_tree:")],
}
nav = st.navigation(pages)
from ui.common import BRAND_SVG  # noqa: E402

with st.sidebar:
    st.markdown(f"<div class='nb-brand'>{BRAND_SVG}<b>NeuroBridge</b></div>", unsafe_allow_html=True)
    st.caption("From biological hypotheses to clinically relevant evidence.")
nav.run()
