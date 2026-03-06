"""JobGraph -- AI Job Discovery Demo."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

st.set_page_config(
    page_title="JobGraph -- AI Job Discovery",
    page_icon="\U0001f50d",
    layout="wide",
)

# Import after path setup
from retrieval.search import search
from retrieval.resume_parser import parse_pdf
from app.components.job_card import render_job_card
from app.components.skill_chart import render_skill_chart, render_match_summary


def main():
    st.title("JobGraph")
    st.caption("AI-Powered Job Discovery using Graph Neural Networks")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **JobGraph** transforms job postings into a knowledge graph
        and uses a Graph Neural Network (HGT) to power semantic job discovery.

        **How it works:**
        1. Jobs, skills, and companies form a heterogeneous graph
        2. An HGT model learns rich embeddings
        3. Your query is matched against 481 jobs
        4. Results are re-ranked with BM25 for precision
        """)

        st.markdown("---")
        top_k = st.slider("Number of results", 5, 20, 10)

    # Input tabs
    tab1, tab2 = st.tabs(["Paste Resume / Query", "Upload PDF"])

    resume_text = ""

    with tab1:
        resume_text = st.text_area(
            "Describe your background or paste your resume:",
            height=200,
            placeholder="e.g., 5 years Python, ML engineering, experience with PyTorch and NLP. Looking for senior roles in AI/fintech...",
        )

    with tab2:
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if uploaded_file:
            try:
                resume_text = parse_pdf(uploaded_file)
                st.success(f"Extracted {len(resume_text)} characters from PDF")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:1000] + ("..." if len(resume_text) > 1000 else ""))
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")

    # Search
    if st.button("Find Matching Jobs", type="primary", use_container_width=True):
        if not resume_text.strip():
            st.warning("Please enter some text or upload a resume first.")
            return

        with st.spinner("Searching across thousands of jobs..."):
            try:
                results = search(resume_text, top_k=top_k)
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        if not results:
            st.info("No matching jobs found. Try a different query.")
            return

        st.markdown(f"### Found {len(results)} matching jobs")
        st.markdown("---")

        # Match summary metrics
        render_match_summary(results)

        # Skill chart
        with st.expander("Skill Overlap Analysis", expanded=True):
            render_skill_chart(results)

        # Job cards
        st.markdown("### Results")
        for i, result in enumerate(results, 1):
            render_job_card(result, rank=i)


if __name__ == "__main__":
    main()
