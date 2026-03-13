"""JobGraph -- AI Job Discovery Demo."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="JobGraph -- AI Job Discovery",
    page_icon="\U0001f50d",
    layout="wide",
)

# Import after path setup
from retrieval.search import search
from retrieval.resume_parser import parse_pdf, extract_skills_from_text
from app.components.job_card import render_job_card
from app.components.skill_chart import render_skill_chart, render_match_summary
from app.components.skill_roadmap import render_skill_roadmap
from app.components.graph_viz import render_graph_viz
from app.components.market_insights import render_market_insights


@st.cache_data
def load_jobs():
    return pd.read_parquet(project_root / "data" / "processed" / "jobs.parquet")


def search_page():
    """Job search page with filters and results."""
    st.markdown("#### Describe your background or paste your resume")

    # Input tabs
    tab1, tab2 = st.tabs(["Text Input", "Upload PDF"])

    resume_text = ""

    with tab1:
        resume_text = st.text_area(
            "Your background:",
            height=150,
            placeholder="e.g., 5 years Python, ML engineering, experience with PyTorch and NLP. Looking for senior roles in AI...",
            label_visibility="collapsed",
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

        # Map dropdown values to filter params
        seniority_map = {
            "Any": None, "Entry": "entry", "Mid": "mid",
            "Senior": "senior", "Staff+": "staff+",
        }
        role_map = {
            "Any": None, "SWE": "swe", "ML / AI": "ml", "Data": "data",
            "DevOps": "devops", "PM": "pm", "Design": "design", "Sales": "sales",
        }
        location_map = {
            "Any": None, "Remote": "remote", "Hybrid": "hybrid", "Onsite": "onsite",
        }

        sen_filter = seniority_map.get(st.session_state.get("seniority_select", "Any"))
        role_filter = role_map.get(st.session_state.get("role_select", "Any"))
        loc_filter = location_map.get(st.session_state.get("location_select", "Any"))
        top_k = st.session_state.get("top_k_slider", 10)

        with st.spinner("Searching..."):
            try:
                results = search(
                    resume_text, top_k=top_k,
                    seniority_filter=sen_filter,
                    role_family_filter=role_filter,
                    location_filter=loc_filter,
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                return

        if not results:
            st.info("No matching jobs found. Try broadening your filters.")
            return

        st.markdown(f"### Found {len(results)} matching jobs")
        st.markdown("---")

        # Match summary metrics
        render_match_summary(results)

        # Three expanders: skill chart, learning roadmap, knowledge graph
        col_left, col_right = st.columns(2)

        with col_left:
            with st.expander("Skill Overlap Analysis", expanded=True):
                render_skill_chart(results)

        with col_right:
            with st.expander("Learning Roadmap", expanded=True):
                render_skill_roadmap(results)

        # Knowledge graph visualization
        with st.expander("Knowledge Graph View", expanded=False):
            query_skills = extract_skills_from_text(resume_text)
            render_graph_viz(results, query_skills)

        # Job cards
        st.markdown("### Results")
        for i, result in enumerate(results, 1):
            render_job_card(result, rank=i)


def insights_page():
    """Market insights page with aggregate analytics."""
    df = load_jobs()
    render_market_insights(df)


def main():
    st.title("JobGraph")
    st.caption("AI-Powered Job Discovery using Graph Neural Networks")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **JobGraph** transforms job postings into a knowledge graph
        and uses Graph Neural Networks to power semantic job discovery.

        Paste your resume or describe your skills, and JobGraph finds
        the best matching roles with skill-gap analysis.
        """)

        st.markdown("---")
        st.subheader("Filters")
        st.slider("Number of results", 5, 20, 10, key="top_k_slider")
        st.selectbox(
            "Seniority", ["Any", "Entry", "Mid", "Senior", "Staff+"],
            key="seniority_select",
            help="'Any' auto-detects from your query text.",
        )
        st.selectbox(
            "Role Category", ["Any", "SWE", "ML / AI", "Data", "DevOps", "PM", "Design", "Sales"],
            key="role_select",
        )
        st.selectbox(
            "Location", ["Any", "Remote", "Hybrid", "Onsite"],
            key="location_select",
        )

        st.markdown("---")
        df = load_jobs()
        st.caption(f"{len(df)} jobs indexed across {df['company'].nunique()} companies")

    # Main tabs
    tab_search, tab_insights = st.tabs(["Job Search", "Market Insights"])

    with tab_search:
        search_page()

    with tab_insights:
        insights_page()


if __name__ == "__main__":
    main()
