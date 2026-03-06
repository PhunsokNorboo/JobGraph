"""Skill overlap visualization component."""
import streamlit as st
import plotly.graph_objects as go
from extraction.schema import SearchResult


def render_skill_chart(results: list[SearchResult]):
    """Render aggregated skill overlap chart across all results."""
    if not results:
        return

    # Aggregate skill matches and gaps across all results
    match_counts: dict[str, int] = {}
    gap_counts: dict[str, int] = {}

    for r in results:
        for s in r.matched_skills:
            match_counts[s] = match_counts.get(s, 0) + 1
        for s in r.missing_skills:
            gap_counts[s] = gap_counts.get(s, 0) + 1

    # Top 15 most relevant skills
    all_skills: dict[str, int] = {}
    for s, c in match_counts.items():
        all_skills[s] = all_skills.get(s, 0) + c
    for s, c in gap_counts.items():
        all_skills[s] = all_skills.get(s, 0) + c

    top_skills = sorted(all_skills.keys(), key=lambda s: all_skills[s], reverse=True)[:15]

    if not top_skills:
        return

    matched = [match_counts.get(s, 0) for s in top_skills]
    gaps = [gap_counts.get(s, 0) for s in top_skills]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Skills You Have",
        x=top_skills,
        y=matched,
        marker_color="#4CAF50",
    ))
    fig.add_trace(go.Bar(
        name="Skills to Learn",
        x=top_skills,
        y=gaps,
        marker_color="#FF9800",
    ))

    fig.update_layout(
        barmode="stack",
        title="Skill Overlap Across Matched Jobs",
        xaxis_title="Skill",
        yaxis_title="# of Jobs",
        height=400,
        margin=dict(t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_match_summary(results: list[SearchResult]):
    """Render a quick summary of match quality."""
    if not results:
        return

    avg_score = sum(r.similarity_score for r in results) / len(results)
    total_matched = sum(len(r.matched_skills) for r in results)
    total_missing = sum(len(r.missing_skills) for r in results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Match Score", f"{avg_score:.2f}")
    with col2:
        st.metric("Skills Matched", total_matched)
    with col3:
        st.metric("Skills to Learn", total_missing)
