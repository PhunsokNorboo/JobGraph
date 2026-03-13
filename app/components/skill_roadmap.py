"""Skill gap roadmap — shows which skills to learn for maximum job unlocks."""
import streamlit as st
from extraction.schema import SearchResult


def render_skill_roadmap(results: list[SearchResult]):
    """Render a 'learn these skills' roadmap based on search results.

    Ranks missing skills by how many additional jobs they would unlock.
    """
    if not results:
        return

    # Count how many jobs require each missing skill
    skill_job_count: dict[str, int] = {}
    for r in results:
        for skill in r.missing_skills:
            skill_job_count[skill] = skill_job_count.get(skill, 0) + 1

    if not skill_job_count:
        st.success("You already have all the required skills for these jobs!")
        return

    # Sort by impact (most jobs unlocked first)
    top_skills = sorted(skill_job_count.items(), key=lambda x: -x[1])[:5]

    st.markdown("#### Your Learning Roadmap")
    st.caption("Skills ranked by how many matched jobs they unlock")

    for skill, count in top_skills:
        pct = int(count / len(results) * 100)
        html = (
            '<div style="background: #1a1a2e; border-left: 4px solid #FF9800; '
            'padding: 12px 16px; margin-bottom: 8px; border-radius: 0 8px 8px 0;">'
            '<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<span style="color: #fff; font-weight: 600; font-size: 15px;">{skill}</span>'
            f'<span style="background: #FF9800; color: #fff; padding: 2px 10px; '
            f'border-radius: 12px; font-size: 13px; font-weight: 600;">'
            f'{count} {"job" if count == 1 else "jobs"}</span>'
            '</div>'
            f'<div style="background: #2a2a3e; border-radius: 4px; height: 6px; margin-top: 8px;">'
            f'<div style="background: linear-gradient(90deg, #FF9800, #FF5722); '
            f'width: {pct}%; height: 6px; border-radius: 4px;"></div>'
            '</div>'
            f'<span style="color: #888; font-size: 11px;">Required by {pct}% of your matches</span>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
