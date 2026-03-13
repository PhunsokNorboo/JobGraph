"""Job card component for Streamlit UI."""
import streamlit as st
from extraction.schema import SearchResult


def render_job_card(result: SearchResult, rank: int):
    """Render a single job result as a styled card."""
    job = result.job

    badge_colors = {
        "entry": "#4CAF50",
        "mid": "#2196F3",
        "senior": "#FF9800",
        "staff": "#9C27B0",
        "principal": "#F44336",
        "manager": "#795548",
    }
    badge_color = badge_colors.get(job.seniority, "#607D8B")

    # Build salary HTML
    salary_html = ""
    if job.salary_min or job.salary_max:
        if job.salary_min and job.salary_max:
            salary_text = f"${job.salary_min:,} - ${job.salary_max:,}"
        elif job.salary_min:
            salary_text = f"From ${job.salary_min:,}"
        else:
            salary_text = f"Up to ${job.salary_max:,}"
        salary_html = f'<p style="color: #2e7d32; font-weight: 600; margin: 8px 0;">{salary_text}</p>'

    # Build score bar HTML
    score_pct = max(0, min(100, int(result.similarity_score * 100)))
    score_html = (
        '<div style="margin: 8px 0;">'
        '<span style="font-size: 12px; color: #888;">Match Score</span>'
        '<div style="background: #e0e0e0; border-radius: 8px; height: 8px; margin-top: 4px;">'
        f'<div style="background: linear-gradient(90deg, #4CAF50, #2196F3); width: {score_pct}%; height: 8px; border-radius: 8px;"></div>'
        '</div>'
        f'<span style="font-size: 11px; color: #888;">{result.similarity_score:.2f}</span>'
        '</div>'
    )

    # Build matched skills HTML
    matched_html = ""
    if result.matched_skills:
        chips = " ".join(
            f'<span style="background: #e8f5e9; color: #2e7d32; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; display: inline-block;">&#10003; {s}</span>'
            for s in result.matched_skills
        )
        matched_html = f'<div style="margin: 8px 0;">{chips}</div>'

    # Build missing skills HTML
    missing_html = ""
    if result.missing_skills:
        chips = " ".join(
            f'<span style="background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; display: inline-block;">&#9733; {s}</span>'
            for s in result.missing_skills[:5]
        )
        remaining = len(result.missing_skills) - 5
        if remaining > 0:
            chips += f' <span style="font-size: 12px; color: #888;">+{remaining} more</span>'
        missing_html = f'<div style="margin: 8px 0;">{chips}</div>'

    # Build description HTML
    desc_html = ""
    if job.description_summary:
        desc_html = f'<p style="font-size: 13px; color: #555; margin: 8px 0;">{job.description_summary}</p>'

    # Build apply link HTML
    link_html = ""
    if job.raw_url:
        link_html = f'<a href="{job.raw_url}" target="_blank" style="color: #1976D2; font-size: 13px; text-decoration: none;">View Original Posting &rarr;</a>'

    # Location line
    location = job.location_type.title() if job.location_type else ""
    if job.location_city:
        location += f" &bull; {job.location_city}"

    # Assemble full card as one HTML string
    html = (
        '<div style="border: 1px solid #e0e0e0; border-radius: 12px; padding: 20px; margin-bottom: 16px; background: white;">'
        '<div style="display: flex; justify-content: space-between; align-items: center;">'
        '<div>'
        f'<h3 style="margin: 0; color: #1a1a1a;">{rank}. {job.title}</h3>'
        f'<p style="margin: 4px 0; color: #666; font-size: 14px;">{job.company} &bull; {location}</p>'
        '</div>'
        f'<span style="background: {badge_color}; color: white; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 600;">{job.seniority.upper()}</span>'
        '</div>'
        f'{salary_html}'
        f'{score_html}'
        f'{matched_html}'
        f'{missing_html}'
        f'{desc_html}'
        f'{link_html}'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)
