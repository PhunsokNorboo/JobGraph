"""Job market insights — aggregate analytics across all indexed jobs."""
import ast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _parse_skills(val) -> list[str]:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def render_market_insights(df: pd.DataFrame):
    """Render market-level analytics from the full jobs dataset."""

    st.markdown("### Job Market Insights")
    st.caption(f"Aggregate analytics across {len(df)} indexed job postings")

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        st.metric("Companies", df["company"].nunique())
    with col3:
        all_skills = []
        for s in df["required_skills"]:
            all_skills.extend(_parse_skills(s))
        st.metric("Unique Skills", len(set(all_skills)))
    with col4:
        has_salary = df["salary_min"].notna() | df["salary_max"].notna()
        st.metric("With Salary", f"{has_salary.sum()}")

    st.markdown("---")

    # Row 1: Skills demand + Company hiring
    col_left, col_right = st.columns(2)

    with col_left:
        # Top skills in demand
        from collections import Counter
        skill_counts = Counter(all_skills)
        top_20 = skill_counts.most_common(20)
        skills_df = pd.DataFrame(top_20, columns=["Skill", "Demand"])

        fig = px.bar(
            skills_df, x="Demand", y="Skill", orientation="h",
            title="Top 20 Skills in Demand",
            color="Demand",
            color_continuous_scale=["#FF9800", "#E91E63"],
        )
        fig.update_layout(
            height=500, yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=20),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Hiring by company
        company_counts = df["company"].value_counts().head(10)
        fig = px.bar(
            x=company_counts.values, y=company_counts.index, orientation="h",
            title="Top Hiring Companies",
            labels={"x": "Open Positions", "y": "Company"},
            color=company_counts.values,
            color_continuous_scale=["#2196F3", "#9C27B0"],
        )
        fig.update_layout(
            height=500, yaxis=dict(autorange="reversed"),
            margin=dict(t=40, b=20),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Role family + Location + Seniority
    col1, col2, col3 = st.columns(3)

    with col1:
        role_counts = df["role_family"].value_counts()
        fig = px.pie(
            values=role_counts.values, names=role_counts.index,
            title="Roles by Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        loc_counts = df["location_type"].value_counts()
        fig = px.pie(
            values=loc_counts.values, names=loc_counts.index,
            title="Remote vs Hybrid vs Onsite",
            color_discrete_map={
                "remote": "#4CAF50", "hybrid": "#FF9800", "onsite": "#2196F3",
            },
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        sen_order = ["entry", "mid", "senior", "staff"]
        sen_counts = df["seniority"].value_counts().reindex(sen_order).dropna()
        fig = px.bar(
            x=sen_counts.index, y=sen_counts.values,
            title="Seniority Distribution",
            labels={"x": "Level", "y": "Jobs"},
            color=sen_counts.index,
            color_discrete_map={
                "entry": "#4CAF50", "mid": "#2196F3",
                "senior": "#FF9800", "staff": "#9C27B0",
            },
        )
        fig.update_layout(height=350, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Salary distribution
    salary_df = df[df["salary_max"].notna()].copy()
    if len(salary_df) > 5:
        salary_df["salary_max"] = salary_df["salary_max"].astype(float)
        fig = px.box(
            salary_df, x="seniority", y="salary_max",
            title="Salary Distribution by Seniority",
            labels={"salary_max": "Max Salary ($)", "seniority": "Seniority Level"},
            category_orders={"seniority": sen_order},
            color="seniority",
            color_discrete_map={
                "entry": "#4CAF50", "mid": "#2196F3",
                "senior": "#FF9800", "staff": "#9C27B0",
            },
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
