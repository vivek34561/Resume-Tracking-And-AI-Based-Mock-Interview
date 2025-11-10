"""
Resume Analysis Visualizations
Beautiful charts and graphs for resume analysis using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from typing import Dict, List


def create_score_gauge(score: float, title: str = "Overall Match Score") -> go.Figure:
    """
    Create a beautiful gauge chart for overall score.
    
    Args:
        score: Score value (0-100)
        title: Title for the gauge
        
    Returns:
        Plotly figure object
    """
    # Determine color based on score
    if score >= 75:
        color = "#10b981"  # Green
    elif score >= 50:
        color = "#f59e0b"  # Orange
    else:
        color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24, 'color': '#1f2937'}},
        delta={'reference': 70, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#94a3b8"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#f1f5f9",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fed7aa'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1f2937", 'family': "Inter, sans-serif"}
    )
    
    return fig


def create_skills_radar_chart(skill_scores: Dict[str, float]) -> go.Figure:
    """
    Create a radar/spider chart for skill scores.
    
    Args:
        skill_scores: Dictionary of skill names and scores (0-10)
        
    Returns:
        Plotly figure object
    """
    if not skill_scores:
        return None
    
    # Limit to top 8 skills for better visualization
    sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    skills = [item[0] for item in sorted_skills]
    scores = [item[1] for item in sorted_skills]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=skills,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=8, color='#6366f1'),
        name='Your Skills',
        hovertemplate='<b>%{theta}</b><br>Score: %{r}/10<extra></extra>'
    ))
    
    # Add reference line at score 7 (good threshold)
    fig.add_trace(go.Scatterpolar(
        r=[7] * len(skills),
        theta=skills,
        mode='lines',
        line=dict(color='#10b981', width=1, dash='dash'),
        name='Target (7/10)',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                showticklabels=True,
                ticks='outside',
                tickfont=dict(size=10),
                gridcolor='#e2e8f0'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#1f2937')
            ),
            bgcolor='#f8fafc'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        title=dict(
            text="Top Skills Assessment",
            font=dict(size=20, color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        height=450,
        margin=dict(l=80, r=80, t=100, b=80),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_skills_bar_chart(skill_scores: Dict[str, float], top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart for skill scores.
    
    Args:
        skill_scores: Dictionary of skill names and scores
        top_n: Number of top skills to show
        
    Returns:
        Plotly figure object
    """
    if not skill_scores:
        return None
    
    # Sort and get top N skills
    sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    skills = [item[0] for item in sorted_skills]
    scores = [item[1] for item in sorted_skills]
    
    # Color coding
    colors = ['#10b981' if score >= 7 else '#f59e0b' if score >= 4 else '#ef4444' 
              for score in scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=skills,
        x=scores,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='#1f2937', width=1)
        ),
        text=[f'{score}/10' for score in scores],
        textposition='auto',
        textfont=dict(size=12, color='white', family='Inter, sans-serif'),
        hovertemplate='<b>%{y}</b><br>Score: %{x}/10<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Top {len(skills)} Skills Ranking",
            font=dict(size=20, color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Score (out of 10)",
            range=[0, 10],
            showgrid=True,
            gridcolor='#e2e8f0',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12, color='#1f2937'),
            autorange="reversed"
        ),
        height=400 + (len(skills) * 30),
        margin=dict(l=150, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f8fafc',
        hovermode='y'
    )
    
    # Add reference lines
    fig.add_vline(x=7, line_dash="dash", line_color="#10b981", annotation_text="Good", 
                  annotation_position="top")
    fig.add_vline(x=4, line_dash="dash", line_color="#f59e0b", annotation_text="Fair", 
                  annotation_position="top")
    
    return fig


def create_skill_categories_pie(skill_scores: Dict[str, float]) -> go.Figure:
    """
    Create a pie chart showing skill category distribution.
    
    Args:
        skill_scores: Dictionary of skill names and scores
        
    Returns:
        Plotly figure object
    """
    if not skill_scores:
        return None
    
    # Categorize skills
    excellent = sum(1 for score in skill_scores.values() if score >= 8)
    good = sum(1 for score in skill_scores.values() if 5 <= score < 8)
    needs_improvement = sum(1 for score in skill_scores.values() if score < 5)
    
    labels = ['Excellent (8-10)', 'Good (5-7)', 'Needs Improvement (0-4)']
    values = [excellent, good, needs_improvement]
    colors = ['#10b981', '#f59e0b', '#ef4444']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{label}</b><br>%{value} skills<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="Skill Distribution",
            font=dict(size=20, color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=400,
        margin=dict(l=20, r=20, t=80, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=f'Total<br>{sum(values)}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig


def create_comparison_chart(found_skills: List[str], missing_skills: List[str]) -> go.Figure:
    """
    Create a comparison chart for found vs missing skills.
    
    Args:
        found_skills: List of skills found in resume
        missing_skills: List of skills missing from resume
        
    Returns:
        Plotly figure object
    """
    categories = ['Skills Found', 'Skills Missing']
    values = [len(found_skills), len(missing_skills)]
    colors = ['#10b981', '#ef4444']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='#1f2937', width=2)
        ),
        text=values,
        textposition='auto',
        textfont=dict(size=24, color='white', family='Inter, sans-serif'),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Skills Gap Analysis",
            font=dict(size=20, color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=14, color='#1f2937')
        ),
        yaxis=dict(
            title="Number of Skills",
            showgrid=True,
            gridcolor='#e2e8f0',
            tickfont=dict(size=12)
        ),
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f8fafc'
    )
    
    return fig


def create_score_trend_chart(analysis_result: Dict) -> go.Figure:
    """
    Create a waterfall chart showing score breakdown.
    
    Args:
        analysis_result: Analysis result dictionary
        
    Returns:
        Plotly figure object
    """
    # Extract score components (this is illustrative, adapt to your actual data structure)
    base_score = 100
    skill_penalty = -sum(1 for score in analysis_result.get('skill_scores', {}).values() if score < 5) * 5
    experience_bonus = min(len(analysis_result.get('strengths', [])) * 3, 20)
    final_score = analysis_result.get('overall_score', 0)
    
    fig = go.Figure(go.Waterfall(
        name="Score Components",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Base Score", "Skill Gaps", "Strengths", "Final Score"],
        textposition="outside",
        text=[f"{base_score}%", f"{skill_penalty}%", f"+{experience_bonus}%", f"{final_score}%"],
        y=[base_score, skill_penalty, experience_bonus, final_score],
        connector={"line": {"color": "#94a3b8"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6366f1"}}
    ))
    
    fig.update_layout(
        title=dict(
            text="Score Breakdown",
            font=dict(size=20, color='#1f2937'),
            x=0.5,
            xanchor='center'
        ),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f8fafc',
        yaxis=dict(title="Score (%)", showgrid=True, gridcolor='#e2e8f0')
    )
    
    return fig


def display_enhanced_visualizations(analysis_result: Dict):
    """
    Display all enhanced visualizations for resume analysis.
    
    Args:
        analysis_result: Complete analysis result dictionary
    """
    st.markdown("### 📊 Visual Analysis Dashboard")
    st.markdown("---")
    
    # Row 1: Gauge and Pie Chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        score_gauge = create_score_gauge(
            analysis_result.get("overall_score", 0),
            "ATS Match Score"
        )
        st.plotly_chart(score_gauge, use_container_width=True)
    
    with col2:
        if analysis_result.get("skill_scores"):
            pie_chart = create_skill_categories_pie(analysis_result["skill_scores"])
            st.plotly_chart(pie_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Skills Radar Chart
    if analysis_result.get("skill_scores"):
        radar_chart = create_skills_radar_chart(analysis_result["skill_scores"])
        st.plotly_chart(radar_chart, use_container_width=True)
        
        st.markdown("---")
        
        # Row 3: Bar Chart
        bar_chart = create_skills_bar_chart(analysis_result["skill_scores"])
        st.plotly_chart(bar_chart, use_container_width=True)
    
    # Row 4: Gap Analysis
    if analysis_result.get("strengths") or analysis_result.get("missing_skills"):
        st.markdown("---")
        comparison_chart = create_comparison_chart(
            analysis_result.get("strengths", []),
            analysis_result.get("missing_skills", [])
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
