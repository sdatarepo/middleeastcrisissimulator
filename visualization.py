from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_chart(sim_df: pd.DataFrame, title: str, color_col: str = 'country', include_uncertainty: bool = True) -> go.Figure:
    fig = go.Figure()

    if include_uncertainty and {'impact_low', 'impact_high'}.issubset(sim_df.columns):
        for name, grp in sim_df.groupby(color_col):
            grp = grp.sort_values('month')
            if grp['impact_low'].notna().any() and grp['impact_high'].notna().any():
                fig.add_trace(go.Scatter(
                    x=grp['month'], y=grp['impact_high'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=grp['month'], y=grp['impact_low'], mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(100,100,100,0.15)', showlegend=False, hoverinfo='skip'
                ))
            fig.add_trace(go.Scatter(x=grp['month'], y=grp['impact'], mode='lines+markers', name=str(name)))
    else:
        for name, grp in sim_df.groupby(color_col):
            grp = grp.sort_values('month')
            fig.add_trace(go.Scatter(x=grp['month'], y=grp['impact'], mode='lines+markers', name=str(name)))

    fig.update_layout(title=title, xaxis_title='Month after shock', yaxis_title='CPI impact (percentage points)', hovermode='x unified')
    return fig


def africa_map(map_df: pd.DataFrame, title: str) -> go.Figure:
    hover_cols = [c for c in ['country', 'impact', 'peak_impact', 'peak_month', 'category'] if c in map_df.columns]
    fig = px.choropleth(
        map_df,
        locations='country',
        locationmode='country names',
        color='impact',
        scope='africa',
        hover_name='country',
        hover_data={c: True for c in hover_cols if c != 'country'},
        title=title,
        color_continuous_scale=[
            [0.0, '#F7D39C'],  # low
            [0.5, '#EB8146'],  # middle
            [1.0, '#F5270C'],  # high
        ],
    )
    fig.update_geos(fitbounds='locations', visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def cumulative_bar(summary_df: pd.DataFrame, x: str, title: str) -> go.Figure:
    fig = px.bar(summary_df, x=x, y='cumulative_impact', color=x, title=title, text_auto='.2f')
    fig.update_layout(showlegend=False, xaxis_title=x.replace('_', ' ').title(), yaxis_title='Cumulative impact (pp)')
    return fig


def stacked_decomposition(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.bar(df, x='channel', y='impact', color='channel', title=title, text_auto='.2f')
    fig.update_layout(showlegend=False, xaxis_title='Transmission channel', yaxis_title='Impact (pp)')
    return fig
