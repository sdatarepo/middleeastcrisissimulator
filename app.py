from __future__ import annotations

import io
import time
import pandas as pd
import streamlit as st
st.markdown(
    """
    <h1 style="color: rgb(19, 43, 87); font-size: 28px; margin-bottom: 0; margin-top: -60px; padding-top: -60px">
        Middle-East Crisis Impact on Africa Simulator - Scenarios: 1, 3, 6 Months Shock - Permanent Shock
    </h1>
    """,
    unsafe_allow_html=True
)

from data_loader import load_stage2_data, build_reference_lists
from simulation_engine import (
    prepare_parameter_matrices,
    simulate_selection,
    summarize_impacts,
    build_map_dataset,
    dominant_category_table,
    compute_group_comparison,
    decomposition_table,
)
from visualization import line_chart, africa_map, cumulative_bar, stacked_decomposition

st.set_page_config(page_title='Oil Shock Inflation Simulator', layout='wide')
#st.title('Oil Price Shock Impact on Inflation in Africa')
st.caption('Stage 2 Streamlit simulator using precomputed distributed-lag parameters only.')


data = load_stage2_data()
refs = build_reference_lists(data)
matrices = prepare_parameter_matrices(data)
parameter_matrix = matrices['coefficients']
shock_templates = data['shock_templates']

with st.sidebar:
    st.header('Simulation controls')
    countries = st.multiselect('Country selector', refs['countries'], default=['Nigeria'] if 'Nigeria' in refs['countries'] else refs['countries'][:1])
    categories = st.multiselect('CPI category selector', refs['categories'], default=['All Items'] if 'All Items' in refs['categories'] else refs['categories'][:1])
    shock_magnitude = st.slider('Shock magnitude (%)', min_value=0, max_value=100, value=20, step=1)
    scenario = st.selectbox('Scenario selector', ['1 month', '3 months', '6 months', 'permanent'], index=1)
    horizon = st.slider('Time horizon (months)', min_value=1, max_value=12, value=12, step=1)
    show_uncertainty = st.toggle('Show uncertainty (confidence intervals)', value=True)
    autoplay = st.toggle('Autoplay animation', value=False)
    animation_speed = st.slider('Animation speed (seconds)', min_value=0.2, max_value=2.0, value=0.8, step=0.1)

if not countries:
    st.warning('Select at least one country to run the simulator.')
    st.stop()
if not categories:
    st.warning('Select at least one CPI category to run the simulator.')
    st.stop()

sim_df = simulate_selection(
    parameter_matrix=parameter_matrix,
    shock_templates=shock_templates,
    countries=countries,
    categories=categories,
    scenario=scenario,
    magnitude_pct=shock_magnitude,
    horizon=horizon,
    include_uncertainty=show_uncertainty,
)

if sim_df.empty:
    st.error('No simulation output is available for the selected combination. Try a different country or category.')
    st.stop()

summary_df = summarize_impacts(sim_df, ['country', 'category'])

st.subheader('Summary indicators')
metric_cols = st.columns(3)
metric_cols[0].metric('Peak inflation impact (pp)', f"{summary_df['peak_impact'].max():.2f}")
metric_cols[1].metric('Month of peak', f"{int(summary_df.loc[summary_df['peak_impact'].idxmax(), 'peak_month'])}")
metric_cols[2].metric('Cumulative impact (pp)', f"{summary_df['cumulative_impact'].sum():.2f}")


def download_frame_button(df: pd.DataFrame, filename: str, label: str, extra_filters: dict):
    export_df = df.copy()
    for key, value in extra_filters.items():
        export_df[key] = ', '.join(map(str, value)) if isinstance(value, list) else value
    st.download_button(label, data=export_df.to_csv(index=False).encode('utf-8'), file_name=filename, mime='text/csv')


tab_map, tab_country, tab_compare, tab_decomp = st.tabs(['Map', 'Country analysis', 'Comparison', 'Decomposition'])

with tab_map:
    st.subheader('Africa map of simulated CPI impact')
    if horizon == 1:
        month_for_map = 1
        st.caption('Month after shock: 1')
    else:
        month_for_map = st.slider(
            'Month after shock',
            min_value=1,
            max_value=horizon,
            value=min(3, horizon),
            step=1,
            key='map_month'
        )
    map_category = st.selectbox('Category for map', categories, index=0, key='map_category')

    map_sim = simulate_selection(
        parameter_matrix,
        shock_templates,
        countries=refs['countries'],
        categories=[map_category],
        scenario=scenario,
        magnitude_pct=shock_magnitude,
        horizon=horizon,
        include_uncertainty=show_uncertainty,
    )
    map_df = build_map_dataset(map_sim, month_for_map)
    st.plotly_chart(africa_map(map_df, f'{map_category}: month {month_for_map} impact across Africa'), use_container_width=True)

    dom = dominant_category_table(sim_df, month_for_map)
    if not dom.empty:
        st.markdown('**Dominant category among current selections**')
        st.dataframe(dom, use_container_width=True)

    download_frame_button(map_df, 'map_results.csv', 'Download map results', {
        'selected_month': month_for_map,
        'scenario': scenario,
        'shock_magnitude_pct': shock_magnitude,
        'category_filter': [map_category],
    })

    if autoplay:
        placeholder = st.empty()
        for m in range(1, horizon + 1):
            auto_df = build_map_dataset(map_sim, m)
            with placeholder.container():
                st.plotly_chart(africa_map(auto_df, f'Autoplay map: {map_category}, month {m}'), use_container_width=True)
            time.sleep(animation_speed)

with tab_country:
    st.subheader('Country analysis')
    color_mode = 'country' if len(countries) > 1 else 'category'
    if color_mode == 'category':
        chart_df = sim_df.copy()
    else:
        chart_df = sim_df.copy()
    st.plotly_chart(line_chart(chart_df, 'Simulated CPI impact over time', color_col=color_mode, include_uncertainty=show_uncertainty), use_container_width=True)
    st.dataframe(summary_df, use_container_width=True)
    download_frame_button(sim_df, 'country_analysis_results.csv', 'Download country analysis', {
        'countries': countries,
        'categories': categories,
        'scenario': scenario,
        'shock_magnitude_pct': shock_magnitude,
        'horizon': horizon,
    })

with tab_compare:
    st.subheader('Comparison view')
    compare_category = st.selectbox('Comparison category', categories, index=0, key='compare_category')

    compare_countries_df = sim_df[sim_df['category'].eq(compare_category)].copy()
    if not compare_countries_df.empty:
        st.plotly_chart(line_chart(compare_countries_df, f'Country comparison: {compare_category}', color_col='country', include_uncertainty=show_uncertainty), use_container_width=True)
        st.plotly_chart(cumulative_bar(summarize_impacts(compare_countries_df, ['country']), 'country', 'Cumulative country impact'), use_container_width=True)

    group_df = compute_group_comparison(matrices['oil_groups'], shock_templates, scenario, shock_magnitude, horizon, compare_category, show_uncertainty)
    if not group_df.empty:
        st.markdown('**Oil producers vs non-producers**')
        st.plotly_chart(line_chart(group_df, f'Oil-group comparison: {compare_category}', color_col='country', include_uncertainty=show_uncertainty), use_container_width=True)
        st.plotly_chart(cumulative_bar(summarize_impacts(group_df, ['country']), 'country', 'Cumulative impact by oil group'), use_container_width=True)

    africa_avg = data['dlr_category_africa'].copy()
    africa_wide = africa_avg[africa_avg['category'].isin(categories)].pivot_table(index=['country', 'category'], columns='lag', values=['beta', 'ci_low', 'ci_high'], aggfunc='first')
    if not africa_wide.empty:
        africa_wide.columns = [f'{a}_{int(b)}' for a, b in africa_wide.columns]
        africa_wide = africa_wide.reset_index()
        africa_sim = simulate_selection(africa_wide, shock_templates, africa_wide['country'].tolist(), africa_wide['category'].tolist(), scenario, shock_magnitude, horizon, show_uncertainty)
        st.markdown('**Africa average by category**')
        st.plotly_chart(line_chart(africa_sim, 'Africa-average category comparison', color_col='category', include_uncertainty=show_uncertainty), use_container_width=True)

    compare_export = pd.concat([compare_countries_df, group_df], ignore_index=True)
    download_frame_button(compare_export, 'comparison_results.csv', 'Download comparison results', {
        'countries': countries,
        'categories': categories,
        'comparison_category': compare_category,
        'scenario': scenario,
        'shock_magnitude_pct': shock_magnitude,
    })

with tab_decomp:
    st.subheader('Transmission decomposition')
    decomp_country = st.selectbox('Country for decomposition', countries, index=0, key='decomp_country')
    if horizon == 1:
        decomp_month = 1
        st.caption('Decomposition month: 1')
    else:
        decomp_month = st.slider(
            'Decomposition month',
            min_value=1,
            max_value=horizon,
            value=min(3, horizon),
            step=1,
            key='decomp_month'
        )
    decomp_df = decomposition_table(sim_df, decomp_month, decomp_country)
    st.plotly_chart(stacked_decomposition(decomp_df, f'Category-channel contribution for {decomp_country}, month {decomp_month}'), use_container_width=True)

    raw_country_month = sim_df[(sim_df['country'].eq(decomp_country)) & (sim_df['month'].eq(decomp_month))][['country', 'category', 'month', 'impact', 'impact_low', 'impact_high']].sort_values('impact', ascending=False)
    st.dataframe(raw_country_month, use_container_width=True)
    download_frame_button(raw_country_month, 'decomposition_results.csv', 'Download decomposition results', {
        'country': decomp_country,
        'month': decomp_month,
        'scenario': scenario,
        'shock_magnitude_pct': shock_magnitude,
    })

with st.expander('Policy interpretation notes'):
    st.markdown(
        'The simulator maps a user-defined oil price shock into monthly CPI effects using pre-estimated distributed lag coefficients. '
        'Peak impact identifies the maximum simulated inflation pressure, while cumulative impact summarizes the total pass-through over the selected horizon. '
        'Confidence intervals reflect uncertainty inherited from the estimated lag coefficients.'
    )

with st.expander('Data preview'):
    st.dataframe(data['sample_simulation'], use_container_width=True)
    st.caption(f"Loaded Stage 1 data from: {data['base_path']['path'].iloc[0]}")
