from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

SCENARIO_LABELS = {
    '1 month': 'shock_1m',
    '3 months': 'shock_3m',
    '6 months': 'shock_6m',
    'permanent': 'shock_perm',
}

CHANNEL_PRIORITY = [
    'Food and non-alcoholic beverages',
    'Transport',
    'Housing, water, electricity, gas and other fuels',
    'Furnishings, household equipment and routine household maintenance',
    'Restaurants and hotels',
]

CHANNEL_GROUPS = {
    'Food': ['Food and non-alcoholic beverages'],
    'Transport': ['Transport'],
    'Housing': ['Housing, water, electricity, gas and other fuels'],
    'Energy-related': [
        'Housing, water, electricity, gas and other fuels',
        'Transport',
        'Furnishings, household equipment and routine household maintenance',
    ],
}


@dataclass
class SummaryStats:
    peak_impact: float
    peak_month: int
    cumulative_impact: float


def _lag_columns(prefix: str = 'beta', max_lag: int = 12) -> list[str]:
    return [f'{prefix}_{i}' for i in range(1, max_lag + 1)]


def normalize_scenario_label(label: str) -> str:
    return SCENARIO_LABELS.get(label, label)


@st.cache_data(show_spinner=False)
def prepare_parameter_matrices(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    coeff = data['simulation_coefficients_matrix'].copy()
    params = data['dlr_full_parameters'].copy()

    for lag in range(1, 13):
        if f'beta_{lag}' not in coeff.columns:
            coeff[f'beta_{lag}'] = 0.0

    ci_low = params.pivot_table(index=['country', 'category'], columns='lag', values='ci_low', aggfunc='first').reset_index()
    ci_high = params.pivot_table(index=['country', 'category'], columns='lag', values='ci_high', aggfunc='first').reset_index()

    ci_low.columns = ['country', 'category'] + [f'ci_low_{int(c)}' for c in ci_low.columns[2:]]
    ci_high.columns = ['country', 'category'] + [f'ci_high_{int(c)}' for c in ci_high.columns[2:]]

    for lag in range(1, 13):
        low_col = f'ci_low_{lag}'
        high_col = f'ci_high_{lag}'
        if low_col not in ci_low.columns:
            ci_low[low_col] = np.nan
        if high_col not in ci_high.columns:
            ci_high[high_col] = np.nan

    combined = coeff.merge(ci_low, on=['country', 'category'], how='left').merge(ci_high, on=['country', 'category'], how='left')
    return {
        'coefficients': combined,
        'oil_groups': data['dlr_oil_groups'].copy(),
        'country_all_items': data['dlr_country_all_items'].copy(),
        'africa_category': data['dlr_category_africa'].copy(),
    }


def get_shock_vector(shock_templates: pd.DataFrame, scenario: str, horizon: int, magnitude_pct: float) -> np.ndarray:
    scenario = normalize_scenario_label(scenario)
    template = (
        shock_templates.loc[shock_templates['scenario'].eq(scenario), ['month', 'normalized_shock']]
        .sort_values('month')
    )
    if template.empty:
        raise ValueError(f'Unknown scenario: {scenario}')
    vec = template['normalized_shock'].to_numpy(dtype=float)
    if horizon <= len(vec):
        vec = vec[:horizon]
    else:
        pad_value = vec[-1] if scenario == 'shock_perm' else 0.0
        vec = np.concatenate([vec, np.repeat(pad_value, horizon - len(vec))])
    return vec * magnitude_pct


def convolve_coefficients(coefficients: np.ndarray, shock_vector: np.ndarray) -> np.ndarray:
    horizon = len(shock_vector)
    max_lag = len(coefficients)
    impact = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        total = 0.0
        for k in range(max_lag):
            shock_idx = t - k
            if shock_idx >= 0:
                total += coefficients[k] * shock_vector[shock_idx]
        impact[t] = total
    return impact


def _extract_vector(row: pd.Series, prefix: str, max_lag: int = 12) -> np.ndarray:
    return np.array([float(row.get(f'{prefix}_{i}', 0.0) or 0.0) for i in range(1, max_lag + 1)], dtype=float)


def simulate_single(row: pd.Series, shock_templates: pd.DataFrame, scenario: str, magnitude_pct: float, horizon: int, include_uncertainty: bool = True) -> pd.DataFrame:
    shock = get_shock_vector(shock_templates, scenario, horizon, magnitude_pct)
    beta = _extract_vector(row, 'beta')
    impact = convolve_coefficients(beta, shock)

    out = pd.DataFrame({
        'country': row['country'],
        'category': row['category'],
        'month': np.arange(1, horizon + 1),
        'shock_input_pct': shock,
        'impact': impact,
    })

    if include_uncertainty:
        low = _extract_vector(row, 'ci_low')
        high = _extract_vector(row, 'ci_high')
        out['impact_low'] = convolve_coefficients(np.nan_to_num(low, nan=0.0), shock)
        out['impact_high'] = convolve_coefficients(np.nan_to_num(high, nan=0.0), shock)
    else:
        out['impact_low'] = np.nan
        out['impact_high'] = np.nan

    out['peak_impact'] = float(out['impact'].max()) if not out.empty else np.nan
    out['peak_month'] = int(out.loc[out['impact'].idxmax(), 'month']) if not out.empty else np.nan
    out['cumulative_impact'] = float(out['impact'].sum()) if not out.empty else np.nan
    return out


def simulate_selection(parameter_matrix: pd.DataFrame, shock_templates: pd.DataFrame, countries: list[str], categories: list[str], scenario: str, magnitude_pct: float, horizon: int, include_uncertainty: bool = True) -> pd.DataFrame:
    subset = parameter_matrix[
        parameter_matrix['country'].isin(countries) & parameter_matrix['category'].isin(categories)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=['country', 'category', 'month', 'shock_input_pct', 'impact', 'impact_low', 'impact_high'])

    frames = [simulate_single(row, shock_templates, scenario, magnitude_pct, horizon, include_uncertainty) for _, row in subset.iterrows()]
    return pd.concat(frames, ignore_index=True)


def summarize_impacts(sim_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if sim_df.empty:
        return pd.DataFrame(columns=group_cols + ['peak_impact', 'peak_month', 'cumulative_impact'])

    idx = sim_df.groupby(group_cols)['impact'].idxmax()
    peak_rows = sim_df.loc[idx, group_cols + ['month', 'impact']].rename(columns={'month': 'peak_month', 'impact': 'peak_impact'})
    cumulative = sim_df.groupby(group_cols, as_index=False)['impact'].sum().rename(columns={'impact': 'cumulative_impact'})
    return peak_rows.merge(cumulative, on=group_cols, how='left').sort_values(group_cols).reset_index(drop=True)


def dominant_category_table(sim_df: pd.DataFrame, month: int) -> pd.DataFrame:
    df = sim_df.loc[sim_df['month'].eq(month)].copy()
    if df.empty:
        return pd.DataFrame(columns=['country', 'dominant_category', 'impact'])
    idx = df.groupby('country')['impact'].idxmax()
    out = df.loc[idx, ['country', 'category', 'impact']].rename(columns={'category': 'dominant_category'})
    return out.sort_values('country').reset_index(drop=True)


def build_map_dataset(sim_df: pd.DataFrame, month: int) -> pd.DataFrame:
    df = sim_df.loc[sim_df['month'].eq(month)].copy()
    if df.empty:
        return df
    summary = summarize_impacts(sim_df, ['country', 'category'])
    df = df.merge(summary, on=['country', 'category'], how='left')
    return df


def compute_group_comparison(dlr_oil_groups: pd.DataFrame, shock_templates: pd.DataFrame, scenario: str, magnitude_pct: float, horizon: int, category: str, include_uncertainty: bool = True) -> pd.DataFrame:
    wide = dlr_oil_groups[dlr_oil_groups['category'].eq(category)].pivot_table(index=['country', 'category'], columns='lag', values=['beta', 'ci_low', 'ci_high'], aggfunc='first')
    if wide.empty:
        return pd.DataFrame()
    wide.columns = [f'{a}_{int(b)}' for a, b in wide.columns]
    wide = wide.reset_index()
    wide['country'] = wide['country'].replace({'Oil Producer': 'Oil producers', 'Non Oil Producer': 'Non-producers'})
    return simulate_selection(wide, shock_templates, wide['country'].tolist(), [category], scenario, magnitude_pct, horizon, include_uncertainty)


def decomposition_table(sim_df: pd.DataFrame, month: int, country: str) -> pd.DataFrame:
    df = sim_df[(sim_df['country'].eq(country)) & (sim_df['month'].eq(month))].copy()
    if df.empty:
        return pd.DataFrame(columns=['channel', 'impact'])

    rows = []
    for channel, cats in CHANNEL_GROUPS.items():
        value = df[df['category'].isin(cats)]['impact'].sum()
        rows.append({'channel': channel, 'impact': value})
    return pd.DataFrame(rows)
