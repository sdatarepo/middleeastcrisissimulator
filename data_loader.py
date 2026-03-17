from __future__ import annotations

from pathlib import Path
import zipfile
import pandas as pd
import streamlit as st

DATA_DIRNAME = "oil_shock_simulation_outputs_extracted"


def _candidate_roots() -> list[Path]:
    cwd = Path.cwd()
    here = Path(__file__).resolve().parent
    return [
        cwd,
        here,
        Path('/mnt/data'),
        Path.cwd() / 'data',
        here / 'data',
    ]


def ensure_data_extracted() -> Path:
    """Find extracted outputs or unzip them once."""
    expected = ['transformed_data', 'dlr_parameters', 'simulation_outputs']

    for root in _candidate_roots():
        if all((root / x).exists() for x in expected):
            return root
        extracted = root / DATA_DIRNAME
        if all((extracted / x).exists() for x in expected):
            return extracted

    zip_candidates = []
    for root in _candidate_roots():
        zip_candidates.extend(root.glob('oil_shock_simulation_outputs.zip'))
        zip_candidates.extend(root.glob('**/oil_shock_simulation_outputs.zip'))

    seen = set()
    for zip_path in zip_candidates:
        if zip_path in seen:
            continue
        seen.add(zip_path)
        out_dir = zip_path.parent / DATA_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        if all((out_dir / x).exists() for x in expected):
            return out_dir

    raise FileNotFoundError(
        "Could not locate Stage 1 outputs. Place the extracted folders or "
        "oil_shock_simulation_outputs.zip next to app.py or in the working directory."
    )


@st.cache_data(show_spinner=False)
def load_stage2_data() -> dict[str, pd.DataFrame]:
    base = ensure_data_extracted()

    files = {
        'cpi_long': base / 'transformed_data' / 'cpi_long.csv',
        'cpi_inflation_series': base / 'transformed_data' / 'cpi_inflation_series.csv',
        'oil_monthly': base / 'transformed_data' / 'oil_monthly.csv',
        'model_dataset': base / 'transformed_data' / 'model_dataset.csv',
        'dlr_full_parameters': base / 'dlr_parameters' / 'dlr_full_parameters.csv',
        'dlr_country_all_items': base / 'dlr_parameters' / 'dlr_country_all_items.csv',
        'dlr_category_africa': base / 'dlr_parameters' / 'dlr_category_africa.csv',
        'dlr_oil_groups': base / 'dlr_parameters' / 'dlr_oil_groups.csv',
        'simulation_coefficients_matrix': base / 'simulation_outputs' / 'simulation_coefficients_matrix.csv',
        'metadata': base / 'simulation_outputs' / 'metadata.csv',
        'shock_templates': base / 'simulation_outputs' / 'shock_templates.csv',
        'sample_simulation': base / 'simulation_outputs' / 'sample_simulation.csv',
    }

    data: dict[str, pd.DataFrame] = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        for date_col in ['date']:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        data[name] = df

    data['base_path'] = pd.DataFrame({'path': [str(base)]})
    return data


@st.cache_data(show_spinner=False)
def build_reference_lists(data: dict[str, pd.DataFrame]) -> dict[str, list]:
    coeff = data['simulation_coefficients_matrix'].copy()
    full = data['dlr_full_parameters'].copy()
    shock = data['shock_templates'].copy()

    countries = sorted(coeff['country'].dropna().astype(str).unique().tolist())
    countries = [c for c in countries if c not in {'Africa', 'Oil Producer', 'Non Oil Producer'}]
    categories = sorted(coeff['category'].dropna().astype(str).unique().tolist())
    scenarios = shock['scenario'].dropna().astype(str).unique().tolist()
    lags = sorted(full['lag'].dropna().astype(int).unique().tolist())

    return {
        'countries': countries,
        'categories': categories,
        'scenarios': scenarios,
        'lags': lags,
        'max_horizon': int(shock['month'].max()),
    }
