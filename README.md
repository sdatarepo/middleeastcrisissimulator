# Oil Shock Inflation Simulator (Streamlit)

This app uses only the precomputed Stage 1 outputs. It does **not** re-estimate regressions.

## Files included
- `app.py`
- `data_loader.py`
- `simulation_engine.py`
- `visualization.py`
- `requirements.txt`

## Expected data
Place either:
- the extracted Stage 1 folders next to `app.py`, or
- `oil_shock_simulation_outputs.zip` next to `app.py`

The app will automatically locate or extract:
- `transformed_data/`
- `dlr_parameters/`
- `simulation_outputs/`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Africa map of simulated CPI impact
- Country and category comparison
- Oil producers vs non-producers comparison
- Confidence bands from `ci_low` and `ci_high`
- Downloadable filtered results in every tab
- Cached loading and precomputed parameter use only
