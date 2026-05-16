# Transformer Loss-of-Life Simulation under Rooftop-PV Penetration

Reproducible Python codebase for the IEEE ICT-PEP 2026 paper:

> **"Transformer Loss-of-Life Acceleration under High Rooftop-PV Penetration on Indonesian 150/20 kV Substations: A Hybrid IEEE C57.91 + XGBoost Simulation"**

## Quick Start (10 commands)

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd transformer_lol_sim

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e .

# 4. Validate the thermal model against IEEE C57.91-2011 Annex G
python scripts/01_validate_thermal.py

# 5. Generate XGBoost surrogate training data (8-10 min)
python scripts/02_generate_training_data.py

# 6. Train the surrogate model (< 2 min)
python scripts/03_train_surrogate.py

# 7. Run the full Monte Carlo sweep (~ 25 min)
python scripts/04_run_monte_carlo.py --n_runs 1000

# 8. Generate all figures and tables
python scripts/05_generate_figures.py

# 9. (Alternative) Run the full pipeline in one step
python scripts/run_all.py

# 10. (Quick smoke test, 3 min)
python scripts/run_all.py --quick
```

## Key Results

| Metric | Value |
|--------|-------|
| Surrogate RMSE | < 0.26 °C (gate: < 1.5 °C) |
| Surrogate R² | > 0.9996 (gate: > 0.99) |
| PV non-monotonic turnover | ~30–50% penetration |
| Baseline annual LoL | ~0.04 % |
| 75% PV annual LoL | higher than baseline (acceleration effect) |

## Repository Structure

```
transformer_lol_sim/
├── config/                   # Transformer parameters + scenario grid
│   ├── transformer_60mva.yaml
│   └── scenarios.yaml
├── src/transformer_lol/      # Core library
│   ├── thermal_model.py      # IEEE C57.91-2011 ODE (Clause 7)
│   ├── aging.py              # Arrhenius F_AA + loss-of-life
│   ├── pv_synthesis.py       # Haurwitz clear-sky + Markov clouds
│   ├── load_profile.py       # Synthetic Indonesian load profile
│   ├── ambient_profile.py    # Tropical ambient generator
│   ├── surrogate.py          # XGBoost autoregressive model
│   ├── monte_carlo.py        # MC sweep driver
│   ├── scenarios.py          # Scenario definitions
│   └── plotting.py           # 8 publication-quality figures
├── scripts/                  # Executable pipeline scripts
│   ├── 01_validate_thermal.py
│   ├── 02_generate_training_data.py
│   ├── 03_train_surrogate.py
│   ├── 04_run_monte_carlo.py
│   ├── 05_generate_figures.py
│   └── run_all.py
├── tests/                    # pytest test suite (30 tests)
└── results/                  # Generated figures, tables, metrics
    ├── figures/              # 8 PNG + 8 PDF
    ├── tables/               # 3 CSV
    └── metrics.json
```

## Physics Model

The thermal model implements the **IEEE C57.91-2011 Clause 7 differential equations**:

```
dΔΘ_TO/dt = (1/τ_TO) · (ΔΘ_TO_ult − ΔΘ_TO)
dΔΘ_HS/dt = (1/τ_W)  · (ΔΘ_HS_ult − ΔΘ_HS)
```

Validation against Annex G fixtures (two-phase load step test) is run by `scripts/01_validate_thermal.py`.

The **XGBoost surrogate** predicts θ_HS autoregressively from 13 engineered features including load lags, ambient temperature, PV generation, and cyclic time encoding. Training uses teacher forcing; inference uses autoregressive rollout with analytical steady-state initialization.

## Scenarios

30-scenario grid: {PV: 0, 10, 30, 50, 75%} × {ΔT_amb: 0, 1, 2 °C} × {load growth: 0, 4.5%/yr}.
Plus 8 named interpretable scenarios (baseline, high-PV, climate projections).

## Requirements

- Python 3.10+
- numpy, scipy, pandas, xgboost, scikit-learn, matplotlib, pyyaml, pyarrow

See `requirements.txt` for pinned versions.

## Running Tests

```bash
pytest tests/ -v                           # all 30 tests
pytest tests/ -v --ignore=tests/test_end_to_end.py  # fast (< 20s)
```

## License

MIT. See LICENSE for details.
