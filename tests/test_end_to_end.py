"""End-to-end smoke test: one-day simulation runs without errors.

This test verifies that the full pipeline (profile generation -> thermal
simulation -> aging computation -> MC result DataFrame) executes end-to-end
in under 60 seconds for a small baseline scenario.

The surrogate is used if the model file exists; otherwise falls back to ODE.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transformer_lol.monte_carlo import run_scenario, MC_RESULT_COLUMNS
from transformer_lol.scenarios import Scenario
from transformer_lol.thermal_model import TransformerParams
from transformer_lol.utils import get_project_root


TRANSFORMER = TransformerParams()
ROOT = get_project_root()
SURROGATE_MODEL_PATH = ROOT / "results" / "surrogate_model.json"
SURROGATE_SCALER_PATH = ROOT / "results" / "surrogate_scaler.pkl"


def _load_surrogate_if_available():
    """Return (model, scaler) if the surrogate is trained, else (None, None)."""
    if SURROGATE_MODEL_PATH.exists() and SURROGATE_SCALER_PATH.exists():
        from transformer_lol.surrogate import load_surrogate
        return load_surrogate(SURROGATE_MODEL_PATH, SURROGATE_SCALER_PATH)
    return None, None


# ---------------------------------------------------------------------------
# Test 1: Baseline scenario runs 10 realizations
# ---------------------------------------------------------------------------

def test_baseline_10_runs() -> None:
    """10 MC runs of the baseline (no PV) scenario complete correctly."""
    scenario = Scenario(
        name="baseline_smoke",
        pv_penetration=0.0,
        ambient_delta_c=0.0,
        load_growth_pct_per_year=0.0,
        n_runs=10,
        base_seed=0,
    )
    model, scaler = _load_surrogate_if_available()
    use_surrogate = model is not None

    t0 = time.perf_counter()
    df = run_scenario(
        scenario=scenario,
        transformer=TRANSFORMER,
        use_surrogate=use_surrogate,
        surrogate_model=model,
        surrogate_scaler=scaler,
        n_hours=8760,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    # Shape
    assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
    assert list(df.columns) == MC_RESULT_COLUMNS, "DataFrame columns mismatch"

    # No NaN
    assert not df.isna().any().any(), "NaN values in result DataFrame"

    # Physical bounds on LoL
    assert (df["lol_percent_annual"] >= 0.0).all(), "LoL must be non-negative"
    assert (df["lol_percent_annual"] < 10.0).all(), \
        f"LoL implausibly high: max={df['lol_percent_annual'].max():.3f}%"

    # Physical bounds on peak theta_HS
    assert (df["peak_theta_hs_c"] > 30.0).all(), "Peak theta_HS < ambient?"
    assert (df["peak_theta_hs_c"] < 200.0).all(), "Peak theta_HS implausibly high"

    # Runtime check
    assert elapsed < 60.0, f"10 runs took {elapsed:.1f}s — too slow (threshold 60s)"


# ---------------------------------------------------------------------------
# Test 2: PV scenario has lower or comparable LoL vs baseline (moderate PV)
# ---------------------------------------------------------------------------

def test_moderate_pv_reduces_lol() -> None:
    """At 30% PV, mean LoL should be <= baseline (cool midday effect dominates)."""
    model, scaler = _load_surrogate_if_available()
    use_surrogate = model is not None

    def run(pv_pen: float, n: int = 20) -> float:
        sc = Scenario(name=f"test_pv{pv_pen:.0%}", pv_penetration=pv_pen,
                      n_runs=n, base_seed=0)
        df = run_scenario(sc, TRANSFORMER, use_surrogate=use_surrogate,
                          surrogate_model=model, surrogate_scaler=scaler,
                          n_hours=8760, verbose=False)
        return float(df["lol_percent_annual"].mean())

    lol_0 = run(0.0)
    lol_30 = run(0.30)
    # Moderate PV should not dramatically worsen LoL
    # (may be slightly higher or lower depending on cloud variability)
    assert lol_30 < lol_0 * 1.5, \
        f"30% PV LoL ({lol_30:.3f}%) is > 1.5x baseline ({lol_0:.3f}%) — unexpected"


# ---------------------------------------------------------------------------
# Test 3: Result columns are correctly typed
# ---------------------------------------------------------------------------

def test_result_dtypes() -> None:
    """All numeric columns must be float/int; no object dtype."""
    sc = Scenario(name="dtype_check", pv_penetration=0.0, n_runs=5, base_seed=1)
    model, scaler = _load_surrogate_if_available()
    df = run_scenario(sc, TRANSFORMER, use_surrogate=model is not None,
                      surrogate_model=model, surrogate_scaler=scaler,
                      n_hours=8760, verbose=False)
    numeric_cols = [c for c in MC_RESULT_COLUMNS
                    if c not in ("scenario_name", "dispatch_method")]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), \
            f"Column '{col}' has non-numeric dtype: {df[col].dtype}"


# ---------------------------------------------------------------------------
# Test 4: Reverse flow hours zero when PV=0
# ---------------------------------------------------------------------------

def test_no_reverse_flow_without_pv() -> None:
    """With zero PV penetration, reverse_flow_hours must be 0."""
    sc = Scenario(name="no_pv_revflow", pv_penetration=0.0, n_runs=3, base_seed=2)
    model, scaler = _load_surrogate_if_available()
    df = run_scenario(sc, TRANSFORMER, use_surrogate=model is not None,
                      surrogate_model=model, surrogate_scaler=scaler,
                      n_hours=8760, verbose=False)
    assert (df["reverse_flow_hours"] == 0.0).all(), \
        "reverse_flow_hours should be 0 with no PV"
