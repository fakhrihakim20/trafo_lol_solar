"""Unit tests for the Jember data layer (Phase 4C).

Tests verify:
1. ``load_jember_location()`` returns sensible coordinates and climate.
2. ``load_jember_ambient`` falls back to synthesis with Jember climate
   when no BMKG CSV is present (mean ≈ 26.5 °C ± 1).
3. ``load_jember_pv`` falls back to Haurwitz at Jember latitude when no
   atlite cutout is present; per-unit values respect the penetration cap.
4. ``load_jember_load`` falls back to East Java template; peak ≤ peak_pu.
5. delta_c uplift is added on the ambient series.
"""

from __future__ import annotations

import numpy as np
import pytest

from transformer_lol.data_jember import (
    JemberLocation,
    load_jember_ambient,
    load_jember_load,
    load_jember_location,
    load_jember_pv,
)


# ---------------------------------------------------------------------------
# Test 1: location dataclass
# ---------------------------------------------------------------------------

def test_load_jember_location() -> None:
    loc = load_jember_location()
    assert isinstance(loc, JemberLocation)
    assert loc.name == "Jember"
    assert loc.country == "Indonesia"
    # Jember lat is around -8.17, lon around 113.7
    assert -9.0 < loc.latitude < -7.0, f"latitude {loc.latitude} not in Jember range"
    assert 112.0 < loc.longitude < 115.0
    # Climate sanity: cooler than Jakarta's 28°C
    assert 25.0 < loc.annual_mean_c < 28.0


# ---------------------------------------------------------------------------
# Test 2: ambient fallback uses Jember climate
# ---------------------------------------------------------------------------

def test_load_jember_ambient_fallback_climate() -> None:
    """Without a BMKG CSV, the loader falls back to Jember-tuned synthesis."""
    arr = load_jember_ambient(
        n_hours=8760, delta_c=0.0, seed=42,
        bmkg_csv_path="/nonexistent/bmkg.csv",  # force fallback
    )
    assert arr.shape == (8760,)
    mean = float(arr.mean())
    # Jember annual mean ≈ 26.5 °C; allow ±1.5 for noise + diurnal averaging
    assert 25.0 < mean < 28.0, f"Jember ambient mean {mean:.2f} out of expected range"
    # No NaNs / inf
    assert np.all(np.isfinite(arr))


def test_load_jember_ambient_delta_c_uplift() -> None:
    """delta_c must shift every value by the same amount."""
    base = load_jember_ambient(n_hours=240, delta_c=0.0, seed=7,
                                bmkg_csv_path="/nonexistent/bmkg.csv")
    warm = load_jember_ambient(n_hours=240, delta_c=2.0, seed=7,
                                bmkg_csv_path="/nonexistent/bmkg.csv")
    np.testing.assert_allclose(warm - base, 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Test 3: PV fallback respects penetration_pu
# ---------------------------------------------------------------------------

def test_load_jember_pv_fallback_caps_at_penetration() -> None:
    """PV fallback values must lie in [0, penetration_pu] (modulo small noise)."""
    pv = load_jember_pv(
        n_hours=24 * 14, penetration_pu=0.50, seed=42,
        cutout_path="/nonexistent/cutout.nc",
    )
    assert pv.shape == (24 * 14,)
    assert pv.min() >= 0.0
    assert pv.max() <= 0.50 + 1e-6, f"pv max {pv.max():.4f} > penetration 0.50"


def test_load_jember_pv_zero_penetration_zero_output() -> None:
    pv = load_jember_pv(n_hours=24, penetration_pu=0.0, seed=0,
                         cutout_path="/nonexistent/cutout.nc")
    assert np.all(pv == 0.0)


# ---------------------------------------------------------------------------
# Test 4: Load fallback respects peak_pu
# ---------------------------------------------------------------------------

def test_load_jember_load_fallback_peak() -> None:
    load = load_jember_load(
        n_hours=24 * 30, peak_pu=0.85, seed=42,
        pln_csv_path="/nonexistent/pln.csv",
    )
    assert load.shape == (24 * 30,)
    # East Java synthesis: peak ≤ peak_pu × 1.0 (template normalised to 1.0 at hour 21)
    # plus small Gaussian noise (sigma = 2.5 % of peak)
    assert load.max() <= 0.85 * 1.10, f"load max {load.max():.3f} too high"
    # Min should be a meaningful fraction of peak (not zero)
    assert load.min() > 0.10
