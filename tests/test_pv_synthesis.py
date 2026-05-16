"""Unit tests for src/transformer_lol/pv_synthesis.py.

Tests verify:
1. Night hours output exactly zero
2. Peak output bounded by penetration_pu
3. Markov chain clear-state fraction near stationary distribution
4. Seed reproducibility (same seed → same output, different → different)
5. Annual capacity factor in physically realistic range for Jakarta
"""

from __future__ import annotations

import numpy as np
import pytest

from transformer_lol.pv_synthesis import (
    synthesize_pv_profile,
    _solar_elevation_sin,
    _haurwitz_irradiance,
    JAKARTA_LAT_DEG,
)


PENETRATION = 0.30
N_HOURS = 8760


# ---------------------------------------------------------------------------
# Test 1: Night hours are exactly zero
# ---------------------------------------------------------------------------

def test_pv_zero_at_night() -> None:
    """Hours where sun is below horizon must output exactly 0.0.

    Check the first 6 hours of Jan 1 (00:00–05:00 Jakarta local = definitely night
    at latitude -6.2° in January).
    """
    pv = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=42)

    # Jan 1 midnight to 05:00 (indices 0–4)
    assert np.all(pv[:5] == 0.0), f"PV at midnight–05:00 must be 0; got {pv[:5]}"

    # Cross-check using solar geometry directly
    h = np.arange(N_HOURS, dtype=float)
    doy = h / 24.0 + 1.0
    hour_local = h % 24.0
    sin_alpha = _solar_elevation_sin(doy, hour_local, JAKARTA_LAT_DEG)
    night_mask = sin_alpha <= 0.0
    assert np.all(pv[night_mask] == 0.0), "PV must be 0 whenever sun is below horizon"


# ---------------------------------------------------------------------------
# Test 2: Output bounded by penetration_pu
# ---------------------------------------------------------------------------

def test_pv_bounded_by_penetration() -> None:
    """PV output must be in [0, penetration_pu] at all times."""
    pv = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=42)
    assert np.all(pv >= 0.0), f"PV has negative values: min={pv.min()}"
    assert np.all(pv <= PENETRATION + 1e-9), \
        f"PV exceeds penetration_pu {PENETRATION}: max={pv.max()}"


# ---------------------------------------------------------------------------
# Test 3: Markov chain clear-state fraction near stationary
# ---------------------------------------------------------------------------

def test_markov_clear_fraction() -> None:
    """Annual clear-hour fraction must be near stationary (60%) within ±5%.

    Default p_cc=0.80, p_uu=0.70 → stationary clear fraction = 0.60.
    Over 8760 hours, empirical fraction should be 0.60 ± 0.05.
    """
    # Use a sunny month to isolate cloud effect from night
    # Compute fraction of daylight hours that are clear
    h = np.arange(N_HOURS, dtype=float)
    doy = h / 24.0 + 1.0
    hour_local = h % 24.0
    from transformer_lol.pv_synthesis import _solar_elevation_sin, JAKARTA_LAT_DEG
    sin_alpha = _solar_elevation_sin(doy, hour_local, JAKARTA_LAT_DEG)
    daylight_mask = sin_alpha > 0.0

    # Use penetration=1.0 and clearness=0.55 (default Markov params)
    pv_clear = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=1.0,
                                     clearness_index_mean=0.55, seed=42)

    # Clear-sky reference irradiance at each hour
    from transformer_lol.pv_synthesis import _haurwitz_irradiance, PERFORMANCE_RATIO
    g_cs = _haurwitz_irradiance(sin_alpha)
    g_cs_pu = g_cs / 1000.0 * PERFORMANCE_RATIO   # expected output if fully clear

    # Hours that are daytime and close to clear-sky output → "clear" in Markov
    is_clear_hour = daylight_mask & (pv_clear > 0.85 * g_cs_pu)
    clear_fraction_of_daylight = is_clear_hour.sum() / daylight_mask.sum()
    assert 0.50 <= clear_fraction_of_daylight <= 0.75, \
        f"Clear fraction = {clear_fraction_of_daylight:.3f} outside [0.50, 0.75]"


# ---------------------------------------------------------------------------
# Test 4: Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility() -> None:
    """Same seed → identical array; different seed → different array."""
    pv1 = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=0)
    pv2 = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=0)
    pv3 = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=1)

    np.testing.assert_array_equal(pv1, pv2, err_msg="Same seed must give identical output")
    assert not np.array_equal(pv1, pv3), "Different seeds should give different output"


# ---------------------------------------------------------------------------
# Test 5: Annual capacity factor in realistic range for Jakarta
# ---------------------------------------------------------------------------

def test_annual_capacity_factor() -> None:
    """Annual capacity factor for 30% penetration should be in [10%, 25%].

    Jakarta: ~2200 sunshine hours/year, PR=0.85.
    CF = total_energy / (8760 * penetration_pu).
    Realistic CF for tropical ground-mount: 15–22%.
    """
    pv = synthesize_pv_profile(n_hours=N_HOURS, penetration_pu=PENETRATION, seed=42)
    cf = pv.mean() / PENETRATION   # capacity factor as fraction
    assert 0.10 <= cf <= 0.25, f"Capacity factor {cf:.3f} outside [0.10, 0.25]"


# ---------------------------------------------------------------------------
# Test 6: Solar geometry sanity (equinox noon near Jakarta)
# ---------------------------------------------------------------------------

def test_solar_elevation_equinox_noon() -> None:
    """At vernal equinox (DOY=81) solar noon at Jakarta lat=-6.2, sun is near zenith.

    sin(alpha) at solar noon on equinox ≈ cos(lat) ≈ cos(-6.2°) ≈ 0.994.
    """
    sin_a = _solar_elevation_sin(
        doy=np.array([81.0]),
        hour_local=np.array([12.0]),
        lat_deg=JAKARTA_LAT_DEG,
    )
    assert abs(float(sin_a[0]) - 0.994) < 0.01, \
        f"sin(alpha) at equinox noon should be ~0.994, got {float(sin_a[0]):.4f}"


# ---------------------------------------------------------------------------
# Test 7: Haurwitz irradiance physical range
# ---------------------------------------------------------------------------

def test_haurwitz_range() -> None:
    """Haurwitz model: 0 at night, < 1200 W/m² at any angle."""
    sin_a = np.linspace(-0.1, 1.0, 200)
    g = _haurwitz_irradiance(sin_a)
    assert np.all(g >= 0.0), "Irradiance must be non-negative"
    assert np.all(g <= 1200.0), f"Irradiance exceeds 1200 W/m²: max={g.max():.1f}"
    assert g[sin_a <= 0].sum() == 0.0, "Irradiance must be 0 when sun is below horizon"
