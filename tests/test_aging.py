"""Unit tests for src/transformer_lol/aging.py.

Tests verify the Arrhenius aging acceleration factor (F_AA) formula
and the loss-of-life integrators per IEEE C57.91-2011 Clause 5.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transformer_lol.aging import (
    aging_acceleration_factor,
    b_from_moisture,
    faa_scalar,
    loss_of_life_hours,
    loss_of_life_percent,
    RATED_LIFE_HOURS,
    THETA_HS_RATED,
    B_CONSTANT,
    B_MOISTURE_SLOPE,
)


# ---------------------------------------------------------------------------
# Test 1: F_AA exactly 1.0 at rated temperature
# ---------------------------------------------------------------------------

def test_f_aa_at_rated_temp() -> None:
    """F_AA(110 °C) must equal 1.0 within 1e-9.

    C57.91-2011 Clause 5: 110 °C is the reference temperature.
    At this temperature exp(B*(1/T_ref - 1/T_ref)) = exp(0) = 1.
    """
    faa = faa_scalar(THETA_HS_RATED)
    assert math.isclose(faa, 1.0, abs_tol=1e-9), \
        f"F_AA(110 °C) must be exactly 1.0, got {faa}"


# ---------------------------------------------------------------------------
# Test 2: F_AA at 120 °C (exact formula value)
# ---------------------------------------------------------------------------

def test_f_aa_at_120c() -> None:
    """F_AA(120 °C) from the exact formula.

    Note: IEEE text loosely states ≈ 2, but the formula gives 2.7089.
    This test uses the exact value, not the approximation.
    """
    faa = faa_scalar(120.0)
    expected = math.exp(B_CONSTANT * (1.0/(THETA_HS_RATED + 273.15)
                                       - 1.0/(120.0 + 273.15)))
    assert math.isclose(faa, expected, rel_tol=1e-6), \
        f"F_AA(120 °C): expected {expected:.6f}, got {faa:.6f}"
    # Also verify it is in physically meaningful range (> 2, < 5)
    assert 2.0 < faa < 5.0, f"F_AA(120 °C) = {faa:.4f} outside expected range (2, 5)"


# ---------------------------------------------------------------------------
# Test 3: Monotonic ordering
# ---------------------------------------------------------------------------

def test_f_aa_ordering() -> None:
    """F_AA must be strictly increasing with temperature.

    Physical requirement: higher temperature → faster aging.
    """
    temps = [80.0, 98.0, 110.0, 120.0, 140.0, 160.0]
    faas = [faa_scalar(t) for t in temps]
    for i in range(len(faas) - 1):
        assert faas[i] < faas[i + 1], \
            f"F_AA not monotonically increasing: F_AA({temps[i]})={faas[i]:.4f} " \
            f">= F_AA({temps[i+1]})={faas[i+1]:.4f}"

    # Below 110 °C → F_AA < 1 (slower aging than reference)
    assert faa_scalar(98.0) < 1.0, "F_AA(98) should be < 1.0"
    # Above 110 °C → F_AA > 1 (faster aging than reference)
    assert faa_scalar(120.0) > 1.0, "F_AA(120) should be > 1.0"


# ---------------------------------------------------------------------------
# Test 4: LoL at rated life span
# ---------------------------------------------------------------------------

def test_lol_100_percent_at_180000h() -> None:
    """180 000 hours at 110 °C must give exactly 100 % LoL.

    C57.91-2011 Clause 5: rated life is defined at 110 °C continuous operation.
    F_AA(110°C) = 1.0, so aging hours = 180 000 × 1.0 = 180 000 → 100%.
    """
    # Scalar input → treated as constant temperature for one step
    lol = loss_of_life_percent(
        theta_hs_c=110.0,
        dt_hours=RATED_LIFE_HOURS,   # one timestep of 180 000 hours duration
        rated_life_hours=RATED_LIFE_HOURS,
    )
    assert math.isclose(lol, 100.0, abs_tol=0.001), \
        f"180 000 h at 110 °C should give 100 % LoL, got {lol:.6f} %"


def test_lol_distributed_rated() -> None:
    """Array of 180 000 steps at 110 °C for 1 hour each → 100 % LoL."""
    n = int(RATED_LIFE_HOURS)  # 180 000 hourly steps
    theta_hs = np.full(n, 110.0)
    lol = loss_of_life_percent(theta_hs, dt_hours=1.0, rated_life_hours=RATED_LIFE_HOURS)
    assert math.isclose(lol, 100.0, abs_tol=0.001), \
        f"180 000 hourly steps at 110 °C → 100 % LoL, got {lol:.6f} %"


# ---------------------------------------------------------------------------
# Test 5: Physical bounds
# ---------------------------------------------------------------------------

def test_f_aa_positive() -> None:
    """F_AA must be strictly positive for all finite temperatures."""
    temps = np.linspace(40.0, 200.0, 50)
    faas = aging_acceleration_factor(temps)
    assert np.all(faas > 0.0), "F_AA must be positive for all temperatures"


# ---------------------------------------------------------------------------
# Test 6: Moisture-dependent B-constant (IEEE C57.91-2011 Annex E)
# ---------------------------------------------------------------------------

def test_moisture_b_constant() -> None:
    """B(w) linear model per IEEE C57.91-2011 Annex E.

    Fixture values (from the standard):
      w=1% (dry service):  B = 15 000 K  (= B_CONSTANT)
      w=2%:                B ≈ 14 579 K
      w=3% (wet season):   B = 14 158 K

    Higher moisture → lower B → less temperature-sensitive aging.
    At sub-rated temperatures (typical operation), lower B gives higher F_AA
    (less aging suppression), so wet insulation ages faster day-to-day even
    though it would age slower than dry at severe overloads.
    """
    # Endpoint values from IEEE C57.91-2011 Annex E
    assert math.isclose(b_from_moisture(1.0), B_CONSTANT, abs_tol=0.1), \
        f"B at 1% should equal B_CONSTANT={B_CONSTANT}, got {b_from_moisture(1.0)}"
    assert math.isclose(b_from_moisture(3.0), 14_158.0, abs_tol=0.1), \
        f"B at 3% should be 14 158, got {b_from_moisture(3.0)}"
    assert math.isclose(b_from_moisture(2.0), B_CONSTANT + B_MOISTURE_SLOPE, abs_tol=1.0), \
        f"B at 2% should be ~{B_CONSTANT + B_MOISTURE_SLOPE:.0f}"

    # Clamping: outside [1, 3] % should saturate
    assert math.isclose(b_from_moisture(0.0), b_from_moisture(1.0), abs_tol=0.01), \
        "B below 1% should clamp to B(1%)"
    assert math.isclose(b_from_moisture(5.0), b_from_moisture(3.0), abs_tol=0.01), \
        "B above 3% should clamp to B(3%)"

    # At sub-rated temperature (90 °C), lower B → higher F_AA (wetter ages faster)
    faa_dry = float(aging_acceleration_factor(90.0, moisture_water_pct=1.0))
    faa_wet = float(aging_acceleration_factor(90.0, moisture_water_pct=3.0))
    assert faa_wet > faa_dry, \
        "At sub-rated temps, wetter insulation (lower B) must give higher F_AA"

    # At overload (120 °C > rated), lower B → lower F_AA (wetter ages slower)
    faa_dry_hot = float(aging_acceleration_factor(120.0, moisture_water_pct=1.0))
    faa_wet_hot = float(aging_acceleration_factor(120.0, moisture_water_pct=3.0))
    assert faa_wet_hot < faa_dry_hot, \
        "At overload temps, wetter insulation (lower B) must give lower F_AA"

    # None argument falls back to standard B=15 000
    faa_default = float(aging_acceleration_factor(120.0, moisture_water_pct=None))
    faa_std = float(aging_acceleration_factor(120.0))
    assert math.isclose(faa_default, faa_std, rel_tol=1e-9), \
        "moisture_water_pct=None must give same result as no argument"


def test_lol_zero_at_zero_temperature_effect() -> None:
    """Very cold temperatures (F_AA << 1) give near-zero LoL."""
    theta_hs = np.full(24, 40.0)   # very cold transformer
    lol = loss_of_life_percent(theta_hs, dt_hours=1.0)
    assert lol < 0.01, f"LoL at 40 °C should be negligible, got {lol:.6f} %"
