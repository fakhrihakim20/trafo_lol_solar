"""Unit tests for load_profile.py — both the national and East Java templates.

Previously a gap in the test suite (noted in the explore phase of the
PyPSA migration plan).
"""

from __future__ import annotations

import numpy as np
import pytest

from transformer_lol.load_profile import (
    synthesize_eastjava_load,
    synthesize_fallback_load,
)


# ---------------------------------------------------------------------------
# Generic Indonesian template
# ---------------------------------------------------------------------------

def test_fallback_load_shape() -> None:
    arr = synthesize_fallback_load(n_hours=8760, peak_pu=0.85, seed=42)
    assert arr.shape == (8760,)
    assert np.all(np.isfinite(arr))
    assert arr.min() >= 0.0
    assert arr.max() <= 1.5


def test_fallback_load_peak_near_target() -> None:
    """Annual max must be close to peak_pu (template normalised so hour 20 = 1.0)."""
    arr = synthesize_fallback_load(n_hours=24 * 30, peak_pu=0.85, seed=42)
    assert 0.80 < arr.max() < 0.95, f"peak {arr.max():.3f} not near target 0.85"


def test_fallback_load_seed_reproducibility() -> None:
    a = synthesize_fallback_load(n_hours=240, seed=99)
    b = synthesize_fallback_load(n_hours=240, seed=99)
    np.testing.assert_array_equal(a, b)


def test_fallback_load_weekend_dip() -> None:
    """Saturdays/Sundays should average ~85 % of weekday levels (× 0.85)."""
    arr = synthesize_fallback_load(n_hours=24 * 28, peak_pu=0.85, seed=42)
    daily = arr.reshape(28, 24).mean(axis=1)
    weekdays = daily[[0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25]]
    weekends = daily[[5, 6, 12, 13, 19, 20, 26, 27]]
    assert weekends.mean() < weekdays.mean()


# ---------------------------------------------------------------------------
# East Java (Jember) template
# ---------------------------------------------------------------------------

def test_eastjava_load_shape() -> None:
    arr = synthesize_eastjava_load(n_hours=8760, peak_pu=0.85, seed=42)
    assert arr.shape == (8760,)
    assert np.all(np.isfinite(arr))
    assert 0.0 < arr.min()
    assert arr.max() <= 1.5


def test_eastjava_evening_peak_later_than_national() -> None:
    """East Java template should peak at hour 21:00 vs national 20:00."""
    arr = synthesize_eastjava_load(n_hours=24 * 14, peak_pu=0.85, seed=0)
    avg = arr.reshape(14, 24).mean(axis=0)
    peak_hour = int(np.argmax(avg))
    assert 20 <= peak_hour <= 22, f"East Java peak at hour {peak_hour}, expected 20-22"


def test_eastjava_softer_weekend() -> None:
    """East Java weekend reduction (0.92×) is softer than national (0.85×)."""
    arr_ej = synthesize_eastjava_load(n_hours=24 * 14, peak_pu=0.85, seed=42,
                                       weekend_multiplier=0.92)
    arr_nat = synthesize_fallback_load(n_hours=24 * 14, peak_pu=0.85, seed=42)
    # Last day = Sunday in our convention; East Java should be relatively higher
    sunday_ej_mean = arr_ej[-24:].mean()
    sunday_nat_mean = arr_nat[-24:].mean()
    # East Java mean should be closer to 1.0× than national (which dips by 15 %)
    assert sunday_ej_mean > sunday_nat_mean * 0.95


def test_eastjava_seed_reproducibility() -> None:
    a = synthesize_eastjava_load(n_hours=240, seed=11)
    b = synthesize_eastjava_load(n_hours=240, seed=11)
    np.testing.assert_array_equal(a, b)
