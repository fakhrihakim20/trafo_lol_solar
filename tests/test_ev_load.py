"""Unit tests for EV charging load synthesis (ev_load.py).

Tests verify:
1. Zero penetration → zero output (regression gate)
2. Slow overnight: peak hour is within 02:00 ± 2 h window
3. Slow overnight: zero output during 10:00–18:00 (no daytime slow charging)
4. Fast opportunistic: zero output during 22:00–06:00 (no night fast charging)
5. Fast opportunistic: non-zero output during 10:00–14:00 (business hours)
6. Combined wrapper: sum equals slow + fast individually
7. Seed reproducibility: same seed → identical output
"""

from __future__ import annotations

import numpy as np
import pytest

from transformer_lol.ev_load import (
    synthesize_ev_fast,
    synthesize_ev_slow,
    synthesize_ev_total,
)


N_HOURS = 24 * 7   # one week
PENE = 0.10        # 10% penetration


# ---------------------------------------------------------------------------
# Test 1: Zero penetration → all zeros (regression gate)
# ---------------------------------------------------------------------------

def test_zero_penetration_slow() -> None:
    """synthesize_ev_slow with penetration=0 must return all zeros."""
    out = synthesize_ev_slow(N_HOURS, penetration_pu=0.0, seed=0)
    assert np.all(out == 0.0), "Zero penetration must give zero output (slow)"


def test_zero_penetration_fast() -> None:
    """synthesize_ev_fast with penetration=0 must return all zeros."""
    out = synthesize_ev_fast(N_HOURS, penetration_pu=0.0, seed=0)
    assert np.all(out == 0.0), "Zero penetration must give zero output (fast)"


# ---------------------------------------------------------------------------
# Test 2: Slow charging — peak near 02:00
# ---------------------------------------------------------------------------

def test_slow_peak_near_02h() -> None:
    """Slow charging daily average must peak within 02:00 ± 2 h.

    We average over the 7-day week to smooth daily jitter, then find the
    hour-of-day with the maximum load.
    """
    out = synthesize_ev_slow(N_HOURS, penetration_pu=PENE, seed=0)
    # Reshape to (days, 24) and average across days
    days = len(out) // 24
    avg = out[:days * 24].reshape(days, 24).mean(axis=0)
    peak_hour = int(np.argmax(avg))
    # 02:00 ± 2 h, accounting for midnight wrap-around
    delta = min(abs(peak_hour - 2), 24 - abs(peak_hour - 2))
    assert delta <= 2, (
        f"Slow EV peak should be at 02:00 ± 2 h, got hour {peak_hour} "
        f"(delta={delta})"
    )


# ---------------------------------------------------------------------------
# Test 3: Slow charging — zero during midday
# ---------------------------------------------------------------------------

def test_slow_zero_during_midday() -> None:
    """Slow charging should be essentially zero between 10:00 and 18:00.

    At σ=2.5 h and peak at 02:00, the load at 10:00 is
    exp(-0.5*(8/2.5)²) ≈ 0.00034 of peak — effectively zero.
    """
    out = synthesize_ev_slow(N_HOURS, penetration_pu=PENE, seed=0)
    # Extract all 10:00–17:00 hours across days
    hours_of_day = np.arange(N_HOURS) % 24
    midday_mask = (hours_of_day >= 10) & (hours_of_day < 18)
    max_midday = out[midday_mask].max()
    assert max_midday < PENE * 0.01, (
        f"Slow EV should be ~0 at midday; got max={max_midday:.5f} "
        f"(threshold={PENE*0.01:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: Fast charging — zero at night
# ---------------------------------------------------------------------------

def test_fast_zero_at_night() -> None:
    """Fast charging must be zero between 22:00 and 06:00."""
    out = synthesize_ev_fast(N_HOURS, penetration_pu=PENE, seed=0)
    hours_of_day = np.arange(N_HOURS) % 24
    night_mask = (hours_of_day >= 22) | (hours_of_day < 6)
    assert np.all(out[night_mask] == 0.0), (
        "Fast EV charging must be zero between 22:00 and 06:00"
    )


# ---------------------------------------------------------------------------
# Test 5: Fast charging — non-zero during business hours
# ---------------------------------------------------------------------------

def test_fast_nonzero_in_business_hours() -> None:
    """Fast charging must be non-zero during 10:00–14:00 (active window)."""
    out = synthesize_ev_fast(N_HOURS, penetration_pu=PENE, seed=0)
    hours_of_day = np.arange(N_HOURS) % 24
    biz_mask = (hours_of_day >= 10) & (hours_of_day < 14)
    assert out[biz_mask].mean() > PENE * 0.05, (
        "Fast EV should have meaningful load during 10:00–14:00 business hours"
    )


# ---------------------------------------------------------------------------
# Test 6: Combined wrapper equals sum of parts
# ---------------------------------------------------------------------------

def test_total_equals_sum() -> None:
    """synthesize_ev_total must equal slow + fast computed separately."""
    seed = 77
    slow = synthesize_ev_slow(N_HOURS, PENE * 0.7, seed=seed)
    fast = synthesize_ev_fast(N_HOURS, PENE * 0.3, seed=seed + 1)
    total = synthesize_ev_total(N_HOURS, PENE * 0.7, PENE * 0.3, seed=seed)
    np.testing.assert_allclose(
        total, slow + fast, atol=1e-12,
        err_msg="synthesize_ev_total must equal sum of slow and fast",
    )


# ---------------------------------------------------------------------------
# Test 7: Seed reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility() -> None:
    """Same seed must produce identical output across two calls."""
    a = synthesize_ev_slow(N_HOURS, PENE, seed=42)
    b = synthesize_ev_slow(N_HOURS, PENE, seed=42)
    np.testing.assert_array_equal(a, b, err_msg="Same seed must be reproducible (slow)")

    c = synthesize_ev_fast(N_HOURS, PENE, seed=99)
    d = synthesize_ev_fast(N_HOURS, PENE, seed=99)
    np.testing.assert_array_equal(c, d, err_msg="Same seed must be reproducible (fast)")
