"""Unit tests for battery storage dispatch (battery.py).

Tests verify:
1. Zero capacity → all-zeros no-op (regression gate)
2. Full charge during reverse-flow: battery absorbs excess PV
3. SoC bounds: never exceeds [0.10, 0.90]
4. Round-trip efficiency: energy delivered ≤ energy stored × η
5. Net-load reduction: battery reduces peak net load vs no-battery
6. Lifetime fraction: non-zero after cycling, zero for idle battery
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transformer_lol.battery import (
    BatteryParams,
    BATTERY_EFFICIENCY,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    dispatch_battery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_profile(n_hours: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """24-hour square-wave: load=0.9 pu all day, PV=1.0 pu for hours 8-16."""
    load = np.full(n_hours, 0.9)
    pv = np.zeros(n_hours)
    pv[8:16] = 1.0  # excess PV in daylight → reverse flow
    return load, pv


PARAMS_5MWH = BatteryParams(
    capacity_mwh=5.0,
    max_power_mw=2.0,
    discharge_threshold_pu=0.85,
    transformer_rated_mva=60.0,
    soc_init=0.50,
)


# ---------------------------------------------------------------------------
# Test 1: Zero capacity is a no-op
# ---------------------------------------------------------------------------

def test_no_battery_is_noop() -> None:
    """capacity_mwh=0 must return net_load = clip(load-pv, 0) unchanged."""
    load, pv = _simple_profile()
    params = BatteryParams(capacity_mwh=0.0, max_power_mw=0.0)
    result = dispatch_battery(load, pv, params)

    expected_net = np.clip(load - pv, 0.0, None)
    np.testing.assert_allclose(
        result["net_load"], expected_net, atol=1e-12,
        err_msg="Zero-capacity battery must not alter net load",
    )
    assert np.all(result["battery_p_pu"] == 0.0), "No battery → zero power"
    assert result["lifetime_fraction_used"] == 0.0


# ---------------------------------------------------------------------------
# Test 2: Battery absorbs reverse-flow during PV surplus
# ---------------------------------------------------------------------------

def test_charges_during_reverse_flow() -> None:
    """When PV > load, battery must charge (battery_p < 0 in the convention)."""
    load, pv = _simple_profile()  # pv=1.0 > load=0.9 during hours 8–16
    result = dispatch_battery(load, pv, PARAMS_5MWH)

    # During surplus hours, battery power should be negative (charging)
    for h in range(8, 16):
        assert result["battery_p_pu"][h] <= 0.0, (
            f"Battery should charge during PV surplus at hour {h}; "
            f"got battery_p={result['battery_p_pu'][h]:.4f}"
        )

    # After charging, SoC should be higher than initial 0.50
    assert result["soc"].max() > 0.50, "SoC must rise during charging"


# ---------------------------------------------------------------------------
# Test 3: SoC bounds strictly respected
# ---------------------------------------------------------------------------

def test_soc_bounds() -> None:
    """SoC must stay within [SoC_min, SoC_max] at all times."""
    # Aggressive profile: large PV surplus then heavy load
    n = 48
    load = np.full(n, 1.2)   # heavy load all day
    pv = np.zeros(n)
    pv[0:12] = 2.0            # massive PV surplus first 12 h
    pv[12:] = 0.0

    result = dispatch_battery(load, pv, PARAMS_5MWH)

    assert result["soc"].min() >= BATTERY_SOC_MIN - 1e-9, (
        f"SoC min {result['soc'].min():.4f} < SoC_min {BATTERY_SOC_MIN}"
    )
    assert result["soc"].max() <= BATTERY_SOC_MAX + 1e-9, (
        f"SoC max {result['soc'].max():.4f} > SoC_max {BATTERY_SOC_MAX}"
    )


# ---------------------------------------------------------------------------
# Test 4: Round-trip efficiency — energy out ≤ energy in × η
# ---------------------------------------------------------------------------

def test_round_trip_efficiency() -> None:
    """Total energy discharged ≤ energy charged × η (starting from empty battery).

    Starting at SoC_min ensures no pre-stored energy can be discharged without
    first being charged in this run.  The invariant is:
        energy_out ≤ energy_in × η
    because each unit of energy stored must pass through the η=0.9 discharge
    efficiency before delivery.
    """
    # Start with empty battery (SoC_min) so no pre-stored energy skews the count
    params_empty = BatteryParams(
        capacity_mwh=5.0,
        max_power_mw=2.0,
        discharge_threshold_pu=0.85,
        transformer_rated_mva=60.0,
        soc_init=BATTERY_SOC_MIN,   # start empty
    )
    load, pv = _simple_profile(48)  # 2-day cycle
    result = dispatch_battery(load, pv, params_empty, dt_hours=1.0)

    rated_mva = params_empty.transformer_rated_mva
    battery_p = result["battery_p_pu"]

    energy_charged_mwh = float(np.sum(battery_p[battery_p < 0]) * (-rated_mva))
    energy_discharged_mwh = float(np.sum(battery_p[battery_p > 0]) * rated_mva)

    # Discharged ≤ charged × η (some energy may remain in battery, further reducing ratio)
    assert energy_discharged_mwh <= energy_charged_mwh * BATTERY_EFFICIENCY + 1e-6, (
        f"Discharged {energy_discharged_mwh:.3f} MWh > "
        f"charged {energy_charged_mwh:.3f} × η={BATTERY_EFFICIENCY}"
    )


# ---------------------------------------------------------------------------
# Test 5: Battery reduces peak net load
# ---------------------------------------------------------------------------

def test_battery_reduces_peak_load() -> None:
    """With battery, peak net load must be ≤ peak net load without battery."""
    n = 24
    load = np.full(n, 0.95)    # above threshold → triggers discharge
    pv = np.zeros(n)
    pv[8:14] = 1.1             # PV surplus for charging

    no_batt = BatteryParams(capacity_mwh=0.0, max_power_mw=0.0)
    with_batt = PARAMS_5MWH

    res_no = dispatch_battery(load, pv, no_batt)
    res_with = dispatch_battery(load, pv, with_batt)

    peak_no = res_no["net_load"].max()
    peak_with = res_with["net_load"].max()

    assert peak_with <= peak_no + 1e-9, (
        f"Battery should reduce peak; no_batt={peak_no:.4f}, with_batt={peak_with:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Lifetime fraction is non-zero after cycling
# ---------------------------------------------------------------------------

def test_lifetime_fraction_nonzero_after_cycling() -> None:
    """After charge/discharge cycling, lifetime_fraction_used must be > 0."""
    load, pv = _simple_profile(24 * 30)   # 30-day simulation with regular cycling
    result = dispatch_battery(load, pv, PARAMS_5MWH)
    assert result["lifetime_fraction_used"] > 0.0, (
        "Battery that cycled should have lifetime_fraction_used > 0"
    )
    # And less than 1 (shouldn't use full life in 30 days)
    assert result["lifetime_fraction_used"] < 1.0
