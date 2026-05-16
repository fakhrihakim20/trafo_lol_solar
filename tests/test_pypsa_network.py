"""Unit tests for the PyPSA Network builder (Phase 4A).

Tests verify:
1. Network is constructed with correct components for a typical scenario.
2. Snapshots are set to the requested length and frequency.
3. PV time-series sums match the input profile (no off-by-one in p_max_pu scaling).
4. Battery component is omitted when capacity_mwh = 0.
5. Profile length mismatch raises ValueError.
6. Custom snapshots are honoured when provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transformer_lol.pypsa_network import (
    DEFAULT_TRANSFORMER_X,
    PYPSA_BATTERY_ETA_DISPATCH,
    PYPSA_BATTERY_ETA_STORE,
    build_pypsa_network,
    network_summary,
)
from transformer_lol.scenarios import Scenario
from transformer_lol.thermal_model import TransformerParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profiles(n_hours: int = 168, pv_pen: float = 0.50) -> dict[str, np.ndarray]:
    """Synthetic 1-week profiles for testing."""
    rng = np.random.default_rng(42)
    hours = np.arange(n_hours)
    # Daily double-peak load
    load = 0.7 + 0.15 * np.sin(2 * np.pi * (hours % 24) / 24.0 + 1.0) + 0.02 * rng.standard_normal(n_hours)
    load = np.clip(load, 0.0, 1.5)
    # PV: triangular daily curve, peak at noon
    pv = np.clip(np.sin(np.pi * (hours % 24) / 24.0), 0.0, 1.0) * pv_pen
    ev = np.zeros(n_hours)
    return {"load_pu": load, "pv_pu": pv, "ev_pu": ev}


def _make_scenario(pv_pen: float = 0.50, batt_mwh: float = 5.0, batt_mw: float = 2.0) -> Scenario:
    return Scenario(
        name="test_scenario",
        pv_penetration=pv_pen,
        battery_capacity_mwh=batt_mwh,
        battery_power_mw=batt_mw,
        n_runs=1,
    )


def _make_transformer() -> TransformerParams:
    return TransformerParams()  # default 60 MVA ONAF


# ---------------------------------------------------------------------------
# Test 1: Standard network construction
# ---------------------------------------------------------------------------

def test_network_components_present() -> None:
    """A standard scenario must produce 2 buses, 1 transformer, 2 generators,
    1 load, and 1 storage unit."""
    n = build_pypsa_network(_make_scenario(), _make_transformer(), _make_profiles())
    summary = network_summary(n)

    assert summary["buses"] == ["grid_150kv", "mv_20kv"]
    assert summary["transformers"] == ["tfr_60mva"]
    assert set(summary["generators"]) == {"grid_slack", "pv_jember"}
    assert summary["loads"] == ["demand_jember"]
    assert summary["storage_units"] == ["battery"]
    assert summary["transformer_s_nom_mva"] == pytest.approx(60.0)
    assert summary["pv_p_nom_mw"] == pytest.approx(30.0)  # 50 % of 60 MVA
    assert summary["battery_p_nom_mw"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Test 2: Snapshot count matches profile length
# ---------------------------------------------------------------------------

def test_snapshots_match_profile_length() -> None:
    """Network snapshots must equal the profile length."""
    profiles = _make_profiles(n_hours=240)
    n = build_pypsa_network(_make_scenario(), _make_transformer(), profiles)
    assert len(n.snapshots) == 240


# ---------------------------------------------------------------------------
# Test 3: PV time-series scaling round-trip
# ---------------------------------------------------------------------------

def test_pv_p_max_pu_round_trip() -> None:
    """PV available energy in MWh must equal input pv_pu × rated_mva summed.

    Available PV energy per snapshot = p_nom × p_max_pu × Δt.
    For 1-h snapshots, the sum equals total available MWh which must match
    the input profile pv_pu * rated_mva (in MWh).
    """
    transformer = _make_transformer()
    pv_pen = 0.50
    profiles = _make_profiles(pv_pen=pv_pen)
    scenario = _make_scenario(pv_pen=pv_pen)

    n = build_pypsa_network(scenario, transformer, profiles)
    pv_avail_mwh = (n.generators.loc["pv_jember", "p_nom"]
                    * n.generators_t.p_max_pu["pv_jember"].sum())
    expected_mwh = float(profiles["pv_pu"].sum() * transformer.rated_mva)
    assert pv_avail_mwh == pytest.approx(expected_mwh, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 4: Zero battery capacity → no storage unit
# ---------------------------------------------------------------------------

def test_no_battery_when_capacity_zero() -> None:
    """battery_capacity_mwh = 0 must omit the StorageUnit."""
    scenario = _make_scenario(batt_mwh=0.0, batt_mw=0.0)
    n = build_pypsa_network(scenario, _make_transformer(), _make_profiles())
    assert len(n.storage_units) == 0


# ---------------------------------------------------------------------------
# Test 5: Battery efficiencies match plan
# ---------------------------------------------------------------------------

def test_battery_efficiency_split() -> None:
    """PyPSA splits round-trip efficiency; both sides must use the constants."""
    n = build_pypsa_network(_make_scenario(), _make_transformer(), _make_profiles())
    eta_store = float(n.storage_units.loc["battery", "efficiency_store"])
    eta_dispatch = float(n.storage_units.loc["battery", "efficiency_dispatch"])
    assert eta_store == pytest.approx(PYPSA_BATTERY_ETA_STORE)
    assert eta_dispatch == pytest.approx(PYPSA_BATTERY_ETA_DISPATCH)
    # Round-trip ≈ 0.9025 — within 0.5 % of legacy battery.BATTERY_EFFICIENCY = 0.90.
    assert eta_store * eta_dispatch == pytest.approx(0.9025, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 6: Mismatched profile lengths raise
# ---------------------------------------------------------------------------

def test_mismatched_profile_lengths_raise() -> None:
    """Passing profiles with unequal lengths must raise ValueError."""
    profiles = _make_profiles(n_hours=168)
    profiles["pv_pu"] = profiles["pv_pu"][:100]  # truncate
    with pytest.raises(ValueError, match="equal length"):
        build_pypsa_network(_make_scenario(), _make_transformer(), profiles)


# ---------------------------------------------------------------------------
# Test 7: Custom snapshots are honoured
# ---------------------------------------------------------------------------

def test_custom_snapshots_tz_naive() -> None:
    """User-provided tz-naive snapshots must be used verbatim."""
    snaps = pd.date_range("2025-06-01", periods=168, freq="h")
    n = build_pypsa_network(
        _make_scenario(), _make_transformer(), _make_profiles(),
        snapshots=snaps,
    )
    assert n.snapshots.equals(snaps)


def test_tz_aware_snapshots_get_localized() -> None:
    """tz-aware snapshots must be auto-stripped (PyPSA ≥ 1.0 doesn't accept tz)."""
    snaps_tz = pd.date_range("2025-06-01", periods=168, freq="h", tz="Asia/Jakarta")
    n = build_pypsa_network(
        _make_scenario(), _make_transformer(), _make_profiles(),
        snapshots=snaps_tz,
    )
    assert getattr(n.snapshots, "tz", None) is None
    assert len(n.snapshots) == 168


# ---------------------------------------------------------------------------
# Test 8: Transformer s_nom matches transformer.rated_mva
# ---------------------------------------------------------------------------

def test_transformer_s_nom_propagates() -> None:
    """transformer.rated_mva must flow into the PyPSA Transformer.s_nom."""
    from dataclasses import replace
    t = _make_transformer()
    t30 = replace(t, rated_mva=30.0)
    n = build_pypsa_network(_make_scenario(), t30, _make_profiles())
    assert float(n.transformers.loc["tfr_60mva", "s_nom"]) == pytest.approx(30.0)
    assert float(n.transformers.loc["tfr_60mva", "x"]) == pytest.approx(DEFAULT_TRANSFORMER_X)
