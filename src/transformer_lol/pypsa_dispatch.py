"""LP-optimised battery dispatch via PyPSA (Phase 4B).

Drop-in alternative to ``battery.dispatch_battery`` that solves a linear
program (HiGHS) over the full horizon to minimise the cost of imports from
the slack grid.  The threshold heuristic in ``battery.py`` makes a greedy
per-timestep decision; the LP here sees the entire 8760-h profile at once
and can, e.g., charge the battery in advance of an evening peak.

Both functions return the same dict shape so they are interchangeable from
the Monte Carlo driver:

    {
        "net_load":               np.ndarray (T,) p.u. — clamped to ≥ 0
        "battery_p_pu":           np.ndarray (T,) p.u. — + = discharge
        "soc":                    np.ndarray (T,) fraction of capacity
        "lifetime_fraction_used": float in [0, 1]
    }

Notes
-----
* The LP uses PyPSA's StorageUnit with ``efficiency_store = efficiency_dispatch
  = 0.95``, giving round-trip 0.9025 (≈ the heuristic's 0.90).
* SoC is *not* explicitly bounded to [0.10, 0.90] because PyPSA's StorageUnit
  treats the range [0, max_hours · p_nom] as usable capacity.  To preserve
  the heuristic's degradation-aware bounds, the effective capacity passed to
  PyPSA is ``capacity_mwh × (SoC_max - SoC_min)`` and the initial SoC is
  re-mapped accordingly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from .battery import (
    BATTERY_CYCLE_LIFE,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    BatteryParams,
)
from .pypsa_network import build_pypsa_network


_log = logging.getLogger(__name__)


def dispatch_battery_pypsa_lp(
    load_pu: np.ndarray,
    pv_pu: np.ndarray,
    params: BatteryParams,
    dt_hours: float = 1.0,
    solver_name: str = "highs",
    ev_pu: np.ndarray | None = None,
    scenario: Any | None = None,
    transformer: Any | None = None,
) -> dict[str, np.ndarray]:
    """LP battery dispatch — drop-in for :func:`battery.dispatch_battery`.

    Parameters
    ----------
    load_pu : np.ndarray, shape (T,)
        Gross load (excluding EV) [p.u. of transformer MVA].
    pv_pu : np.ndarray, shape (T,)
        PV generation [p.u.].
    params : BatteryParams
        Battery sizing.  ``capacity_mwh = 0`` returns the no-op result
        (matches the heuristic's behaviour for the regression gate).
    dt_hours : float
        Snapshot length [hours].  Default 1.0.
    solver_name : str
        LP solver name passed to ``Network.optimize()``.  Default "highs".
    ev_pu : np.ndarray, optional
        EV charging load [p.u.].  Added to load before dispatch.
    scenario, transformer : optional
        Forwarded to ``build_pypsa_network`` if supplied.  When omitted, a
        minimal scenario / transformer with the required fields is synthesised
        from ``params`` so the function is callable in isolation (tests).

    Returns
    -------
    dict
        Same keys as :func:`battery.dispatch_battery`.

    Notes
    -----
    The LP minimises the cost of slack-grid imports plus a tiny battery cycling
    cost.  Because the slack generator is the only "expensive" source, the LP
    naturally prefers PV → battery discharge → slack, in that order.
    """
    T = len(load_pu)
    pv_pu = np.asarray(pv_pu, dtype=float)
    load_pu = np.asarray(load_pu, dtype=float)
    if ev_pu is None:
        ev_pu = np.zeros(T, dtype=float)
    else:
        ev_pu = np.asarray(ev_pu, dtype=float)

    # No-op path matches the heuristic's behaviour exactly.
    if params.capacity_mwh <= 0.0 or params.max_power_mw <= 0.0:
        return {
            "net_load": np.clip(load_pu + ev_pu - pv_pu, 0.0, None),
            "battery_p_pu": np.zeros(T),
            "soc": np.full(T, params.soc_init),
            "lifetime_fraction_used": 0.0,
        }

    # Synthesise a minimal scenario / transformer if the caller didn't pass one.
    rated_mva = float(getattr(transformer, "rated_mva", params.transformer_rated_mva))
    # Map heuristic soc_init (fraction of total capacity, in [SoC_min, SoC_max])
    # onto fraction of usable capacity for PyPSA.
    usable_window = BATTERY_SOC_MAX - BATTERY_SOC_MIN
    soc_init_in_window = (params.soc_init - BATTERY_SOC_MIN) / usable_window
    soc_init_in_window = max(0.0, min(1.0, soc_init_in_window))
    if scenario is None:
        scenario = _MinimalScenario(
            pv_penetration=float(np.max(pv_pu)) if len(pv_pu) else 0.0,
            battery_capacity_mwh=params.capacity_mwh * usable_window,
            battery_power_mw=params.max_power_mw,
            battery_soc_init_pu=soc_init_in_window,
        )
    if transformer is None:
        transformer = _MinimalTransformer(rated_mva=rated_mva)

    profiles = {"load_pu": load_pu, "pv_pu": pv_pu, "ev_pu": ev_pu}
    n = build_pypsa_network(scenario, transformer, profiles)

    # Solve the LP.  PyPSA prints lots of INFO via linopy / HiGHS — quiet it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prev_levels = {
            name: logging.getLogger(name).level
            for name in ("linopy", "linopy.io", "linopy.model", "linopy.constants",
                         "pypsa.consistency", "pypsa.optimization.optimize")
        }
        for name in prev_levels:
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            status, condition = n.optimize(solver_name=solver_name, log_to_console=False)
        finally:
            for name, lvl in prev_levels.items():
                logging.getLogger(name).setLevel(lvl)

    if status != "ok" or condition != "optimal":
        raise RuntimeError(
            f"PyPSA LP did not converge: status={status}, condition={condition}"
        )

    # Extract dispatched series.  PyPSA sign convention: storage_units_t.p > 0 = discharge.
    battery_p_mw = n.storage_units_t.p["battery"].to_numpy()
    battery_p_pu = battery_p_mw / rated_mva

    # Net load on the MV bus (load + EV) - PV - battery_discharge, clamped to ≥ 0.
    net_load = np.clip(load_pu + ev_pu - pv_pu - battery_p_pu, 0.0, None)

    # Map PyPSA's SoC (MWh) onto the [SoC_min, SoC_max] degradation-aware fraction.
    # PyPSA capacity = nominal × (SoC_max - SoC_min); raw nominal capacity is
    # params.capacity_mwh.  Re-scale to the [SoC_min, SoC_max] window.
    raw_soc_mwh = n.storage_units_t.state_of_charge["battery"].to_numpy()
    usable_mwh = params.capacity_mwh * (BATTERY_SOC_MAX - BATTERY_SOC_MIN)
    if usable_mwh > 0:
        soc_fraction_in_window = raw_soc_mwh / usable_mwh   # in [0, 1] of usable
        soc = BATTERY_SOC_MIN + soc_fraction_in_window * (BATTERY_SOC_MAX - BATTERY_SOC_MIN)
    else:
        soc = np.full(T, params.soc_init)

    # Battery aging: mirror the bookkeeping in battery.dispatch_battery so that
    # both methods produce comparable lifetime metrics.
    cumulative_throughput_mwh = float(np.sum(np.abs(battery_p_pu)) * rated_mva * dt_hours)
    equivalent_cycles = cumulative_throughput_mwh / (2.0 * usable_mwh) if usable_mwh > 0 else 0.0
    lifetime_fraction = equivalent_cycles / BATTERY_CYCLE_LIFE

    return {
        "net_load": net_load,
        "battery_p_pu": battery_p_pu,
        "soc": soc,
        "lifetime_fraction_used": float(lifetime_fraction),
    }


# ---------------------------------------------------------------------------
# Minimal stand-in dataclasses for standalone use (tests, notebooks)
# ---------------------------------------------------------------------------

class _MinimalScenario:
    """Just enough fields for ``build_pypsa_network`` to work without the real Scenario."""
    def __init__(self, pv_penetration: float, battery_capacity_mwh: float,
                 battery_power_mw: float, battery_soc_init_pu: float = 0.5) -> None:
        self.pv_penetration = float(pv_penetration)
        self.battery_capacity_mwh = float(battery_capacity_mwh)
        self.battery_power_mw = float(battery_power_mw)
        self.battery_soc_init_pu = float(battery_soc_init_pu)


class _MinimalTransformer:
    def __init__(self, rated_mva: float) -> None:
        self.rated_mva = float(rated_mva)
