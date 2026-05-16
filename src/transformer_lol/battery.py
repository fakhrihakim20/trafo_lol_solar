"""Grid-connected battery storage dispatch — simple threshold rule.

No model-predictive control.  The battery follows two rules per timestep:

  1. **Charge** when PV generation exceeds transformer load (absorb reverse flow).
  2. **Discharge** when net load exceeds ``discharge_threshold_pu`` (peak shaving).

State of Charge (SoC) is clamped to [SoC_min, SoC_max] = [0.10, 0.90] to
limit degradation.  Round-trip efficiency η = 0.90; 10% of energy is lost
per charge-discharge cycle.  Charge and discharge are also limited by
``max_power_pu`` (rated converter power as a fraction of transformer MVA).

Per-cycle battery aging: each equivalent full cycle (capacity/max_discharge)
consumes 1/6000 of the battery's useful life (LiFePO4 reference, ≈6000 cycles
to 80% capacity).  ``lifetime_fraction_used`` is the fraction of total battery
life consumed over the simulation period.

References
----------
Muratori, M. & Rizzoni, G. (2016). *Residential Demand Response Protocol*.
    LBNL-2001284.  Lawrence Berkeley National Laboratory.
IEA Global EV Outlook 2024.  OECD/IEA, Paris.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Battery parameter dataclass
# ---------------------------------------------------------------------------

BATTERY_CYCLE_LIFE: float = 6_000.0   # LiFePO4 reference cycles to 80% capacity
BATTERY_EFFICIENCY: float = 0.90      # round-trip efficiency (charge × discharge)
BATTERY_SOC_MIN: float = 0.10         # lower SoC bound (protect against deep discharge)
BATTERY_SOC_MAX: float = 0.90         # upper SoC bound (protect against overcharge)


@dataclass
class BatteryParams:
    """Parameters for the simple-threshold battery storage dispatch model.

    Parameters
    ----------
    capacity_mwh : float
        Usable battery capacity [MWh].  0 = no battery (dispatch is a no-op).
    max_power_mw : float
        Maximum charge/discharge converter power [MW].
    discharge_threshold_pu : float
        Net-load level (p.u. of transformer rated MVA) above which the battery
        discharges to shave peak load.  Typically 0.85.
    transformer_rated_mva : float
        Transformer rated power [MVA] — converts MW to p.u.
    soc_init : float
        Initial State of Charge [fraction].  Default 0.5.
    """
    capacity_mwh: float = 0.0
    max_power_mw: float = 0.0
    discharge_threshold_pu: float = 0.85
    transformer_rated_mva: float = 60.0
    soc_init: float = 0.50

    def __post_init__(self) -> None:
        if self.capacity_mwh < 0:
            raise ValueError("capacity_mwh must be >= 0")
        if self.max_power_mw < 0:
            raise ValueError("max_power_mw must be >= 0")
        if not (0.0 <= self.soc_init <= 1.0):
            raise ValueError(f"soc_init must be in [0, 1], got {self.soc_init}")


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------

def dispatch_battery(
    load_pu: np.ndarray,
    pv_pu: np.ndarray,
    params: BatteryParams,
    dt_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Run the simple threshold-rule battery dispatch over a load/PV profile.

    At each timestep:
      • If PV > load (reverse flow): charge battery until SoC = SoC_max.
      • Else if net load > discharge_threshold: discharge battery until
        net load = threshold or SoC = SoC_min.

    Both charge and discharge are rate-limited by max_power_pu.

    Parameters
    ----------
    load_pu : np.ndarray, shape (T,)
        Gross load (including EV, growth scaling) [p.u. of transformer MVA].
    pv_pu : np.ndarray, shape (T,)
        PV generation [p.u.].
    params : BatteryParams
    dt_hours : float
        Timestep [hours].

    Returns
    -------
    dict with keys:
        ``net_load``          : np.ndarray (T,) — load p.u. after battery dispatch
        ``battery_p_pu``      : np.ndarray (T,) — battery power [p.u.]; + = discharge
        ``soc``               : np.ndarray (T,) — SoC at end of each timestep [0–1]
        ``lifetime_fraction_used`` : float — fraction of LiFePO4 life consumed

    Notes
    -----
    When ``params.capacity_mwh == 0`` (no battery), returns the original
    load unchanged and all-zero battery arrays.  This is the regression gate.
    """
    T = len(load_pu)

    if params.capacity_mwh <= 0.0 or params.max_power_mw <= 0.0:
        return {
            "net_load": np.clip(load_pu - pv_pu, 0.0, None),
            "battery_p_pu": np.zeros(T),
            "soc": np.full(T, params.soc_init),
            "lifetime_fraction_used": 0.0,
        }

    rated_mva = params.transformer_rated_mva
    cap_mwh = params.capacity_mwh
    max_p_pu = params.max_power_mw / rated_mva          # [p.u.]
    threshold_pu = params.discharge_threshold_pu
    eta = BATTERY_EFFICIENCY
    soc = params.soc_init

    battery_p = np.zeros(T)    # positive = discharging (delivering to bus)
    soc_arr = np.empty(T)
    cumulative_charge_mwh = 0.0  # track energy throughput for aging

    for t in range(T):
        net_before = load_pu[t] - pv_pu[t]  # positive = load > PV, negative = excess PV

        if net_before < 0.0:
            # Excess PV → charge battery
            pv_surplus_pu = -net_before   # positive quantity
            charge_demand_pu = min(pv_surplus_pu, max_p_pu)
            # Limit by SoC headroom (1-to-1: stored = input, losses accounted on discharge)
            max_charge_pu = (BATTERY_SOC_MAX - soc) * cap_mwh / (rated_mva * dt_hours)
            charge_pu = min(charge_demand_pu, max_charge_pu)
            charge_pu = max(charge_pu, 0.0)
            delta_soc = charge_pu * rated_mva * dt_hours / cap_mwh   # 1-to-1
            soc = min(soc + delta_soc, BATTERY_SOC_MAX)
            battery_p[t] = -charge_pu          # negative = absorbing power
            cumulative_charge_mwh += charge_pu * rated_mva * dt_hours

        elif net_before > threshold_pu:
            # Peak load → discharge battery
            deficit_pu = net_before - threshold_pu
            discharge_demand_pu = min(deficit_pu, max_p_pu)
            # Limit by SoC floor; energy delivered = SoC_withdrawn × cap × η
            max_delivered_pu = (soc - BATTERY_SOC_MIN) * cap_mwh * eta / (rated_mva * dt_hours)
            discharge_pu = min(discharge_demand_pu, max_delivered_pu)
            discharge_pu = max(discharge_pu, 0.0)
            delta_soc = discharge_pu * rated_mva * dt_hours / (cap_mwh * eta)
            soc = max(soc - delta_soc, BATTERY_SOC_MIN)
            battery_p[t] = discharge_pu        # positive = delivering power
            cumulative_charge_mwh += discharge_pu * rated_mva * dt_hours

        soc_arr[t] = soc

    # Net load after battery dispatch (clamp to zero — battery can't inject below zero)
    net_load_final = np.clip(load_pu - pv_pu - battery_p, 0.0, None)

    # Battery aging: cycles = total energy throughput / (2 × usable capacity)
    # (factor of 2 because a full cycle = charge + discharge = 2× capacity throughput)
    usable_mwh = cap_mwh * (BATTERY_SOC_MAX - BATTERY_SOC_MIN)
    equivalent_cycles = cumulative_charge_mwh / (2.0 * usable_mwh) if usable_mwh > 0 else 0.0
    lifetime_fraction = equivalent_cycles / BATTERY_CYCLE_LIFE

    return {
        "net_load": net_load_final,
        "battery_p_pu": battery_p,
        "soc": soc_arr,
        "lifetime_fraction_used": float(lifetime_fraction),
    }
