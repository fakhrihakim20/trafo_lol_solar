"""EV charging load synthesis — aggregate slow and fast charging profiles.

Two charging patterns are modelled (shapes adapted from NREL / LBNL references):

1. **Slow overnight (residential)** — NREL TP-5400-72956 (2019).
   Each residential EV charges 6–8 h starting ~22:00 at 3.3–7.4 kW.
   Aggregate across hundreds of EVs → smooth Gaussian centred at 02:00.

2. **Fast opportunistic (commercial)** — LBNL-2001284 (2020).
   DC fast chargers (~50 kW) used randomly between 08:00–20:00.
   Each session lasts 20–40 min; aggregate → roughly uniform spread across
   business hours with a mild midday peak.

Both functions return a per-unit array normalised to the transformer's rated
MVA via `penetration_pu`.  Add both to the gross load before computing net load:

    ev_total = synthesize_ev_slow(n_hours, ppu, seed) + synthesize_ev_fast(...)
    gross_load = base_load + ev_total

Profile shapes are calibrated from published US/EU patterns and scaled by
analogy to Indonesian conditions (assumed shape-invariant; only magnitude
differs via penetration_pu).  Document this assumption in the paper.

References
----------
Muratori, M. et al. (2019). *Electric Vehicle Charging Infrastructure Trends*.
    NREL TP-5400-72956.  National Renewable Energy Laboratory, Golden, CO.
Muratori, M. & Rizzoni, G. (2016). *Residential Demand Response Protocol*.
    LBNL-2001284.  Lawrence Berkeley National Laboratory.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Slow overnight charging (residential)
# ---------------------------------------------------------------------------

_SLOW_PEAK_HOUR: float = 2.0   # hour-of-day peak: 02:00 (off-peak tariff slot)
_SLOW_SIGMA: float = 2.5       # Gaussian width [hours] — NREL aggregate shape
_SLOW_CHARGE_KW: float = 5.0   # average slow charger power [kW]


def synthesize_ev_slow(
    n_hours: int,
    penetration_pu: float,
    seed: int | None = None,
    rated_mva: float = 60.0,
) -> np.ndarray:
    """Aggregate slow-overnight EV charging load [p.u. of transformer MVA].

    Shape: Gaussian centred at 02:00 with σ=2.5 h, normalised so the peak of
    the annual average daily profile equals ``penetration_pu``.

    Parameters
    ----------
    n_hours : int
        Simulation length [hours].  Typically 8760.
    penetration_pu : float
        Peak aggregate charging load as a fraction of transformer rated MVA.
        E.g. 0.10 = 10% PV-equivalent penetration (6 MVA at 60 MVA base).
    seed : int or None
        RNG seed for run-to-run stochastic variation (±5% daily scale factor).
        None → uses numpy default (not reproducible).
    rated_mva : float
        Transformer rated power [MVA] — used only for documentation; shape
        is normalised to ``penetration_pu`` directly.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        Per-unit charging load.  Non-negative; peak ≈ penetration_pu on
        high-occupancy nights.
    """
    if penetration_pu <= 0.0:
        return np.zeros(n_hours)

    rng = np.random.default_rng(seed)

    hours_of_day = np.arange(n_hours, dtype=float) % 24.0
    # Gaussian centred at peak hour, wrapped on 24-h circle
    delta = (hours_of_day - _SLOW_PEAK_HOUR) % 24.0
    delta = np.where(delta > 12.0, delta - 24.0, delta)  # fold to [-12, 12]
    profile = np.exp(-0.5 * (delta / _SLOW_SIGMA) ** 2)

    # Normalise so the deterministic peak == 1
    profile /= profile.max()

    # Daily random scale factor: ±5% day-to-day variation (occupancy jitter)
    n_days = (n_hours + 23) // 24
    daily_scale = 1.0 + 0.05 * rng.standard_normal(n_days)
    daily_scale = np.clip(daily_scale, 0.5, 1.5)
    # Tile to hourly resolution
    scale_hourly = np.repeat(daily_scale, 24)[:n_hours]

    return np.clip(profile * scale_hourly * penetration_pu, 0.0, None)


# ---------------------------------------------------------------------------
# Fast opportunistic charging (commercial)
# ---------------------------------------------------------------------------

_FAST_ACTIVE_START: int = 8   # hour-of-day: DC fast chargers active from 08:00
_FAST_ACTIVE_END: int = 20    # hour-of-day: DC fast chargers active until 20:00
_FAST_ACTIVE_HOURS: int = _FAST_ACTIVE_END - _FAST_ACTIVE_START   # 12 h window
_FAST_MIDDAY_BOOST: float = 0.15  # midday soft peak (lunch break): +15% at noon


def synthesize_ev_fast(
    n_hours: int,
    penetration_pu: float,
    seed: int | None = None,
) -> np.ndarray:
    """Aggregate fast-opportunistic EV charging load [p.u. of transformer MVA].

    Shape: roughly uniform across 08:00–20:00 with a mild Gaussian boost at
    12:00 (lunch break), and zero outside that window.  Hour-by-hour Poisson
    jitter models session arrivals per LBNL-2001284.

    Parameters
    ----------
    n_hours : int
        Simulation length [hours].
    penetration_pu : float
        Peak aggregate charging load [p.u.].
    seed : int or None
        RNG seed.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        Per-unit charging load.  Zero at night; peak ≈ penetration_pu midday.
    """
    if penetration_pu <= 0.0:
        return np.zeros(n_hours)

    rng = np.random.default_rng(seed)

    hours_of_day = np.arange(n_hours, dtype=float) % 24.0
    # Base: uniform within active window
    in_window = (hours_of_day >= _FAST_ACTIVE_START) & (hours_of_day < _FAST_ACTIVE_END)
    profile = np.where(in_window, 1.0 / _FAST_ACTIVE_HOURS, 0.0)

    # Midday Gaussian boost centred at 12:00
    delta_noon = (hours_of_day - 12.0)
    midday_boost = _FAST_MIDDAY_BOOST * np.exp(-0.5 * (delta_noon / 2.0) ** 2)
    midday_boost = np.where(in_window, midday_boost, 0.0)
    profile = profile + midday_boost

    # Normalise to peak = 1
    peak = profile.max()
    if peak > 0:
        profile /= peak

    # Hourly Poisson jitter: each hour gets a multiplicative noise ~ Normal(1, 0.1)
    jitter = 1.0 + 0.10 * rng.standard_normal(n_hours)
    jitter = np.clip(jitter, 0.0, 2.0)
    profile = profile * jitter

    return np.clip(profile * penetration_pu, 0.0, None)


# ---------------------------------------------------------------------------
# Combined convenience wrapper
# ---------------------------------------------------------------------------

def synthesize_ev_total(
    n_hours: int,
    slow_penetration_pu: float,
    fast_penetration_pu: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sum of slow and fast EV charging profiles.

    Parameters
    ----------
    n_hours : int
    slow_penetration_pu : float
        Slow (overnight residential) peak load [p.u.].
    fast_penetration_pu : float
        Fast (commercial) peak load [p.u.].
    seed : int or None
        RNG seed.  Slow and fast use seed and seed+1 respectively.

    Returns
    -------
    np.ndarray, shape (n_hours,)
    """
    seed_s = seed
    seed_f = None if seed is None else seed + 1
    slow = synthesize_ev_slow(n_hours, slow_penetration_pu, seed=seed_s)
    fast = synthesize_ev_fast(n_hours, fast_penetration_pu, seed=seed_f)
    return slow + fast
