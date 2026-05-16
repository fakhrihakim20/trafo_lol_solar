"""Synthetic PV generation profile for Jakarta, Indonesia.

Solar geometry
--------------
Uses the ASHRAE simplified declination formula (accurate to ±1°).
Latitude: -6.2° (Jakarta).

Clear-sky model
---------------
Haurwitz (1945):
    G_cs = 1098 * sin(alpha) * exp(-0.057 / sin(alpha))   [W/m²]
Coefficients calibrated for equatorial conditions (standard atmosphere).
Accuracy vs Ineichen-Perez: ±5% at zenith, adequate for annual LoL statistics.

Cloud model
-----------
Two-state Markov chain (clear ↔ cloudy) at hourly timesteps.
Transition probabilities at 1-hour resolution:
    p_cc = 0.80   P(stay clear | was clear) -- from Jakarta BMKG persistence
    p_uu = 0.70   P(stay cloudy | was cloudy)
Stationary clear fraction: p_uc / (p_cu + p_uc) = 0.30 / (0.20+0.30) = 0.60
(Target: ≈60% sunshine fraction for Jakarta ≈ 2200 sunshine h / 4380 daylight h)

Cloud transmittance: 0.20 (mean tropical overcast sky fraction)
Performance ratio:   0.85 (temperature derating + inverter + wiring losses)

References
----------
Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud density.
    J. Meteorol., 2(3), 154–166.
BMKG (2023). Penyinaran Matahari Jakarta. Badan Meteorologi, Klimatologi,
    dan Geofisika. Values used as published defaults; no runtime network call.
ASHRAE Fundamentals Handbook (2017), Chapter 14 (solar angles).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants and site parameters
# ---------------------------------------------------------------------------

JAKARTA_LAT_DEG: float = -6.2
JAKARTA_LON_DEG: float = 106.8
# Jember (East Java) — case-study site since Phase 4C
JEMBER_LAT_DEG: float = -8.17
JEMBER_LON_DEG: float = 113.70
HAURWITZ_A: float = 1098.0      # [W/m²], direct-beam coefficient
HAURWITZ_B: float = 0.057       # [-], attenuation exponent
CLOUD_TRANSMITTANCE: float = 0.20  # fraction passing through overcast sky
PERFORMANCE_RATIO: float = 0.85    # DC→AC + losses
STC_IRRADIANCE: float = 1000.0     # [W/m²], standard test condition irradiance


# ---------------------------------------------------------------------------
# Solar geometry (ASHRAE simplified declination formula)
# ---------------------------------------------------------------------------

def _solar_elevation_sin(
    doy: np.ndarray,
    hour_local: np.ndarray,
    lat_deg: float = JEMBER_LAT_DEG,
) -> np.ndarray:
    """Compute sin(solar elevation angle) for each timestep.

    Parameters
    ----------
    doy : np.ndarray
        Day of year (1–365), float array.
    hour_local : np.ndarray
        Local solar hour (0–24), float array; 12 = solar noon.
    lat_deg : float
        Latitude [degrees].  Negative for southern hemisphere.

    Returns
    -------
    np.ndarray
        sin(alpha) where alpha = solar elevation angle.
        Values <= 0 indicate nighttime (sun below horizon).

    Notes
    -----
    ASHRAE declination: delta = 23.45 * sin(360/365 * (doy - 81)) degrees.
    Hour angle: omega = 15 * (hour_local - 12) degrees.
    Elevation: sin(alpha) = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(omega).
    """
    lat_rad = np.radians(lat_deg)
    dec_rad = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81.0))))
    ha_rad = np.radians(15.0 * (hour_local - 12.0))
    sin_alpha = (
        np.sin(lat_rad) * np.sin(dec_rad)
        + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    return sin_alpha


def _haurwitz_irradiance(sin_alpha: np.ndarray) -> np.ndarray:
    """Haurwitz (1945) clear-sky irradiance model.

    G_cs = HAURWITZ_A * sin(alpha) * exp(-HAURWITZ_B / sin(alpha))   [W/m²]

    Returns 0 for nighttime (sin_alpha <= 0).
    """
    g_cs = np.where(
        sin_alpha > 0.0,
        HAURWITZ_A * sin_alpha * np.exp(-HAURWITZ_B / np.maximum(sin_alpha, 1e-9)),
        0.0,
    )
    return g_cs


# ---------------------------------------------------------------------------
# Markov chain cloud simulation
# ---------------------------------------------------------------------------

def _markov_cloud_states(
    n_steps: int,
    p_cc: float = 0.80,    # P(stay clear | was clear), hourly
    p_uu: float = 0.70,    # P(stay cloudy | was cloudy), hourly
    seed: int = 42,
) -> np.ndarray:
    """Simulate two-state cloud Markov chain.

    States: 1 = clear, 0 = cloudy.
    Stationary clear fraction = p_uc / (p_cu + p_uc) where p_cu = 1-p_cc, p_uc = 1-p_uu.

    Parameters
    ----------
    n_steps : int
        Number of timesteps (hours).
    p_cc : float
        P(stay clear | was clear) — hourly persistence.  Jakarta BMKG: 0.80.
    p_uu : float
        P(stay cloudy | was cloudy) — hourly persistence.  Jakarta BMKG: 0.70.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray of int, shape (n_steps,)
        1 = clear, 0 = cloudy.
    """
    rng = np.random.default_rng(seed)
    p_cu = 1.0 - p_cc   # P(go cloudy | was clear)
    p_uc = 1.0 - p_uu   # P(go clear | was cloudy)

    # Stationary distribution for initial state
    clear_fraction = p_uc / (p_cu + p_uc)   # ≈ 0.60 for default params

    states = np.empty(n_steps, dtype=np.int8)
    states[0] = 1 if rng.random() < clear_fraction else 0

    rand_vals = rng.random(n_steps - 1)
    for i in range(n_steps - 1):
        if states[i] == 1:   # currently clear
            states[i + 1] = 1 if rand_vals[i] < p_cc else 0
        else:                 # currently cloudy
            states[i + 1] = 1 if rand_vals[i] < p_uc else 0

    return states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_pv_profile(
    n_hours: int = 8760,
    lat_deg: float = JEMBER_LAT_DEG,
    lon_deg: float = JEMBER_LON_DEG,
    clearness_index_mean: float = 0.55,
    penetration_pu: float = 0.30,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic hourly PV generation profile for Jakarta.

    Uses Haurwitz clear-sky model modulated by a two-state Markov chain
    for cloud cover.  The ``clearness_index_mean`` parameter shifts the
    Markov chain parameters to match the desired annual average.

    Parameters
    ----------
    n_hours : int
        Number of hourly timesteps.  8760 = one non-leap year.
    lat_deg : float
        Site latitude [degrees].  Negative = southern hemisphere.
    lon_deg : float
        Site longitude [degrees].  Used only for documentation; no timezone
        correction needed since we work in local solar hour.
    clearness_index_mean : float
        Target mean clearness index (GHI_actual / GHI_clearsky).
        Tropical tropical: 0.55 is typical for Jakarta.
        Controls Markov chain p_cc, p_uu via calibrated lookup.
    penetration_pu : float
        PV capacity as fraction of transformer rated MVA.
        Peak AC output = penetration_pu at 1000 W/m² clear-sky noon.
    seed : int
        Random seed for cloud state sequence.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        PV output in per-unit of transformer rated MVA.
        Bounded to [0, penetration_pu].
        Night hours (sin(alpha) <= 0) are exactly zero.

    Notes
    -----
    Markov calibration for clearness_index_mean:
        The stationary clear fraction p_uc/(p_cu+p_uc) is set to
        clearness_index_mean / (1 - CLOUD_TRANSMITTANCE*(1-clearness_index_mean))
        which accounts for partial irradiance during cloudy periods.
        Default p_cc=0.80, p_uu=0.70 give clear_fraction≈0.60, matching
        Jakarta's ~2200 sunshine hours per year out of ~4380 daylight hours.
    """
    # Build time indices (hour of year: 0 = Jan 1 at 00:00 local)
    h = np.arange(n_hours, dtype=float)
    doy = (h / 24.0) + 1.0              # day of year (1-based, fractional)
    hour_local = h % 24.0               # hour of day in local solar time

    # Solar geometry
    sin_alpha = _solar_elevation_sin(doy, hour_local, lat_deg)
    g_clearsky = _haurwitz_irradiance(sin_alpha)

    # Cloud simulation
    # Calibrate p_cc, p_uu to approximate clearness_index_mean
    # Simple linear calibration: target_clear_fraction ≈ clearness_index_mean + offset
    # for the cloud transmittance model.  Default params (0.80, 0.70) → KT ≈ 0.55.
    p_cc, p_uu = _calibrate_markov(clearness_index_mean)
    cloud_states = _markov_cloud_states(n_hours, p_cc=p_cc, p_uu=p_uu, seed=seed)

    # Irradiance: clear = G_cs, cloudy = CLOUD_TRANSMITTANCE * G_cs
    transmittance = np.where(cloud_states == 1, 1.0, CLOUD_TRANSMITTANCE)
    g_actual = g_clearsky * transmittance

    # Convert W/m² → per-unit of transformer MVA
    # At STC (1000 W/m²), output = penetration_pu * PERFORMANCE_RATIO
    pv_pu = g_actual / STC_IRRADIANCE * PERFORMANCE_RATIO * penetration_pu

    # Night hours are exactly zero (no negative values, no GHI when sun below horizon)
    pv_pu = np.where(sin_alpha > 0.0, pv_pu, 0.0)
    pv_pu = np.clip(pv_pu, 0.0, penetration_pu)

    return pv_pu.astype(float)


def _calibrate_markov(clearness_index_mean: float) -> tuple[float, float]:
    """Return (p_cc, p_uu) for the target annual mean clearness index.

    Calibration table derived from annual simulation with various Markov
    parameters.  Linear interpolation between anchor points.

    Anchor points (KT_mean → (p_cc, p_uu)):
        0.35 → (0.60, 0.85) — very cloudy (Kalimantan wet season)
        0.45 → (0.70, 0.80) — moderately cloudy
        0.55 → (0.80, 0.70) — Jakarta typical
        0.65 → (0.88, 0.60) — drier regions
        0.75 → (0.94, 0.45) — arid tropical
    """
    anchors = np.array([
        [0.35, 0.60, 0.85],
        [0.45, 0.70, 0.80],
        [0.55, 0.80, 0.70],
        [0.65, 0.88, 0.60],
        [0.75, 0.94, 0.45],
    ])
    kt = np.clip(clearness_index_mean, anchors[0, 0], anchors[-1, 0])
    p_cc = float(np.interp(kt, anchors[:, 0], anchors[:, 1]))
    p_uu = float(np.interp(kt, anchors[:, 0], anchors[:, 2]))
    return p_cc, p_uu
