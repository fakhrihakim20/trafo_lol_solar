"""Tropical ambient temperature profile generator for Jakarta, Indonesia.

Composite model:
    T_a(h) = T_mean
           - A_seasonal * cos(2*pi*(h - h_peak_season) / 8760)
           + A_daily    * sin(2*pi*(h % 24 - h_peak_day) / 24)
           + N(0, sigma)

Parameters calibrated to Jakarta BMKG climate data:
  - Annual mean: 28.0 °C
  - Seasonal amplitude: 1.5 °C peak in April (h ≈ 2160)
  - Diurnal amplitude: 4.5 °C peak at 14:00 local (WIB)
  - Noise std: 0.8 °C

References
----------
BMKG (2023). Rata-rata suhu udara Jakarta. Badan Meteorologi, Klimatologi,
dan Geofisika, Indonesia.  https://bmkg.go.id (values used as published defaults;
actual file not fetched at runtime to satisfy no-network constraint).
"""

from __future__ import annotations

import numpy as np


def synthesize_tropical_ambient(
    n_hours: int = 8760,
    annual_mean_c: float = 26.5,
    daily_amplitude_c: float = 5.0,
    seasonal_amplitude_c: float = 1.2,
    noise_std_c: float = 0.8,
    delta_c: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic tropical ambient temperature profile.

    Defaults are calibrated to **Jember (East Java)** since the Phase 4C
    PyPSA migration; the prior Jakarta-tuned defaults (28.0, 4.5, 1.5) are
    available by passing them explicitly.  Real BMKG data can be loaded via
    ``data_jember.load_jember_ambient`` instead.

    Parameters
    ----------
    n_hours : int
        Number of hourly timesteps.  8760 = one non-leap year.
    annual_mean_c : float
        Annual mean temperature [°C].  Jember: 26.5;  Jakarta: 28.0.
    daily_amplitude_c : float
        Diurnal temperature swing (half-range) [°C].  Jember: 5.0;  Jakarta: 4.5.
    seasonal_amplitude_c : float
        Seasonal temperature swing (half-range) [°C].  Jember: 1.2;  Jakarta: 1.5.
    noise_std_c : float
        Std-dev of Gaussian noise [°C].  Default 0.8.
    delta_c : float
        Climate scenario offset added uniformly to all values [°C].
        Use 0.0 for present day, 1.0 for +1 °C future scenario.
    seed : int
        Random seed for Gaussian noise.  Pass different seeds for MC runs.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        Hourly ambient temperature [°C].

    Notes
    -----
    Seasonal term: minus cosine so that peak is near h ≈ 2160 (≈ April in WIB).
    Diurnal term: sine shifted so that peak is at 14:00 local time.
    Gaussian noise: std = 0.8 °C (short-term variability).
    """
    rng = np.random.default_rng(seed)
    h = np.arange(n_hours, dtype=float)

    # Seasonal component: peak near April (hour 2160 of year for April 1st noon)
    h_peak_season = 2160.0
    seasonal = -seasonal_amplitude_c * np.cos(2.0 * np.pi * (h - h_peak_season) / 8760.0)

    # Diurnal component: peak at 14:00 local (h % 24 relative to midnight)
    h_peak_day = 14.0
    diurnal = daily_amplitude_c * np.sin(2.0 * np.pi * (h % 24.0 - h_peak_day) / 24.0)

    # Gaussian noise
    noise = rng.normal(0.0, noise_std_c, size=n_hours)

    return (annual_mean_c + seasonal + diurnal + noise + delta_c).astype(float)
