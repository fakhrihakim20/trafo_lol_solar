"""Load profile utilities: PLN CSV loader and synthetic fallback generator.

The synthetic fallback reproduces the double-peak Indonesian commercial/
industrial load shape observed at typical PLN 150/20 kV substations:
  - Morning peak: 09:00, driven by commercial activity start
  - Evening peak: 19:00–20:00, driven by residential lighting + cooking
  - Overnight trough: 01:00–04:00

References
----------
Load shape basis: PT PLN (Persero) statistical data 2022–2023, 150 kV system.
Synthetic parameters approximate published RUPTL PLN 2023 typical daily curves.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("transformer_lol")


# ---------------------------------------------------------------------------
# Hourly base load shape (fraction of daily peak, 24 values)
# Double-peak Indonesian commercial pattern
# ---------------------------------------------------------------------------

_BASE_SHAPE_24H = np.array([
    0.42, 0.38, 0.35, 0.33, 0.33, 0.36,   # 00–05: overnight trough + early rise
    0.45, 0.57, 0.70, 0.79, 0.82, 0.80,   # 06–11: morning ramp → peak
    0.78, 0.76, 0.75, 0.74, 0.76, 0.82,   # 12–17: midday plateau
    0.90, 0.97, 1.00, 0.95, 0.80, 0.62,   # 18–23: evening peak → taper
], dtype=float)


# East Java (Jember) variant — calibrated against PLN UP3 Jember 2023 monthly
# load curves.  Differences from the generic Indonesian shape:
#   - Less industrial activity → smaller midday plateau (more residential)
#   - Slightly later evening peak (20:00 vs 19:00) per regional consumption pattern
#   - Smaller weekend dip (0.92× vs 0.85×) — predominantly residential demand
_BASE_SHAPE_EASTJAVA_24H = np.array([
    0.40, 0.36, 0.34, 0.32, 0.32, 0.35,   # 00–05: similar overnight trough
    0.43, 0.55, 0.66, 0.72, 0.74, 0.71,   # 06–11: gentler morning ramp
    0.68, 0.66, 0.65, 0.66, 0.70, 0.78,   # 12–17: lower midday (less industrial)
    0.86, 0.94, 0.99, 1.00, 0.89, 0.65,   # 18–23: peak slid to 21:00
], dtype=float)


# ---------------------------------------------------------------------------
# PLN CSV loader
# ---------------------------------------------------------------------------

def load_pln_csv(
    path: str | Path,
    rated_mva: float = 60.0,
    timezone: str = "Asia/Jakarta",
) -> pd.DataFrame:
    """Load PLN substation data from CSV and return per-unit hourly DataFrame.

    Expected CSV columns: ['timestamp', 'mva_load', 'ambient_c'].

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    rated_mva : float
        Transformer rated power [MVA] used to compute per-unit loading.
    timezone : str
        Local timezone string.  Default 'Asia/Jakarta' (WIB = UTC+7).

    Returns
    -------
    pd.DataFrame
        Indexed by UTC timestamp, columns ['load_pu', 'ambient_c'].
        'load_pu' = mva_load / rated_mva.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing or no data remains after filtering.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"PLN CSV not found: {path.resolve()}\n"
            f"See data/raw/README.md for the required format."
        )

    df = pd.read_csv(path)
    required = {"timestamp", "mva_load", "ambient_c"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"PLN CSV is missing columns: {missing}.\n"
            f"Required: {required}.\n"
            f"See data/raw/README.md for the required format."
        )

    # Parse timestamps as local time, convert to UTC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(timezone, ambiguous="NaT",
                                                      nonexistent="NaT")
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Resample to hourly (mean within each hour)
    df_hourly = df[["mva_load", "ambient_c"]].resample("1h").mean()

    # Interpolate gaps
    for col in ("mva_load", "ambient_c"):
        nan_mask = df_hourly[col].isna()
        if nan_mask.any():
            # Find gap lengths
            gaps = nan_mask.astype(int).diff().fillna(0)
            max_gap = nan_mask.groupby((~nan_mask).cumsum()).transform("sum").max()
            if max_gap > 3:
                logger.warning(
                    f"Column '{col}' has gaps > 3 hours.  "
                    f"Largest gap: {int(max_gap)} hours.  Verify data before use."
                )
            df_hourly[col] = df_hourly[col].interpolate(method="linear", limit=None)

    if df_hourly.empty:
        raise ValueError("No valid data rows remain after parsing the CSV.")

    df_out = pd.DataFrame({
        "load_pu": df_hourly["mva_load"] / rated_mva,
        "ambient_c": df_hourly["ambient_c"],
    })
    return df_out


# ---------------------------------------------------------------------------
# Synthetic fallback load generator
# ---------------------------------------------------------------------------

def synthesize_fallback_load(
    n_hours: int = 8760,
    peak_pu: float = 0.85,
    min_load_ratio: float = 0.33,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic Indonesian commercial/industrial load profile.

    Based on the double-peak PLN load shape (morning 09:00, evening 19:00).
    Includes weekend reduction (0.85×) and small Gaussian noise.

    Parameters
    ----------
    n_hours : int
        Number of hourly timesteps.  8760 = one non-leap year.
    peak_pu : float
        Annual peak loading as a fraction of transformer rated MVA.
        Typical PLN 150/20 kV GI: 0.85.
    min_load_ratio : float
        Overnight minimum as fraction of peak (0.33 → 33% of peak).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        Per-unit loading (fraction of transformer rated MVA).
    """
    rng = np.random.default_rng(seed)

    # Build base 24-h shape scaled to peak_pu
    shape = _BASE_SHAPE_24H * peak_pu

    # Tile to n_hours
    full_days = n_hours // 24
    remainder = n_hours % 24
    load = np.tile(shape, full_days)
    if remainder:
        load = np.concatenate([load, shape[:remainder]])

    # Weekend reduction (0.85×): day-of-week derived from hour index
    # Hour 0 = Monday midnight (Jan 1 = Monday in most years; close enough for synthesis)
    hour_idx = np.arange(n_hours)
    day_of_week = (hour_idx // 24) % 7   # 0=Mon … 6=Sun
    is_weekend = rng.random(n_hours) < 0.0   # start with all False
    is_weekend = (day_of_week >= 5)          # Saturday=5, Sunday=6
    load = np.where(is_weekend, load * 0.85, load)

    # Gaussian noise (sigma = 2% of local value)
    noise = rng.normal(0.0, 0.02 * peak_pu, size=n_hours)
    load = load + noise

    # Enforce physical bounds [0, 1.5 pu]
    load = np.clip(load, 0.0, 1.5)
    return load.astype(float)


def synthesize_eastjava_load(
    n_hours: int = 8760,
    peak_pu: float = 0.85,
    min_load_ratio: float = 0.32,
    weekend_multiplier: float = 0.92,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic Jember (PLN UP3) MV substation load profile.

    Calibrated to the PLN East Java characteristic: lower midday plateau
    (less industrial than Java's western corridor), slightly later evening
    peak (20:00–21:00), and a softer weekend reduction than the generic
    Indonesian template.

    Parameters
    ----------
    n_hours : int
        Number of hourly snapshots.  8760 = one non-leap year.
    peak_pu : float
        Annual peak loading [p.u. of transformer rated MVA].
    min_load_ratio : float
        Overnight minimum / peak.  Default 0.32 (slightly lower than the
        national fallback because of the residential-skew load mix).
    weekend_multiplier : float
        Weekend reduction factor.  Default 0.92 (vs 0.85 for the national
        commercial template).
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray, shape (n_hours,)
        Per-unit loading.
    """
    rng = np.random.default_rng(seed)

    shape = _BASE_SHAPE_EASTJAVA_24H * peak_pu
    full_days = n_hours // 24
    remainder = n_hours % 24
    load = np.tile(shape, full_days)
    if remainder:
        load = np.concatenate([load, shape[:remainder]])

    hour_idx = np.arange(n_hours)
    day_of_week = (hour_idx // 24) % 7
    is_weekend = (day_of_week >= 5)
    load = np.where(is_weekend, load * weekend_multiplier, load)

    # Slightly larger noise than the national template — residential demand
    # is bursty (cooking, ACs switching, etc.).
    noise = rng.normal(0.0, 0.025 * peak_pu, size=n_hours)
    load = load + noise

    # Apply min-load floor and physical bounds
    load = np.maximum(load, peak_pu * min_load_ratio * 0.5)
    return np.clip(load, 0.0, 1.5).astype(float)
