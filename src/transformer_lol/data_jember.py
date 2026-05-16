"""Jember data layer (Phase 4C of the PyPSA migration roadmap).

Centralises loaders for the Jember case study:

* ``load_jember_location()`` — read coordinates, climate, and solar reference
  from ``config/jember_location.yaml``.
* ``load_jember_ambient(n_hours, seed)`` — try to read a BMKG CSV; fall back
  to ``ambient_profile.synthesize_tropical_ambient`` with Jember climate
  defaults when the CSV is absent.
* ``load_jember_pv(n_hours, penetration_pu, seed)`` — try to read an atlite
  ERA5 cutout for Jember; fall back to ``pv_synthesis.synthesize_pv_profile``
  (Haurwitz + Markov clouds) at Jember latitude when the cutout is absent.
* ``load_jember_load(n_hours, peak_pu, seed)`` — try to read a PLN East Java
  CSV; fall back to ``load_profile.synthesize_eastjava_load``.

All loaders return ``np.ndarray`` of shape ``(n_hours,)`` so that downstream
code (``monte_carlo.py``) is data-source-agnostic.

Real-data sources expected at the following paths (configurable via
``config/paths.yaml``):

* ``data/raw/atlite_cutout_jember.nc``  — atlite ERA5 cutout (download via
  ``scripts/00_fetch_era5_cutout.py``; requires Copernicus CDS account).
* ``data/raw/bmkg_jember_temperature.csv`` — BMKG portal CSV; user downloads.
* ``data/raw/pln_eastjava_load.csv`` — PLN internal data; user uploads if
  available.

When any file is missing the loader silently falls back to the synthetic
Jember profile so the pipeline is never blocked by missing external data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ambient_profile import synthesize_tropical_ambient
from .load_profile import synthesize_eastjava_load
from .pv_synthesis import synthesize_pv_profile
from .utils import get_project_root, load_yaml


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Location dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JemberLocation:
    name: str
    region: str
    country: str
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    bmkg_station_id: str
    pln_substation_name: str
    annual_mean_c: float
    daily_amplitude_c: float
    seasonal_amplitude_c: float
    noise_std_c: float
    solar_annual_capacity_factor: float
    solar_haurwitz_kt_mean: float


def load_jember_location(config_path: str | Path | None = None) -> JemberLocation:
    """Read ``config/jember_location.yaml`` into a JemberLocation dataclass."""
    if config_path is None:
        config_path = get_project_root() / "config" / "jember_location.yaml"
    cfg = load_yaml(config_path)
    loc = cfg.get("location", {})
    clim = cfg.get("climate", {})
    sol = cfg.get("solar", {})
    return JemberLocation(
        name=str(loc.get("name", "Jember")),
        region=str(loc.get("region", "East Java")),
        country=str(loc.get("country", "Indonesia")),
        latitude=float(loc.get("latitude", -8.17)),
        longitude=float(loc.get("longitude", 113.70)),
        elevation_m=float(loc.get("elevation_m", 89)),
        timezone=str(loc.get("timezone", "Asia/Jakarta")),
        bmkg_station_id=str(loc.get("bmkg_station_id", "96935")),
        pln_substation_name=str(loc.get("pln_substation_name", "Jember")),
        annual_mean_c=float(clim.get("annual_mean_c", 26.5)),
        daily_amplitude_c=float(clim.get("daily_amplitude_c", 5.0)),
        seasonal_amplitude_c=float(clim.get("seasonal_amplitude_c", 1.2)),
        noise_std_c=float(clim.get("noise_std_c", 0.8)),
        solar_annual_capacity_factor=float(sol.get("annual_capacity_factor", 0.18)),
        solar_haurwitz_kt_mean=float(sol.get("haurwitz_kt_mean", 0.55)),
    )


# ---------------------------------------------------------------------------
# Default file paths (overridable from config/paths.yaml or args)
# ---------------------------------------------------------------------------

def _default_paths() -> dict[str, Path]:
    root = get_project_root()
    return {
        "atlite_cutout": root / "data" / "raw" / "atlite_cutout_jember.nc",
        "bmkg_csv":      root / "data" / "raw" / "bmkg_jember_temperature.csv",
        "pln_csv":       root / "data" / "raw" / "pln_eastjava_load.csv",
    }


# ---------------------------------------------------------------------------
# Ambient temperature loader (real BMKG CSV → fallback to synthesis)
# ---------------------------------------------------------------------------

def load_jember_ambient(
    n_hours: int = 8760,
    delta_c: float = 0.0,
    seed: int = 42,
    bmkg_csv_path: str | Path | None = None,
    location: JemberLocation | None = None,
) -> np.ndarray:
    """Return a Jember ambient-temperature time-series of length n_hours.

    Tries to load BMKG data from ``bmkg_csv_path``; on missing file or load
    failure, falls back to ``synthesize_tropical_ambient`` calibrated to
    Jember climate.

    Parameters
    ----------
    n_hours : int
    delta_c : float
        Climate-scenario uplift added to the final series.
    seed : int
        Used by the synthesis fallback only.
    bmkg_csv_path : str or Path, optional
        Override the default ``data/raw/bmkg_jember_temperature.csv``.
    location : JemberLocation, optional
        If None, loads from ``config/jember_location.yaml``.
    """
    if location is None:
        location = load_jember_location()
    if bmkg_csv_path is None:
        bmkg_csv_path = _default_paths()["bmkg_csv"]
    bmkg_csv_path = Path(bmkg_csv_path)

    if bmkg_csv_path.exists():
        try:
            return _load_bmkg_csv(bmkg_csv_path, n_hours) + delta_c
        except Exception as exc:   # noqa: BLE001
            _log.warning(f"Could not parse BMKG CSV {bmkg_csv_path}: {exc}; "
                         f"falling back to synthesis.")
    else:
        _log.info(f"No BMKG file at {bmkg_csv_path} — using synthetic Jember climate.")

    return synthesize_tropical_ambient(
        n_hours=n_hours,
        annual_mean_c=location.annual_mean_c,
        daily_amplitude_c=location.daily_amplitude_c,
        seasonal_amplitude_c=location.seasonal_amplitude_c,
        noise_std_c=location.noise_std_c,
        delta_c=delta_c,
        seed=seed,
    )


def _load_bmkg_csv(path: Path, n_hours: int) -> np.ndarray:
    """Parse a BMKG temperature CSV and return an n_hours array.

    BMKG portal export format:  ``Tanggal,Tn,Tx,Tavg,...`` with daily values.
    We linearly interpolate daily Tavg to hourly resolution and add a
    sinusoidal diurnal component (peak at 14:00) of amplitude
    ``(Tx - Tn) / 2``.
    """
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["Tanggal"], dayfirst=True)
    df = df.set_index("Tanggal").sort_index()
    # Forward-fill missing daily rows up to 3 days
    df = df.resample("D").mean().interpolate(limit=3)
    if "Tavg" not in df.columns or "Tn" not in df.columns or "Tx" not in df.columns:
        raise ValueError(f"BMKG CSV missing Tavg/Tn/Tx columns: {list(df.columns)}")

    # Build hourly index of length n_hours starting from the first daily row
    start = df.index[0]
    hours = pd.date_range(start, periods=n_hours, freq="h")
    daily_mean = df["Tavg"].reindex(hours.normalize()).to_numpy()
    daily_amp = ((df["Tx"] - df["Tn"]) / 2.0).reindex(hours.normalize()).to_numpy()
    diurnal = daily_amp * np.sin(2.0 * np.pi * (np.arange(n_hours) % 24 - 8) / 24.0)
    return daily_mean + diurnal


# ---------------------------------------------------------------------------
# PV loader (atlite cutout → fallback to Haurwitz)
# ---------------------------------------------------------------------------

def load_jember_pv(
    n_hours: int = 8760,
    penetration_pu: float = 0.50,
    seed: int = 42,
    cutout_path: str | Path | None = None,
    location: JemberLocation | None = None,
) -> np.ndarray:
    """Return a Jember PV per-unit time-series of length n_hours.

    Tries to load an atlite cutout; on missing file or atlite import error,
    falls back to ``synthesize_pv_profile`` at Jember latitude.

    The fallback path is currently the production code path because ERA5
    cutout download requires a Copernicus CDS account (out of repo scope).
    The atlite branch is wired for when the cutout becomes available.
    """
    if location is None:
        location = load_jember_location()
    if cutout_path is None:
        cutout_path = _default_paths()["atlite_cutout"]
    cutout_path = Path(cutout_path)

    if cutout_path.exists():
        try:
            return _load_atlite_pv(cutout_path, n_hours, penetration_pu, location)
        except Exception as exc:  # noqa: BLE001
            _log.warning(f"Could not load atlite cutout {cutout_path}: {exc}; "
                         f"falling back to Haurwitz synthesis.")
    else:
        _log.info(f"No atlite cutout at {cutout_path} — using Haurwitz synthesis at "
                  f"lat={location.latitude:.2f}.")

    return synthesize_pv_profile(
        n_hours=n_hours,
        lat_deg=location.latitude,
        penetration_pu=penetration_pu,
        clearness_index_mean=location.solar_haurwitz_kt_mean,
        seed=seed,
    )


def _load_atlite_pv(
    cutout_path: Path,
    n_hours: int,
    penetration_pu: float,
    location: JemberLocation,
) -> np.ndarray:
    """Run atlite's PV conversion at the Jember coordinate.

    Reads the cutout, builds a single-pixel mask at (lat, lon), runs
    ``cutout.pv(...)`` with a fixed-tilt south-facing array (latitude tilt),
    and rescales the resulting capacity-factor series to ``penetration_pu``.
    """
    import atlite
    cutout = atlite.Cutout(path=str(cutout_path))
    # Single-pixel layout at Jember
    layout = cutout.layout_from_capacity_list(
        capacity_data=np.array([[location.longitude, location.latitude, 1.0]]),
    )
    pv_series = cutout.pv(
        panel="CSi",
        orientation={"slope": abs(location.latitude), "azimuth": 0.0},
        layout=layout,
        per_unit=True,
    )
    cf = np.asarray(pv_series.values).flatten()
    if len(cf) < n_hours:
        # Tile if cutout is shorter than requested horizon
        cf = np.tile(cf, int(np.ceil(n_hours / len(cf))))
    cf = cf[:n_hours]
    # cf is per-unit of installed capacity; scale to penetration_pu of transformer MVA
    return cf * penetration_pu


# ---------------------------------------------------------------------------
# Load profile loader (PLN CSV → fallback to East Java template)
# ---------------------------------------------------------------------------

def load_jember_load(
    n_hours: int = 8760,
    peak_pu: float = 0.85,
    seed: int = 42,
    pln_csv_path: str | Path | None = None,
    location: JemberLocation | None = None,
) -> np.ndarray:
    """Return a Jember load per-unit time-series of length n_hours.

    Tries to read a PLN East Java CSV; on missing file, falls back to the
    Jember-tuned synthesis template ``synthesize_eastjava_load``.
    """
    if location is None:
        location = load_jember_location()
    if pln_csv_path is None:
        pln_csv_path = _default_paths()["pln_csv"]
    pln_csv_path = Path(pln_csv_path)

    if pln_csv_path.exists():
        try:
            return _load_pln_csv(pln_csv_path, n_hours, peak_pu)
        except Exception as exc:   # noqa: BLE001
            _log.warning(f"Could not parse PLN CSV {pln_csv_path}: {exc}; "
                         f"falling back to East Java template.")
    else:
        _log.info(f"No PLN file at {pln_csv_path} — using synthetic East Java template.")

    return synthesize_eastjava_load(n_hours=n_hours, peak_pu=peak_pu, seed=seed)


def _load_pln_csv(path: Path, n_hours: int, peak_pu: float) -> np.ndarray:
    """Parse a PLN substation export to an hourly per-unit profile.

    Expected schema:  ``Timestamp,Load_MW``  (or ``Tanggal,P_MW``).
    The series is rescaled so its annual peak == ``peak_pu``.
    """
    import pandas as pd
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp") or cols.get("tanggal")
    p_col = cols.get("load_mw") or cols.get("p_mw")
    if ts_col is None or p_col is None:
        raise ValueError(f"PLN CSV must have Timestamp+Load_MW columns; got {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index().resample("h").mean().interpolate(limit=3)
    series = df[p_col].to_numpy()
    if len(series) < n_hours:
        series = np.tile(series, int(np.ceil(n_hours / len(series))))
    series = series[:n_hours]
    series = series / series.max() * peak_pu
    return np.clip(series, 0.0, 1.5)
