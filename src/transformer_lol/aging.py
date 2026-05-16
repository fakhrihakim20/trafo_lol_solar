"""Insulation aging model — IEEE C57.91-2011 Clause 5.

Arrhenius aging acceleration factor (FAA) and loss-of-life (LoL) integrator
for thermally upgraded Kraft paper insulation.

Key constants (C57.91-2011 Clause 5)
--------------------------------------
B_CONSTANT     = 15 000 K          (Arrhenius activation energy / gas constant)
THETA_HS_RATED = 110 °C            (reference hot-spot; FAA = 1.0)
RATED_LIFE_H   = 180 000 hours     (expected insulation life at 110 °C continuous)

FAA formula (C57.91-2011 Eq. 3)
---------------------------------
F_AA(θ_HS) = exp[ B · (1/(θ_HS_rated + 273.15) − 1/(θ_HS + 273.15)) ]

Note on F_AA(120 °C)
---------------------
The IEEE standard text loosely states F_AA ≈ 2 at 120 °C.  The exact formula gives
F_AA(120 °C) = 2.7089.  Tests in this repo use the exact value, not the approximation.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (C57.91-2011 Clause 5)
# ---------------------------------------------------------------------------

B_CONSTANT: float = 15_000.0        # Arrhenius constant [K] — standard value
THETA_HS_RATED: float = 110.0       # reference hot-spot temperature [°C]
RATED_LIFE_HOURS: float = 180_000.0 # rated insulation life [hours]

# Moisture-dependent B-constant (IEEE C57.91-2011 Annex E)
# Linear fit over [1, 3] % water content:
#   w=1% (dry service):  B = 15 000 K  (= B_CONSTANT, standard value)
#   w=2%:                B ≈ 14 579 K
#   w=3% (wet season):   B = 14 158 K
# Higher moisture → lower B → less temperature-sensitive aging.
# Valid range: w ∈ [1, 3] % (clamped outside this range).
B_MOISTURE_SLOPE: float = -421.0    # dB / d(w %)  [K / %]  (Annex E linear fit, negative)


def b_from_moisture(water_content_pct: float) -> float:
    """Effective Arrhenius B-constant at a given insulation water content.

    IEEE C57.91-2011 Annex E linear model, clamped to [1, 3] %.
    At 1 % (dry service) returns B_CONSTANT = 15 000 K exactly.
    At 3 % (wet season)  returns 14 158 K.

    Parameters
    ----------
    water_content_pct : float
        Insulation water content [% by dry weight].

    Returns
    -------
    float
        Effective B [K].
    """
    w = max(1.0, min(3.0, water_content_pct))
    return B_CONSTANT + B_MOISTURE_SLOPE * (w - 1.0)


# ---------------------------------------------------------------------------
# Aging acceleration factor
# ---------------------------------------------------------------------------

def aging_acceleration_factor(
    theta_hs_c: float | np.ndarray,
    moisture_water_pct: float | None = None,
) -> float | np.ndarray:
    """Arrhenius aging acceleration factor F_AA.

    IEEE C57.91-2011 Eq. (3), Clause 5.

    Parameters
    ----------
    theta_hs_c : float or np.ndarray
        Hot-spot temperature [°C].
    moisture_water_pct : float or None
        Insulation water content [% by dry weight].  When provided, the
        effective B-constant is derived from the moisture model (Annex E):
        B(w) = 14 158 + 948·(w−1) K.  When None, uses B=15 000 K (standard).

    Returns
    -------
    float or np.ndarray
        F_AA.  F_AA = 1.0 exactly at θ_HS = 110 °C (with standard B).

    Examples
    --------
    >>> import math
    >>> math.isclose(aging_acceleration_factor(110.0), 1.0, abs_tol=1e-9)
    True
    >>> round(aging_acceleration_factor(120.0), 4)
    2.7089
    """
    B = b_from_moisture(moisture_water_pct) if moisture_water_pct is not None else B_CONSTANT
    return np.exp(
        B * (
            1.0 / (THETA_HS_RATED + 273.15)
            - 1.0 / (np.asarray(theta_hs_c, dtype=float) + 273.15)
        )
    )


# ---------------------------------------------------------------------------
# Loss-of-life integrators
# ---------------------------------------------------------------------------

def loss_of_life_hours(
    theta_hs_c: float | np.ndarray,
    dt_hours: float,
    moisture_water_pct: float | None = None,
) -> float:
    """Equivalent aging hours consumed over a temperature history.

    Integrates F_AA · Δt and returns the total equivalent aging in hours.
    This is NOT a percentage — divide by RATED_LIFE_HOURS to get LoL%.

    IEEE C57.91-2011 Clause 5, Eq. (4):
        Equivalent aging hours = Σ F_AA(θ_HS[i]) · Δt

    Parameters
    ----------
    theta_hs_c : float or np.ndarray
        Hot-spot temperature history [°C].  Scalar → treated as constant.
    dt_hours : float
        Timestep duration [hours].
    moisture_water_pct : float or None
        Passed to aging_acceleration_factor().  See that function's docstring.

    Returns
    -------
    float
        Equivalent aging hours consumed.
    """
    faa = aging_acceleration_factor(
        np.asarray(theta_hs_c, dtype=float), moisture_water_pct=moisture_water_pct
    )
    return float(np.sum(faa) * dt_hours)


def loss_of_life_percent(
    theta_hs_c: float | np.ndarray,
    dt_hours: float,
    rated_life_hours: float = RATED_LIFE_HOURS,
    moisture_water_pct: float | None = None,
) -> float:
    """Loss-of-life as a percentage of rated insulation life.

    IEEE C57.91-2011 Clause 5, Eq. (5):
        LoL [%] = 100 · Σ(F_AA · Δt) / rated_life_hours

    Parameters
    ----------
    theta_hs_c : float or np.ndarray
        Hot-spot temperature history [°C].
    dt_hours : float
        Timestep duration [hours].
    rated_life_hours : float
        Rated insulation life [hours].  Default: 180 000 h (thermally upgraded Kraft).
    moisture_water_pct : float or None
        Passed to aging_acceleration_factor().  See that function's docstring.

    Returns
    -------
    float
        LoL [%].
    """
    equiv_hours = loss_of_life_hours(theta_hs_c, dt_hours, moisture_water_pct=moisture_water_pct)
    return 100.0 * equiv_hours / rated_life_hours


# ---------------------------------------------------------------------------
# Convenience: scalar F_AA (for validation / standalone checks)
# ---------------------------------------------------------------------------

def faa_scalar(theta_hs_c: float) -> float:
    """Scalar version of aging_acceleration_factor, returns a Python float.

    Useful in validation scripts and tests where a numpy scalar is inconvenient.
    """
    return float(aging_acceleration_factor(theta_hs_c))
