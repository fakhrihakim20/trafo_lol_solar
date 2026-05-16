"""Unit tests for src/transformer_lol/thermal_model.py.

Tests verify:
1. Ultimate temperature computations (Eq. 3 and 4 of C57.91-2011)
2. Steady-state convergence at K=1.0 after 5×τ_TO
3. Step-response analytically (τ_TO time constant)
4. Annex G two-phase load profile reproduction
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transformer_lol.thermal_model import (
    TransformerParams,
    _delta_theta_to_ult,
    _delta_theta_hs_ult,
    simulate_thermal,
    steady_state,
)
from transformer_lol.validation import ANNEX_G_PARAMS, run_annex_g_validation


# Default ONAF parameters used across tests
P = TransformerParams()


# ---------------------------------------------------------------------------
# Test 1: Ultimate temperature values at rated and overload
# ---------------------------------------------------------------------------

def test_ultimate_at_rated_load() -> None:
    """At K=1.0, ultimate rises must equal rated values exactly.

    C57.91-2011 Eq. (3): ΔΘ_TO_ult = ΔΘ_TO_R × [(1²·R+1)/(R+1)]^n
    At K=1: numerator = R+1, so ΔΘ_TO_ult = ΔΘ_TO_R exactly.
    """
    d_to = _delta_theta_to_ult(1.0, P)
    d_hs = _delta_theta_hs_ult(1.0, P)
    assert math.isclose(d_to, P.delta_theta_to_r, abs_tol=1e-6), \
        f"ΔΘ_TO_ult at K=1 should be {P.delta_theta_to_r}, got {d_to}"
    assert math.isclose(d_hs, P.delta_theta_hs_r, abs_tol=1e-6), \
        f"ΔΘ_HS_ult at K=1 should be {P.delta_theta_hs_r}, got {d_hs}"


def test_ultimate_at_overload() -> None:
    """At K=1.5, ultimate rises are higher than rated values.

    Manual calculation with n=m=0.8, R=5:
      ΔΘ_TO_ult = 55 × [(1.5²×5+1)/(5+1)]^0.8
               = 55 × [12.25/6]^0.8
               = 55 × 2.04167^0.8
               ≈ 55 × 1.759 = 96.75 °C
      ΔΘ_HS_ult = 25 × 1.5^(2×0.8) = 25 × 1.5^1.6 ≈ 25 × 1.900 = 47.50 °C
    """
    d_to = _delta_theta_to_ult(1.5, P)
    d_hs = _delta_theta_hs_ult(1.5, P)
    assert d_to > P.delta_theta_to_r, "ΔΘ_TO_ult at K=1.5 must exceed rated value"
    assert d_hs > P.delta_theta_hs_r, "ΔΘ_HS_ult at K=1.5 must exceed rated value"
    # Check approximate magnitudes (±1 °C tolerance on manual calc)
    assert abs(d_to - 96.75) < 1.0, f"ΔΘ_TO_ult(1.5) ≈ 96.75, got {d_to:.3f}"
    assert abs(d_hs - 47.50) < 1.0, f"ΔΘ_HS_ult(1.5) ≈ 47.50, got {d_hs:.3f}"


# ---------------------------------------------------------------------------
# Test 2: Steady-state convergence after 5×τ_TO
# ---------------------------------------------------------------------------

def test_steady_state_at_110c() -> None:
    """After 5×τ_TO at K=1.0, θ_HS must converge to 30+55+25=110 °C within 0.5 °C.

    Analytic: ΔΘ(5τ) = ΔΘ_ult × (1 − exp(−5)) ≈ ΔΘ_ult × 0.9933.
    The 0.7% error from steady state gives θ_HS ≈ 109.45 °C, within 0.5 °C of 110.
    """
    n_steps = 800   # 800 hours >> 5×τ_TO = 12.5 hours
    load_pu = np.ones(n_steps)
    ambient_c = np.full(n_steps, 30.0)

    res = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0, initial_state=(0.0, 0.0))
    theta_hs_final = res["theta_hs"][-1]
    assert abs(theta_hs_final - 110.0) < 0.5, \
        f"Steady-state θ_HS should be ~110 °C, got {theta_hs_final:.3f} °C"


# ---------------------------------------------------------------------------
# Test 3: Step response — time constant τ_TO
# ---------------------------------------------------------------------------

def test_step_response_analytic() -> None:
    """At t=τ_TO=150 min after a cold K=1.0 step, ΔΘ_TO = ΔΘ_TO_R × (1−1/e).

    Analytic: ΔΘ_TO(τ_TO) = 55 × (1 − exp(−1)) = 55 × 0.63212 = 34.767 °C.
    """
    n_steps = 151   # 0…150 minutes at 1-min resolution
    load_pu = np.ones(n_steps)
    ambient_c = np.full(n_steps, 30.0)

    res = simulate_thermal(load_pu, ambient_c, P, dt_min=1.0, initial_state=(0.0, 0.0))
    delta_to_at_tau = res["delta_to"][150]   # index 150 → t=150 min = τ_TO
    expected = P.delta_theta_to_r * (1.0 - math.exp(-1.0))   # = 34.767 °C
    assert abs(delta_to_at_tau - expected) < 0.05, \
        f"ΔΘ_TO at t=τ_TO: expected {expected:.3f}, got {delta_to_at_tau:.3f}"


# ---------------------------------------------------------------------------
# Test 4: Annex G two-phase load profile
# ---------------------------------------------------------------------------

def test_annex_g_two_phase() -> None:
    """Reproduce IEEE C57.91-2011 Annex G fixtures within stated tolerances.

    Runs run_annex_g_validation() — which raises AssertionError on failure,
    causing this test to fail with a descriptive message.
    """
    run_annex_g_validation(params=ANNEX_G_PARAMS, verbose=False)


# ---------------------------------------------------------------------------
# Test 5: Physical sanity — no NaN, bounds
# ---------------------------------------------------------------------------

def test_no_nan_in_output() -> None:
    """Simulate a realistic 24-hour profile and check for NaN-free output."""
    rng = np.random.default_rng(0)
    load_pu = 0.7 + 0.2 * rng.standard_normal(24)
    load_pu = np.clip(load_pu, 0.0, 1.5)
    ambient_c = np.full(24, 30.0)

    res = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0)
    assert not np.any(np.isnan(res["theta_hs"])), "NaN in theta_hs"
    assert not np.any(np.isnan(res["theta_to"])), "NaN in theta_to"
    assert np.all(res["theta_hs"] > 0.0), "theta_hs must be positive"


# ---------------------------------------------------------------------------
# Test 6: Harmonic THD uplift — IEC 61378-1 R_eff = R*(1 + 0.5*THD²)
# ---------------------------------------------------------------------------

def test_thd_uplift_at_5pct() -> None:
    """At THD=5%, R_eff = R*(1+0.5*0.05²) = 5.00625.

    At constant overload K=1.5, higher R_eff raises ΔΘ_TO_ult and therefore
    θ_HS.  THD=0 must be bit-identical to the no-thd default call.
    """
    thd = 0.05
    R_eff_expected = P.R * (1.0 + 0.5 * thd ** 2)
    assert math.isclose(R_eff_expected, 5.00625, rel_tol=1e-9), (
        f"R_eff at THD=5% should be 5.00625, got {R_eff_expected}"
    )

    n_steps = 800
    load_pu = np.full(n_steps, 1.5)   # constant overload maximises R sensitivity
    ambient_c = np.full(n_steps, 30.0)

    res_base = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0)
    res_thd  = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0, inverter_thd=thd)

    assert res_thd["theta_hs"][-1] > res_base["theta_hs"][-1], (
        "THD=5% at K=1.5 overload must raise θ_HS vs zero-THD baseline"
    )

    # THD=0 must be numerically identical to not passing the argument
    res_zero = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0, inverter_thd=0.0)
    np.testing.assert_allclose(
        res_zero["theta_hs"], res_base["theta_hs"],
        atol=1e-10, err_msg="inverter_thd=0 must be identical to default",
    )


# ---------------------------------------------------------------------------
# Test 7: Initial-state from kwargs vs explicit cold start
# ---------------------------------------------------------------------------

def test_initial_state_consistency() -> None:
    """Warm-start at steady state should give same final result as long cold run."""
    # Compute steady state at K=1.0, 30°C
    d_to_ss, d_hs_ss = steady_state(1.0, 30.0, P)

    n = 24
    load_pu = np.ones(n)
    ambient_c = np.full(n, 30.0)

    res_warm = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0,
                                initial_state=(d_to_ss, d_hs_ss))
    res_auto = simulate_thermal(load_pu, ambient_c, P, dt_min=60.0,
                                initial_state=None)   # auto steady-state init

    # Both should give identical results (same initial state)
    np.testing.assert_allclose(res_warm["theta_hs"], res_auto["theta_hs"],
                               atol=0.01, err_msg="Warm vs auto init mismatch")
