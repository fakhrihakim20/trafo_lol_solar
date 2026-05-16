"""IEEE C57.91-2011 Clause 7 transformer thermal model — differential (ODE) form.

State variables
---------------
y[0] = ΔΘ_TO   top-oil temperature rise over ambient   [°C]
y[1] = ΔΘ_HS   hot-spot temperature rise over top-oil  [°C]

ODEs (all time in minutes, matching τ_TO and τ_W units)
-------------------------------------------------------
dΔΘ_TO/dt = (1/τ_TO) · (ΔΘ_TO_ult(K) − ΔΘ_TO)   [Eq. 1]
dΔΘ_HS/dt = (1/τ_W)  · (ΔΘ_HS_ult(K) − ΔΘ_HS)   [Eq. 2]

Ultimate (steady-state) values at loading K
--------------------------------------------
ΔΘ_TO_ult = ΔΘ_TO_R · [(K²·R + 1) / (R + 1)]^n   [Eq. 3]
ΔΘ_HS_ult = ΔΘ_HS_R · K^(2m)                      [Eq. 4]
Θ_HS      = Θ_A + ΔΘ_TO + ΔΘ_HS                   [Eq. 5]

References
----------
IEEE Std C57.91-2011, Clause 7 (differential thermal model),
Table 4 (ONAF default parameters).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Cooling controller parameters
# ---------------------------------------------------------------------------

@dataclass
class CoolingControllerParams:
    """Parameters for the ONAN/ONAF two-state cooling mode controller.

    IEEE C57.91-2011 Table 4 supplies the per-mode thermal parameters.
    Hysteresis band (theta_to_off_c, theta_to_on_c) prevents rapid flapping.

    Parameters
    ----------
    theta_to_on_c : float
        Top-oil threshold [°C] for ONAN → ONAF transition (fan start).
    theta_to_off_c : float
        Top-oil threshold [°C] for ONAF → ONAN transition (fan stop).
    fan_mttf_hours : float
        Mean time to fan failure [hours].  Used in stochastic fan-failure draws.
    onan_tau_to_min, onan_delta_theta_to_r, onan_n
        ONAN thermal parameters (natural convection, C57.91 Table 4).
    onaf_tau_to_min, onaf_delta_theta_to_r, onaf_n
        ONAF thermal parameters (forced air, C57.91 Table 4).
    """
    theta_to_on_c: float = 70.0
    theta_to_off_c: float = 60.0
    fan_mttf_hours: float = 8760.0
    onan_tau_to_min: float = 240.0
    onan_delta_theta_to_r: float = 55.0
    onan_n: float = 0.8
    onaf_tau_to_min: float = 150.0
    onaf_delta_theta_to_r: float = 55.0
    onaf_n: float = 0.8

    def __post_init__(self) -> None:
        if self.theta_to_off_c >= self.theta_to_on_c:
            raise ValueError(
                f"Hysteresis requires theta_to_off_c < theta_to_on_c; "
                f"got {self.theta_to_off_c} >= {self.theta_to_on_c}"
            )

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "CoolingControllerParams":
        """Construct from the 'cooling_states' sub-dict of transformer_60mva.yaml."""
        cs = cfg.get("cooling_states", {})
        onan = cs.get("ONAN", {})
        onaf = cs.get("ONAF", {})
        ctl = cs.get("controller", {})
        return cls(
            theta_to_on_c=float(ctl.get("theta_to_on_c", 70.0)),
            theta_to_off_c=float(ctl.get("theta_to_off_c", 60.0)),
            fan_mttf_hours=float(ctl.get("fan_mttf_hours", 8760.0)),
            onan_tau_to_min=float(onan.get("tau_to_min", 240.0)),
            onan_delta_theta_to_r=float(onan.get("delta_theta_to_r", 55.0)),
            onan_n=float(onan.get("n", 0.8)),
            onaf_tau_to_min=float(onaf.get("tau_to_min", 150.0)),
            onaf_delta_theta_to_r=float(onaf.get("delta_theta_to_r", 55.0)),
            onaf_n=float(onaf.get("n", 0.8)),
        )


# ---------------------------------------------------------------------------
# Transformer parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class TransformerParams:
    """Nameplate and thermal parameters for one transformer unit.

    All values with physical meaning cite their IEEE C57.91-2011 source.

    Parameters
    ----------
    rated_mva : float
        Transformer rated power [MVA].
    delta_theta_to_r : float
        Rated top-oil rise over ambient ΔΘ_TO_R [°C]. C57.91 Table 4 ONAF: 55.
    delta_theta_hs_r : float
        Rated hot-spot rise over top-oil ΔΘ_HS_R [°C]. C57.91 Table 4 ONAF: 25.
    tau_to_min : float
        Top-oil thermal time constant τ_TO [minutes]. C57.91 Table 4 ONAF: 150.
    tau_w_min : float
        Winding thermal time constant τ_W [minutes]. C57.91 Table 4 ONAF: 7.
    R : float
        Ratio of load losses to no-load losses at rated current. C57.91: 5.
    n : float
        Oil temperature exponent (ONAF). C57.91 Table 4: 0.8.
    m : float
        Winding temperature exponent (ONAF). C57.91 Table 4: 0.8.
    cooling_mode : str
        One of "ONAN", "ONAF", "OFAF".
    insulation_water_pct : float or None
        Insulation water content [% by dry weight] for moisture-dependent
        Arrhenius B-constant (IEEE C57.91-2011 Annex E).  None → use standard
        B=15 000 K.  Typical range: 1 % (dry) – 3 % (wet service).
    cooling_ctrl_params : CoolingControllerParams or None
        If provided, enables ONAN/ONAF mode switching via CoolingController.
        None (default) uses fixed cooling_mode throughout (backwards compatible).
    reverse_flow_loss_uplift : float
        Optional equivalent-loss sensitivity for reverse power flow.  The
        production paper configuration keeps this at zero because measured
        directional reverse-current loss data are unavailable.
    """
    rated_mva: float = 60.0
    delta_theta_to_r: float = 55.0
    delta_theta_hs_r: float = 25.0
    tau_to_min: float = 150.0
    tau_w_min: float = 7.0
    R: float = 5.0
    n: float = 0.8
    m: float = 0.8
    cooling_mode: str = "ONAF"
    insulation_water_pct: float | None = None
    cooling_ctrl_params: CoolingControllerParams | None = None
    reverse_flow_loss_uplift: float = 0.0

    def __post_init__(self) -> None:
        if self.cooling_mode not in ("ONAN", "ONAF", "OFAF"):
            raise ValueError(f"cooling_mode must be ONAN/ONAF/OFAF, got {self.cooling_mode!r}")
        if self.tau_to_min <= 0 or self.tau_w_min <= 0:
            raise ValueError("Time constants must be positive.")
        if self.reverse_flow_loss_uplift < 0.0:
            raise ValueError("reverse_flow_loss_uplift must be non-negative.")

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "TransformerParams":
        """Construct from the 'transformer' sub-dict of transformer_60mva.yaml."""
        t = cfg.get("transformer", cfg)
        water_pct = t.get("insulation_water_content_pct")
        ctrl = (
            CoolingControllerParams.from_yaml(cfg.get("transformer", cfg))
            if "cooling_states" in t else None
        )
        return cls(
            rated_mva=t["rated_mva"],
            delta_theta_to_r=t["delta_theta_to_r"],
            delta_theta_hs_r=t["delta_theta_hs_r"],
            tau_to_min=t["tau_to_min"],
            tau_w_min=t["tau_w_min"],
            R=t["R"],
            n=t["n"],
            m=t["m"],
            cooling_mode=t["cooling_mode"],
            insulation_water_pct=float(water_pct) if water_pct is not None else None,
            cooling_ctrl_params=ctrl,
            reverse_flow_loss_uplift=float(t.get("reverse_flow_loss_uplift", 0.0)),
        )


# ---------------------------------------------------------------------------
# ONAN/ONAF cooling controller (two-state with hysteresis)
# ---------------------------------------------------------------------------

class CoolingController:
    """Stateful ONAN/ONAF two-state controller with hysteresis.

    State machine
    -------------
    ONAN --(θ_TO > theta_to_on AND fans_ok)--> ONAF
    ONAF --(θ_TO < theta_to_off)-------------> ONAN
    ANY  --(fans_ok = False)-----------------> ONAN  (degraded/fan-failure mode)

    Usage
    -----
    Create one controller per simulation run (stateful — do NOT share between runs).
    Call .step(theta_to) at each timestep to update and retrieve the active mode.
    Call .active_params(base_params) to obtain the TransformerParams for that mode.
    """

    def __init__(self, ctrl_params: CoolingControllerParams, fans_ok: bool = True) -> None:
        self._p = ctrl_params
        self.fans_ok = fans_ok
        self._mode: str = "ONAN"  # always start in natural-convection mode

    @property
    def mode(self) -> str:
        """Current cooling mode: 'ONAN' or 'ONAF'."""
        return self._mode

    def step(self, theta_to: float) -> str:
        """Update cooling state based on current top-oil temperature.

        Parameters
        ----------
        theta_to : float
            Current top-oil temperature [°C].

        Returns
        -------
        str
            Active mode after this update: 'ONAN' or 'ONAF'.
        """
        if not self.fans_ok:
            self._mode = "ONAN"
        elif self._mode == "ONAN" and theta_to > self._p.theta_to_on_c:
            self._mode = "ONAF"
        elif self._mode == "ONAF" and theta_to < self._p.theta_to_off_c:
            self._mode = "ONAN"
        return self._mode

    def active_params(self, base: "TransformerParams") -> "TransformerParams":
        """Return a TransformerParams configured for the current cooling mode.

        Only tau_to_min, delta_theta_to_r, n, and cooling_mode are overridden;
        all other fields (R, m, rated_mva, etc.) are inherited from *base*.
        """
        if self._mode == "ONAN":
            return replace(
                base,
                tau_to_min=self._p.onan_tau_to_min,
                delta_theta_to_r=self._p.onan_delta_theta_to_r,
                n=self._p.onan_n,
                cooling_mode="ONAN",
            )
        return replace(
            base,
            tau_to_min=self._p.onaf_tau_to_min,
            delta_theta_to_r=self._p.onaf_delta_theta_to_r,
            n=self._p.onaf_n,
            cooling_mode="ONAF",
        )


# ---------------------------------------------------------------------------
# Ultimate (steady-state) temperature computations
# ---------------------------------------------------------------------------

def _delta_theta_to_ult(K: float, params: TransformerParams) -> float:
    """Steady-state top-oil rise at loading K.  C57.91-2011 Eq. (3).

    ΔΘ_TO_ult = ΔΘ_TO_R · [(K²·R + 1) / (R + 1)]^n
    """
    return params.delta_theta_to_r * ((K**2 * params.R + 1.0) / (params.R + 1.0)) ** params.n


def _delta_theta_hs_ult(K: float, params: TransformerParams) -> float:
    """Steady-state hot-spot rise over top-oil at loading K.  C57.91-2011 Eq. (4).

    ΔΘ_HS_ult = ΔΘ_HS_R · K^(2m)
    """
    return params.delta_theta_hs_r * K ** (2.0 * params.m)


def steady_state(
    K: float,
    ambient_c: float,
    params: TransformerParams,
    inverter_thd: float = 0.0,
) -> tuple[float, float]:
    """Return steady-state (ΔΘ_TO, ΔΘ_HS) at loading K and ambient temperature.

    Parameters
    ----------
    K : float
        Per-unit loading (I/I_rated).
    ambient_c : float
        Ambient temperature [°C].
    params : TransformerParams
    inverter_thd : float
        Total harmonic distortion fraction of inverter output current (0–1).
        Applies harmonic uplift R_eff = R·(1 + 0.5·THD²) per IEC 61378-1.
        Default 0 preserves existing behaviour.

    Returns
    -------
    (delta_to, delta_hs) : tuple[float, float]
        Top-oil rise [°C], hot-spot rise over top-oil [°C].
    """
    if inverter_thd > 0.0:
        R_eff = params.R * (1.0 + 0.5 * inverter_thd ** 2)
        params = replace(params, R=R_eff)
    return _delta_theta_to_ult(K, params), _delta_theta_hs_ult(K, params)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def _make_ode(
    load_interp: interp1d,
    ambient_interp: interp1d,
    params: TransformerParams,
) -> Any:
    """Return the ODE function dy/dt for scipy solve_ivp.

    Time axis is in minutes.  All temperature rises are in °C.
    """
    tau_to = params.tau_to_min
    tau_w = params.tau_w_min

    def ode(t: float, y: np.ndarray) -> list[float]:
        K = float(load_interp(t))
        # Clamp K to non-negative (PV can push net load to zero but not negative loading)
        K = max(K, 0.0)
        d_to_ult = _delta_theta_to_ult(K, params)
        d_hs_ult = _delta_theta_hs_ult(K, params)
        d_to, d_hs = y[0], y[1]
        dy0 = (d_to_ult - d_to) / tau_to   # C57.91-2011 Eq. (1)
        dy1 = (d_hs_ult - d_hs) / tau_w    # C57.91-2011 Eq. (2)
        return [dy0, dy1]

    return ode


# ---------------------------------------------------------------------------
# Switching simulation (analytical ZOH solution per interval)
# ---------------------------------------------------------------------------

def _simulate_with_switching(
    load_pu: np.ndarray,
    ambient_c: np.ndarray,
    base_params: "TransformerParams",
    dt_min: float,
    y0: np.ndarray,
    controller: CoolingController,
) -> dict[str, Any]:
    """Step-by-step thermal simulation with per-interval mode switching.

    At each hourly interval the controller observes the current top-oil
    temperature and may toggle the cooling mode.  Within each interval the
    parameters are constant, so the ODE has an exact analytical solution:

        ΔΘ_TO(t+Δt) = ΔΘ_TO_ult + (ΔΘ_TO(t) − ΔΘ_TO_ult) · exp(−Δt / τ_TO)

    This is equivalent to `solve_ivp` in the limit of zero numerical error for
    piecewise-constant forcing, but ~1000× faster per step.

    Parameters
    ----------
    load_pu : np.ndarray, shape (T,)
    ambient_c : np.ndarray, shape (T,)
    base_params : TransformerParams
        Base (nominal) transformer parameters; R must already include any THD uplift.
    dt_min : float
        Timestep [minutes].
    y0 : np.ndarray, shape (2,)
        Initial (ΔΘ_TO, ΔΘ_HS).
    controller : CoolingController
        Stateful controller; mutated in-place by this function.

    Returns
    -------
    dict with keys: theta_hs, theta_to, delta_to, delta_hs, mode, solver_info.
        ``mode`` is a (T,) array of strings "ONAN"/"ONAF" per timestep.
    """
    T = len(load_pu)
    delta_to_arr = np.empty(T)
    delta_hs_arr = np.empty(T)
    mode_arr = np.empty(T, dtype="U4")

    d_to, d_hs = float(y0[0]), float(y0[1])

    for t in range(T):
        # θ_TO at start of this interval → informs mode decision
        theta_to_now = ambient_c[t] + d_to
        controller.step(theta_to_now)
        active = controller.active_params(base_params)
        mode_arr[t] = controller.mode

        K = max(float(load_pu[t]), 0.0)
        d_to_ult = _delta_theta_to_ult(K, active)
        d_hs_ult = _delta_theta_hs_ult(K, active)

        # Exact analytical solution for linear ODE with constant K in [t, t+Δt]
        alpha_to = np.exp(-dt_min / active.tau_to_min)
        alpha_hs = np.exp(-dt_min / active.tau_w_min)

        d_to = d_to_ult + (d_to - d_to_ult) * alpha_to
        d_hs = d_hs_ult + (d_hs - d_hs_ult) * alpha_hs

        delta_to_arr[t] = d_to
        delta_hs_arr[t] = d_hs

    theta_to_arr = ambient_c + delta_to_arr
    theta_hs_arr = ambient_c + delta_to_arr + delta_hs_arr

    return {
        "theta_hs": theta_hs_arr,
        "theta_to": theta_to_arr,
        "delta_to": delta_to_arr,
        "delta_hs": delta_hs_arr,
        "mode": mode_arr,
        "solver_info": {
            "nfev": T,
            "njev": 0,
            "status": 0,
            "message": "analytic switching simulation",
        },
    }


# ---------------------------------------------------------------------------
# Public simulation API
# ---------------------------------------------------------------------------

def simulate_thermal(
    load_pu: np.ndarray,
    ambient_c: np.ndarray,
    params: TransformerParams,
    dt_min: float = 60.0,
    initial_state: tuple[float, float] | None = None,
    inverter_thd: float = 0.0,
    cooling_controller: CoolingController | None = None,
) -> dict[str, Any]:
    """Simulate transformer thermal response using IEEE C57.91-2011 differential model.

    Solves the two-state ODE:
        dΔΘ_TO/dt = (1/τ_TO) · (ΔΘ_TO_ult(K) − ΔΘ_TO)
        dΔΘ_HS/dt = (1/τ_W)  · (ΔΘ_HS_ult(K) − ΔΘ_HS)

    Time is in minutes throughout to be consistent with τ_TO and τ_W units.
    scipy.integrate.solve_ivp RK45 is used with max_step = dt_min / 3 to
    resolve transients from step changes in loading or PV clouds.

    Parameters
    ----------
    load_pu : np.ndarray, shape (T,)
        Per-unit loading K = I/I_rated at each timestep.  Values at
        t = 0, dt_min, 2*dt_min, …, (T-1)*dt_min.
    ambient_c : np.ndarray, shape (T,)
        Ambient temperature [°C] at each timestep.
    params : TransformerParams
        Thermal and nameplate parameters.
    dt_min : float
        Timestep size [minutes].  Default 60 (hourly).
    initial_state : tuple[float, float] or None
        (ΔΘ_TO_0, ΔΘ_HS_0) initial rises [°C].
        If None, initialized to steady state at the first load/ambient sample.
    inverter_thd : float
        Total harmonic distortion fraction of inverter output current (0–1).
        Applies harmonic uplift R_eff = R·(1 + 0.5·THD²) per IEC 61378-1.
        Default 0 preserves existing behaviour exactly.
    cooling_controller : CoolingController or None
        If provided, enables ONAN/ONAF mode switching via the analytical
        step-by-step solver (see _simulate_with_switching).  If None, uses the
        original single-span RK45 solve with fixed cooling mode.

    Returns
    -------
    dict with keys:
        ``theta_hs``  : np.ndarray, shape (T,) — hot-spot temperature [°C]
        ``theta_to``  : np.ndarray, shape (T,) — top-oil temperature [°C]
        ``delta_to``  : np.ndarray, shape (T,) — top-oil rise over ambient [°C]
        ``delta_hs``  : np.ndarray, shape (T,) — hot-spot rise over top-oil [°C]
        ``solver_info``: dict — scipy solve_ivp metadata (nfev, etc.)

    Raises
    ------
    ValueError
        If load_pu and ambient_c have different lengths, or if any values
        are outside physically plausible bounds.
    """
    load_pu = np.asarray(load_pu, dtype=float)
    ambient_c = np.asarray(ambient_c, dtype=float)

    # Apply harmonic loss uplift: R_eff = R*(1 + 0.5*THD²)  [IEC 61378-1]
    if inverter_thd > 0.0:
        R_eff = params.R * (1.0 + 0.5 * inverter_thd ** 2)
        params = replace(params, R=R_eff)

    T = len(load_pu)
    if len(ambient_c) != T:
        raise ValueError(
            f"load_pu length {T} != ambient_c length {len(ambient_c)}."
        )
    if T < 2:
        raise ValueError("Need at least 2 timesteps.")

    if np.any(load_pu < 0):
        raise ValueError("load_pu contains negative values.")
    if np.any(np.isnan(load_pu)) or np.any(np.isnan(ambient_c)):
        raise ValueError("NaN values detected in inputs.")

    # Time axis in minutes
    t_arr = np.arange(T, dtype=float) * dt_min

    # Zero-order hold interpolation (matches SCADA piecewise-constant data)
    # bounds_error=False + fill_value extrapolates final value past last point
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        load_interp = interp1d(t_arr, load_pu, kind="previous",
                               bounds_error=False,
                               fill_value=(load_pu[0], load_pu[-1]))
        ambient_interp = interp1d(t_arr, ambient_c, kind="previous",
                                  bounds_error=False,
                                  fill_value=(ambient_c[0], ambient_c[-1]))

    # Initial state: steady-state at first sample if not specified
    if initial_state is None:
        d_to_ss, d_hs_ss = steady_state(load_pu[0], ambient_c[0], params)
        y0 = np.array([d_to_ss, d_hs_ss], dtype=float)
    else:
        y0 = np.array(initial_state, dtype=float)

    # Dispatch to switching solver when a CoolingController is provided
    if cooling_controller is not None:
        return _simulate_with_switching(
            load_pu, ambient_c, params, dt_min, y0, cooling_controller
        )

    # Build ODE (ambient_interp not used in ODE itself — Θ_A added post-hoc)
    ode = _make_ode(load_interp, ambient_interp, params)

    # Solve over full time span; evaluate at each SCADA timestep
    # max_step = dt_min / 3 ensures ≥3 internal steps per interval, resolving
    # cloud-induced 15-min transients without excessive computation.
    max_step = max(dt_min / 3.0, 0.1)   # minimum 0.1 min regardless of dt
    sol = solve_ivp(
        ode,
        t_span=(t_arr[0], t_arr[-1]),
        y0=y0,
        method="RK45",
        t_eval=t_arr,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    delta_to = sol.y[0]   # ΔΘ_TO at each timestep
    delta_hs = sol.y[1]   # ΔΘ_HS at each timestep

    # Θ_HS = Θ_A + ΔΘ_TO + ΔΘ_HS  (C57.91-2011 Eq. 5)
    theta_to = ambient_c + delta_to
    theta_hs = ambient_c + delta_to + delta_hs

    solver_info = {
        "nfev": sol.nfev,
        "njev": getattr(sol, "njev", 0),
        "status": sol.status,
        "message": sol.message,
    }

    return {
        "theta_hs": theta_hs,
        "theta_to": theta_to,
        "delta_to": delta_to,
        "delta_hs": delta_hs,
        "solver_info": solver_info,
    }
