"""Monte Carlo sweep driver for transformer loss-of-life scenarios.

Each scenario runs n_runs independent realizations.  Every run uses
seed = scenario.base_seed + run_idx to guarantee reproducibility while
ensuring stochastic independence between runs.

Two execution modes:
  use_surrogate=False  ODE physics (slow, ground truth, for spot-checking)
  use_surrogate=True   XGBoost surrogate (fast, used for full sweep)

Both modes return an identical DataFrame schema.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .aging import loss_of_life_percent, aging_acceleration_factor
from .battery import BatteryParams, dispatch_battery
from .data_jember import (
    load_jember_ambient,
    load_jember_load,
    load_jember_pv,
)
from .ev_load import synthesize_ev_total
from .pypsa_dispatch import dispatch_battery_pypsa_lp
from .thermal_model import TransformerParams, CoolingController, simulate_thermal
from .scenarios import Scenario


# ---------------------------------------------------------------------------
# Output column specification
# ---------------------------------------------------------------------------

MC_RESULT_COLUMNS: list[str] = [
    "run_idx",
    "scenario_name",
    "pv_penetration",
    "ambient_delta_c",
    "load_growth_pct",
    "peak_theta_hs_c",
    "mean_theta_hs_c",
    "lol_percent_annual",
    "lol_hours_annual",
    "hours_above_110c",
    "hours_above_120c",
    "reverse_flow_hours",
    "peak_net_load_pu",
    "mean_f_aa",
    "battery_lifetime_fraction",   # fraction of LiFePO4 life consumed (0 if no battery)
    "dispatch_method",             # "threshold" (heuristic) or "pypsa_lp" (Phase 4B)
]


# ---------------------------------------------------------------------------
# Loading convention helpers
# ---------------------------------------------------------------------------

def effective_thermal_loading(
    load_pu: np.ndarray,
    pv_pu: np.ndarray,
    battery_p_pu: np.ndarray | None = None,
    reverse_flow_loss_uplift: float = 0.0,
) -> np.ndarray:
    """Return non-negative equivalent transformer loading for thermal aging.

    Forward import uses the usual net loading ``load - PV - battery`` clipped at
    zero.  When ``reverse_flow_loss_uplift`` is positive, reverse export is
    represented as a small equivalent loading: an uplift of 0.025 means reverse
    current contributes 2.5% of the same-current load-loss term, so the
    equivalent current is ``sqrt(0.025) * abs(reverse_export)``.

    The default uplift is zero because the paper's headline runs do not have
    measured directional reverse-current loss data.
    """
    load_pu = np.asarray(load_pu, dtype=float)
    pv_pu = np.asarray(pv_pu, dtype=float)
    if battery_p_pu is None:
        battery_p_pu = np.zeros_like(load_pu)
    else:
        battery_p_pu = np.asarray(battery_p_pu, dtype=float)

    signed_net = load_pu - pv_pu - battery_p_pu
    forward_load = np.clip(signed_net, 0.0, None)
    if reverse_flow_loss_uplift <= 0.0:
        return forward_load

    reverse_export = np.clip(-signed_net, 0.0, None)
    reverse_equiv = np.sqrt(reverse_flow_loss_uplift) * reverse_export
    return np.maximum(forward_load, reverse_equiv)


# ---------------------------------------------------------------------------
# Single-run simulation
# ---------------------------------------------------------------------------

def run_single_mc(
    scenario: Scenario,
    run_idx: int,
    transformer: TransformerParams,
    use_surrogate: bool = True,
    surrogate_model: Any = None,
    surrogate_scaler: Any = None,
    n_hours: int = 8760,
    dt_hours: float = 1.0,
) -> dict[str, float | int | str]:
    """Execute one Monte Carlo realization and return result metrics.

    Parameters
    ----------
    scenario : Scenario
        Scenario definition (PV penetration, ambient delta, load growth).
    run_idx : int
        Run index; seed = scenario.base_seed + run_idx.
    transformer : TransformerParams
        Transformer parameters.
    use_surrogate : bool
        If True, use XGBoost surrogate for fast inference.
        If False, use ODE physics solver.
    surrogate_model, surrogate_scaler
        Required if use_surrogate=True.
    n_hours : int
        Simulation length in hours.
    dt_hours : float
        Timestep [hours].

    Returns
    -------
    dict
        One-row result with keys matching MC_RESULT_COLUMNS.
    """
    seed = scenario.base_seed + run_idx

    # Load growth multiplier
    load_scale = 1.0 + scenario.load_growth_pct_per_year / 100.0

    # Generate stochastic profiles via the Jember data layer (Phase 4C).
    # When real BMKG/PLN/atlite files are present, these loaders use them;
    # otherwise they fall back to Jember-tuned synthesis (still deterministic by seed).
    load_base = load_jember_load(n_hours, peak_pu=0.85, seed=seed)
    load_pu = np.clip(load_base * load_scale, 0.0, 1.5)
    ambient_c = load_jember_ambient(
        n_hours, delta_c=scenario.ambient_delta_c, seed=seed
    )
    pv_gen = load_jember_pv(
        n_hours, penetration_pu=scenario.pv_penetration, seed=seed
    )

    # Add EV charging load (zero if no EVs configured)
    ev_load = synthesize_ev_total(
        n_hours,
        slow_penetration_pu=scenario.ev_slow_penetration_pu,
        fast_penetration_pu=scenario.ev_fast_penetration_pu,
        seed=seed + 10_000,  # offset to avoid seed collision with load/PV
    )
    load_pu = np.clip(load_pu + ev_load, 0.0, 1.5)

    # Battery storage dispatch (no-op if capacity_mwh=0)
    battery_params = BatteryParams(
        capacity_mwh=scenario.battery_capacity_mwh,
        max_power_mw=scenario.battery_power_mw,
        discharge_threshold_pu=scenario.battery_threshold_pu,
        transformer_rated_mva=transformer.rated_mva,
        soc_init=0.50,
    )
    if scenario.dispatch_method == "pypsa_lp":
        batt_result = dispatch_battery_pypsa_lp(
            load_pu, pv_gen, battery_params, dt_hours=dt_hours,
            solver_name=scenario.pypsa_solver,
            scenario=scenario, transformer=transformer,
        )
    else:
        batt_result = dispatch_battery(load_pu, pv_gen, battery_params, dt_hours=dt_hours)
    net_load = batt_result["net_load"]
    thermal_load = effective_thermal_loading(
        load_pu,
        pv_gen,
        battery_p_pu=batt_result["battery_p_pu"],
        reverse_flow_loss_uplift=scenario.reverse_flow_loss_uplift,
    )
    battery_lifetime_frac = batt_result["lifetime_fraction_used"]

    # Apply harmonic loss uplift to transformer params if THD > 0
    thd = scenario.inverter_thd
    transformer_eff = (
        replace(transformer, R=transformer.R * (1.0 + 0.5 * thd ** 2))
        if thd > 0.0 else transformer
    )

    # Build cooling controller if the transformer has mode-switching params
    cooling_ctrl = None
    if transformer_eff.cooling_ctrl_params is not None:
        fans_ok = scenario.cooling_failure_mode != "fan_failed"
        cooling_ctrl = CoolingController(transformer_eff.cooling_ctrl_params, fans_ok=fans_ok)

    # Thermal simulation
    if use_surrogate:
        if surrogate_model is None or surrogate_scaler is None:
            raise ValueError("surrogate_model and surrogate_scaler must be provided "
                             "when use_surrogate=True.")
        from .surrogate import predict_trajectory
        result = predict_trajectory(
            surrogate_model, surrogate_scaler,
            load_pu, ambient_c, pv_gen,
            transformer=transformer_eff,
            dt_hours=dt_hours,
            net_load_override=thermal_load,
        )
        theta_hs = result["theta_hs"]
    else:
        result = simulate_thermal(
            thermal_load, ambient_c, transformer_eff, dt_min=dt_hours * 60.0,
            cooling_controller=cooling_ctrl,
        )
        theta_hs = result["theta_hs"]

    # Compute metrics (use moisture-dependent B if configured)
    moisture = transformer.insulation_water_pct
    f_aa = aging_acceleration_factor(theta_hs, moisture_water_pct=moisture)
    lol_pct = loss_of_life_percent(theta_hs, dt_hours=dt_hours, moisture_water_pct=moisture)
    lol_hours = lol_pct / 100.0 * 180_000.0

    # Reverse flow: hours when PV generation exceeds transformer load
    reverse_flow_hours = float(np.sum(pv_gen > load_pu) * dt_hours)

    return {
        "run_idx": run_idx,
        "scenario_name": scenario.name,
        "pv_penetration": scenario.pv_penetration,
        "ambient_delta_c": scenario.ambient_delta_c,
        "load_growth_pct": scenario.load_growth_pct_per_year,
        "peak_theta_hs_c": float(theta_hs.max()),
        "mean_theta_hs_c": float(theta_hs.mean()),
        "lol_percent_annual": float(lol_pct),
        "lol_hours_annual": float(lol_hours),
        "hours_above_110c": float(np.sum(theta_hs > 110.0) * dt_hours),
        "hours_above_120c": float(np.sum(theta_hs > 120.0) * dt_hours),
        "reverse_flow_hours": reverse_flow_hours,
        "peak_net_load_pu": float(net_load.max()),
        "mean_f_aa": float(f_aa.mean()),
        "battery_lifetime_fraction": battery_lifetime_frac,
        "dispatch_method": scenario.dispatch_method,
    }


# ---------------------------------------------------------------------------
# Full scenario sweep
# ---------------------------------------------------------------------------

def _run_scenario_batch(
    scenario: Scenario,
    transformer: TransformerParams,
    surrogate_model: Any,
    surrogate_scaler: Any,
    n_hours: int = 8760,
    dt_hours: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fast batch surrogate path: predict all n_runs simultaneously.

    Generates all profiles upfront, then calls predict_trajectories_batch which
    predicts every run's theta_HS in a single vectorised per-timestep XGBoost
    call.  Roughly 100x faster than the sequential single-run loop.
    """
    from .surrogate import predict_trajectories_batch

    n = scenario.n_runs
    load_scale = 1.0 + scenario.load_growth_pct_per_year / 100.0
    thd = scenario.inverter_thd
    transformer_eff = (
        replace(transformer, R=transformer.R * (1.0 + 0.5 * thd ** 2))
        if thd > 0.0 else transformer
    )
    # Note: CoolingController is not applied to the surrogate batch path here.
    # After Phase 2 surrogate retraining (scripts 02+03), the surrogate will
    # implicitly capture mode-switching dynamics in its training data.

    if verbose:
        print(f"  Generating {n} profiles ...", flush=True)

    load_arr = np.empty((n, n_hours), dtype=float)
    ambient_arr = np.empty((n, n_hours), dtype=float)
    pv_arr = np.empty((n, n_hours), dtype=float)

    for i in range(n):
        seed = scenario.base_seed + i
        # Phase 4C: route through Jember data loaders.  These prefer real data
        # (atlite cutout, BMKG CSV, PLN CSV) when present and otherwise fall
        # back to Jember-tuned synthesis — same numpy interface either way.
        base_load = np.clip(
            load_jember_load(n_hours, peak_pu=0.85, seed=seed) * load_scale,
            0.0, 1.5,
        )
        ev_load = synthesize_ev_total(
            n_hours,
            slow_penetration_pu=scenario.ev_slow_penetration_pu,
            fast_penetration_pu=scenario.ev_fast_penetration_pu,
            seed=seed + 10_000,
        )
        load_arr[i] = np.clip(base_load + ev_load, 0.0, 1.5)
        ambient_arr[i] = load_jember_ambient(
            n_hours, delta_c=scenario.ambient_delta_c, seed=seed
        )
        pv_arr[i] = load_jember_pv(
            n_hours, penetration_pu=scenario.pv_penetration, seed=seed
        )

    # Battery dispatch per-run (vectorised loop; no-op when capacity=0)
    battery_params = BatteryParams(
        capacity_mwh=scenario.battery_capacity_mwh,
        max_power_mw=scenario.battery_power_mw,
        discharge_threshold_pu=scenario.battery_threshold_pu,
        transformer_rated_mva=transformer_eff.rated_mva,
        soc_init=0.50,
    )
    net_load_arr = np.empty_like(load_arr)
    battery_lifetime_arr = np.zeros(n)
    use_lp = (scenario.dispatch_method == "pypsa_lp")
    if use_lp and verbose:
        print(f"  PyPSA LP dispatch ({n} runs, solver={scenario.pypsa_solver}) ...",
              flush=True)
    for i in range(n):
        if use_lp:
            br = dispatch_battery_pypsa_lp(
                load_arr[i], pv_arr[i], battery_params, dt_hours=dt_hours,
                solver_name=scenario.pypsa_solver,
                scenario=scenario, transformer=transformer_eff,
            )
        else:
            br = dispatch_battery(load_arr[i], pv_arr[i], battery_params, dt_hours=dt_hours)
        net_load_arr[i] = effective_thermal_loading(
            load_arr[i],
            pv_arr[i],
            battery_p_pu=br["battery_p_pu"],
            reverse_flow_loss_uplift=scenario.reverse_flow_loss_uplift,
        )
        battery_lifetime_arr[i] = br["lifetime_fraction_used"]

    if verbose:
        print(f"  Batch surrogate prediction ({n} x {n_hours} steps) ...", flush=True)

    result = predict_trajectories_batch(
        surrogate_model, surrogate_scaler, load_arr, ambient_arr, pv_arr,
        transformer=transformer_eff, dt_hours=dt_hours,
        net_load_arr=net_load_arr,
    )
    theta_hs_all = result["theta_hs"]  # (n, T)

    # Vectorised metrics (use moisture-dependent B if configured)
    moisture = transformer.insulation_water_pct
    f_aa_all = aging_acceleration_factor(theta_hs_all, moisture_water_pct=moisture)  # (n, T)
    lol_pct_all = f_aa_all.mean(axis=1) * n_hours * dt_hours / 180_000.0 * 100.0

    rows: list[dict] = []
    for i in range(n):
        lol_hours = lol_pct_all[i] / 100.0 * 180_000.0
        reverse_flow = float(np.sum(pv_arr[i] > load_arr[i]) * dt_hours)
        rows.append({
            "run_idx": i,
            "scenario_name": scenario.name,
            "pv_penetration": scenario.pv_penetration,
            "ambient_delta_c": scenario.ambient_delta_c,
            "load_growth_pct": scenario.load_growth_pct_per_year,
            "peak_theta_hs_c": float(theta_hs_all[i].max()),
            "mean_theta_hs_c": float(theta_hs_all[i].mean()),
            "lol_percent_annual": float(lol_pct_all[i]),
            "lol_hours_annual": float(lol_hours),
            "hours_above_110c": float(np.sum(theta_hs_all[i] > 110.0) * dt_hours),
            "hours_above_120c": float(np.sum(theta_hs_all[i] > 120.0) * dt_hours),
            "reverse_flow_hours": reverse_flow,
            "peak_net_load_pu": float(net_load_arr[i].max()),
            "mean_f_aa": float(f_aa_all[i].mean()),
            "battery_lifetime_fraction": float(battery_lifetime_arr[i]),
            "dispatch_method": scenario.dispatch_method,
        })

    return pd.DataFrame(rows, columns=MC_RESULT_COLUMNS)


def run_scenario(
    scenario: Scenario,
    transformer: TransformerParams,
    use_surrogate: bool = True,
    surrogate_model: Any = None,
    surrogate_scaler: Any = None,
    n_hours: int = 8760,
    dt_hours: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all Monte Carlo realizations for one scenario.

    Parameters
    ----------
    scenario : Scenario
        Scenario to simulate.
    transformer : TransformerParams
        Transformer parameters.
    use_surrogate : bool
        Use surrogate (True) or ODE (False).
    surrogate_model, surrogate_scaler
        Required if use_surrogate=True.
    n_hours : int
        Hours per simulation (default 8760 = 1 year).
    dt_hours : float
        Timestep [hours].
    verbose : bool
        Print progress every 100 runs.

    Returns
    -------
    pd.DataFrame
        One row per run, columns = MC_RESULT_COLUMNS.
    """
    # Fast path: batch all n_runs through the surrogate simultaneously
    if use_surrogate and surrogate_model is not None:
        return _run_scenario_batch(
            scenario=scenario,
            transformer=transformer,
            surrogate_model=surrogate_model,
            surrogate_scaler=surrogate_scaler,
            n_hours=n_hours,
            dt_hours=dt_hours,
            verbose=verbose,
        )

    # Slow path: sequential ODE (for ground-truth spot-checks)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i in range(scenario.n_runs):
        row = run_single_mc(
            scenario=scenario,
            run_idx=i,
            transformer=transformer,
            use_surrogate=False,
            surrogate_model=None,
            surrogate_scaler=None,
            n_hours=n_hours,
            dt_hours=dt_hours,
        )
        rows.append(row)
        if verbose and (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (scenario.n_runs - i - 1) / rate
            print(f"  {scenario.name}: {i+1}/{scenario.n_runs} runs "
                  f"({rate:.1f} runs/s, ~{remaining:.0f}s left)", flush=True)

    return pd.DataFrame(rows, columns=MC_RESULT_COLUMNS)


def run_all_scenarios(
    scenarios: list[Scenario],
    transformer: TransformerParams,
    use_surrogate: bool = True,
    surrogate_model: Any = None,
    surrogate_scaler: Any = None,
    n_hours: int = 8760,
    dt_hours: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full Monte Carlo sweep across all scenarios.

    Parameters
    ----------
    scenarios : list[Scenario]
    transformer : TransformerParams
    use_surrogate, surrogate_model, surrogate_scaler
        Passed to run_scenario().
    n_hours, dt_hours : int, float
    verbose : bool

    Returns
    -------
    pd.DataFrame
        All runs from all scenarios concatenated, columns = MC_RESULT_COLUMNS.
    """
    all_frames: list[pd.DataFrame] = []
    for j, sc in enumerate(scenarios):
        if verbose:
            print(f"\n[{j+1}/{len(scenarios)}] Scenario: {sc.name} "
                  f"(PV={sc.pv_penetration:.0%}, dT={sc.ambient_delta_c:+.1f}C, "
                  f"grow={sc.load_growth_pct_per_year:.1f}%/yr, "
                  f"n={sc.n_runs} runs)")
        df_sc = run_scenario(
            sc, transformer,
            use_surrogate=use_surrogate,
            surrogate_model=surrogate_model,
            surrogate_scaler=surrogate_scaler,
            n_hours=n_hours,
            dt_hours=dt_hours,
            verbose=verbose,
        )
        all_frames.append(df_sc)

    return pd.concat(all_frames, ignore_index=True)
