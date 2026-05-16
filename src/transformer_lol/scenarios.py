"""Scenario grid builder for Monte Carlo sweep.

Loads named scenarios from config/scenarios.yaml and programmatically
expands the full parameter grid for the paper sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .utils import load_yaml, get_project_root


@dataclass
class Scenario:
    """One simulation scenario definition.

    Parameters
    ----------
    name : str
        Unique scenario identifier (used as DataFrame column group label).
    pv_penetration : float
        PV capacity as fraction of transformer rated MVA (0–1).
    ambient_delta_c : float
        Climate scenario temperature offset added to ambient profile [°C].
    load_growth_pct_per_year : float
        Annual load growth rate [%/year].  Applied as a uniform multiplier
        over the simulation year: effective_K = base_K * (1 + growth/100).
    inverter_thd : float
        Total harmonic distortion fraction of PV inverter output current (0–1).
        Passed to simulate_thermal(); R_eff = R*(1+0.5*THD²) per IEC 61378-1.
        Default 0.05 (5% THD, typical for modern string inverters).
    reverse_flow_loss_uplift : float
        Optional sensitivity knob for directional reverse-flow losses.  When
        positive, reverse export magnitude is converted to an equivalent
        thermal loading using sqrt(uplift) scaling.
    cooling_failure_mode : str
        Fan failure scenario.  "none" = fans always operational (default).
        "fan_failed" = fans permanently failed → transformer runs in ONAN.
    ev_slow_penetration_pu : float
        Peak aggregate slow-overnight EV charging load [p.u.].  0 = no EVs.
    ev_fast_penetration_pu : float
        Peak aggregate fast-opportunistic EV charging load [p.u.].  0 = no EVs.
    battery_capacity_mwh : float
        Battery storage capacity [MWh].  0 = no battery.
    battery_power_mw : float
        Battery maximum charge/discharge power [MW].  0 = no battery.
    battery_threshold_pu : float
        Discharge trigger: battery discharges when net load > threshold [p.u.].
    n_runs : int
        Number of Monte Carlo realizations for this scenario.
    base_seed : int
        Base random seed; run i uses seed = base_seed + i.
    description : str
        Human-readable label for figure legends.
    """
    name: str
    pv_penetration: float
    ambient_delta_c: float = 0.0
    load_growth_pct_per_year: float = 0.0
    inverter_thd: float = 0.05
    reverse_flow_loss_uplift: float = 0.0
    cooling_failure_mode: str = "none"
    ev_slow_penetration_pu: float = 0.0
    ev_fast_penetration_pu: float = 0.0
    battery_capacity_mwh: float = 0.0
    battery_power_mw: float = 0.0
    battery_threshold_pu: float = 0.85
    dispatch_method: str = "threshold"   # "threshold" (heuristic) | "pypsa_lp" (Phase 4B)
    pypsa_solver: str = "highs"          # solver passed to Network.optimize()
    n_runs: int = 1000
    base_seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.pv_penetration <= 1.5):
            raise ValueError(f"pv_penetration {self.pv_penetration} out of [0, 1.5]")
        if self.n_runs < 1:
            raise ValueError("n_runs must be >= 1")
        if self.reverse_flow_loss_uplift < 0.0:
            raise ValueError("reverse_flow_loss_uplift must be non-negative")
        if self.cooling_failure_mode not in ("none", "fan_failed"):
            raise ValueError(
                f"cooling_failure_mode must be 'none' or 'fan_failed', "
                f"got {self.cooling_failure_mode!r}"
            )
        if self.dispatch_method not in ("threshold", "pypsa_lp"):
            raise ValueError(
                f"dispatch_method must be 'threshold' or 'pypsa_lp', "
                f"got {self.dispatch_method!r}"
            )


def load_scenarios(
    config_path: str | Path | None = None,
) -> list[Scenario]:
    """Load all scenarios from config/scenarios.yaml.

    Returns the named scenarios AND the full programmatic grid
    (PV × ambient_delta × load_growth), deduplicating any overlaps.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to scenarios.yaml.  Defaults to config/scenarios.yaml.

    Returns
    -------
    list[Scenario]
        All unique scenarios, named scenarios first, then grid entries.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "scenarios.yaml"
    cfg = load_yaml(config_path)

    mc_defaults = cfg.get("mc_defaults", {})
    n_runs_default = mc_defaults.get("n_runs", 1000)
    base_seed_default = mc_defaults.get("base_seed", 42)
    thd_default = float(mc_defaults.get("inverter_thd", 0.05))
    reverse_uplift_default = float(mc_defaults.get("reverse_flow_loss_uplift", 0.0))
    fail_default = str(mc_defaults.get("cooling_failure_mode", "none"))
    dispatch_default = str(mc_defaults.get("dispatch_method", "threshold"))
    solver_default = str(mc_defaults.get("pypsa_solver", "highs"))

    seen_names: set[str] = set()
    scenarios: list[Scenario] = []

    # Named scenarios first (from YAML 'scenarios' section)
    for name, params in cfg.get("scenarios", {}).items():
        if name in seen_names:
            continue
        scenarios.append(Scenario(
            name=name,
            pv_penetration=float(params.get("pv_penetration", 0.0)),
            ambient_delta_c=float(params.get("ambient_delta_c", 0.0)),
            load_growth_pct_per_year=float(params.get("load_growth_pct_per_year", 0.0)),
            inverter_thd=float(params.get("inverter_thd", thd_default)),
            reverse_flow_loss_uplift=float(
                params.get("reverse_flow_loss_uplift", reverse_uplift_default)
            ),
            cooling_failure_mode=str(params.get("cooling_failure_mode", fail_default)),
            ev_slow_penetration_pu=float(params.get("ev_slow_penetration_pu", 0.0)),
            ev_fast_penetration_pu=float(params.get("ev_fast_penetration_pu", 0.0)),
            battery_capacity_mwh=float(params.get("battery_capacity_mwh", 0.0)),
            battery_power_mw=float(params.get("battery_power_mw", 0.0)),
            battery_threshold_pu=float(params.get("battery_threshold_pu", 0.85)),
            dispatch_method=str(params.get("dispatch_method", dispatch_default)),
            pypsa_solver=str(params.get("pypsa_solver", solver_default)),
            n_runs=int(params.get("n_runs", n_runs_default)),
            base_seed=int(params.get("base_seed", base_seed_default)),
            description=params.get("description", ""),
        ))
        seen_names.add(name)

    # Full grid expansion
    grid = cfg.get("grid", {})
    pv_levels = grid.get("pv_levels", [0.0, 0.10, 0.30, 0.50, 0.75])
    ambient_deltas = grid.get("ambient_deltas", [0.0, 1.0, 2.0])
    load_growths = grid.get("load_growths", [0.0, 4.5])

    for pv in pv_levels:
        for da in ambient_deltas:
            for lg in load_growths:
                name = _make_grid_name(pv, da, lg)
                if name in seen_names:
                    continue
                scenarios.append(Scenario(
                    name=name,
                    pv_penetration=float(pv),
                    ambient_delta_c=float(da),
                    load_growth_pct_per_year=float(lg),
                    inverter_thd=thd_default,
                    reverse_flow_loss_uplift=reverse_uplift_default,
                    cooling_failure_mode=fail_default,
                    n_runs=n_runs_default,
                    base_seed=base_seed_default,
                    description=f"PV={pv*100:.0f}%, dT={da:+.1f}C, grow={lg:.1f}%/yr",
                ))
                seen_names.add(name)

    return scenarios


def _make_grid_name(pv: float, da: float, lg: float) -> str:
    """Generate a deterministic scenario name from grid parameters."""
    pv_str = f"pv{int(round(pv * 100)):03d}"
    da_str = f"dt{int(round(da * 10)):02d}"
    lg_str = f"gr{int(round(lg * 10)):03d}"
    return f"{pv_str}_{da_str}_{lg_str}"


def get_named_scenarios(config_path: str | Path | None = None) -> list[Scenario]:
    """Return only the named (non-grid) scenarios from the YAML file.

    Useful for the paper's narrative scenarios where human-readable labels matter.
    """
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "scenarios.yaml"
    cfg = load_yaml(config_path)
    mc_defaults = cfg.get("mc_defaults", {})
    n_runs_default = mc_defaults.get("n_runs", 1000)
    base_seed_default = mc_defaults.get("base_seed", 42)
    thd_default = float(mc_defaults.get("inverter_thd", 0.05))
    reverse_uplift_default = float(mc_defaults.get("reverse_flow_loss_uplift", 0.0))
    fail_default = str(mc_defaults.get("cooling_failure_mode", "none"))
    dispatch_default = str(mc_defaults.get("dispatch_method", "threshold"))
    solver_default = str(mc_defaults.get("pypsa_solver", "highs"))

    return [
        Scenario(
            name=name,
            pv_penetration=float(p.get("pv_penetration", 0.0)),
            ambient_delta_c=float(p.get("ambient_delta_c", 0.0)),
            load_growth_pct_per_year=float(p.get("load_growth_pct_per_year", 0.0)),
            inverter_thd=float(p.get("inverter_thd", thd_default)),
            reverse_flow_loss_uplift=float(
                p.get("reverse_flow_loss_uplift", reverse_uplift_default)
            ),
            cooling_failure_mode=str(p.get("cooling_failure_mode", fail_default)),
            ev_slow_penetration_pu=float(p.get("ev_slow_penetration_pu", 0.0)),
            ev_fast_penetration_pu=float(p.get("ev_fast_penetration_pu", 0.0)),
            battery_capacity_mwh=float(p.get("battery_capacity_mwh", 0.0)),
            battery_power_mw=float(p.get("battery_power_mw", 0.0)),
            battery_threshold_pu=float(p.get("battery_threshold_pu", 0.85)),
            dispatch_method=str(p.get("dispatch_method", dispatch_default)),
            pypsa_solver=str(p.get("pypsa_solver", solver_default)),
            n_runs=int(p.get("n_runs", n_runs_default)),
            base_seed=int(p.get("base_seed", base_seed_default)),
            description=p.get("description", ""),
        )
        for name, p in cfg.get("scenarios", {}).items()
    ]
