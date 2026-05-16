"""PyPSA Network builder for the Jember 60 MVA two-bus case study.

Phase 4A of the PyPSA migration roadmap.  Constructs a standard
PyPSA Network from existing time-series profiles so that the power-system
layer becomes inspectable, exportable, and (in Phase 4B) optimisable.

Network topology
----------------
::

    grid_150kv ──[Transformer 60 MVA ONAF]── mv_20kv
                                                ├── Generator: pv_jember
                                                ├── Load: demand_jember (load + EV)
                                                └── StorageUnit: battery (optional)

The slack generator at ``grid_150kv`` represents the upstream HV grid;
its marginal cost is set high enough that PV is always used first.

The constructed Network is consumed by ``pypsa_dispatch.dispatch_battery_pypsa_lp``
(Phase 4B) and by visualisation helpers in ``plotting.fig12_pypsa_network``
(Phase 4D).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import pypsa


# Standard reference values — IEC 60076 typical defaults for a 60 MVA OD distribution transformer.
# These are NOT used by the LP economic dispatch (no losses or voltage limits enforced),
# only by power-flow analysis if added later.
DEFAULT_TRANSFORMER_X: float = 0.10   # per-unit reactance
DEFAULT_TRANSFORMER_R: float = 0.005  # per-unit resistance

# Allow the LP to push up to 2 × rated through the transformer.  The IEEE C57.91 thermal
# model handles overloads up to ~K = 1.5 by accepting extra aging; we don't want PyPSA's
# power-flow s_nom constraint to make the LP infeasible for the same operating regime.
DEFAULT_TRANSFORMER_S_MAX_PU: float = 2.0

# PyPSA splits round-trip efficiency in two: η_store × η_dispatch.  Setting both to 0.95
# yields 0.9025 ≈ 0.90 (matches battery.BATTERY_EFFICIENCY within 0.3 %).
PYPSA_BATTERY_ETA_STORE: float = 0.95
PYPSA_BATTERY_ETA_DISPATCH: float = 0.95

# Tiny non-zero marginal cost on the battery discourages pointless cycling at zero shadow price.
PYPSA_BATTERY_MARGINAL_COST: float = 0.01

# Slack-bus generator marginal cost.  Larger than PV (=0) so PV is dispatched first;
# smaller than load-shedding cost so the slack always serves residual demand.
SLACK_MARGINAL_COST: float = 1.0
SLACK_P_NOM_MW: float = 1.0e4   # large enough to be effectively unconstrained


def build_pypsa_network(
    scenario: Any,
    transformer: Any,
    profiles: Mapping[str, np.ndarray],
    snapshots: pd.DatetimeIndex | None = None,
    network_name: str = "jember_60mva",
) -> pypsa.Network:
    """Build the Jember 60 MVA HV/MV PyPSA network for one scenario.

    Parameters
    ----------
    scenario : Scenario
        Scenario dataclass providing PV penetration, EV penetration, and
        battery sizing (capacity_mwh, power_mw, threshold_pu).
    transformer : TransformerParams
        Transformer parameters; only ``rated_mva`` is used by the network model.
    profiles : Mapping[str, np.ndarray]
        Hourly time-series, all shape (T,):
            ``load_pu``  : gross load (p.u. of transformer MVA), excluding EV.
            ``pv_pu``    : PV generation (p.u.).
            ``ev_pu``    : EV charging load (p.u.); pass zeros if no EV.
        All arrays must have the same length T.
    snapshots : pd.DatetimeIndex, optional
        Snapshot timestamps (must be timezone-naive — PyPSA ≥ 1.0 rejects
        tz-aware snapshots).  Timezone alignment is handled upstream by the
        data loaders; snapshots here are pure ordering labels.
        Defaults to T hourly snapshots starting at 2024-01-01 00:00.
    network_name : str
        ``Network.name`` attribute.

    Returns
    -------
    pypsa.Network
        Network with two buses, one transformer, two generators (slack + PV),
        one load, and (if ``scenario.battery_capacity_mwh > 0``) one storage
        unit.  Snapshots are set; component time-series are populated.

    Notes
    -----
    The Network is *not* optimised here.  Call ``dispatch_battery_pypsa_lp``
    (Phase 4B) to run an LP dispatch and extract net_load.

    The slack generator ensures the LP is always feasible regardless of
    renewable shortfall; in normal operation it carries the residual load
    that PV + battery cannot supply.
    """
    load_pu = np.asarray(profiles["load_pu"], dtype=float)
    pv_pu = np.asarray(profiles["pv_pu"], dtype=float)
    ev_pu = np.asarray(profiles.get("ev_pu", np.zeros_like(load_pu)), dtype=float)

    if not (len(load_pu) == len(pv_pu) == len(ev_pu)):
        raise ValueError(
            f"profiles must have equal length; got load={len(load_pu)}, "
            f"pv={len(pv_pu)}, ev={len(ev_pu)}"
        )
    T = len(load_pu)

    if snapshots is None:
        snapshots = pd.date_range("2024-01-01 00:00", periods=T, freq="h")
    else:
        if len(snapshots) != T:
            raise ValueError(f"snapshots length {len(snapshots)} != profile length {T}")
        if getattr(snapshots, "tz", None) is not None:
            # PyPSA ≥ 1.0 rejects tz-aware snapshots; strip the timezone here
            # so callers can still pass tz-aware indexes from upstream loaders.
            snapshots = snapshots.tz_localize(None)

    rated_mva = float(transformer.rated_mva)

    n = pypsa.Network()
    n.name = network_name
    n.set_snapshots(snapshots)

    # --- Buses ---------------------------------------------------------------
    n.add("Bus", "grid_150kv", v_nom=150.0)
    n.add("Bus", "mv_20kv", v_nom=20.0)

    # --- Transformer ---------------------------------------------------------
    n.add(
        "Transformer", "tfr_60mva",
        bus0="grid_150kv", bus1="mv_20kv",
        s_nom=rated_mva,
        s_max_pu=DEFAULT_TRANSFORMER_S_MAX_PU,
        x=DEFAULT_TRANSFORMER_X,
        r=DEFAULT_TRANSFORMER_R,
        model="t",
    )

    # --- Slack generator on HV grid -----------------------------------------
    n.add(
        "Generator", "grid_slack",
        bus="grid_150kv",
        p_nom=SLACK_P_NOM_MW,
        marginal_cost=SLACK_MARGINAL_COST,
        control="Slack",
    )

    # --- PV generator on MV bus ---------------------------------------------
    pv_penetration = float(getattr(scenario, "pv_penetration", 0.0))
    pv_p_nom = pv_penetration * rated_mva
    if pv_p_nom > 0:
        # p_max_pu is fraction of p_nom available at each snapshot; pv_pu is fraction
        # of transformer MVA, so divide by penetration to express it as fraction of p_nom.
        p_max_pu = np.clip(pv_pu / pv_penetration, 0.0, 1.0)
        n.add(
            "Generator", "pv_jember",
            bus="mv_20kv",
            p_nom=pv_p_nom,
            marginal_cost=0.0,
            p_max_pu=pd.Series(p_max_pu, index=snapshots),
        )
    else:
        # Zero-PV scenario: still register the component with p_nom=0 so consumers
        # always see a 'pv_jember' generator, even if it produces nothing.
        n.add(
            "Generator", "pv_jember",
            bus="mv_20kv",
            p_nom=0.0,
            marginal_cost=0.0,
        )

    # --- Load on MV bus ------------------------------------------------------
    total_demand_pu = load_pu + ev_pu
    n.add(
        "Load", "demand_jember",
        bus="mv_20kv",
        p_set=pd.Series(total_demand_pu * rated_mva, index=snapshots),
    )

    # --- Battery storage (optional) -----------------------------------------
    cap_mwh = float(getattr(scenario, "battery_capacity_mwh", 0.0))
    p_mw = float(getattr(scenario, "battery_power_mw", 0.0))
    if cap_mwh > 0 and p_mw > 0:
        max_hours = cap_mwh / p_mw
        # Initial SoC as a fraction of usable capacity.  Defaults to 0.5;
        # callers (e.g. dispatch_battery_pypsa_lp) can override via
        # ``scenario.battery_soc_init_pu`` to mirror the heuristic's soc_init.
        soc_init_pu = float(getattr(scenario, "battery_soc_init_pu", 0.5))
        soc_init_pu = max(0.0, min(1.0, soc_init_pu))
        n.add(
            "StorageUnit", "battery",
            bus="mv_20kv",
            p_nom=p_mw,
            max_hours=max_hours,
            efficiency_store=PYPSA_BATTERY_ETA_STORE,
            efficiency_dispatch=PYPSA_BATTERY_ETA_DISPATCH,
            state_of_charge_initial=soc_init_pu * cap_mwh,
            cyclic_state_of_charge=False,
            marginal_cost=PYPSA_BATTERY_MARGINAL_COST,
        )

    return n


def network_summary(n: pypsa.Network) -> dict[str, Any]:
    """Return a small dict describing the Network — useful for tests and logs."""
    return {
        "name": n.name,
        "n_snapshots": int(len(n.snapshots)),
        "buses": list(n.buses.index),
        "transformers": list(n.transformers.index),
        "generators": list(n.generators.index),
        "loads": list(n.loads.index),
        "storage_units": list(n.storage_units.index),
        "transformer_s_nom_mva": float(n.transformers.loc["tfr_60mva", "s_nom"])
            if "tfr_60mva" in n.transformers.index else None,
        "pv_p_nom_mw": float(n.generators.loc["pv_jember", "p_nom"])
            if "pv_jember" in n.generators.index else None,
        "battery_p_nom_mw": float(n.storage_units.loc["battery", "p_nom"])
            if "battery" in n.storage_units.index else 0.0,
    }
