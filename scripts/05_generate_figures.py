"""Script 05: Generate all 13 publication-quality figures and 3 CSV tables.

Reads Monte Carlo results from results/tables/mc_results_all_scenarios.parquet
and the trained surrogate for accuracy figures.  Does NOT re-run simulations.

Usage
-----
    python scripts/05_generate_figures.py

Output
------
    results/figures/fig01_system_schematic.png/.pdf
    results/figures/fig02_annex_g_validation.png/.pdf
    results/figures/fig03_daily_trajectories.png/.pdf
    results/figures/fig04_annual_lol_boxplots.png/.pdf
    results/figures/fig05_pv_lol_nonmonotonic.png/.pdf
    results/figures/fig06_sensitivity_heatmap.png/.pdf
    results/figures/fig07_surrogate_accuracy.png/.pdf
    results/figures/fig08_runtime_comparison.png/.pdf
    results/figures/fig09_cooling_mode_timeline.png/.pdf
    results/figures/fig10_battery_dispatch_timeline.png/.pdf
    results/figures/fig11_dispatch_comparison.png/.pdf
    results/figures/fig12_pypsa_network.png/.pdf
    results/figures/fig13_ev_battery_thermal.png/.pdf
    results/tables/table1_scenario_definitions.csv
    results/tables/table2_lol_statistics.csv
    results/tables/table3_surrogate_performance.csv
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from transformer_lol.plotting import (
    fig01_system_schematic,
    fig02_annex_g_validation,
    fig03_daily_trajectories,
    fig04_annual_lol_boxplots,
    fig05_pv_lol_nonmonotonic,
    fig06_sensitivity_heatmap,
    fig07_surrogate_accuracy,
    fig08_runtime_comparison,
    fig09_cooling_mode_timeline,
    fig10_battery_dispatch_timeline,
    fig11_dispatch_comparison,
    fig12_pypsa_network,
    fig13_ev_battery_thermal,
)
from transformer_lol.utils import get_project_root, setup_logging, load_yaml
from transformer_lol.thermal_model import TransformerParams


def main() -> int:
    root = get_project_root()
    log = setup_logging(root / "results" / "logs")
    log.info("=== Script 05: Generate Figures and Tables ===")

    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load MC results
    # -----------------------------------------------------------------------
    mc_path = tab_dir / "mc_results_all_scenarios.parquet"
    if not mc_path.exists():
        log.error(f"MC results not found: {mc_path}")
        log.error("Run scripts/04_run_monte_carlo.py first.")
        return 1

    mc_df = pd.read_parquet(mc_path)
    log.info(f"Loaded {len(mc_df):,} MC rows from {mc_path}")

    # -----------------------------------------------------------------------
    # Figures 1-3: standalone (no MC data needed)
    # -----------------------------------------------------------------------
    log.info("Generating fig01 (system schematic) ...")
    fig01_system_schematic(fig_dir)

    log.info("Generating fig02 (Annex G validation) ...")
    fig02_annex_g_validation(fig_dir)

    log.info("Generating fig03 (daily trajectories) ...")
    fig03_daily_trajectories(fig_dir)

    # -----------------------------------------------------------------------
    # Figures 4-6: require MC data
    # -----------------------------------------------------------------------
    log.info("Generating fig04 (LoL boxplots) ...")
    # Use only the 8 named scenarios for the boxplot — more readable than 38 boxes
    named = ["baseline", "pv10_tropical_today", "pv30_tropical_today",
             "pv50_tropical_today", "pv75_tropical_today",
             "pv30_warmer_2030", "pv50_warmer_2030", "pv50_warmer_2040"]
    mc_named = mc_df[mc_df["scenario_name"].isin(named)]
    fig04_annual_lol_boxplots(fig_dir, mc_named if len(mc_named) > 0 else mc_df)

    log.info("Generating fig05 (PV-LoL vs climate and load growth) ...")
    # Pass full mc_df — fig05 internally splits by load_growth_pct for its two panels
    fig05_pv_lol_nonmonotonic(fig_dir, mc_df)

    log.info("Generating fig06 (sensitivity heatmap) ...")
    fig06_sensitivity_heatmap(fig_dir, mc_df)

    # -----------------------------------------------------------------------
    # Figure 7: surrogate accuracy
    # -----------------------------------------------------------------------
    model_path = root / "results" / "surrogate_model.json"
    scaler_path = root / "results" / "surrogate_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        log.info("Generating fig07 (surrogate accuracy) ...")
        from transformer_lol.surrogate import (
            load_surrogate, build_training_set, FEATURE_NAMES
        )
        from transformer_lol.thermal_model import TransformerParams

        model, scaler = load_surrogate(model_path, scaler_path)
        cfg = load_yaml(root / "config" / "transformer_60mva.yaml")
        transformer = TransformerParams.from_yaml(cfg)

        # Generate a small held-out test set for the scatter plot
        test_df = build_training_set(n_trajectories=30, hours_per_trajectory=48,
                                      transformer=transformer, seed=99999)
        X_test = test_df[FEATURE_NAMES].values.astype(float)
        y_true = test_df["theta_hs_t"].values.astype(float)
        X_test_s = scaler.transform(X_test)
        X_test_df = pd.DataFrame(X_test_s, columns=FEATURE_NAMES)
        y_pred = model.predict(X_test_df)
        fig07_surrogate_accuracy(fig_dir, y_true, y_pred)
    else:
        log.warning("Surrogate model not found — skipping fig07. "
                    "Run scripts/03_train_surrogate.py first.")

    # -----------------------------------------------------------------------
    # Figure 8: runtime comparison
    # -----------------------------------------------------------------------
    log.info("Benchmarking ODE vs surrogate runtimes ...")
    timing_data = _benchmark_runtimes(root)
    fig08_runtime_comparison(fig_dir, timing_data)

    # -----------------------------------------------------------------------
    # Figure 9: cooling-mode timeline (Phase 2 addition)
    # -----------------------------------------------------------------------
    log.info("Generating fig09 (cooling-mode timeline) ...")
    cfg_t = load_yaml(root / "config" / "transformer_60mva.yaml")
    transformer_fig09 = TransformerParams.from_yaml(cfg_t)
    fig09_cooling_mode_timeline(fig_dir, transformer_fig09, n_days=7, pv_penetration=0.75)

    # -----------------------------------------------------------------------
    # Figure 10: battery dispatch timeline (Phase 3B addition)
    # -----------------------------------------------------------------------
    log.info("Generating fig10 (battery dispatch timeline) ...")
    fig10_battery_dispatch_timeline(
        fig_dir,
        pv_penetration=0.75,
        battery_capacity_mwh=5.0,
        battery_power_mw=2.0,
        transformer_rated_mva=transformer_fig09.rated_mva,
        seed=42,
    )

    # -----------------------------------------------------------------------
    # Figure 11 + 12: PyPSA dispatch comparison + network schematic (Phase 4B)
    # -----------------------------------------------------------------------
    cmp_path = tab_dir / "dispatch_comparison.csv"
    if cmp_path.exists():
        log.info("Generating fig11 (threshold vs PyPSA-LP dispatch comparison) ...")
        df_compare = pd.read_csv(cmp_path)
        fig11_dispatch_comparison(fig_dir, df_compare)
    else:
        log.warning(f"{cmp_path.name} not found — run scripts/06_compare_dispatch_methods.py "
                    "to enable fig11.")

    log.info("Generating fig12 (PyPSA network schematic) ...")
    fig12_pypsa_network(fig_dir)

    log.info("Generating fig13 (EV battery thermal comparison) ...")
    fig13_ev_battery_thermal(fig_dir, transformer_fig09)

    # -----------------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------------
    log.info("Generating CSV tables ...")

    # Table 1: scenario definitions
    scen_df = mc_df.drop_duplicates(subset="scenario_name")[
        ["scenario_name", "pv_penetration", "ambient_delta_c", "load_growth_pct"]
    ].copy()
    scen_df.columns = ["scenario", "pv_penetration", "ambient_delta_c", "load_growth_pct_yr"]
    scen_df.to_csv(tab_dir / "table1_scenario_definitions.csv", index=False)

    # Table 2: LoL statistics per scenario
    stats = mc_df.groupby("scenario_name").agg(
        pv_penetration=("pv_penetration", "first"),
        ambient_delta_c=("ambient_delta_c", "first"),
        n_runs=("run_idx", "count"),
        lol_pct_mean=("lol_percent_annual", "mean"),
        lol_pct_std=("lol_percent_annual", "std"),
        lol_pct_p5=("lol_percent_annual", lambda x: x.quantile(0.05)),
        lol_pct_p95=("lol_percent_annual", lambda x: x.quantile(0.95)),
        peak_ths_mean=("peak_theta_hs_c", "mean"),
        reverse_flow_h_mean=("reverse_flow_hours", "mean"),
    ).reset_index()
    stats = stats.round(4)
    stats.to_csv(tab_dir / "table2_lol_statistics.csv", index=False)

    # Table 3: surrogate performance metrics
    metrics_path = root / "results" / "metrics.json"
    surr_rows = []
    if metrics_path.exists():
        with open(metrics_path) as fh:
            metrics = json.load(fh)
        for key in ("surrogate_rmse_c", "surrogate_r2", "surrogate_lol_abs_err_pct"):
            if key in metrics:
                surr_rows.append({"metric": key, "value": metrics[key]})
    surr_df = pd.DataFrame(surr_rows)
    surr_df.to_csv(tab_dir / "table3_surrogate_performance.csv", index=False)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    figs = sorted(fig_dir.glob("fig*.png"))
    print(f"\n{'='*60}")
    print(f"Figures generated: {len(figs)}/13")
    for f in figs:
        print(f"  {f.name}")
    tabs = sorted(tab_dir.glob("table*.csv"))
    print(f"\nTables generated: {len(tabs)}/3")
    for t in tabs:
        print(f"  {t.name}")
    print(f"\n[OK] All outputs written to results/")
    return 0


def _benchmark_runtimes(root: Path) -> dict[str, dict[int, float]]:
    """Time ODE (sequential) vs surrogate (batched MC) for 1, 100, 1000 runs."""
    from transformer_lol.thermal_model import TransformerParams, simulate_thermal
    from transformer_lol.load_profile import synthesize_fallback_load
    from transformer_lol.ambient_profile import synthesize_tropical_ambient
    from transformer_lol.pv_synthesis import synthesize_pv_profile
    from transformer_lol.scenarios import Scenario
    from transformer_lol.monte_carlo import run_scenario

    transformer = TransformerParams()
    n_hours = 8760

    # Time one ODE run (single call to simulate_thermal)
    load = synthesize_fallback_load(n_hours, seed=0)
    ambient = synthesize_tropical_ambient(n_hours, seed=0)
    t0 = time.perf_counter()
    simulate_thermal(load, ambient, transformer, dt_min=60.0)
    ode_per_run = time.perf_counter() - t0

    # Time batch surrogate MC for 1, 10, 100 runs; extrapolate to 1000
    timing_data: dict[str, dict[int, float]] = {
        "ODE (RK45)": {1: ode_per_run, 100: 100 * ode_per_run, 1000: 1000 * ode_per_run},
    }

    model_path = root / "results" / "surrogate_model.json"
    scaler_path = root / "results" / "surrogate_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        from transformer_lol.surrogate import load_surrogate
        model, scaler = load_surrogate(model_path, scaler_path)

        # Measure batch surrogate: time at n=10 and n=100, extrapolate to 1000
        results_surr: dict[int, float] = {}
        for n_bench in (10, 100):
            sc_b = Scenario(name="bench", pv_penetration=0.3, ambient_delta_c=0.0,
                            load_growth_pct_per_year=0.0, n_runs=n_bench, base_seed=77)
            t0 = time.perf_counter()
            run_scenario(sc_b, transformer, use_surrogate=True,
                         surrogate_model=model, surrogate_scaler=scaler,
                         n_hours=n_hours, verbose=False)
            results_surr[n_bench] = time.perf_counter() - t0

        # Extrapolate to 1000 using measured scaling: t ≈ a + b*n
        # Solve: t10 = a + 10b, t100 = a + 100b → b = (t100-t10)/90, a = t10-10b
        t10, t100 = results_surr[10], results_surr[100]
        b = (t100 - t10) / 90.0
        a = t10 - 10 * b
        t1000 = a + 1000 * b

        timing_data["XGBoost Surrogate\n(batch MC)"] = {
            1: results_surr[10] / 10.0,
            100: results_surr[100],
            1000: max(t1000, results_surr[100] * 3),  # conservative lower bound
        }
    return timing_data


if __name__ == "__main__":
    sys.exit(main())
