"""Script 04: Run Monte Carlo sweep across all scenarios.

Uses the trained XGBoost surrogate (default) or the physics ODE (--use_ode flag).
Saves results to results/tables/mc_results_all_scenarios.parquet + .csv.

Usage
-----
    python scripts/04_run_monte_carlo.py [--n_runs N] [--use_ode] [--scenario NAME]

Options
-------
    --n_runs N       Override n_runs for all scenarios (default: from config)
    --use_ode        Use ODE solver instead of surrogate (slow; for spot-checking)
    --scenario NAME  Run only the named scenario (default: all scenarios)

Output
------
    results/tables/mc_results_all_scenarios.parquet
    results/tables/mc_results_all_scenarios.csv
    results/metrics.json    (updated with Monte Carlo headline numbers)
"""

from __future__ import annotations

import argparse
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

from transformer_lol.monte_carlo import run_all_scenarios, run_scenario
from transformer_lol.scenarios import load_scenarios, get_named_scenarios
from transformer_lol.surrogate import load_surrogate
from transformer_lol.thermal_model import TransformerParams
from transformer_lol.utils import get_project_root, setup_logging, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Monte Carlo sweep")
    parser.add_argument("--n_runs", type=int, default=None,
                        help="Override n_runs for all scenarios")
    parser.add_argument("--use_ode", action="store_true",
                        help="Use ODE solver instead of surrogate (slow)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run only this named scenario")
    args = parser.parse_args()

    root = get_project_root()
    log = setup_logging(root / "results" / "logs")
    log.info("=== Script 04: Monte Carlo Sweep ===")

    # Load transformer parameters
    cfg = load_yaml(root / "config" / "transformer_60mva.yaml")
    transformer = TransformerParams.from_yaml(cfg)

    # Load surrogate (unless --use_ode)
    surrogate_model, surrogate_scaler = None, None
    if not args.use_ode:
        model_path = root / "results" / "surrogate_model.json"
        scaler_path = root / "results" / "surrogate_scaler.pkl"
        if not model_path.exists():
            log.error("Surrogate model not found. Run scripts/03_train_surrogate.py first.")
            log.error("Or use --use_ode flag to run with the physics solver.")
            return 1
        surrogate_model, surrogate_scaler = load_surrogate(model_path, scaler_path)
        log.info(f"Surrogate loaded from {model_path}")

    # Load scenarios
    if args.scenario:
        all_sc = load_scenarios()
        scenarios = [s for s in all_sc if s.name == args.scenario]
        if not scenarios:
            log.error(f"Scenario '{args.scenario}' not found.")
            log.error(f"Available: {[s.name for s in all_sc[:8]]} ...")
            return 1
    else:
        scenarios = load_scenarios()

    if args.n_runs is not None:
        for sc in scenarios:
            sc.n_runs = args.n_runs

    log.info(f"Running {len(scenarios)} scenarios, "
             f"{'ODE' if args.use_ode else 'surrogate'} mode")
    total_runs = sum(s.n_runs for s in scenarios)
    log.info(f"Total runs: {total_runs:,}")

    t0 = time.perf_counter()
    mc_df = run_all_scenarios(
        scenarios=scenarios,
        transformer=transformer,
        use_surrogate=not args.use_ode,
        surrogate_model=surrogate_model,
        surrogate_scaler=surrogate_scaler,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    log.info(f"Completed {len(mc_df):,} runs in {elapsed:.1f}s "
             f"({len(mc_df)/elapsed:.0f} runs/s)")

    # Save results
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "mc_results_all_scenarios.parquet"
    csv_path = out_dir / "mc_results_all_scenarios.csv"
    mc_df.to_parquet(parquet_path, index=False)
    mc_df.to_csv(csv_path, index=False)
    log.info(f"Results saved: {parquet_path}")

    # Compute headline numbers
    baseline_rows = mc_df[mc_df["pv_penetration"] == 0.0]
    high_pv_rows = mc_df[mc_df["pv_penetration"] == 0.75]
    jember_baseline_rows = mc_df[mc_df["scenario_name"] == "baseline"]
    jember_pv50_rows = mc_df[mc_df["scenario_name"] == "pv50_tropical_today"]
    jember_pv75_rows = mc_df[mc_df["scenario_name"] == "pv75_tropical_today"]

    # Turnover point: PV level where mean LoL is minimized (among dT=0 scenarios)
    dt0_rows = mc_df[mc_df["ambient_delta_c"] == 0.0]
    pivot = dt0_rows.groupby("pv_penetration")["lol_percent_annual"].mean()
    turnover_pv = float(pivot.idxmin()) if len(pivot) > 0 else float("nan")

    # Phase 1B exit gate: compare LoL at dry (1% moisture, B=15 000) vs wet (3% moisture, B=14 158).
    # Approximated from stored mean_f_aa: scale by exp((B2-B1) * (1/T_rated - 1/T_mean)).
    _b_dry_lol = None
    _b_wet_lol = None
    if len(baseline_rows) > 0:
        from transformer_lol.aging import B_CONSTANT, b_from_moisture
        B_WET = b_from_moisture(3.0)   # 14 158 K at 3% moisture
        mean_ths = float(baseline_rows["mean_theta_hs_c"].mean())
        inv_delta = 1.0 / (110.0 + 273.15) - 1.0 / (mean_ths + 273.15)
        _mean_faa_dry = float(baseline_rows["mean_f_aa"].mean())
        _mean_faa_wet = _mean_faa_dry * float(np.exp((B_WET - B_CONSTANT) * inv_delta))
        n_hours_yr = 8760
        _b_dry_lol = round(_mean_faa_dry * n_hours_yr / 180_000.0 * 100.0, 4)
        _b_wet_lol = round(_mean_faa_wet * n_hours_yr / 180_000.0 * 100.0, 4)

    metrics_path = root / "results" / "metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    metrics.update({
        "n_mc_runs_per_scenario": int(scenarios[0].n_runs) if scenarios else 0,
        "n_scenarios": len(scenarios),
        "total_runs": len(mc_df),
        "runtime_seconds": round(elapsed, 1),
        "legacy_all_zero_pv_lol_pct_mean": round(float(baseline_rows["lol_percent_annual"].mean()), 4)
                                           if len(baseline_rows) > 0 else None,
        "legacy_all_zero_pv_lol_pct_std": round(float(baseline_rows["lol_percent_annual"].std()), 4)
                                          if len(baseline_rows) > 0 else None,
        "legacy_all_pv75_lol_pct_mean": round(float(high_pv_rows["lol_percent_annual"].mean()), 4)
                                        if len(high_pv_rows) > 0 else None,
        "jember_baseline_lol_pct_mean": round(float(jember_baseline_rows["lol_percent_annual"].mean()), 4)
                                        if len(jember_baseline_rows) > 0 else None,
        "jember_baseline_lol_pct_std": round(float(jember_baseline_rows["lol_percent_annual"].std()), 4)
                                       if len(jember_baseline_rows) > 0 else None,
        "jember_pv50_today_lol_pct_mean": round(float(jember_pv50_rows["lol_percent_annual"].mean()), 4)
                                           if len(jember_pv50_rows) > 0 else None,
        "jember_pv75_today_lol_pct_mean": round(float(jember_pv75_rows["lol_percent_annual"].mean()), 4)
                                           if len(jember_pv75_rows) > 0 else None,
        "jember_pv50_today_lol_reduction_pct": round(
            (1.0 - float(jember_pv50_rows["lol_percent_annual"].mean())
             / float(jember_baseline_rows["lol_percent_annual"].mean())) * 100.0, 1
        ) if len(jember_baseline_rows) > 0 and len(jember_pv50_rows) > 0 else None,
        "jember_pv75_today_lol_reduction_pct": round(
            (1.0 - float(jember_pv75_rows["lol_percent_annual"].mean())
             / float(jember_baseline_rows["lol_percent_annual"].mean())) * 100.0, 1
        ) if len(jember_baseline_rows) > 0 and len(jember_pv75_rows) > 0 else None,
        "turnover_pv_penetration_pct": round(turnover_pv * 100, 1)
                                        if not np.isnan(turnover_pv) else None,
        "headline_with_constant_B": _b_dry_lol,
        "headline_with_wet_B": _b_wet_lol,
    })

    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info(f"Metrics updated: {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Monte Carlo Summary")
    print(f"{'='*60}")
    print(f"  Scenarios:       {len(scenarios)}")
    print(f"  Total runs:      {len(mc_df):,}")
    print(f"  Wall-clock:      {elapsed:.1f}s ({len(mc_df)/elapsed:.0f} runs/s)")
    if len(baseline_rows) > 0:
        print(f"  Baseline LoL:    {baseline_rows['lol_percent_annual'].mean():.3f}% (mean)")
    if len(high_pv_rows) > 0:
        print(f"  High-PV LoL:     {high_pv_rows['lol_percent_annual'].mean():.3f}% (mean)")
    if not np.isnan(turnover_pv):
        print(f"  Turnover point:  ~{turnover_pv*100:.0f}% PV penetration")
    print(f"\n[OK] Results saved to {parquet_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
