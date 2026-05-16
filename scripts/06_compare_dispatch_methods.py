"""Script 06: Compare threshold vs PyPSA-LP battery dispatch (Phase 4B).

Runs the named battery scenarios twice — once with the existing threshold-rule
heuristic, once with the PyPSA LP optimisation — and writes a side-by-side
comparison CSV for the paper.

The LP path is ~10× slower per run, so this script defaults to a reduced
``n_runs=100`` instead of the full 1000 used by script 04.

Usage
-----
    python scripts/06_compare_dispatch_methods.py [--n_runs 100] [--scenarios A,B,C]

Output
------
    results/tables/dispatch_comparison.csv
    results/tables/dispatch_comparison.parquet
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import replace
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from transformer_lol.monte_carlo import run_scenario
from transformer_lol.scenarios import get_named_scenarios
from transformer_lol.surrogate import load_surrogate
from transformer_lol.thermal_model import TransformerParams
from transformer_lol.utils import get_project_root, setup_logging, load_yaml


# Scenarios that have a non-trivial battery and benefit from a comparison.
DEFAULT_COMPARE_SCENARIOS: tuple[str, ...] = (
    "pv75_battery5mwh_today",
    "pv75_battery5mwh_ev30_warmer_2030",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare threshold vs PyPSA-LP dispatch.")
    parser.add_argument("--n_runs", type=int, default=100,
                        help="Reduced n_runs (LP is slow). Default 100.")
    parser.add_argument("--scenarios", type=str, default=None,
                        help=f"Comma-separated scenario names. "
                             f"Default: {','.join(DEFAULT_COMPARE_SCENARIOS)}")
    parser.add_argument("--n_hours", type=int, default=8760)
    args = parser.parse_args()

    target_names = (
        tuple(args.scenarios.split(","))
        if args.scenarios else DEFAULT_COMPARE_SCENARIOS
    )

    root = get_project_root()
    log = setup_logging(root / "results" / "logs")
    log.info("=== Script 06: Compare Dispatch Methods (Phase 4B) ===")

    cfg = load_yaml(root / "config" / "transformer_60mva.yaml")
    transformer = TransformerParams.from_yaml(cfg)
    log.info(f"Transformer: {transformer.rated_mva} MVA")

    surrogate_path = root / "results" / "surrogate_model.json"
    scaler_path = root / "results" / "surrogate_scaler.pkl"
    if not (surrogate_path.exists() and scaler_path.exists()):
        log.error("Surrogate model not found. Run scripts/03_train_surrogate.py first.")
        return 1
    model, scaler = load_surrogate(surrogate_path, scaler_path)

    all_named = get_named_scenarios()
    scenarios_by_name = {s.name: s for s in all_named}

    rows: list[pd.DataFrame] = []
    for name in target_names:
        sc_orig = scenarios_by_name.get(name)
        if sc_orig is None:
            log.warning(f"Scenario {name!r} not found in config; skipping.")
            continue
        if sc_orig.battery_capacity_mwh <= 0:
            log.warning(f"Scenario {name!r} has no battery; skipping.")
            continue

        for method in ("threshold", "pypsa_lp"):
            sc = replace(sc_orig, dispatch_method=method, n_runs=args.n_runs)
            log.info(f"  [{name}] dispatch={method} n_runs={args.n_runs} ...")
            t0 = time.perf_counter()
            df = run_scenario(
                sc, transformer,
                use_surrogate=True,
                surrogate_model=model, surrogate_scaler=scaler,
                n_hours=args.n_hours,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            log.info(f"    {len(df)} runs in {elapsed:.1f}s "
                     f"(LoL mean={df['lol_percent_annual'].mean():.4f}%, "
                     f"peak θ_HS={df['peak_theta_hs_c'].mean():.2f} C, "
                     f"battery cycles={df['battery_lifetime_fraction'].mean()*6000:.1f})")
            rows.append(df)

    if not rows:
        log.error("No results to write.")
        return 1

    df_all = pd.concat(rows, ignore_index=True)

    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dispatch_comparison.csv"
    parquet_path = out_dir / "dispatch_comparison.parquet"
    df_all.to_csv(csv_path, index=False)
    df_all.to_parquet(parquet_path, index=False)
    log.info(f"Wrote {csv_path} ({len(df_all)} rows)")

    # Summary table for the log
    summary = df_all.groupby(["scenario_name", "dispatch_method"]).agg(
        lol_pct_mean=("lol_percent_annual", "mean"),
        lol_pct_std=("lol_percent_annual", "std"),
        peak_ths_mean=("peak_theta_hs_c", "mean"),
        battery_lifetime_mean=("battery_lifetime_fraction", "mean"),
    ).round(4)
    print("\n" + "=" * 70)
    print("Dispatch comparison summary")
    print("=" * 70)
    print(summary.to_string())
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
