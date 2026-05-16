"""Script 02: Generate surrogate training data via physics ODE simulation.

Runs diverse (load, PV, ambient) trajectories through the IEEE C57.91
thermal model to produce a labelled dataset for XGBoost training.

Usage
-----
    python scripts/02_generate_training_data.py [--n_traj N] [--hours H]

Options
-------
    --n_traj N    Number of trajectories (default: 200)
    --hours H     Hours per trajectory (default: 168, i.e. 1 week)

Output
------
    data/processed/surrogate_training_data.parquet

The output contains FEATURE_NAMES + ['theta_hs_t', 'trajectory_id'] columns.
Row count ≈ n_traj * (hours - 2) after NaN rows are dropped.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Allow running from either transformer_lol_sim/ or its parent
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from transformer_lol.surrogate import build_training_set, FEATURE_NAMES
from transformer_lol.thermal_model import TransformerParams
from transformer_lol.utils import get_project_root, setup_logging, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate surrogate training data")
    parser.add_argument("--n_traj", type=int, default=200,
                        help="Number of trajectories (default: 200)")
    parser.add_argument("--hours", type=int, default=168,
                        help="Hours per trajectory (default: 168 = 1 week)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Master random seed (default: 0)")
    parser.add_argument("--dt_hours", type=float, default=1.0,
                        help="Timestep size in hours (default: 1.0)")
    args = parser.parse_args()

    root = get_project_root()
    log = setup_logging(root / "results" / "logs")
    log.info("=== Script 02: Generate Surrogate Training Data ===")

    # Load transformer parameters
    cfg = load_yaml(root / "config" / "transformer_60mva.yaml")
    transformer = TransformerParams.from_yaml(cfg)
    log.info(f"Transformer: {transformer.rated_mva} MVA {transformer.cooling_mode}")

    # Output path
    out_path = root / "data" / "processed" / "surrogate_training_data.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Generating {args.n_traj} trajectories x {args.hours} hours ...")
    t0 = time.perf_counter()

    df = build_training_set(
        n_trajectories=args.n_traj,
        hours_per_trajectory=args.hours,
        transformer=transformer,
        dt_hours=args.dt_hours,
        seed=args.seed,
    )

    elapsed = time.perf_counter() - t0
    log.info(f"Generated {len(df):,} rows in {elapsed:.1f}s")

    # Sanity checks
    import numpy as np
    if df["theta_hs_t"].isna().any():
        raise RuntimeError("NaN values in target column theta_hs_t.")
    ths_range = (df["theta_hs_t"].min(), df["theta_hs_t"].max())
    log.info(f"theta_hs_t range: {ths_range[0]:.1f} — {ths_range[1]:.1f} °C")
    if not (30.0 <= ths_range[0]) or not (ths_range[1] <= 200.0):
        log.warning("theta_hs_t outside expected physical range [30, 200] — check inputs")

    df.to_parquet(out_path, index=False)
    log.info(f"Saved to {out_path}")

    # Print summary statistics
    print("\nDataset summary:")
    print(f"  Rows:        {len(df):,}")
    print(f"  Trajectories:{df['trajectory_id'].nunique()}")
    print(f"  Features:    {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES[:4])} ...)")
    print(f"  Target:      theta_hs_t [{ths_range[0]:.1f}, {ths_range[1]:.1f}] degC")
    print(f"  Output:      {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
