"""Script 06: Export predicted-vs-simulated theta_HS pairs for the companion website.

Reads the trained surrogate (results/surrogate_model.json + scaler) and
re-runs the same 30-trajectory hold-out used by script 05 to produce a tidy
CSV that the Next.js website turns into the Fig. 2 scatter/CDF charts.

Output: results/tables/surrogate_holdout.csv with columns
    theta_hs_true, theta_hs_pred, abs_error

Usage
-----
    python scripts/06_export_holdout_predictions.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from transformer_lol.surrogate import (
    FEATURE_NAMES,
    load_surrogate,
    _build_single_step_features,
)
from transformer_lol.thermal_model import TransformerParams, simulate_thermal
from transformer_lol.ambient_profile import synthesize_tropical_ambient
from transformer_lol.load_profile import synthesize_fallback_load
from transformer_lol.pv_synthesis import synthesize_pv_profile
from transformer_lol.utils import get_project_root, load_yaml


def main() -> int:
    root = get_project_root()
    print("=== Script 06: Export holdout predictions for the companion website ===")

    cfg = load_yaml(root / "config" / "transformer_60mva.yaml")
    transformer = TransformerParams.from_yaml(cfg)

    model_path = root / "results" / "surrogate_model.json"
    scaler_path = root / "results" / "surrogate_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        print(f"[FAIL] surrogate model not found at {model_path}")
        print("       Run scripts/03_train_surrogate.py first.")
        return 1

    model, scaler = load_surrogate(model_path, scaler_path)

    # 30 trajectories from a seed range not used in training (training used 0..199,
    # script 03 LoL validation used 9999..). We pick a fresh range here.
    n_holdout = 30
    n_hours = 168
    seed_base = 50_000

    rows = []
    # Teacher-forced one-step-ahead evaluation, matching the script-05 hold-out
    # reported in the paper (~0.25 degC RMSE). The surrogate receives the TRUE
    # theta_HS/theta_TO lags from the ODE at each step (not its own prior
    # predictions), so this isolates the model's regression accuracy from
    # autoregressive drift.
    warm = 3
    for i in range(n_holdout):
        s = seed_base + i
        pv_pen = float(np.random.default_rng(s).uniform(0.0, 0.75))

        load = synthesize_fallback_load(n_hours, seed=s)
        ambient = synthesize_tropical_ambient(n_hours, seed=s)
        pv = synthesize_pv_profile(n_hours, penetration_pu=pv_pen, seed=s)
        net_load = np.clip(load - pv, 0.0, 1.5)

        truth = simulate_thermal(net_load, ambient, transformer, dt_min=60.0)
        theta_hs_true = truth["theta_hs"]
        theta_to_true = truth["theta_to"]

        for t in range(warm, n_hours):
            feat = _build_single_step_features(
                t=t,
                load_pu=net_load,
                ambient_c=ambient,
                pv_share=pv,
                theta_hs_arr=theta_hs_true,
                theta_to_arr=theta_to_true,
                dt_hours=1.0,
            )
            feat_s = scaler.transform(feat)
            feat_df = pd.DataFrame(feat_s, columns=FEATURE_NAMES)
            pred_val = float(model.predict(feat_df)[0])
            true_val = float(theta_hs_true[t])
            rows.append({
                "trajectory": i,
                "hour": t,
                "theta_hs_true": true_val,
                "theta_hs_pred": pred_val,
                "abs_error": abs(pred_val - true_val),
            })

    df = pd.DataFrame(rows)
    out_path = root / "results" / "tables" / "surrogate_holdout.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    rmse = float(np.sqrt(((df.theta_hs_pred - df.theta_hs_true) ** 2).mean()))
    within = float((df.abs_error <= 1.5).mean())
    worst = float(df.abs_error.max())
    print(f"  rows written:     {len(df)}")
    print(f"  RMSE:             {rmse:.4f} degC")
    print(f"  within +/-1.5 C:  {within * 100:.2f} %")
    print(f"  worst case error: {worst:.3f} degC")
    print(f"  written -> {out_path}")
    print("[OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
