"""Script 01: Validate thermal model against IEEE C57.91-2011 Annex G.

Runs the two-phase step-load test from C57.91-2011 Annex G and checks that
the implementation reproduces the reference temperature trajectory within
the stated tolerances.

Usage
-----
    python scripts/01_validate_thermal.py

Exit codes
----------
    0 — All Annex G fixtures PASS
    1 — One or more fixtures FAIL (details printed to stdout)

Outputs
-------
    results/figures/fig02_annex_g_validation.png
    results/figures/fig02_annex_g_validation.pdf
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Force UTF-8 output on Windows so non-ASCII chars in log messages don't crash
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Allow running from either transformer_lol_sim/ or its parent
_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from transformer_lol.thermal_model import simulate_thermal
from transformer_lol.aging import aging_acceleration_factor
from transformer_lol.validation import (
    ANNEX_G_PARAMS,
    ANNEX_G_AMBIENT,
    ANNEX_G_FIXTURES,
    run_annex_g_validation,
    _build_annex_g_load,
)
from transformer_lol.utils import get_project_root, setup_logging


def main() -> int:
    root = get_project_root()
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(root / "results" / "logs")

    log.info("=== Script 01: IEEE C57.91-2011 Annex G Validation ===")

    # -----------------------------------------------------------------------
    # 1. Run the simulation
    # -----------------------------------------------------------------------
    load_pu, ambient_c = _build_annex_g_load()
    n_steps = len(load_pu)
    t_min = np.arange(n_steps, dtype=float)

    result = simulate_thermal(
        load_pu=load_pu,
        ambient_c=ambient_c,
        params=ANNEX_G_PARAMS,
        dt_min=1.0,
        initial_state=(0.0, 0.0),
    )

    theta_hs = result["theta_hs"]
    theta_to = result["theta_to"]
    delta_to = result["delta_to"]
    delta_hs = result["delta_hs"]

    # -----------------------------------------------------------------------
    # 2. Run validation (prints pass/fail per fixture)
    # -----------------------------------------------------------------------
    print("\nIEEE C57.91-2011 Annex G Fixture Comparison")
    print("=" * 60)

    try:
        run_annex_g_validation(params=ANNEX_G_PARAMS, verbose=True)
        validation_passed = True
    except AssertionError as exc:
        print(f"\n[ERROR] {exc}")
        validation_passed = False

    # -----------------------------------------------------------------------
    # 3. Print summary table
    # -----------------------------------------------------------------------
    print("\nDetailed results at fixture timesteps:")
    print(f"  {'t (min)':>8}  {'K':>5}  {'θ_HS (°C)':>11}  {'ΔΘ_TO (°C)':>12}  {'F_AA':>8}")
    print("  " + "-" * 56)
    for name, fx in ANNEX_G_FIXTURES.items():
        idx = int(fx["t_min"])
        k_val = 0.6 if idx <= 60 else 1.2
        print(
            f"  {idx:>8.0f}  {k_val:>5.1f}  {theta_hs[idx]:>11.3f}  "
            f"{delta_to[idx]:>12.3f}  {aging_acceleration_factor(theta_hs[idx]):>8.4f}"
        )

    # -----------------------------------------------------------------------
    # 4. Generate fig02_annex_g_validation figure
    # -----------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.dpi": 300,
        "lines.linewidth": 1.2, "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True)

    ax1.plot(t_min, theta_to, label="θ_TO (implementation)", color="tab:blue")
    ax1.plot(t_min, theta_hs, label="θ_HS (implementation)", color="tab:red")
    ax1.axvline(60, color="gray", linestyle="--", linewidth=0.8, label="K step at t=60 min")

    # Mark reference fixtures
    ref_t = [fx["t_min"] for fx in ANNEX_G_FIXTURES.values() if "theta_hs" in fx]
    ref_ths = [fx["theta_hs"] for fx in ANNEX_G_FIXTURES.values() if "theta_hs" in fx]
    ax1.scatter(ref_t, ref_ths, color="tab:red", marker="x", s=60, zorder=5,
                label="Annex G reference θ_HS", linewidths=1.5)

    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("IEEE C57.91-2011 Annex G Validation\n"
                  "Two-phase load step: K=0.6→1.2, ambient=30°C", fontsize=8)

    ax2.plot(t_min, load_pu, color="tab:green", linewidth=1.2)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Loading K (p.u.)")
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = fig_dir / f"fig02_annex_g_validation.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        log.info(f"Saved {path}")
    plt.close()

    # -----------------------------------------------------------------------
    # 5. Exit
    # -----------------------------------------------------------------------
    if validation_passed:
        print("\n[OK] Thermal model validated. Proceed to Phase 2.")
        return 0
    else:
        print("\n[FAIL] Thermal model did NOT pass Annex G. Fix thermal_model.py.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
