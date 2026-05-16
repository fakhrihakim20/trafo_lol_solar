"""run_all.py — Execute the full pipeline from raw inputs to figures.

Orchestrates scripts 01–05 in sequence via subprocess.run.
Each script must return exit code 0 for the pipeline to continue.

Usage
-----
    python scripts/run_all.py [--n_runs N] [--n_traj N] [--quick]

Options
-------
    --n_runs N    Override MC run count per scenario (default: 1000)
    --n_traj N    Override training trajectory count (default: 200)
    --quick       Quick mode: n_runs=10, n_traj=20 (for CI / smoke-test)

Exit code 0 = all scripts succeeded.
Exit code 1 = at least one script failed (check logs/).
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here.parent
sys.path.insert(0, str(_root / "src"))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def run_step(script: str, extra_args: list[str] | None = None) -> bool:
    """Run one pipeline script; return True if it succeeded."""
    cmd = [sys.executable, str(_here / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(_root))
    elapsed = time.perf_counter() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n[{status}] {script} ({elapsed:.1f}s)")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full simulation pipeline")
    parser.add_argument("--n_runs", type=int, default=1000,
                        help="MC runs per scenario (default: 1000)")
    parser.add_argument("--n_traj", type=int, default=200,
                        help="Training trajectories (default: 200)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: n_runs=10, n_traj=20")
    args = parser.parse_args()

    if args.quick:
        args.n_runs = 10
        args.n_traj = 20
        print("[Quick mode] n_runs=10, n_traj=20")

    t_start = time.perf_counter()
    failed: list[str] = []

    steps = [
        ("01_validate_thermal.py", []),
        ("02_generate_training_data.py", ["--n_traj", str(args.n_traj)]),
        ("03_train_surrogate.py", []),
        ("04_run_monte_carlo.py", ["--n_runs", str(args.n_runs)]),
        ("05_generate_figures.py", []),
    ]

    for script, extra in steps:
        ok = run_step(script, extra)
        if not ok:
            failed.append(script)
            print(f"\n[ABORT] {script} failed. Stopping pipeline.")
            break

    total_time = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {total_time:.1f}s")
    if failed:
        print(f"FAILED steps: {failed}")
        print("Check results/logs/transformer_lol.log for details.")
        return 1
    else:
        print("All steps succeeded.")
        print("\nOutputs:")
        print(f"  Figures:  transformer_lol_sim/results/figures/  (8 PNG + 8 PDF)")
        print(f"  Tables:   transformer_lol_sim/results/tables/   (3 CSV)")
        print(f"  Metrics:  transformer_lol_sim/results/metrics.json")
        return 0


if __name__ == "__main__":
    sys.exit(main())
