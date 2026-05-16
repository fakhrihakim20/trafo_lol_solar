"""Publication-quality figure generation for the IEEE ICT-PEP 2026 paper.

All figures use matplotlib only (no seaborn, no plotly).
IEEE two-column format:
  Single-column: 3.5 inches wide
  Double-column: 7.16 inches wide
Each figure is saved as both PNG@300 DPI and PDF (vector).

Figure inventory
----------------
fig01_system_schematic      System block diagram (single-col)
fig02_annex_g_validation    Annex G reproduction check (single-col)
fig03_daily_trajectories    48h traces for 4 PV scenarios (double-col)
fig04_annual_lol_boxplots   Annual LoL box plots by scenario (double-col)
fig05_pv_lol_nonmonotonic   LoL vs PV% under climate & load-growth scenarios (double-col)
fig06_sensitivity_heatmap   Heatmap: LoL on PV x ambient grid (single-col)
fig07_surrogate_accuracy    Surrogate scatter + error CDF (double-col)
fig08_runtime_comparison    ODE vs surrogate wall-clock (single-col)
fig09_cooling_mode_timeline θ_TO + cooling-mode overlay for one representative week (double-col)
fig10_battery_dispatch_timeline Load, PV, battery power, and SoC over a 48 h trace (double-col)
fig11_dispatch_comparison   Threshold vs PyPSA-LP battery dispatch — LoL boxplot + cycles bar (double-col)
fig12_pypsa_network         PyPSA Network schematic for the Jember 2-bus case study (single-col)
fig13_ev_battery_thermal    EV and battery thermal trade-off comparison (single-col)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# IEEE two-column style defaults
# ---------------------------------------------------------------------------

IEEE_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

W_SINGLE = 3.5    # inches, IEEE single-column
W_DOUBLE = 7.16   # inches, IEEE double-column


def _apply_rc() -> None:
    matplotlib.rcParams.update(IEEE_RC)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save figure as PNG (300 DPI) and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 1: System schematic
# ---------------------------------------------------------------------------

def fig01_system_schematic(output_dir: Path) -> None:
    """Draw a block diagram of the simulation system (matplotlib patches).

    150 kV grid -> Transformer -> 20 kV bus -> Aggregated PV + Load
    """
    _apply_rc()
    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, color="lightyellow"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.5, fontweight="bold")

    def arrow(x1, y, x2):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    # Blocks
    box(0.2, 1.5, 1.6, 1.0, "150 kV\nGrid", color="#d0e8ff")
    arrow(1.8, 2.0, 2.5)
    box(2.5, 1.5, 2.0, 1.0, "60 MVA\nONAF\nTransformer", color="#fff3cd")
    arrow(4.5, 2.0, 5.2)
    box(5.2, 1.5, 1.6, 1.0, "20 kV\nBus", color="#d0e8ff")

    # PV and load branches from 20 kV bus
    ax.plot([6.0, 7.0, 7.0], [2.0, 2.0, 2.8], color="black", lw=0.8)
    box(7.0, 2.8, 1.8, 0.9, "Rooftop PV\n(% penetration)", color="#d4edda")
    ax.plot([6.0, 7.0, 7.0], [2.0, 2.0, 0.9], color="black", lw=0.8)
    box(7.0, 0.1, 1.8, 0.9, "MV Load\n(commercial)", color="#f8d7da")

    # Thermal model annotation
    ax.annotate(
        "IEEE C57.91\nthermal model",
        xy=(3.5, 1.5), xytext=(3.5, 0.3),
        ha="center", fontsize=6.5,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="gray"),
        color="gray",
    )

    ax.set_title("Simulation System Overview", fontsize=8, pad=4)
    plt.tight_layout()
    _save(fig, output_dir, "fig01_system_schematic")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Annex G validation
# ---------------------------------------------------------------------------

def fig02_annex_g_validation(output_dir: Path) -> None:
    """Re-generate Annex G validation figure (theta_HS and theta_TO vs time)."""
    from transformer_lol.thermal_model import simulate_thermal
    from transformer_lol.aging import aging_acceleration_factor
    from transformer_lol.validation import (
        ANNEX_G_PARAMS, ANNEX_G_FIXTURES, ANNEX_G_AMBIENT, _build_annex_g_load
    )

    _apply_rc()
    load_pu, ambient_c = _build_annex_g_load()
    n = len(load_pu)
    t_min = np.arange(n, dtype=float)

    result = simulate_thermal(load_pu, ambient_c, ANNEX_G_PARAMS, dt_min=1.0,
                               initial_state=(0.0, 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_SINGLE, 4.0), sharex=True)

    ax1.plot(t_min, result["theta_to"], label="Theta_TO (ODE)", color="tab:blue", lw=1.2)
    ax1.plot(t_min, result["theta_hs"], label="Theta_HS (ODE)", color="tab:red", lw=1.2)
    ax1.axvline(60, color="gray", ls="--", lw=0.8, label="K step t=60 min")

    # Reference scatter points
    ref_t = [fx["t_min"] for fx in ANNEX_G_FIXTURES.values() if "theta_hs" in fx]
    ref_ths = [fx["theta_hs"] for fx in ANNEX_G_FIXTURES.values() if "theta_hs" in fx]
    ax1.scatter(ref_t, ref_ths, color="red", marker="x", s=40, zorder=5,
                label="Analytic reference", linewidths=1.5)

    ax1.set_ylabel("Temperature (degC)")
    ax1.legend(fontsize=7)
    ax1.set_title("IEEE C57.91-2011 Annex G Validation\n"
                  "K=0.6 to K=1.2 step, ambient=30 degC", fontsize=8)

    ax2.step(t_min, load_pu, color="tab:green", lw=1.0, where="post")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Loading K (p.u.)")
    ax2.set_ylim(0, 1.5)

    plt.tight_layout()
    _save(fig, output_dir, "fig02_annex_g_validation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Daily trajectories for 4 PV levels
# ---------------------------------------------------------------------------

def fig03_daily_trajectories(output_dir: Path, n_days: int = 2) -> None:
    """4-panel 48h traces: load, PV, net load, theta_HS for 0/30/50/75% PV."""
    from transformer_lol.thermal_model import TransformerParams, simulate_thermal
    from transformer_lol.load_profile import synthesize_fallback_load
    from transformer_lol.ambient_profile import synthesize_tropical_ambient
    from transformer_lol.pv_synthesis import synthesize_pv_profile

    _apply_rc()
    P = TransformerParams()
    n = 48   # 48 hours
    pv_levels = [0.0, 0.30, 0.50, 0.75]
    colors_load = "tab:blue"
    colors_pv = "tab:orange"
    colors_net = "tab:green"
    colors_hs = "tab:red"

    fig, axes = plt.subplots(4, 1, figsize=(W_DOUBLE, 6.0), sharex=True)

    for idx, pv_pen in enumerate(pv_levels):
        ax = axes[idx]
        load = synthesize_fallback_load(n, seed=42)
        ambient = synthesize_tropical_ambient(n, seed=42)
        pv = synthesize_pv_profile(n, penetration_pu=pv_pen, seed=42)
        net = np.clip(load - pv, 0.0, 1.5)
        result = simulate_thermal(net, ambient, P, dt_min=60.0, initial_state=(0.0, 0.0))

        t = np.arange(n)
        ax2 = ax.twinx()
        ax.plot(t, load, color=colors_load, lw=1.0, label="Load", alpha=0.7)
        ax.plot(t, pv, color=colors_pv, lw=1.0, label="PV", alpha=0.7)
        ax.plot(t, net, color=colors_net, lw=1.2, label="Net load")
        ax2.plot(t, result["theta_hs"], color=colors_hs, lw=1.2, ls="--",
                 label="Theta_HS")
        ax.set_ylabel(f"PV={pv_pen:.0%}\n[p.u.]", fontsize=7)
        ax2.set_ylabel("Theta_HS [degC]", fontsize=7, color=colors_hs)
        ax2.tick_params(axis="y", colors=colors_hs)
        ax.set_ylim(0, 1.4)
        ax2.set_ylim(40, 140)
        ax.set_yticks([0, 0.5, 1.0])
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6,
                      loc="upper right", ncol=2)

    axes[-1].set_xlabel("Hour of Day")
    axes[-1].set_xticks(range(0, 49, 6))
    axes[-1].set_xticklabels([f"{h % 24:02d}:00" for h in range(0, 49, 6)], fontsize=7)
    fig.suptitle("48h Load, PV, Net Load and Hot-Spot Temperature\n"
                 "for Four PV Penetration Levels", fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "fig03_daily_trajectories")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Annual LoL boxplots
# ---------------------------------------------------------------------------

def fig04_annual_lol_boxplots(output_dir: Path, mc_df: pd.DataFrame) -> None:
    """Bar chart of mean annual LoL per named scenario with ±1σ error bars.

    Preferred over box plots because MC variance is very tight (~0.0003%),
    making box-plot IQRs nearly invisible at publication resolution.
    Scenarios are ordered to tell the narrative: PV benefit (left), then
    climate+growth headwinds that overwhelm the benefit (right).
    """
    _apply_rc()

    # Canonical narrative order
    scenario_order = [
        "baseline",
        "pv10_tropical_today", "pv30_tropical_today",
        "pv50_tropical_today", "pv75_tropical_today",
        "pv30_warmer_2030", "pv50_warmer_2030", "pv50_warmer_2040",
    ]
    present = [s for s in scenario_order if s in mc_df["scenario_name"].values]
    # Fall back to whatever is in the data if named scenarios are missing
    if not present:
        present = list(mc_df["scenario_name"].unique())

    stats = {
        s: mc_df[mc_df["scenario_name"] == s]["lol_percent_annual"]
        for s in present
    }
    means = [grp.mean() for grp in stats.values()]
    stds = [grp.std() for grp in stats.values()]

    # Baseline reference (first entry or min-PV, min-delta row)
    baseline_lol = means[0] if means else None

    colors_map = {0.0: "tab:blue", 1.0: "tab:orange", 2.0: "tab:red"}
    bar_colors = []
    for sc in present:
        delta = float(mc_df[mc_df["scenario_name"] == sc]["ambient_delta_c"].iloc[0])
        bar_colors.append(colors_map.get(delta, "gray"))

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.2))
    positions = np.arange(len(present))

    bars = ax.bar(positions, means, color=bar_colors, alpha=0.75,
                  edgecolor="black", linewidth=0.6, width=0.6)
    ax.errorbar(positions, means, yerr=stds, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3, capthick=0.8)

    # Today's baseline reference
    if baseline_lol is not None:
        ax.axhline(baseline_lol, color="black", ls="--", lw=1.0, alpha=0.8,
                   label=f"Baseline: {baseline_lol:.4f}%")

    # Divider between 'tropical today' and 'warmer future' groups
    if "pv75_tropical_today" in present and "pv30_warmer_2030" in present:
        div_x = (present.index("pv75_tropical_today") + present.index("pv30_warmer_2030")) / 2
        ax.axvline(div_x, color="gray", ls=":", lw=0.8, alpha=0.6)

    ax.set_xticks(positions)
    labels = [s.replace("_", "\n") for s in present]
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Annual LoL (%)")
    ax.set_title("Annual Transformer Loss-of-Life: Named Scenarios\n"
                 "(mean \u00b1 1\u03c3, 1000 MC runs each)", fontsize=8)

    patches = [mpatches.Patch(color=c, alpha=0.75, label=f"\u0394T = {k:+.0f}\u00b0C")
               for k, c in colors_map.items()]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, fontsize=7, loc="upper right")
    plt.tight_layout()
    _save(fig, output_dir, "fig04_annual_lol_boxplots")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: PV–LoL non-monotonic relationship (core finding)
# ---------------------------------------------------------------------------

def fig05_pv_lol_nonmonotonic(output_dir: Path, mc_df: pd.DataFrame) -> None:
    """LoL vs PV penetration under climate and load-growth scenarios.

    Panel (a): No load growth — PV alone monotonically reduces LoL across all
               climate offsets, with widening gap at high penetration.
    Panel (b): With projected load growth (4.5 %/yr) — horizontal reference at
               today's no-PV baseline reveals that >= 50% PV is needed to offset
               combined climate and growth headwinds at +1 degC.

    Both panels share the Y axis so the magnitude shift is visually obvious.
    IEEE C57.91 Annex G validated ODE + XGBoost surrogate, 1000 MC runs/scenario.
    """
    _apply_rc()

    pv_vals = sorted(mc_df["pv_penetration"].unique())
    delta_vals = sorted(mc_df["ambient_delta_c"].unique())
    growth_vals = sorted(mc_df["load_growth_pct"].unique())
    palette = ["tab:blue", "tab:orange", "tab:red"]

    # Today's no-PV, no-growth, no-climate-delta baseline for reference
    base_mask = (
        (mc_df["pv_penetration"] == 0.0)
        & (mc_df["ambient_delta_c"] == 0.0)
        & (mc_df["load_growth_pct"] == min(growth_vals))
    )
    baseline_lol = mc_df.loc[base_mask, "lol_percent_annual"].mean() if base_mask.any() else None

    # Two growth levels; if only one exists, duplicate for a single-panel layout
    low_growth = min(growth_vals)
    high_growth = max(growth_vals)
    panel_growths = [low_growth, high_growth] if low_growth != high_growth else [low_growth]

    fig, axes = plt.subplots(1, len(panel_growths), figsize=(W_DOUBLE, 3.2),
                              sharey=True, constrained_layout=True)
    if len(panel_growths) == 1:
        axes = [axes]

    panel_labels = ["(a)", "(b)", "(c)"]

    for ax_idx, (growth, ax) in enumerate(zip(panel_growths, axes)):
        sub_g = mc_df[mc_df["load_growth_pct"] == growth]

        for delta, color in zip(delta_vals, palette):
            sub = sub_g[sub_g["ambient_delta_c"] == delta]
            means, stds, pv_pts = [], [], []
            for pv in pv_vals:
                grp = sub[sub["pv_penetration"] == pv]["lol_percent_annual"]
                if len(grp) > 0:
                    means.append(grp.mean())
                    stds.append(grp.std())
                    pv_pts.append(pv * 100)
            if not means:
                continue

            pv_arr = np.array(pv_pts)
            mean_arr = np.array(means)
            std_arr = np.array(stds)

            ax.plot(pv_arr, mean_arr, "o-", color=color, lw=1.4, ms=4,
                    label=f"\u0394T = {delta:+.0f}\u00b0C")
            ax.fill_between(pv_arr, mean_arr - std_arr, mean_arr + std_arr,
                            color=color, alpha=0.12)

        # Baseline reference line
        if baseline_lol is not None:
            ax.axhline(baseline_lol, color="black", ls="--", lw=1.0, alpha=0.8,
                       label="Today baseline\n(0% PV, no growth)")

        ax.set_xlabel("PV Penetration (%)")
        if ax_idx == 0:
            ax.set_ylabel("Annual LoL (%)")

        growth_str = (f"+{growth:.1f}%/yr load growth"
                      if growth > 0 else "No load growth")
        ax.set_title(f"{panel_labels[ax_idx]} {growth_str}", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle(
        "Rooftop-PV Effect on Transformer LoL: Climate and Load-Growth Context",
        fontsize=8,
    )
    _save(fig, output_dir, "fig05_pv_lol_nonmonotonic")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Sensitivity heatmap
# ---------------------------------------------------------------------------

def fig06_sensitivity_heatmap(output_dir: Path, mc_df: pd.DataFrame) -> None:
    """Heatmap of mean annual LoL on (PV% x ambient delta) grid."""
    _apply_rc()
    pv_vals = sorted(mc_df["pv_penetration"].unique())
    delta_vals = sorted(mc_df["ambient_delta_c"].unique())

    matrix = np.zeros((len(delta_vals), len(pv_vals)))
    for i, da in enumerate(delta_vals):
        for j, pv in enumerate(pv_vals):
            grp = mc_df[
                (mc_df["ambient_delta_c"] == da) & (mc_df["pv_penetration"] == pv)
            ]["lol_percent_annual"]
            matrix[i, j] = grp.mean() if len(grp) > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
    plt.colorbar(im, ax=ax, label="Mean Annual LoL (%)")

    ax.set_xticks(range(len(pv_vals)))
    ax.set_xticklabels([f"{pv*100:.0f}%" for pv in pv_vals], fontsize=7)
    ax.set_yticks(range(len(delta_vals)))
    ax.set_yticklabels([f"{da:+.1f}C" for da in delta_vals], fontsize=7)
    ax.set_xlabel("PV Penetration")
    ax.set_ylabel("Ambient Delta")
    ax.set_title("Mean Annual LoL Sensitivity\n(PV Penetration x Ambient Delta)", fontsize=8)

    for i in range(len(delta_vals)):
        for j in range(len(pv_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if val < matrix.max()*0.6 else "white")

    plt.tight_layout()
    _save(fig, output_dir, "fig06_sensitivity_heatmap")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Surrogate accuracy
# ---------------------------------------------------------------------------

def fig07_surrogate_accuracy(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """2-panel: (a) physics vs surrogate scatter; (b) error empirical CDF."""
    _apply_rc()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.0))

    # Panel (a): scatter
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.scatter(y_true, y_pred, s=0.3, alpha=0.2, color="tab:blue", rasterized=True)
    ax1.plot([vmin, vmax], [vmin, vmax], color="red", lw=1.0, ls="--", label="y=x")
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax1.text(0.05, 0.95, f"RMSE={rmse:.3f} degC\nR^2={r2:.5f}",
             transform=ax1.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax1.set_xlabel("Physics Theta_HS (degC)")
    ax1.set_ylabel("Surrogate Theta_HS (degC)")
    ax1.set_title("(a) Surrogate vs Physics", fontsize=8)
    ax1.legend(fontsize=7)

    # Panel (b): error CDF
    errors = np.abs(y_pred - y_true)
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax2.plot(sorted_err, cdf, color="tab:blue", lw=1.2)
    ax2.axvline(1.5, color="red", ls="--", lw=0.8, label="|err|=1.5 degC threshold")
    pct_within = 100.0 * np.mean(errors < 1.5)
    ax2.text(0.6, 0.3, f"{pct_within:.1f}% within\n1.5 degC", transform=ax2.transAxes,
             fontsize=7, color="red")
    ax2.set_xlabel("|Prediction Error| (degC)")
    ax2.set_ylabel("CDF")
    ax2.set_title("(b) Error Distribution", fontsize=8)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    _save(fig, output_dir, "fig07_surrogate_accuracy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Runtime comparison
# ---------------------------------------------------------------------------

def fig08_runtime_comparison(
    output_dir: Path,
    timing_data: dict[str, dict[int, float]] | None = None,
) -> None:
    """Bar chart comparing ODE vs surrogate wall-clock for 1/100/1000 runs.

    Parameters
    ----------
    timing_data : dict, optional
        {solver_name: {n_runs: wall_clock_seconds}}.
        If None, uses realistic placeholder values for illustration.
    """
    _apply_rc()

    if timing_data is None:
        timing_data = {
            "ODE (RK45)":    {1: 0.08, 100: 8.0, 1000: 80.0},
            "XGBoost Surrogate": {1: 0.001, 100: 0.12, 1000: 1.2},
        }

    n_runs_list = [1, 100, 1000]
    x = np.arange(len(n_runs_list))
    width = 0.35
    colors_map = {"ODE (RK45)": "tab:blue", "XGBoost Surrogate": "tab:orange"}

    fig, ax = plt.subplots(figsize=(W_SINGLE, 2.8))
    for i, (name, timings) in enumerate(timing_data.items()):
        vals = [timings.get(n, float("nan")) for n in n_runs_list]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=colors_map.get(name, "gray"), alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}s" if val < 10 else f"{val:.0f}s",
                        ha="center", va="bottom", fontsize=6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} run{'s' if n > 1 else ''}" for n in n_runs_list])
    ax.set_ylabel("Wall-clock time (s, log scale)")
    ax.set_title("Computational Cost: ODE vs Surrogate", fontsize=8)
    ax.legend(fontsize=7)
    plt.tight_layout()
    _save(fig, output_dir, "fig08_runtime_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Cooling-mode timeline overlay (Phase 2 addition)
# ---------------------------------------------------------------------------

def fig09_cooling_mode_timeline(
    output_dir: Path,
    transformer: Any,
    n_days: int = 7,
    pv_penetration: float = 0.75,
    ambient_delta_c: float = 0.0,
    seed: int = 42,
) -> None:
    """Overlay θ_TO, θ_HS, and ONAN/ONAF mode bands for one representative week.

    Two-panel figure (double-column):
      Panel (a): θ_TO and θ_HS trajectories; shaded bands show active mode.
      Panel (b): Net load (p.u.) and PV generation — context for the mode switches.

    Parameters
    ----------
    output_dir : Path
    transformer : TransformerParams
        Must have cooling_ctrl_params set; otherwise the figure is skipped.
    n_days : int
        Number of days to plot (default 7 = one week).
    pv_penetration : float
        PV penetration fraction for the sample profile.
    ambient_delta_c : float
        Climate delta added to the ambient profile.
    seed : int
        RNG seed for synthetic profiles.
    """
    from .thermal_model import CoolingController, simulate_thermal
    from .load_profile import synthesize_fallback_load
    from .ambient_profile import synthesize_tropical_ambient
    from .pv_synthesis import synthesize_pv_profile

    if transformer.cooling_ctrl_params is None:
        return  # nothing to show without mode-switching

    _apply_rc()
    n_hours = n_days * 24

    load_base = synthesize_fallback_load(n_hours, peak_pu=0.85, seed=seed)
    ambient_c = synthesize_tropical_ambient(n_hours, delta_c=ambient_delta_c, seed=seed)
    pv_gen = synthesize_pv_profile(n_hours, penetration_pu=pv_penetration, seed=seed)
    net_load = np.clip(load_base - pv_gen, 0.0, 1.5)

    ctrl = CoolingController(transformer.cooling_ctrl_params, fans_ok=True)
    result = simulate_thermal(
        net_load, ambient_c, transformer, dt_min=60.0, cooling_controller=ctrl
    )

    hours = np.arange(n_hours)
    theta_to = result["theta_to"]
    theta_hs = result["theta_hs"]
    mode = result["mode"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_DOUBLE, 4.5), sharex=True)

    # ---- Panel (a): temperatures + mode shading ----
    onaf_mask = mode == "ONAF"
    # shade ONAF intervals
    in_onaf = False
    start = 0
    for i in range(n_hours + 1):
        currently_onaf = i < n_hours and onaf_mask[i]
        if currently_onaf and not in_onaf:
            start = i
            in_onaf = True
        elif not currently_onaf and in_onaf:
            ax1.axvspan(start, i, alpha=0.12, color="#1f77b4", label="ONAF active" if start == 0 or not any(onaf_mask[:start]) else "_")
            in_onaf = False

    ax1.plot(hours, theta_to, color="#1f77b4", lw=1.2, label=r"$\Theta_{TO}$")
    ax1.plot(hours, theta_hs, color="#d62728", lw=1.2, label=r"$\Theta_{HS}$")
    ax1.axhline(transformer.cooling_ctrl_params.theta_to_on_c, ls="--", color="gray",
                lw=0.8, label=f"Fan ON ({transformer.cooling_ctrl_params.theta_to_on_c:.0f} °C)")
    ax1.axhline(transformer.cooling_ctrl_params.theta_to_off_c, ls=":", color="gray",
                lw=0.8, label=f"Fan OFF ({transformer.cooling_ctrl_params.theta_to_off_c:.0f} °C)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"ONAN/ONAF Mode Switching — {n_days}-day trace, PV={pv_penetration*100:.0f}%",
        fontsize=9,
    )
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(bottom=min(ambient_c.min() - 2, 25))

    # ---- Panel (b): load and PV ----
    ax2.fill_between(hours, net_load, 0, alpha=0.35, color="#2ca02c", label="Net load")
    ax2.fill_between(hours, pv_gen, 0, alpha=0.25, color="#ff7f0e", label="PV gen")
    ax2.plot(hours, load_base, color="k", lw=0.8, alpha=0.6, label="Gross load")
    ax2.axhline(1.0, ls=":", color="gray", lw=0.6)
    ax2.set_ylabel("Loading (p.u.)")
    ax2.set_xlabel("Hour of simulation")
    ax2.legend(fontsize=7, ncol=3)
    ax2.set_ylim(0, 1.4)

    plt.tight_layout()
    _save(fig, output_dir, "fig09_cooling_mode_timeline")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Battery dispatch timeline
# ---------------------------------------------------------------------------

def fig10_battery_dispatch_timeline(
    output_dir: Path,
    pv_penetration: float = 0.75,
    battery_capacity_mwh: float = 5.0,
    battery_power_mw: float = 2.0,
    transformer_rated_mva: float = 60.0,
    discharge_threshold_pu: float = 0.85,
    seed: int = 42,
    n_hours: int = 48,
) -> None:
    """Three-panel 48 h battery dispatch timeline.

    Panel (a): Gross load, PV generation, and net load after battery.
    Panel (b): Battery power [p.u.]; positive = discharge, negative = charge.
    Panel (c): State of Charge [fraction].

    Parameters
    ----------
    output_dir : Path
    pv_penetration : float
        PV capacity fraction of transformer rated MVA.
    battery_capacity_mwh, battery_power_mw : float
        Battery sizing.
    transformer_rated_mva : float
        Transformer rated MVA (for p.u. conversion).
    discharge_threshold_pu : float
        Load threshold above which battery discharges.
    seed : int
        RNG seed for synthetic profiles.
    n_hours : int
        Trace length (default 48 = two days).
    """
    from .battery import BatteryParams, dispatch_battery
    from .load_profile import synthesize_fallback_load
    from .pv_synthesis import synthesize_pv_profile

    _apply_rc()

    load_pu = synthesize_fallback_load(n_hours, peak_pu=0.85, seed=seed)
    pv_pu = synthesize_pv_profile(n_hours, penetration_pu=pv_penetration, seed=seed)

    params = BatteryParams(
        capacity_mwh=battery_capacity_mwh,
        max_power_mw=battery_power_mw,
        discharge_threshold_pu=discharge_threshold_pu,
        transformer_rated_mva=transformer_rated_mva,
        soc_init=0.50,
    )
    result = dispatch_battery(load_pu, pv_pu, params, dt_hours=1.0)

    hours = np.arange(n_hours)
    battery_p = result["battery_p_pu"]
    soc = result["soc"]
    net_load = result["net_load"]

    fig, axes = plt.subplots(3, 1, figsize=(W_DOUBLE, 5.5), sharex=True)
    ax1, ax2, ax3 = axes

    # ---- Panel (a): load / PV / net load ----
    ax1.fill_between(hours, pv_pu, 0, alpha=0.30, color="#ff7f0e", label="PV")
    ax1.plot(hours, load_pu, color="k", lw=1.0, label="Gross load")
    ax1.plot(hours, net_load, color="#2ca02c", lw=1.2, ls="--", label="Net load")
    ax1.axhline(discharge_threshold_pu, ls=":", color="#d62728", lw=0.8,
                label=f"Discharge threshold ({discharge_threshold_pu:.0%})")
    ax1.set_ylabel("Loading (p.u.)")
    ax1.set_title(
        f"Battery Dispatch — 48 h trace, PV={pv_penetration*100:.0f}%, "
        f"{battery_capacity_mwh:.0f} MWh / {battery_power_mw:.0f} MW",
        fontsize=9,
    )
    ax1.legend(fontsize=7, ncol=2)
    ax1.set_ylim(0, 1.3)

    # ---- Panel (b): battery power ----
    charge_p = np.where(battery_p < 0, battery_p, 0.0)
    discharge_p = np.where(battery_p > 0, battery_p, 0.0)
    ax2.fill_between(hours, discharge_p, 0, alpha=0.45, color="#1f77b4", label="Discharge (+)")
    ax2.fill_between(hours, charge_p, 0, alpha=0.45, color="#ff7f0e", label="Charge (−)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylabel("Battery power (p.u.)")
    ax2.legend(fontsize=7, ncol=2)
    max_p_pu = battery_power_mw / transformer_rated_mva
    ax2.set_ylim(-max_p_pu * 1.4, max_p_pu * 1.4)

    # ---- Panel (c): SoC ----
    from .battery import BATTERY_SOC_MIN, BATTERY_SOC_MAX
    ax3.fill_between(hours, soc, BATTERY_SOC_MIN, where=(soc > BATTERY_SOC_MIN),
                     alpha=0.40, color="#2ca02c", label="SoC")
    ax3.plot(hours, soc, color="#2ca02c", lw=1.2)
    ax3.axhline(BATTERY_SOC_MIN, ls="--", color="#d62728", lw=0.8,
                label=f"SoC min ({BATTERY_SOC_MIN:.0%})")
    ax3.axhline(BATTERY_SOC_MAX, ls="--", color="#1f77b4", lw=0.8,
                label=f"SoC max ({BATTERY_SOC_MAX:.0%})")
    ax3.set_ylabel("State of Charge")
    ax3.set_xlabel("Hour of simulation")
    ax3.legend(fontsize=7, ncol=3)
    ax3.set_ylim(0.0, 1.0)

    plt.tight_layout()
    _save(fig, output_dir, "fig10_battery_dispatch_timeline")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: Threshold vs PyPSA-LP dispatch comparison (Phase 4B)
# ---------------------------------------------------------------------------

def fig11_dispatch_comparison(output_dir: Path, df_compare: pd.DataFrame) -> None:
    """Two-panel comparison of threshold-rule vs PyPSA LP battery dispatch.

    Panel (a): annual LoL boxplot per scenario × dispatch method (n_runs × scenarios).
    Panel (b): mean equivalent battery cycles per scenario × dispatch method.

    Parameters
    ----------
    output_dir : Path
    df_compare : pd.DataFrame
        Long-format DataFrame from `dispatch_comparison.csv` produced by
        `scripts/06_compare_dispatch_methods.py`.  Required columns:
        ``scenario_name``, ``dispatch_method``, ``lol_percent_annual``,
        ``battery_lifetime_fraction``.
    """
    _apply_rc()

    methods = ["threshold", "pypsa_lp"]
    method_labels = {"threshold": "Threshold rule", "pypsa_lp": "PyPSA LP"}
    method_colors = {"threshold": "#1f77b4", "pypsa_lp": "#d62728"}

    scenarios = list(df_compare["scenario_name"].unique())
    n_sc = len(scenarios)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.8))

    # ---- Panel (a): LoL boxplot ----
    positions = []
    box_data = []
    box_colors = []
    box_labels = []
    width = 0.35
    for i, sc in enumerate(scenarios):
        for j, m in enumerate(methods):
            sub = df_compare[
                (df_compare["scenario_name"] == sc) & (df_compare["dispatch_method"] == m)
            ]["lol_percent_annual"].to_numpy()
            if len(sub) == 0:
                continue
            positions.append(i + (j - 0.5) * (width + 0.05))
            box_data.append(sub)
            box_colors.append(method_colors[m])
            box_labels.append(method_labels[m])

    bp = ax1.boxplot(box_data, positions=positions, widths=width, patch_artist=True,
                     showfliers=False, medianprops=dict(color="black", lw=1.0))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax1.set_xticks(range(n_sc))
    ax1.set_xticklabels([_short_label(s) for s in scenarios], rotation=15, ha="right",
                        fontsize=7)
    ax1.set_ylabel("Annual LoL (%)")
    ax1.set_title("(a) Annual LoL: threshold vs PyPSA LP", fontsize=9)
    # Legend (use one box per method)
    handles = [mpatches.Patch(facecolor=method_colors[m], alpha=0.6,
                              label=method_labels[m]) for m in methods]
    ax1.legend(handles=handles, fontsize=7, loc="upper left")

    # ---- Panel (b): equivalent battery cycles bar ----
    cycle_means = []
    bar_x = []
    bar_color = []
    for i, sc in enumerate(scenarios):
        for j, m in enumerate(methods):
            sub = df_compare[
                (df_compare["scenario_name"] == sc) & (df_compare["dispatch_method"] == m)
            ]
            if len(sub) == 0:
                continue
            mean_cycles = float(sub["battery_lifetime_fraction"].mean()) * 6000.0
            cycle_means.append(mean_cycles)
            bar_x.append(i + (j - 0.5) * (width + 0.05))
            bar_color.append(method_colors[m])

    ax2.bar(bar_x, cycle_means, width=width, color=bar_color, alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(n_sc))
    ax2.set_xticklabels([_short_label(s) for s in scenarios], rotation=15, ha="right",
                        fontsize=7)
    ax2.set_ylabel("Equivalent annual cycles")
    ax2.set_title("(b) Battery throughput", fontsize=9)
    ax2.legend(handles=handles, fontsize=7, loc="upper left")

    plt.tight_layout()
    _save(fig, output_dir, "fig11_dispatch_comparison")
    plt.close(fig)


def _short_label(scenario_name: str) -> str:
    """Compact tick label for the comparison figure."""
    return (scenario_name
            .replace("pv75_battery5mwh_", "5 MWh, ")
            .replace("today", "today")
            .replace("ev30_warmer_2030", "+EV+2030"))


# ---------------------------------------------------------------------------
# Figure 12: PyPSA Network schematic (Phase 4B)
# ---------------------------------------------------------------------------

def fig12_pypsa_network(output_dir: Path) -> None:
    """Single-column schematic of the Jember 2-bus PyPSA network.

    Drawn with matplotlib patches rather than ``network.plot()`` so the figure
    is deterministic (no geographic data) and IEEE-format compliant.

    Layout notes
    ------------
    * Canvas: 3.5 × 4.0 in  (IEEE single-column width × taller height)
    * Coordinate space: x in [0, 11],  y in [0, 6]
      → 1 x-unit ≈ 0.318 in,  1 y-unit ≈ 0.667 in
    * Transformer box: 2.60 units ≈ 0.83 in — wide enough for
      "Transformer / 60 MVA ONAF" at 6.5 pt without overflow.
    * PV and Battery boxes centred on the Bus-20 kV x-centre (7.00).
    """
    _apply_rc()

    fig, ax = plt.subplots(figsize=(W_SINGLE, 4.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    FONT = 6.5    # pt — fits comfortably inside all boxes
    LW   = 1.0    # connector line-width

    # ── helpers ───────────────────────────────────────────────────────────────
    def box(x, y, w, h, label, color="#fff3cd"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10",
            facecolor=color, edgecolor="black", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=FONT, fontweight="bold", linespacing=1.35)

    def hline(x1, x2, y):
        ax.plot([x1, x2], [y, y], color="black", lw=LW)

    def vline(x, y1, y2):
        ax.plot([x, x], [y1, y2], color="black", lw=LW)

    # ── geometry constants ────────────────────────────────────────────────────
    BH  = 0.75          # uniform box height for all components
    BY  = 4.00          # bottom-y of main horizontal bus row
    MID = BY + BH / 2   # 4.375 — y of horizontal connector lines

    PV_Y  = 2.20        # bottom-y of PV generator box
    BAT_Y = 0.55        # bottom-y of battery box

    # ── Slack Generator (above Bus 150 kV) ───────────────────────────────────
    SLK_X = 0.25;  SLK_W = 1.90;  SLK_H = 0.60
    SLK_CX = SLK_X + SLK_W / 2   # x-centre = 1.20
    SLK_BOTTOM = BY + BH + 0.30   # y of Slack box bottom = 5.05
    box(SLK_X, SLK_BOTTOM, SLK_W, SLK_H, "Slack\nGenerator", color="#e9ecef")
    vline(SLK_CX, BY + BH, SLK_BOTTOM)   # vertical stub: Bus-150 top → Slack bottom

    # ── Bus 150 kV ────────────────────────────────────────────────────────────
    B1_X = 0.25;  B1_W = 1.90
    box(B1_X, BY, B1_W, BH, "Bus\n150 kV", color="#d0e8ff")

    # ── Transformer 60 MVA ONAF ───────────────────────────────────────────────
    TF_X = 2.80;  TF_W = 2.60
    hline(B1_X + B1_W, TF_X, MID)             # Bus-150 → Transformer
    box(TF_X, BY, TF_W, BH, "Transformer\n60 MVA ONAF", color="#fff3cd")

    # ── Bus 20 kV ─────────────────────────────────────────────────────────────
    B2_X = 6.15;  B2_W = 1.70
    B2_CX = B2_X + B2_W / 2    # x-centre = 7.00
    hline(TF_X + TF_W, B2_X, MID)             # Transformer → Bus-20
    box(B2_X, BY, B2_W, BH, "Bus\n20 kV", color="#d0e8ff")

    # ── Load (demand + EV) ────────────────────────────────────────────────────
    LD_X = 8.50;  LD_W = 2.25
    hline(B2_X + B2_W, LD_X, MID)             # Bus-20 → Load
    box(LD_X, BY, LD_W, BH, "Load\n(demand+EV)", color="#f8d7da")

    # ── PV Generator (below Bus 20 kV, centred on B2_CX) ─────────────────────
    PV_W = 2.60;  PV_H = BH
    PV_X = B2_CX - PV_W / 2    # 5.70
    vline(B2_CX, BY, PV_Y + PV_H)             # Bus-20 bottom → PV top
    box(PV_X, PV_Y, PV_W, PV_H, "Generator\nPV (p_max_pu)", color="#d4edda")

    # ── StorageUnit Battery (below PV Generator) ──────────────────────────────
    BAT_W = 2.60;  BAT_H = BH
    BAT_X = B2_CX - BAT_W / 2  # 5.70
    vline(B2_CX, PV_Y, BAT_Y + BAT_H)         # PV bottom → Battery top
    box(BAT_X, BAT_Y, BAT_W, BAT_H, "StorageUnit\nBattery", color="#cfe2ff")

    ax.set_title("PyPSA Network — Jember 60 MVA", fontsize=8.0, pad=6)
    plt.tight_layout(pad=0.4)
    _save(fig, output_dir, "fig12_pypsa_network")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 13: EV + Battery + Thermal interaction timeline (paper §IV.E)
# ---------------------------------------------------------------------------

def fig13_ev_battery_thermal(
    output_dir: Path,
    transformer: Any,
    n_hours: int = 72,
    pv_penetration: float = 0.95,
    ev_slow_pu: float = 0.30,
    ev_fast_pu: float = 0.12,
    battery_capacity_mwh: float = 10.0,
    battery_power_mw: float = 5.0,
    discharge_threshold_pu: float = 0.70,
    base_load_peak_pu: float = 0.75,
    seed: int = 7,
) -> None:
    r"""3-panel single-column figure: EV charging + battery dispatch + thermal response.

    A 72-hour trace at high PV penetration with EV30 (slow + fast charging),
    plus a threshold-rule battery.  Demonstrates how the battery shaves the
    EV-driven evening / overnight load plateau and the resulting reduction
    in hot-spot temperature versus the no-battery baseline.

    Panels:
      (a) Loading: gross load (incl. EV), PV generation, EV component, net load after battery
      (b) Battery power (red = discharge, blue = charge) and SoC trajectory
      (c) Hot-spot temperature θ_HS with battery vs no-battery
    """
    from .thermal_model import simulate_thermal
    from .load_profile import synthesize_fallback_load
    from .ambient_profile import synthesize_tropical_ambient
    from .pv_synthesis import synthesize_pv_profile
    from .ev_load import synthesize_ev_total
    from .battery import dispatch_battery, BatteryParams

    _apply_rc()

    # ── Profiles ────────────────────────────────────────────────────────────
    base_load = synthesize_fallback_load(n_hours, peak_pu=base_load_peak_pu, seed=seed)
    ev_load = synthesize_ev_total(
        n_hours,
        slow_penetration_pu=ev_slow_pu,
        fast_penetration_pu=ev_fast_pu,
        seed=seed,
    )
    gross_load = base_load + ev_load
    pv_gen = synthesize_pv_profile(n_hours, penetration_pu=pv_penetration, seed=seed)
    ambient = synthesize_tropical_ambient(n_hours, delta_c=0.0, seed=seed)

    # ── Battery dispatch (threshold rule) ───────────────────────────────────
    bat_params = BatteryParams(
        capacity_mwh=battery_capacity_mwh,
        max_power_mw=battery_power_mw,
        discharge_threshold_pu=discharge_threshold_pu,
        transformer_rated_mva=getattr(transformer, "rated_mva", 60.0),
    )
    bat = dispatch_battery(gross_load, pv_gen, bat_params, dt_hours=1.0)
    net_with_bat = bat["net_load"]
    bat_p = bat["battery_p_pu"]            # + = discharge, − = charge
    soc = bat["soc"]

    # No-battery comparison
    net_no_bat = np.clip(gross_load - pv_gen, 0.0, None)

    # ── Thermal sims ────────────────────────────────────────────────────────
    th_no = simulate_thermal(net_no_bat, ambient, transformer, dt_min=60.0)
    th_yes = simulate_thermal(net_with_bat, ambient, transformer, dt_min=60.0)

    hours = np.arange(n_hours)

    # ── Plot (single column, 3 stacked panels) ──────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(W_SINGLE, 5.4), sharex=True)

    # Panel (a): loads
    ax1 = axes[0]
    ax1.plot(hours, gross_load, color="#444444", lw=1.1, label="Gross load (incl. EV)")
    ax1.fill_between(hours, 0, ev_load, color="#9467bd", alpha=0.45, label="EV load")
    ax1.fill_between(hours, 0, pv_gen, color="#ff7f0e", alpha=0.40, label="PV gen")
    ax1.plot(hours, net_with_bat, color="#1f77b4", lw=1.4, label="Net load (w/ battery)")
    ax1.set_ylabel("Loading (p.u.)", fontsize=8)
    ax1.set_title("(a) Load composition vs. PV \\& battery offset", fontsize=8.5)
    ax1.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.85)
    y_top = max(float(gross_load.max()), float(pv_gen.max())) * 1.18
    ax1.set_ylim(0, y_top)

    # Panel (b): battery power + SoC twin axis
    ax2 = axes[1]
    bar_colors = ["#d62728" if x > 0 else "#1f77b4" for x in bat_p]
    ax2.bar(hours, bat_p, color=bar_colors, alpha=0.80, width=0.85, edgecolor="none")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Battery P (p.u.)", fontsize=7)
    ax2_soc = ax2.twinx()
    ax2_soc.plot(hours, soc, color="#2ca02c", lw=1.6)
    ax2_soc.set_ylabel("SoC", color="#2ca02c", fontsize=8)
    ax2_soc.tick_params(axis="y", colors="#2ca02c", labelsize=7)
    ax2_soc.set_ylim(0, 1)
    ax2.set_title("(b) Battery dispatch (red disch / blue chg) + SoC (green)", fontsize=8.5)

    # Panel (c): theta_HS comparison
    ax3 = axes[2]
    ax3.plot(hours, th_no["theta_hs"], color="#d62728", lw=1.4, ls="--",
             label=r"$\theta_{HS}$ no battery")
    ax3.plot(hours, th_yes["theta_hs"], color="#1f77b4", lw=1.4,
             label=r"$\theta_{HS}$ with battery")
    ax3.set_ylabel(r"$\theta_{HS}$ (°C)", fontsize=8)
    ax3.set_xlabel("Hour of simulation", fontsize=8)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_title("(c) Hot-spot temperature: battery shaves the EV-driven peak",
                  fontsize=8.5)

    plt.tight_layout(pad=0.4)
    _save(fig, output_dir, "fig13_ev_battery_thermal")
    plt.close(fig)
