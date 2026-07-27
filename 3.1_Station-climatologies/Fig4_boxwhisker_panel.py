"""
Figure panel: box-whisker distributions (3x2)
=============================================
Combines the per-variable box-whisker plots into one panel image.
Reuses the computation logic from plot_boxwhisker.py; only the drawing
is retargeted to a shared grid of axes.

Usage:
    python3 figure_boxwhisker_panel.py

Output: {workdir}/analysis/basic-stats/figure_boxwhisker_panel.png
"""

import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"

# Panel layout: which variable goes in which cell, row-major (left->right, top->bottom)
# Edit this list to change the panel. Length must be <= NROWS*NCOLS.
PANEL_VARS = [
    "Precipitation", "Temperature",
    "Max_Temperature", "Min_Temperature",
    "Relative_Humidity", "Wind_Speed",
]
NROWS = 3
NCOLS = 2

# Master station order — every subplot uses these fixed x-slots, so columns
# line up across the panel and a shared x-axis is meaningful. Stations without
# data for a given variable simply leave their slot blank.
_STATION_SEQ = ["Giordan", "Xagħra", "Xewkija",
                "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]

# Per-variable exclusions: these stations keep their slot but are never drawn.
EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},
}

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}
LOG_SCALE_VARS = {"Precipitation"}

# Row labels in the CSV
MIN_ROW, MAX_ROW    = "Min", "Max"
P25_ROW, P75_ROW    = "p25", "p75"
MEDIAN_ROW, MEAN_ROW = "Median", "Average"

UNITS = {
    "Max_Temperature": "°C", "Min_Temperature": "°C", "Precipitation": "mm/day",
    "Pressure": "hPa", "Relative_Humidity": "%", "Solar": "W/m²",
    "Temperature": "°C", "Wind_Direction": "°", "Wind_Gust": "m/s",
    "Wind_Speed": "m/s",
}
VARLABELS = {
    "Max_Temperature": "Max Temperature", "Min_Temperature": "Min Temperature",
    "Relative_Humidity": "Relative Humidity", "Solar": "Solar Radiance",
    "Wind_Direction": "Wind Direction", "Wind_Gust": "Wind Gust",
    "Wind_Speed": "Wind Speed",
}

BOX_FACECOLOR   = "#cce5ff"
BOX_EDGECOLOR   = "#2166ac"
MEDIAN_LINE_CLR = "#2166ac"
MEAN_MARKER_CLR = "#d6604d"
CAP_COLOR       = "#2166ac"

# ---------------------------------------------------------------------------
# Computation helpers — copied unchanged from plot_boxwhisker.py
# ---------------------------------------------------------------------------

def load_stats(varname):
    filepath = f"{workdir}/analysis/basic-stats/stats_{varname}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    return pd.read_csv(filepath, header=0, index_col=0)

def safe_float(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def get_station_stats(df, station):
    def read(row):
        if station not in df.columns or row not in df.index:
            return None
        return safe_float(df.loc[row, station])
    return {"min": read(MIN_ROW), "p25": read(P25_ROW), "median": read(MEDIAN_ROW),
            "mean": read(MEAN_ROW), "p75": read(P75_ROW), "max": read(MAX_ROW)}

def get_precip_stats_from_raw(station):
    filepath = f"{workdir}/Data/Daily_{station}.csv"
    empty = {k: None for k in ["min", "p25", "median", "mean", "p75", "max"]}
    if not os.path.exists(filepath):
        return empty
    raw = pd.read_csv(filepath, header=0)
    precip_col = next((c for c in raw.columns
                       if "precip" in c.lower() or "rain" in c.lower()), None)
    if precip_col is None:
        return empty
    vals  = pd.to_numeric(raw[precip_col].replace("-", np.nan), errors="coerce")
    rainy = vals[vals >= 0.1].dropna()
    if rainy.empty:
        return empty
    return {"min": float(rainy.min()), "p25": float(rainy.quantile(0.25)),
            "median": float(rainy.median()), "mean": float(rainy.mean()),
            "p75": float(rainy.quantile(0.75)), "max": float(rainy.max())}

def stats_for_variable(varname):
    """Return an ordered list of (station, stats_dict) over the full _STATION_SEQ,
    with None-filled dicts where a station is absent or excluded."""
    try:
        df = load_stats(varname)
    except FileNotFoundError:
        df = None
    excluded = EXCLUDE_STATIONS.get(varname, set())
    out = []
    for station in _STATION_SEQ:
        if station in excluded:
            out.append((station, {k: None for k in
                        ["min", "p25", "median", "mean", "p75", "max"]}))
            continue
        if varname == "Precipitation":
            st = get_precip_stats_from_raw(station)
        elif df is not None:
            st = get_station_stats(df, station)
        else:
            st = {k: None for k in ["min", "p25", "median", "mean", "p75", "max"]}
        out.append((station, st))
    return out

# ---------------------------------------------------------------------------
# Drawing — one variable onto a given axis
# ---------------------------------------------------------------------------

def draw_boxplot(ax, varname, ordered_stats, show_xlabels):
    use_log   = varname in LOG_SCALE_VARS
    box_half  = 0.3
    whisker_w = 0.15
    n = len(ordered_stats)

    for xi, (station, st) in enumerate(ordered_stats):
        p25, p75 = st["p25"], st["p75"]
        vmin, vmax = st["min"], st["max"]
        median, mean = st["median"], st["mean"]

        if use_log:
            if vmin is not None and vmin <= 0: vmin = None
            if p25  is not None and p25  <= 0: p25  = None

        if p25 is not None and p75 is not None:
            ax.add_patch(plt.Rectangle((xi - box_half, p25), box_half * 2, p75 - p25,
                         facecolor=BOX_FACECOLOR, edgecolor=BOX_EDGECOLOR,
                         linewidth=1.5, zorder=3))
        if median is not None and p25 is not None and p75 is not None:
            ax.plot([xi - box_half, xi + box_half], [median, median],
                    color=MEDIAN_LINE_CLR, linewidth=1.0, linestyle="--", zorder=4)
        if p25 is not None and vmin is not None:
            ax.plot([xi, xi], [vmin, p25], color=BOX_EDGECOLOR, linewidth=1.5, zorder=2)
            ax.plot([xi - whisker_w, xi + whisker_w], [vmin, vmin],
                    color=CAP_COLOR, linewidth=1.5, zorder=2)
        if p75 is not None and vmax is not None:
            ax.plot([xi, xi], [p75, vmax], color=BOX_EDGECOLOR, linewidth=1.5, zorder=2)
            ax.plot([xi - whisker_w, xi + whisker_w], [vmax, vmax],
                    color=CAP_COLOR, linewidth=1.5, zorder=2)
        if mean is not None and (not use_log or mean > 0):
            ax.plot(xi, mean, marker="^", color=MEAN_MARKER_CLR, markersize=8,
                    zorder=5, markeredgecolor="white", markeredgewidth=0.5)

    if use_log:
        ax.set_yscale("log")

    unit = UNITS.get(varname, "")
    vlab = VARLABELS.get(varname, varname)
    if varname == "Precipitation":
        vlab_axis = "Precipitation [≥0.1]"
    else:
        vlab_axis = vlab
    ax.set_ylabel(f"{vlab_axis} ({unit})", fontsize=10)

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_xticks(range(n))
    if show_xlabels:
        ax.set_xticklabels(
            [DISPLAY_NAME_OVERRIDES.get(s, s).replace("-", " ")
             for s, _ in ordered_stats],
            rotation=40, ha="right", fontsize=9)
        ax.set_xlabel("Station", fontsize=11)
    else:
        ax.set_xticklabels([])          # keep ticks, drop labels
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# ---------------------------------------------------------------------------
# Main — assemble the panel
# ---------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 6.0, NROWS * 3.6),
                             squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, varname in enumerate(PANEL_VARS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        ordered = stats_for_variable(varname)
        is_bottom_row = (idx // NCOLS) == (len(PANEL_VARS) - 1) // NCOLS or \
                        idx >= len(PANEL_VARS) - NCOLS
        draw_boxplot(ax, varname, ordered, show_xlabels=is_bottom_row)

        vlab = VARLABELS.get(varname, varname)
        ax.set_title(f"({letters[idx]})", fontsize=12, fontweight="bold", loc="left")

    # Hide any unused cells
    for idx in range(len(PANEL_VARS), NROWS * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    # Single shared legend across the very top
    handles = [
        mpatches.Patch(facecolor=BOX_FACECOLOR, edgecolor=BOX_EDGECOLOR,
                       linewidth=1.5, label="IQR (p25–p75)"),
        Line2D([0], [0], color=BOX_EDGECOLOR, linewidth=1.5, label="Min / Max"),
        Line2D([0], [0], color=MEDIAN_LINE_CLR, linewidth=1.0, linestyle="--",
               label="Median"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=MEAN_MARKER_CLR,
               markersize=9, label="Mean"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=10, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])   # leave headroom for the legend
    out_path = f"{workdir}/analysis/basic-stats/figure_boxwhisker_panel.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
