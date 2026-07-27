"""
Box-Whisker Plot Generator
============================
Produces box-whisker plots per variable showing distribution across stations.

Usage:
    python3 plot_boxwhisker.py

Reads: stats_{varname}.csv  (same files as plot_data_completion.py)

Box-whisker structure drawn from CSV rows:
    Min      → lower whisker tip
    p25      → lower quartile (box bottom)
    Median   → solid centre line inside box
    Average  → triangle marker (mean)
    p75      → upper quartile (box top)
    Max      → upper whisker tip
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"

VARNAMES = [
    "Max_Temperature", "Min_Temperature", "Precipitation", "Pressure", "Relative_Humidity",
    "Solar", "Temperature", "Wind_Direction", "Wind_Gust", "Wind_Speed",
]

# Row labels in the CSV
MIN_ROW    = "Min"
MAX_ROW    = "Max"
P25_ROW    = "p25"
P75_ROW    = "p75"
MEDIAN_ROW = "Median"
MEAN_ROW   = "Average"

# Y-axis unit labels
UNITS = {
    "Max_Temperature":   "°C",
    "Min_Temperature":   "°C",
    "Precipitation":     "mm",
    "Pressure":          "hPa",
    "Relative_Humidity": "%",
    "Solar":             "W/m²",
    "Temperature":       "°C",
    "Wind_Direction":    "°",
    "Wind_Gust":         "m/s",
    "Wind_Speed":        "m/s",
}

# Human-readable variable labels
VARLABELS = {
    "Max_Temperature":   "Max Temperature",
    "Min_Temperature":   "Min Temperature",
    "Relative_Humidity": "Relative Humidity",
    "Solar":             "Solar Radiance",
    "Wind_Direction":    "Wind Direction",
    "Wind_Gust":         "Wind Gust",
    "Wind_Speed":        "Wind Speed",
}

# Variables that should use a log Y scale
LOG_SCALE_VARS = {"Precipitation"}

# ---------------------------------------------------------------------------
# Manual station order
# ---------------------------------------------------------------------------
# For each varname, list station names in the order you want them displayed
# left-to-right. Any station present in the CSV but NOT listed here will be
# appended at the end in their original CSV order.
# Leave a dict entry as an empty list [] to use the original CSV order.

_STATION_SEQ = ["Giordan", "Xagħra", "Xewkija", 
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]
STATION_ORDER = {v: _STATION_SEQ for v in VARNAMES}
# Pressure: omit Giordan (see note)
STATION_ORDER["Pressure"] = [s for s in _STATION_SEQ if s != "Giordan"]
DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BOX_FACECOLOR   = "#cce5ff"
BOX_EDGECOLOR   = "#2166ac"
MEDIAN_LINE_CLR = "#2166ac"   # near-black solid line for median
MEAN_MARKER_CLR = "#d6604d"   # red triangle for mean
CAP_COLOR       = "#2166ac"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stats(varname: str) -> pd.DataFrame:
    filepath = f"{workdir}/analysis/basic-stats/stats_{varname}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected file not found: {filepath}")
    df = pd.read_csv(filepath, header=0, index_col=0)
    return df


def safe_float(val):
    """Return float or None if missing/empty."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_station_stats(df: pd.DataFrame, station: str) -> dict:
    def read(row_label):
        if row_label not in df.index:
            return None
        return safe_float(df.loc[row_label, station])

    return {
        "min":    read(MIN_ROW),
        "p25":    read(P25_ROW),
        "median": read(MEDIAN_ROW),
        "mean":   read(MEAN_ROW),
        "p75":    read(P75_ROW),
        "max":    read(MAX_ROW),
    }


def apply_station_order(varname: str, stations: list, stats_list: list):
    """
    Reorder stations according to STATION_ORDER[varname].
    Stations not mentioned are appended at the end in original CSV order.
    """
    order = STATION_ORDER.get(varname, [])
    if not order:
        return stations, stats_list

    paired  = list(zip(stations, stats_list))
    print(f"  [order debug] available stations: {[s for s, _ in paired]}")
    lookup  = {s: st for s, st in paired}
    listed  = set(order)

    reordered = [(name, lookup[name]) for name in order if name in lookup]

    if not reordered:
        return stations, stats_list

    new_stations, new_stats = zip(*reordered)
    return list(new_stations), list(new_stats)

def get_precip_stats_from_raw(station: str) -> dict:
    """
    For precipitation only: read {workdir}/Data/Daily_{station}.csv,
    treat values < 0.1 mm as missing, and compute stats from rainy days only.
    Returns the same dict shape as get_station_stats.
    """
    filepath = f"{workdir}/Data/Daily_{station}.csv"
    empty = {"min": None, "p25": None, "median": None, "mean": None, "p75": None, "max": None}

    if not os.path.exists(filepath):
        print(f"    [warn] Raw file not found for {station}: {filepath}")
        return empty

    raw = pd.read_csv(filepath, header=0)

    # Find the precipitation column — adjust this string to match your actual header
    precip_col = None
    for col in raw.columns:
        if "precip" in col.lower() or "rain" in col.lower():
            precip_col = col
            break
    if precip_col is None:
        print(f"    [warn] No precipitation column found in {filepath}. Columns: {list(raw.columns)}")
        return empty

    # Coerce to numeric, replace sentinel "-" with NaN
    vals = pd.to_numeric(raw[precip_col].replace("-", np.nan), errors="coerce")

    # Keep only rainy days (>= 0.1 mm)
    rainy = vals[vals >= 0.1].dropna()

    if rainy.empty:
        print(f"    [warn] No rainy days found for {station}")
        return empty

    return {
        "min":    float(rainy.min()),
        "p25":    float(rainy.quantile(0.25)),
        "median": float(rainy.median()),
        "mean":   float(rainy.mean()),
        "p75":    float(rainy.quantile(0.75)),
        "max":    float(rainy.max()),
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_boxplot(
    varname: str,
    stations: list,
    stats_list: list,
    output_dir: str = ".",
) -> None:
    # Apply manual ordering
    stations, stats_list = apply_station_order(varname, stations, stats_list)

    # Filter out stations where all values are None
    valid = [(s, st) for s, st in zip(stations, stats_list)
             if any(v is not None for v in st.values())]

    if not valid:
        print(f"  [skip] No valid data for {varname}, skipping plot.")
        return

    stations_valid, stats_valid = zip(*valid)
    n = len(stations_valid)

    use_log = varname in LOG_SCALE_VARS

    fig_width = max(9, n * 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7))   # taller than before for legend room

    box_half  = 0.3
    whisker_w = 0.15

    for xi, (station, st) in enumerate(zip(stations_valid, stats_valid)):
        p25    = st["p25"]
        p75    = st["p75"]
        vmin   = st["min"]
        vmax   = st["max"]
        median = st["median"]
        mean   = st["mean"]

        # On log scale, non-positive lower values can't be plotted
        if use_log:
            if vmin is not None and vmin <= 0:
                vmin = None
            if p25  is not None and p25  <= 0:
                p25  = None

        # --- Box (p25 to p75) ---
        if p25 is not None and p75 is not None:
            box = plt.Rectangle(
                (xi - box_half, p25),
                box_half * 2,
                p75 - p25,
                facecolor=BOX_FACECOLOR,
                edgecolor=BOX_EDGECOLOR,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(box)

        # --- Median line (solid, inside box) ---
        if median is not None and p25 is not None and p75 is not None:
            ax.plot(
                [xi - box_half, xi + box_half], [median, median],
                color=MEDIAN_LINE_CLR, linewidth=1.0,
                linestyle="--", zorder=4,
            )

        # --- Lower whisker ---
        if p25 is not None and vmin is not None:
            ax.plot([xi, xi], [vmin, p25],
                    color=BOX_EDGECOLOR, linewidth=1.5, zorder=2)
            ax.plot([xi - whisker_w, xi + whisker_w], [vmin, vmin],
                    color=CAP_COLOR, linewidth=1.5, zorder=2)

        # --- Upper whisker ---
        if p75 is not None and vmax is not None:
            ax.plot([xi, xi], [p75, vmax],
                    color=BOX_EDGECOLOR, linewidth=1.5, zorder=2)
            ax.plot([xi - whisker_w, xi + whisker_w], [vmax, vmax],
                    color=CAP_COLOR, linewidth=1.5, zorder=2)

        # --- Mean marker (triangle) ---
        if mean is not None:
            if not use_log or mean > 0:
                ax.plot(
                    xi, mean,
                    marker="^",
                    color=MEAN_MARKER_CLR,
                    markersize=8,
                    zorder=5,
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                )

    # Axes formatting
    unit = UNITS.get(varname, "units")
    vlab = VARLABELS.get(varname, varname)
    if vlab == "Precipitation":
        vlab = f"{vlab} (≥0.1 mm/day)"     

    if use_log:
        ax.set_yscale("log")

    ax.set_xlabel("Station Name", fontsize=12)
    ax.set_ylabel(f"{vlab} ({unit})", fontsize=12)
    ax.set_title(
        f"Distribution: {vlab}",
        fontsize=13,
        fontweight="bold",
        pad=36,    # extra top padding so the legend (placed above axes) doesn't overlap the title
    )
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [DISPLAY_NAME_OVERRIDES.get(s, s).replace("-", " ") for s in stations_valid],
        rotation=40, ha="right", fontsize=9
    )
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.autoscale_view(scalex=False)

    # Legend — floated just above the axes, in a single row of 4 items
    box_patch    = mpatches.Patch(facecolor=BOX_FACECOLOR, edgecolor=BOX_EDGECOLOR,
                                   linewidth=1.5, label="IQR (p25–p75)")
    whisker_line = Line2D([0], [0], color=BOX_EDGECOLOR, linewidth=1.5,
                          label="Min / Max")
    median_line  = Line2D([0], [0], color=MEDIAN_LINE_CLR, linewidth=1.0,
                          linestyle="--", label="Median")
    mean_marker  = Line2D([0], [0], marker="^", color="w",
                          markerfacecolor=MEAN_MARKER_CLR, markersize=9,
                          label="Mean")

    ax.legend(
        handles=[box_patch, whisker_line, median_line, mean_marker],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),  # anchor just above the top-left of the axes
        ncol=4,                       # all four items in one horizontal row
        fontsize=9,
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    fname = f"{workdir}/analysis/basic-stats/boxplot_{varname}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)

    for varname in VARNAMES:
        print(f"\n[var] Processing: {varname}")
        try:
            df = load_stats(varname)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        stations = list(df.columns)
        stats_list = []

        for station in stations:
            st = get_station_stats(df, station)
            if varname == "Precipitation":
                st = get_precip_stats_from_raw(station)
            print(
                f"    {station:20s}  "
                f"min={st['min']}  p25={st['p25']}  "
                f"median={st['median']}  mean={st['mean']}  "
                f"p75={st['p75']}  max={st['max']}"
            )
            stats_list.append(st)

        create_boxplot(varname, stations, stats_list, output_dir)

    print(f"\n[done] All plots written to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    main(run_output_dir)
