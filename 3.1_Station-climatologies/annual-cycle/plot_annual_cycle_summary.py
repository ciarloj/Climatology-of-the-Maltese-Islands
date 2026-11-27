"""
Annual Cycle Summary Plot
==========================
Produces one plot per variable showing the smoothed daily climatology
for all stations as individual coloured lines, plus optional group averages.

Usage:
    python3 plot_annual_cycle.py

Output: ../../analysis/annual-cycle/annual-cycle_{varname}.png
Input:  ../../Data/Daily_{station}.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Station configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"

ALL_STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]

GOZO_STATIONS  = ["Giordan", "Xagħra", "Xewkija"]
MALTA_STATIONS = ["Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

VARNAMES = [
    "Max_Temperature", "Min_Temperature", "Precipitation", "Pressure",
    "Relative_Humidity", "Solar", "Temperature",
    "Wind_Direction", "Wind_Gust", "Wind_Speed",
]

VARLABELS = {
    "Max_Temperature":   "Max Temperature",
    "Min_Temperature":   "Min Temperature",
    "Precipitation":     "Precipitation",
    "Pressure":          "Pressure",
    "Relative_Humidity": "Relative Humidity",
    "Solar":             "Solar Radiation",
    "Temperature":       "Temperature",
    "Wind_Direction":    "Wind Direction",
    "Wind_Gust":         "Wind Gust",
    "Wind_Speed":        "Wind Speed",
}

VAR_AXIS = {
    "Max_Temperature":   ("temperature",        "°C"),
    "Min_Temperature":   ("temperature",        "°C"),
    "Temperature":       ("temperature",        "°C"),
    "Precipitation":     ("precipitation",      "mm/day"),
    "Pressure":          ("pressure",           "hPa"),
    "Relative_Humidity": ("relative humidity",  "%"),
    "Solar":             ("solar irradiance",   "W/m²"),
    "Wind_Speed":        ("wind speed",         "m/s"),
    "Wind_Gust":         ("wind gust",          "m/s"),
    "Wind_Direction":    ("wind direction",     "°"),
    "DTR":               ("temperature range",  "°C"),
    "CDD":               ("days",               "days"),
    "Accumulated_Wind":  ("accumulated wind",   "m/s"),  # see note below on unit
}

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

# Mapping from varname to the column header in Daily_{station}.csv
COL_MAP = {
    "Max_Temperature":   "Max_Temperature",
    "Min_Temperature":   "Min_Temperature",
    "Precipitation":     "Precipitation",
    "Pressure":          "Pressure",
    "Relative_Humidity": "Relative_Humidity",
    "Solar":             "Solar",
    "Temperature":       "Temperature",
    "Wind_Direction":    "Wind_Direction",
    "Wind_Gust":         "Wind_Gust",
    "Wind_Speed":        "Wind_Speed",
}

# Name of the date column in each Daily_{station}.csv
DATE_COL = "Date"

# Missing value sentinel in the CSV
MISSING_VAL = "-"

# ---------------------------------------------------------------------------
# Which group averages to show, per variable
# ---------------------------------------------------------------------------

SHOW_AVG = {
    #                            all      malta    gozo
    "Max_Temperature":   {"all": True,  "malta": True,  "gozo": True},
    "Min_Temperature":   {"all": True,  "malta": True,  "gozo": True},
    "Precipitation":     {"all": True,  "malta": True,  "gozo": True},
    "Pressure":          {"all": True,  "malta": False, "gozo": False},
    "Relative_Humidity": {"all": True,  "malta": True,  "gozo": True},
    "Solar":             {"all": True,  "malta": True,  "gozo": True},
    "Temperature":       {"all": True,  "malta": True,  "gozo": True},
    "Wind_Direction":    {"all": False, "malta": False, "gozo": False},
    "Wind_Gust":         {"all": True,  "malta": True,  "gozo": True},
    "Wind_Speed":        {"all": True,  "malta": True,  "gozo": True},
}

# Stations to omit from specific variables (e.g. unreliable pressure record)
EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},
    "Solar": {"Selmun"},
}

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

SMOOTH_WINDOW = 31   # rolling mean window in days; set to 1 to disable

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

STATION_COLOURS = {
    "Giordan":      "#1f78b4",   # blue
    "San-Lawrenz":  "#33a02c",   # green
    "Xagħra":       "#6a3d9a",   # purple
    "Xewkija":      "#a6cee3",   # light blue
    "Selmun":       "#e31a1c",   # red
    "Dingli":       "#ff7f00",   # orange
    "Zokrija":      "#b15928",   # brown
    "Birkirkara":   "#fb9a99",   # pink
    "Valletta":     "#fdbf6f",   # peach
    "Luqa":         "#cab2d6",   # lavender
    "Bengħajsa":    "#b2df8a",   # light green
}

AVG_ALL_COLOUR   = "#000000"   # black
AVG_MALTA_COLOUR = "#d62728"   # strong red
AVG_GOZO_COLOUR  = "#1f77b4"   # strong blue

AVG_LINEWIDTH = 2.5
STN_LINEWIDTH = 1.2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_daily(station):
    """Load Daily_{station}.csv, parse the date column. Returns DataFrame or None."""
    filepath = f"{workdir}/Data/Daily_{station}.csv"
    if not os.path.exists(filepath):
        print(f"  [warn] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath, header=0)
    if DATE_COL not in df.columns:
        print(f"  [warn] '{DATE_COL}' column not found in {filepath}. Columns: {list(df.columns)}")
        return None
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df[df['Date'] >= pd.Timestamp("2007-01-01")]
    df = df.dropna(subset=[DATE_COL])
    return df


def find_col(df, varname, station):
    """Return the matching column name for varname in df, with fallback."""
    target = COL_MAP.get(varname, varname)
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    for col in df.columns:
        if norm(col) == norm(target):
            return col
    print(f"  [warn] Column '{target}' not found for {station}. Available: {list(df.columns)}")
    return None


def daily_climatology(df, col):
    """
    Compute mean value for each day-of-year (1-366).
    Missing sentinel and NaN are excluded.
    Returns a Series indexed 1-366.
    """
    series = df[col].replace(MISSING_VAL, np.nan)
    series = pd.to_numeric(series, errors="coerce")
    doy = df[DATE_COL].dt.dayofyear
    clim = series.groupby(doy).mean()
    clim = clim.reindex(range(1, 367))
    return clim


def smooth(series, window):
    """
    Centred rolling mean with wrap-around padding at year boundaries
    to avoid edge artifacts in January and December.
    """
    if window <= 1:
        return series
    vals = series.values
    pad = window // 2
    padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
    smoothed = pd.Series(padded).rolling(window, center=True, min_periods=1).mean().values
    return pd.Series(smoothed[pad:-pad], index=series.index)


def month_tick_positions():
    """Return (tick_doys, labels) for the first of each month."""
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    return starts, months


def group_mean(clim_dict, members):
    """Average climatologies across a list of station names present in clim_dict."""
    available = [clim_dict[s] for s in members if s in clim_dict]
    if not available:
        return None
    return pd.concat(available, axis=1).mean(axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_annual_cycle_plot(varname, clim_dict, output_dir):
    """Draw the annual cycle plot for one variable."""

    if not clim_dict:
        print(f"  [skip] No data for {varname}.")
        return

    show = SHOW_AVG.get(varname, {"all": False, "malta": False, "gozo": False})
    vlab = VARLABELS.get(varname, varname)
    unit = UNITS.get(varname, "")
    tick_pos, tick_labels = month_tick_positions()
    x = np.arange(1, 367)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.margins(x=0)

    # --- Draw individual station lines ---
    for station in ALL_STATIONS:
        if station not in clim_dict:
            continue
        colour = STATION_COLOURS.get(station, "#888888")
        ax.plot(x, clim_dict[station].reindex(x).values,
                color=colour, linewidth=STN_LINEWIDTH, alpha=0.85, zorder=3)

    # --- Compute group averages (needed for both drawing and legend) ---
    avg_all = avg_malta = avg_gozo = None

    if show.get("all"):
        avg_all = group_mean(clim_dict, [s for s in ALL_STATIONS if s in clim_dict])
        if avg_all is not None:
            ax.plot(x, smooth(avg_all, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_ALL_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    if show.get("malta"):
        avg_malta = group_mean(clim_dict, [s for s in MALTA_STATIONS if s in clim_dict])
        if avg_malta is not None:
            ax.plot(x, smooth(avg_malta, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    if show.get("gozo"):
        avg_gozo = group_mean(clim_dict, [s for s in GOZO_STATIONS if s in clim_dict])
        if avg_gozo is not None:
            ax.plot(x, smooth(avg_gozo, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_GOZO_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    # Fixed layout: 5 cols x 3 rows. Cols 0-3 = stations, col 4 = averages.
    NCOL = 5
    NROW = 3
    FIXED_STATIONS = ALL_STATIONS
    spacer = mlines.Line2D([], [], color="none", label="")

    # Build station handles — greyed out if station has no data for this variable
    station_handles = []
    blnkc = "#ffffff"
    blnkl = 0.0
    for station in FIXED_STATIONS:
        if station in clim_dict:
            station_handles.append(
                mlines.Line2D([], [], color=STATION_COLOURS.get(station, "#888888"),
                              linewidth=STN_LINEWIDTH,
                              label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " "))
                )
        else:
            station_handles.append(
                mlines.Line2D([], [], color=blnkc, linewidth=blnkl,
                              label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " "))
                )

    # Average handles
    avg_handles = []
    avgg = "Avg (Gozo)"
    avgm = "Avg (Malta)"
    avga = "Avg (all stations)"
    if avg_gozo  is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_GOZO_COLOUR,  
        linewidth=AVG_LINEWIDTH, linestyle="--", label=avgg))
    else:   avg_handles.append(mlines.Line2D([], [], color=blnkc, linewidth=0., label=avgg))
    if avg_malta is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_MALTA_COLOUR, 
        linewidth=AVG_LINEWIDTH, linestyle="--", label=avgm))
    else:   avg_handles.append(mlines.Line2D([], [], color=blnkc, linewidth=0., label=avgm))
    if avg_all   is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_ALL_COLOUR,   
        linewidth=AVG_LINEWIDTH, linestyle="--", label=avga))
    else:   avg_handles.append(mlines.Line2D([], [], color=blnkc, linewidth=0., label=avga))

    # Rearrange 11 station handles into a 4-col x 3-row block (row-major display).
    # Matplotlib fills column-first, so we map display position (row, col) to
    # the correct slot index: slot = col * NROW + row.
    # Station i sits at display (row = i//4, col = i%4) → slot = (i%4)*3 + (i//4)
    stn_slots = [spacer] * (4 * NROW)   # 12 slots, last one stays blank
    for i, h in enumerate(station_handles):
        row = i // 4
        col = i % 4
        stn_slots[col * NROW + row] = h

    legend_handles = stn_slots + avg_handles   # 12 + 3 = 15 total
    title_pad = 8 + NROW * 14

    leg = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=NCOL,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
        handlelength=1.8,
    )

    ax.set_title(
        f"{vlab}",
        fontsize=13, fontweight="bold", pad=title_pad,
    )

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("months")

    desc, unit = VAR_AXIS.get(varname, (VARLABELS.get(varname, varname), UNITS.get(varname, "")))
    ax.set_ylabel(f"{desc} ({unit})")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annual-cycle_{varname}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir=f"{workdir}/analysis/annual-cycle"):

    print("[load] Reading daily data files...")
    daily_data = {}
    for station in ALL_STATIONS:
        df = load_daily(station)
        if df is not None:
            daily_data[station] = df
            print(f"  {station:20s}  {len(df)} rows")

    for varname in VARNAMES:
        print(f"\n[var] {varname}")
        clim_dict = {}
        for station, df in daily_data.items():
            if station in EXCLUDE_STATIONS.get(varname, set()):
                print(f"  {station:20s}  [excluded from {varname}]")
                continue
            col = find_col(df, varname, station)
            if col is None:
                continue
            clim = daily_climatology(df, col)
            clim = smooth(clim, SMOOTH_WINDOW)
            clim_dict[station] = clim
            print(f"  {station:20s}  valid doys: {clim.notna().sum()}")

        create_annual_cycle_plot(varname, clim_dict, output_dir)

    print(f"\n[done] All plots written to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_output_dir = sys.argv[1] if len(sys.argv) > 1 else f"{workdir}/analysis/annual-cycle"
    main(run_output_dir)
