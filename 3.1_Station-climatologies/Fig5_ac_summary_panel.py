"""
Figure panel: annual-cycle climatologies (2x2)
==============================================
Combines per-variable annual-cycle plots into one panel image.
Reuses the computation logic from plot_annual_cycle_summary.py; only the
drawing is retargeted to a shared 2x2 grid of axes.

Usage:
    python3 figure_annual_cycle_panel.py

Output: {workdir}/analysis/annual-cycle/figure_annual_cycle_panel.png
"""

import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"

# Panel layout, row-major (left->right, top->bottom)
PANEL_VARS = [
    "Precipitation", "Temperature",
    "Relative_Humidity", "Wind_Speed",
]
NROWS = 2
NCOLS = 2

ALL_STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]
GOZO_STATIONS  = ["Giordan", "Xagħra", "Xewkija"]
MALTA_STATIONS = ["Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

VARLABELS = {
    "Max_Temperature": "Max Temperature", "Min_Temperature": "Min Temperature",
    "Precipitation": "Precipitation", "Pressure": "Pressure",
    "Relative_Humidity": "Relative Humidity", "Solar": "Solar Radiance",
    "Temperature": "Temperature", "Wind_Direction": "Wind Direction",
    "Wind_Gust": "Wind Gust", "Wind_Speed": "Wind Speed",
}
VAR_AXIS = {
    "Max_Temperature": ("temperature", "°C"), "Min_Temperature": ("temperature", "°C"),
    "Temperature": ("temperature", "°C"), "Precipitation": ("precipitation", "mm/day"),
    "Pressure": ("pressure", "hPa"), "Relative_Humidity": ("relative humidity", "%"),
    "Solar": ("solar irradiance", "W/m²"), "Wind_Speed": ("wind speed", "m/s"),
    "Wind_Gust": ("wind gust", "m/s"), "Wind_Direction": ("wind direction", "°"),
}
COL_MAP = {
    "Max_Temperature": "Max_Temperature", "Min_Temperature": "Min_Temperature",
    "Precipitation": "Precipitation", "Pressure": "Pressure",
    "Relative_Humidity": "Relative_Humidity", "Solar": "Solar",
    "Temperature": "Temperature", "Wind_Direction": "Wind_Direction",
    "Wind_Gust": "Wind_Gust", "Wind_Speed": "Wind_Speed",
}

DATE_COL    = "Date"
MISSING_VAL = "-"

SHOW_AVG = {
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
EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},
    "Solar": {"Selmun"},
}

SMOOTH_WINDOW = 31

STATION_COLOURS = {
    "Giordan": "#1f78b4", "San-Lawrenz": "#33a02c", "Xagħra": "#6a3d9a",
    "Xewkija": "#a6cee3", "Selmun": "#e31a1c", "Dingli": "#ff7f00",
    "Zokrija": "#b15928", "Birkirkara": "#fb9a99", "Valletta": "#fdbf6f",
    "Luqa": "#cab2d6", "Bengħajsa": "#b2df8a",
}
AVG_ALL_COLOUR   = "#000000"
AVG_MALTA_COLOUR = "#d62728"
AVG_GOZO_COLOUR  = "#1f77b4"
AVG_LINEWIDTH = 2.5
STN_LINEWIDTH = 1.2

# ---------------------------------------------------------------------------
# Computation helpers — copied unchanged from plot_annual_cycle_summary.py
# ---------------------------------------------------------------------------

def load_daily(station):
    filepath = f"{workdir}/Data/Daily_{station}.csv"
    if not os.path.exists(filepath):
        print(f"  [warn] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath, header=0)
    if DATE_COL not in df.columns:
        return None
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
    return df.dropna(subset=[DATE_COL])

def find_col(df, varname, station):
    target = COL_MAP.get(varname, varname)
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    for col in df.columns:
        if norm(col) == norm(target):
            return col
    return None

def daily_climatology(df, col):
    series = pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")
    doy = df[DATE_COL].dt.dayofyear
    return series.groupby(doy).mean().reindex(range(1, 367))

def smooth(series, window):
    if window <= 1:
        return series
    vals = series.values
    pad = window // 2
    padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
    sm = pd.Series(padded).rolling(window, center=True, min_periods=1).mean().values
    return pd.Series(sm[pad:-pad], index=series.index)

def month_tick_positions():
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    return starts, months

def group_mean(clim_dict, members):
    available = [clim_dict[s] for s in members if s in clim_dict]
    if not available:
        return None
    return pd.concat(available, axis=1).mean(axis=1)

def build_clim_dict(varname, daily_data):
    clim_dict = {}
    for station, df in daily_data.items():
        if station in EXCLUDE_STATIONS.get(varname, set()):
            continue
        col = find_col(df, varname, station)
        if col is None:
            continue
        clim_dict[station] = smooth(daily_climatology(df, col), SMOOTH_WINDOW)
    return clim_dict

# ---------------------------------------------------------------------------
# Drawing — one variable onto a given axis
# ---------------------------------------------------------------------------

def draw_annual_cycle(ax, varname, clim_dict, show_xlabels):
    show = SHOW_AVG.get(varname, {"all": False, "malta": False, "gozo": False})
    tick_pos, tick_labels = month_tick_positions()
    x = np.arange(1, 367)
    ax.margins(x=0)

    for station in ALL_STATIONS:
        if station not in clim_dict:
            continue
        ax.plot(x, clim_dict[station].reindex(x).values,
                color=STATION_COLOURS.get(station, "#888888"),
                linewidth=STN_LINEWIDTH, alpha=0.85, zorder=3)

    if show.get("all"):
        a = group_mean(clim_dict, [s for s in ALL_STATIONS if s in clim_dict])
        if a is not None:
            ax.plot(x, smooth(a, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_ALL_COLOUR, linewidth=AVG_LINEWIDTH, linestyle="--", zorder=5)
    if show.get("malta"):
        a = group_mean(clim_dict, [s for s in MALTA_STATIONS if s in clim_dict])
        if a is not None:
            ax.plot(x, smooth(a, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH, linestyle="--", zorder=5)
    if show.get("gozo"):
        a = group_mean(clim_dict, [s for s in GOZO_STATIONS if s in clim_dict])
        if a is not None:
            ax.plot(x, smooth(a, SMOOTH_WINDOW).reindex(x).values,
                    color=AVG_GOZO_COLOUR, linewidth=AVG_LINEWIDTH, linestyle="--", zorder=5)

    ax.set_xticks(tick_pos)
    if show_xlabels:
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("months")
    else:
        ax.set_xticklabels([])          # keep ticks, drop labels

    desc, unit = VAR_AXIS.get(varname,
                              (VARLABELS.get(varname, varname), ""))
    ax.set_ylabel(f"{desc} ({unit})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

# ---------------------------------------------------------------------------
# Shared legend
# ---------------------------------------------------------------------------

def build_legend_handles():
    """One row-major block: 9 stations then the 3 averages."""
    handles = []
    for station in ALL_STATIONS:
        handles.append(mlines.Line2D([], [], color=STATION_COLOURS.get(station, "#888888"),
                       linewidth=STN_LINEWIDTH,
                       label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " ")))
    handles.append(mlines.Line2D([], [], color=AVG_GOZO_COLOUR,  linewidth=AVG_LINEWIDTH,
                   linestyle="--", label="Avg (Gozo)"))
    handles.append(mlines.Line2D([], [], color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH,
                   linestyle="--", label="Avg (Malta)"))
    handles.append(mlines.Line2D([], [], color=AVG_ALL_COLOUR,   linewidth=AVG_LINEWIDTH,
                   linestyle="--", label="Avg (all stations)"))
    return handles

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[load] Reading daily data files...")
    daily_data = {}
    for station in ALL_STATIONS:
        df = load_daily(station)
        if df is not None:
            daily_data[station] = df

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 6.0, NROWS * 3.8),
                             squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, varname in enumerate(PANEL_VARS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        clim_dict = build_clim_dict(varname, daily_data)
        is_bottom_row = idx >= len(PANEL_VARS) - NCOLS
        draw_annual_cycle(ax, varname, clim_dict, show_xlabels=is_bottom_row)
        ax.set_title(f"({letters[idx]})", fontsize=12, fontweight="bold", loc="left")

    for idx in range(len(PANEL_VARS), NROWS * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    # Single shared legend across the very top, one row
    fig.legend(handles=build_legend_handles(),
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=6, fontsize=9, frameon=True, handlelength=1.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])   # headroom for the legend
    out_path = f"{workdir}/analysis/annual-cycle/figure_annual_cycle_panel.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
