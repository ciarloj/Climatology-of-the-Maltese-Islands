"""
Figure panel: spatial mean maps (3x2)
=====================================
Combines per-variable spatial mean maps into one panel image.
Reuses the computation + map logic from plot_spatial_stats.py; only the
drawing is retargeted to a shared 3x2 grid of axes, with shared outer
longitude (bottom) and latitude (left) labelling.

All panels show the MEAN of each variable.

Usage:
    python3 figure_spatial_panel.py

Output: {workdir}/analysis/spatial/figure_spatial_panel.png
"""

import os
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

# Panel order (left->right, top->bottom). All shown as MEAN.
PANEL_VARS = [
    "Precipitation", "Temperature",
    "Max_Temperature", "Min_Temperature",
    "Relative_Humidity", "Wind_Speed",
]
NROWS = 3
NCOLS = 2
STAT_KEY = "average"          # all panels use the mean

_wind_colours = ["#cce5ff", "#4da6ff", "#0055cc", "#001a4d", "#000000"]
WIND_CMAP = LinearSegmentedColormap.from_list("wind_intensity", _wind_colours)

def dms(d, m, s):
    return d + m / 60 + s / 3600

STATIONS = [
    ("Luqa",      "LQ", dms(35,51,13), dms(14,28,49),  85),
    ("Bengħajsa", "BN", dms(35,48,54), dms(14,31,45),  36),
    ("Dingli",    "DN", dms(35,51,10), dms(14,23, 4), 250),
    ("Msida",     "MS", dms(35,53,33), dms(14,29,22),  66),
    ("Valletta",  "VL", dms(35,54, 9), dms(14,31, 8),  31),
    ("Selmun",    "SL", dms(35,57,34), dms(14,22,53), 129),
    ("Xagħra",    "XR", dms(36, 3, 2), dms(14,16, 2), 155),
    ("Xewkija",   "XW", dms(36, 1,35), dms(14,16,20), 104),
    ("Giordan",   "GD", dms(36, 4,24), dms(14,13, 8), 167),
]

STATION_FILENAMES = {
    "LQ": ["Luqa"], "BN": ["Bengħajsa", "Benghajsa"], "DN": ["Dingli"],
    "MS": ["Msida", "Birkirkara"], "VL": ["Valletta"], "SL": ["Selmun"],
    "XR": ["Xagħra", "Xaghra"], "XW": ["Xewkija"], "GD": ["Giordan"],
}

LABEL_OFFSETS = {
    "LQ": ( 0.008,-0.010), "BN": (-0.020, 0.006), "DN": ( 0.008, 0.005),
    "MS": (-0.010,-0.012), "VL": ( 0.004, 0.004), "SL": (-0.010,-0.014),
    "XR": ( 0.008, 0.005), "XW": ( 0.008,-0.002), "GD": ( 0.008, 0.000),
}

MISSING_VAL = "-"
DATE_COL    = "Date"

COL_MAP = {
    "Max_Temperature": "Max Temperature", "Min_Temperature": "Min Temperature",
    "Precipitation": "Precipitation", "Pressure": "Pressure",
    "Relative_Humidity": "Relative Humidity", "Solar": "Solar",
    "Temperature": "Temperature", "Wind_Direction": "Wind Direction",
    "Wind_Gust": "Wind Gust", "Wind_Speed": "Wind Speed",
}
VARLABELS = {
    "Max_Temperature": "Max Temperature", "Min_Temperature": "Min Temperature",
    "Precipitation": "Precipitation", "Pressure": "Pressure",
    "Relative_Humidity": "Relative Humidity", "Solar": "Solar Radiance",
    "Temperature": "Temperature", "Wind_Direction": "Wind Direction",
    "Wind_Gust": "Wind Gust", "Wind_Speed": "Wind Speed",
}
UNITS = {
    "Max_Temperature": "°C", "Min_Temperature": "°C", "Precipitation": "mm",
    "Pressure": "hPa", "Relative_Humidity": "%", "Solar": "W/m²",
    "Temperature": "°C", "Wind_Direction": "°", "Wind_Gust": "m/s",
    "Wind_Speed": "m/s",
}

ANNUAL_SUM_VARS = {"Precipitation"}
SKIP_STATIONS = {"Pressure": {"GD"}}

DOT_SIZE       = 90
DOT_EDGECOLOR  = "white"
DOT_LINEWIDTH  = 0.8
MISSING_COLOUR = "#aaaaaa"
MISSING_SIZE   = 55
CMAP           = "RdYlBu_r"
VAR_CMAP = {
    "Precipitation": "YlGnBu", "Relative_Humidity": "YlGnBu",
    "Wind_Speed": WIND_CMAP, "Wind_Gust": WIND_CMAP,
    "Temperature": "RdYlBu_r", "Max_Temperature": "RdYlBu_r",
    "Min_Temperature": "RdYlBu_r",
}
LAND_COLOUR   = "#e8d5b0"
SEA_COLOUR    = "#c9e8f5"
BORDER_COLOUR = "#8b7355"
LABEL_FONT    = 6
LABEL_COLOUR  = "#1a1a2e"

COASTLINE_PATH = f"{workdir}/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
OUTPUT_DIR     = f"{workdir}/analysis/spatial"
LON_MIN, LON_MAX = 14.10, 14.62
LAT_MIN, LAT_MAX = 35.77, 36.10

STAT_FUNC = lambda s: s.mean()      # all panels are means

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_station_series(abbr, varname):
    candidates = STATION_FILENAMES.get(abbr, [])
    filepath = None
    for cand in candidates:
        trial = f"{workdir}/Data/Daily_{cand}.csv"
        if os.path.exists(trial):
            filepath = trial
            break
    if filepath is None:
        return None
    try:
        df = pd.read_csv(filepath, header=0)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        col = COL_MAP.get(varname)
        if col is None or col not in df.columns:
            norm = lambda s: s.lower().replace(" ", "_")
            match = [c for c in df.columns if norm(c) == norm(col or varname)]
            if not match:
                return None
            col = match[0]
        series = pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")
        if varname in ANNUAL_SUM_VARS:
            tmp = pd.DataFrame({"date": df[DATE_COL], "val": series})
            return tmp.groupby(tmp["date"].dt.year)["val"].sum().dropna()
        return series.dropna()
    except Exception as e:
        print(f"  [warn] {abbr}/{varname}: {e}")
        return None

def compute_values(varname):
    vals = {}
    for name, abbr, lat, lon, elev in STATIONS:
        if abbr in SKIP_STATIONS.get(varname, set()):
            vals[abbr] = None
            continue
        s = load_station_series(abbr, varname)
        vals[abbr] = float(STAT_FUNC(s)) if (s is not None and len(s) > 0) else None
    return vals

# ---------------------------------------------------------------------------
# Drawing one map onto an axis
# ---------------------------------------------------------------------------

def draw_map(ax, varname, values, malta, show_xlabels, show_ylabels):
    vlab = VARLABELS.get(varname, varname)
    unit = UNITS.get(varname, "")

    valid   = {k: v for k, v in values.items() if v is not None}
    missing = {k for k, v in values.items() if v is None}

    ax.set_facecolor(SEA_COLOUR)
    malta.plot(ax=ax, color=LAND_COLOUR, edgecolor=BORDER_COLOUR, linewidth=0.7)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    raw_cmap = VAR_CMAP.get(varname, CMAP)
    cmap_obj = raw_cmap if isinstance(raw_cmap, mcolors.Colormap) else matplotlib.colormaps[raw_cmap]

    if valid:
        arr = np.array(list(valid.values()))
        vmin, vmax = arr.min(), arr.max()
        pad = (vmax - vmin) * 0.05 if vmax != vmin else 0.5
        norm = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)
    else:
        norm = mcolors.Normalize(0, 1)

    for name, abbr, lat, lon, elev in STATIONS:
        if abbr in missing:
            ax.scatter(lon, lat, c=MISSING_COLOUR, s=MISSING_SIZE,
                       edgecolors="white", linewidths=0.7, zorder=5)

    lons = [s[3] for s in STATIONS if s[1] in valid]
    lats = [s[2] for s in STATIONS if s[1] in valid]
    vv   = [valid[s[1]] for s in STATIONS if s[1] in valid]
    if valid:
        ax.scatter(lons, lats, c=vv, cmap=cmap_obj, norm=norm, s=DOT_SIZE,
                   edgecolors=DOT_EDGECOLOR, linewidths=DOT_LINEWIDTH, zorder=6)

    for name, abbr, lat, lon, elev in STATIONS:
        dx, dy = LABEL_OFFSETS.get(abbr, (0.008, 0.005))
        ax.text(lon + dx, lat + dy, abbr, fontsize=LABEL_FONT,
                fontweight="bold", color=LABEL_COLOUR, zorder=7)

    ax.set_xticks(np.arange(14.1, 14.65, 0.1))
    ax.set_yticks(np.arange(35.8, 36.15, 0.1))
    ax.tick_params(labelsize=7)
    if show_xlabels:
        ax.set_xlabel("Longitude (°E)", fontsize=9)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_ylabel("Latitude (°N)", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")

    # Colourbar in the empty sea in the lower-left, in data coordinates
    cb_x0, cb_x1 = LON_MIN + 0.01, 14.35          # left edge -> 14.35°E
    cb_y0, cb_y1 = LAT_MIN + 0.02, 35.805          # sit low in the sea band
    cax = ax.inset_axes([cb_x0, cb_y0, cb_x1 - cb_x0, cb_y1 - cb_y0],
                        transform=ax.transData)
    cb = ColorbarBase(cax, cmap=cmap_obj, norm=norm, orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    malta = gpd.read_file(COASTLINE_PATH).cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 5.2, NROWS * 4.4),
                             squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, varname in enumerate(PANEL_VARS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        values = compute_values(varname)
        is_bottom = idx >= len(PANEL_VARS) - NCOLS
        is_left   = (col == 0)
        draw_map(ax, varname, values, malta,
                 show_xlabels=is_bottom, show_ylabels=is_left)
        unit = UNITS.get(varname, "")
        ax.set_title(f"({letters[idx]}) {VARLABELS.get(varname, varname)} ({unit})",
                     fontsize=10, fontweight="bold", loc="left")

    for idx in range(len(PANEL_VARS), NROWS * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.05,
                    wspace=0.05, hspace=0.01)
    out_path = os.path.join(OUTPUT_DIR, "figure_spatial_panel.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
