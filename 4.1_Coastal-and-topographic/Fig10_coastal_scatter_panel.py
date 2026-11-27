"""
Figure panel: coastal-distance vs elevation scatter (2x2)
=========================================================
Combines per-variable coastal scatter plots into one panel.
Reuses the computation logic from plot_coastal_scatter.py; only the
drawing is retargeted to a shared 2x2 grid, with per-panel colourbars
placed inside each axes.

Usage:
    python3 Fig10_coastal_scatter_panel.py

Output: {workdir}/analysis/coastal-scatter/Fig10_coastal_scatter_panel.png
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
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

# Panel order (left->right, top->bottom)
PANEL_VARS = [
    ("Min_Temperature", "mean"),
    ("Max_Temperature", "mean"),
    ("DTR",             "mean"),
    ("Wind_Speed",      "mean"),
]
NROWS = 2
NCOLS = 2

# Colourbar inset box, in DATA coordinates (x: km from coast, y: elevation m)
CBAR_BOX = dict(x0=1.5, x1=3.4, y0=20, y1=35)   # move up (e.g. y0=55,y1=95) if it clips VL/BN

SHOW_COAST_ARROWS = True

# ---------------------------------------------------------------------------
# Config copied from plot_coastal_scatter.py
# ---------------------------------------------------------------------------

class HandlerArrow(HandlerPatch):
    def __init__(self, bearing_deg=0, **kwargs):
        self.bearing_deg = bearing_deg
        super().__init__(**kwargs)
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        rad = np.radians(self.bearing_deg)
        cx, cy = width / 2, height / 2
        length = min(width, height) * 1.8
        dx = np.sin(rad) * length / 2
        dy = np.cos(rad) * length / 2
        arrow = FancyArrow(cx - dx, cy - dy, 2 * dx, 2 * dy,
                           width=length * 0.06, head_width=length * 0.40,
                           head_length=length * 0.40, fc="#1a1a2e", ec="#1a1a2e",
                           transform=trans, length_includes_head=True)
        return [arrow]

def dms(d, m, s):
    return d + m / 60 + s / 3600

STATIONS = [
    ("Luqa","LQ",dms(35,51,13),dms(14,28,49),85),
    ("Bengħajsa","BN",dms(35,48,54),dms(14,31,45),36),
    ("Dingli","DN",dms(35,51,10),dms(14,23,4),250),
    ("Msida","MS",dms(35,53,33),dms(14,29,22),66),
    ("Valletta","VL",dms(35,54,9),dms(14,31,8),31),
    ("Selmun","SL",dms(35,57,34),dms(14,22,53),129),
    ("Xagħra","XR",dms(36,3,2),dms(14,16,2),155),
    ("Xewkija","XW",dms(36,1,35),dms(14,16,20),104),
    ("Giordan","GD",dms(36,4,24),dms(14,13,8),167),
]
STATION_FILENAMES = {
    "LQ":["Luqa"], "BN":["Bengħajsa","Benghajsa"], "DN":["Dingli"],
    "MS":["Msida","Birkirkara"], "VL":["Valletta"], "SL":["Selmun"],
    "XR":["Xagħra","Xaghra"], "XW":["Xewkija"], "GD":["Giordan"],
}
URBAN_STATIONS = {"VL", "MS", "LQ", "XR"}

COASTLINE_PATH = f"{workdir}/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
OUTPUT_DIR     = f"{workdir}/analysis/coastal-scatter"
MISSING_VAL = "-"
DATE_COL    = "Date"

COL_MAP = {
    "Max_Temperature":"Max Temperature", "Min_Temperature":"Min Temperature",
    "Precipitation":"Precipitation", "Pressure":"Pressure",
    "Relative_Humidity":"Relative Humidity", "Solar":"Solar",
    "Temperature":"Temperature", "Wind_Direction":"Wind Direction",
    "Wind_Gust":"Wind Gust", "Wind_Speed":"Wind Speed",
}
VARLABELS = {
    "DTR":"Diurnal Temperature Range", "Max_Temperature":"Max Temperature",
    "Min_Temperature":"Min Temperature", "Wind_Speed":"Wind Speed",
}
UNITS = {"DTR":"°C", "Max_Temperature":"°C", "Min_Temperature":"°C", "Wind_Speed":"m/s"}

_wind_colours = ["#cce5ff", "#4da6ff", "#0055cc", "#001a4d", "#000000"]
WIND_CMAP = LinearSegmentedColormap.from_list("wind_intensity", _wind_colours)
VAR_CMAP = {
    "DTR":"RdYlBu_r", "Wind_Speed":WIND_CMAP,
    "Max_Temperature":"RdYlBu_r", "Min_Temperature":"RdYlBu_r",
}
DEFAULT_CMAP = "plasma"

# ---------------------------------------------------------------------------
# Computation copied from plot_coastal_scatter.py
# ---------------------------------------------------------------------------

def _col(df, varname):
    target = COL_MAP.get(varname, varname)
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    m = [c for c in df.columns if norm(c) == norm(target)]
    return m[0] if m else None

def _series(df, varname):
    col = _col(df, varname)
    if col is None:
        return None
    return pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")

def compute_stat(df, varname, stat):
    try:
        if varname == "DTR":
            tmax, tmin = _series(df, "Max_Temperature"), _series(df, "Min_Temperature")
            if tmax is None or tmin is None:
                return None
            dtr = (tmax - tmin).dropna()
            return float(dtr.mean()) if len(dtr) else None
        s = _series(df, varname)
        if s is None:
            return None
        s = s.dropna()
        return float(s.mean()) if len(s) else None
    except Exception:
        return None

def compute_coastal_distances_m():
    print("[coast] Loading GSHHG coastline ...")
    coast = gpd.read_file(COASTLINE_PATH).cx[14.05:14.65, 35.75:36.12]
    coast_utm = coast.to_crs("EPSG:32633")
    coast_geom = unary_union(coast_utm.geometry).boundary
    stn = gpd.GeoDataFrame(
        [{"abbr": a, "geometry": Point(lon, lat)} for _, a, lat, lon, _ in STATIONS],
        crs="EPSG:4326").to_crs("EPSG:32633")
    distances, directions = {}, {}
    for _, row in stn.iterrows():
        pt = row.geometry
        d = pt.distance(coast_geom)
        nearest = coast_geom.interpolate(coast_geom.project(pt))
        bearing = (np.degrees(np.arctan2(nearest.x - pt.x, nearest.y - pt.y)) + 360) % 360
        distances[row["abbr"]] = d
        directions[row["abbr"]] = bearing
    return distances, directions

# ---------------------------------------------------------------------------
# Drawing one scatter onto an axis
# ---------------------------------------------------------------------------

def draw_scatter(ax, varname, stat, stat_values, distances_m, directions,
                 elevations, show_xlabel, show_ylabel, show_legend):
    valid   = {k: v for k, v in stat_values.items() if v is not None}
    missing = {k for k, v in stat_values.items() if v is None}

    raw_cmap = VAR_CMAP.get(varname, DEFAULT_CMAP)
    cmap_obj = raw_cmap if isinstance(raw_cmap, mcolors.Colormap) else matplotlib.colormaps[raw_cmap]

    if valid:
        vals = np.array(list(valid.values()))
        vmin, vmax = vals.min(), vals.max()
        pad = (vmax - vmin) * 0.05 if vmax != vmin else 0.5
        norm = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)
    else:
        norm = mcolors.Normalize(0, 1)

    for abbr in missing:
        if abbr in distances_m:
            m = "s" if abbr in URBAN_STATIONS else "o"
            ax.scatter(distances_m[abbr]/1000, elevations[abbr], c="#aaaaaa",
                       s=70, marker=m, edgecolors="black", linewidths=0.8, zorder=4)
    for abbr, val in valid.items():
        if abbr in distances_m:
            m = "s" if abbr in URBAN_STATIONS else "o"
            ax.scatter(distances_m[abbr]/1000, elevations[abbr],
                       c=[cmap_obj(norm(val))], s=110, marker=m,
                       edgecolors="black", linewidths=0.8, zorder=5)
    for abbr in list(valid) + list(missing):
        if abbr in distances_m:
            ax.annotate(abbr, (distances_m[abbr]/1000, elevations[abbr]),
                        textcoords="offset points", xytext=(5, -10),
                        fontsize=7, fontweight="bold", color="#1a1a2e")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if show_xlabel:
        ax.set_xlabel("Distance from coastline (km)", fontsize=10)
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Elevation (m)", fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.4, linestyle="--")

    # Coast arrows (fixed visual length via axis size)
    if SHOW_COAST_ARROWS:
        fig = ax.figure
        ax_bbox = ax.get_position()
        ax_w_in = ax_bbox.width  * fig.get_size_inches()[0]
        ax_h_in = ax_bbox.height * fig.get_size_inches()[1]
        xr = ax.get_xlim()[1] - ax.get_xlim()[0]
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        alx = 0.24 * xr / ax_w_in
        aly = 0.24 * yr / ax_h_in
        for abbr, bearing in directions.items():
            if abbr not in distances_m:
                continue
            x0, y0 = distances_m[abbr]/1000, elevations[abbr]
            rad = np.radians(bearing)
            ax.annotate("", xy=(x0 + np.sin(rad)*alx, y0 + np.cos(rad)*aly),
                        xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#1a1a2e",
                                        lw=1.1, mutation_scale=7), zorder=3)

    # Per-panel colourbar inset, in data coordinates
    b = CBAR_BOX
    cax = ax.inset_axes([b["x0"], b["y0"], b["x1"] - b["x0"], b["y1"] - b["y0"]],
                        transform=ax.transData)
    cb = ColorbarBase(cax, cmap=cmap_obj, norm=norm, orientation="horizontal")
    cb.ax.tick_params(labelsize=8)
    # no colourbar label, per spec

    # Legend only on the requested panel
    if show_legend:
        legend_elements = [
            Line2D([0],[0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="black", markeredgewidth=1.5, markersize=8,
                   linestyle="none", label="Rural station"),
            Line2D([0],[0], marker="s", color="none", markerfacecolor="none",
                   markeredgecolor="black", markeredgewidth=1.5, markersize=8,
                   linestyle="none", label="Urban station"),
            Line2D([0],[0], marker="o", color="none", markerfacecolor="#aaaaaa",
                   markeredgecolor="black", markeredgewidth=1.0, markersize=8,
                   linestyle="none", label="No data"),
        ]
        handler_map = {}
        if SHOW_COAST_ARROWS:
            coast_patch = FancyArrow(0,0,1,0); north_patch = FancyArrow(0,0,1,0)
            coast_patch.set_label("Direction to nearest coast")
            north_patch.set_label("North (0°)")
            legend_elements += [coast_patch, north_patch]
            handler_map[coast_patch] = HandlerArrow(bearing_deg=45)
            handler_map[north_patch] = HandlerArrow(bearing_deg=0)
        ax.legend(handles=legend_elements, handler_map=handler_map,
                  loc="upper right", fontsize=7.5, framealpha=0.9,
                  title="Legend", title_fontsize=7.5)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    distances_m, directions = compute_coastal_distances_m()
    elevations = {abbr: elev for _, abbr, _, _, elev in STATIONS}

    print("[load] Reading daily data files ...")
    daily_data = {}
    for name, abbr, lat, lon, elev in STATIONS:
        for cand in STATION_FILENAMES.get(abbr, []):
            trial = f"{workdir}/Data/Daily_{cand}.csv"
            if os.path.exists(trial):
                daily_data[abbr] = pd.read_csv(trial, header=0)
                d = pd.read_csv(trial, header=0)
                d[DATE_COL] = pd.to_datetime(d[DATE_COL], dayfirst=True, errors="coerce")
                daily_data[abbr] = d[d[DATE_COL] >= pd.Timestamp("2007-01-01")]
                break

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(11.0, 8.5), squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, (varname, stat) in enumerate(PANEL_VARS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        stat_values = {}
        for abbr in [s[1] for s in STATIONS]:
            stat_values[abbr] = (compute_stat(daily_data[abbr], varname, stat)
                                 if abbr in daily_data else None)
        is_bottom = idx >= len(PANEL_VARS) - NCOLS
        is_left   = (col == 0)
        is_topright = (idx == 1)          # top-right panel = (b)
        draw_scatter(ax, varname, stat, stat_values, distances_m, directions,
                     elevations, show_xlabel=is_bottom, show_ylabel=is_left,
                     show_legend=is_topright)
        ax.set_title(f"({letters[idx]}) {VARLABELS.get(varname, varname)}",
                     fontsize=10, fontweight="bold", loc="left")

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "Fig10_coastal_scatter_panel.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=600)
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
