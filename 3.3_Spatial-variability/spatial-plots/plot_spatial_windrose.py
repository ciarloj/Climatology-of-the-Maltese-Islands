"""
Spatial Wind Rose Map — Malta & Gozo
======================================
Plots a wind rose for each station overlaid on the map of Malta and Gozo.
Stations without sufficient wind data are shown as grey dots.

Usage:
    python3 plot_spatial_windrose.py

Output: ../../analysis/spatial/spatial_windrose.png
Input:  ../../Data/Daily_{station}.csv
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Station data — (name, abbreviation, lat_deg, lon_deg, elevation_m)
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

def dms(d, m, s):
    return d + m / 60 + s / 3600

STATIONS = [
    ("Luqa",        "LQ",  dms(35, 51, 13),  dms(14, 28, 49),  85),
    ("Bengħajsa",   "BN",  dms(35, 48, 54),  dms(14, 31, 45),  36),
    ("Dingli",      "DN",  dms(35, 51, 10),  dms(14, 23,  4), 250),
    ("Msida",       "MS",  dms(35, 53, 33),  dms(14, 29, 22),  66),
    ("Valletta",    "VL",  dms(35, 54,  9),  dms(14, 31,  8),  31),
    ("Selmun",      "SL",  dms(35, 57, 34),  dms(14, 22, 53), 129),
    ("Xagħra",      "XR",  dms(36,  3,  2),  dms(14, 16,  2), 155),
    ("Xewkija",     "XW",  dms(36,  1, 35),  dms(14, 16, 20), 104),
    ("Giordan",     "GD",  dms(36,  4, 24),  dms(14, 13,  8), 167),
]

STATION_FILENAMES = {
    "LQ":  "Luqa",
    "BN":  "Bengħajsa",
    "DN":  "Dingli",
    "MS":  "Msida",
    "VL":  "Valletta",
    "SL":  "Selmun",
    "XR":  "Xagħra",
    "XW":  "Xewkija",
    "GD":  "Giordan",
}

# ---------------------------------------------------------------------------
# Label offsets (lon, lat) for abbreviation text
# ---------------------------------------------------------------------------

LABEL_OFFSETS = {
    "LQ":  ( 0.008,  0.002),  
    "BN":  (-0.017,  0.010),  
    "DN":  ( 0.008,  0.005),
    "MS":  (-0.010, -0.012),
    "VL":  ( 0.004,  0.004),
    "SL":  (-0.010, -0.014),
    "XR":  ( 0.008,  0.005),
    "XW":  ( 0.008,  0.003),  
    "LW":  ( 0.009, -0.018),  
    "MO":  ( 0.008,  0.005),
    "GD":  ( 0.010,  0.002),  
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATE_COL    = "Date"
MISSING_VAL = "-"

COL_WIND_DIR   = "Wind Direction"
COL_WIND_SPEED = "Wind Speed"

# Minimum number of valid observations required to draw a wind rose
MIN_OBS = 365

# Wind rose bins
N_DIR_BINS   = 16          # number of directional sectors (16 = 22.5° each)

# Wind rose inset size in degrees of lat/lon
# Each rose is drawn as an inset axes centred on the station coordinate
ROSE_SIZE = 0.03   # half-width/height in degrees — tune to taste

# Map bounds
LON_MIN, LON_MAX = 14.10, 14.62
LAT_MIN, LAT_MAX = 35.77, 36.10

# Map appearance
LAND_COLOUR    = "#e8d5b0"
SEA_COLOUR     = "#c9e8f5"
BORDER_COLOUR  = "#8b7355"
MISSING_COLOUR = "#aaaaaa"
LABEL_FONT     = 8
LABEL_COLOUR   = "#1a1a2e"

COASTLINE_PATH = f"{workdir}/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
OUTPUT_PATH    = f"{workdir}/analysis/spatial/spatial_windrose.png"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wind_data(abbr):
    """
    Load wind direction and speed for a station.
    Returns (dir_series, spd_series) aligned on valid rows, or (None, None).
    """
    filename = STATION_FILENAMES.get(abbr)
    if filename is None:
        return None, None
    filepath = f"{workdir}/Data/Daily_{filename}.csv"
    if not os.path.exists(filepath):
        return None, None
    try:
        df = pd.read_csv(filepath, header=0)

        # Find columns — try exact match then normalised fallback
        def find(target):
            if target in df.columns:
                return target
            norm = lambda s: s.lower().replace(" ", "_")
            matches = [c for c in df.columns if norm(c) == norm(target)]
            return matches[0] if matches else None

        dir_col = find(COL_WIND_DIR)
        spd_col = find(COL_WIND_SPEED)
        if dir_col is None or spd_col is None:
            return None, None

        wdir = pd.to_numeric(df[dir_col].replace(MISSING_VAL, np.nan), errors="coerce")
        wspd = pd.to_numeric(df[spd_col].replace(MISSING_VAL, np.nan), errors="coerce")

        # Keep only rows where both are valid
        mask = wdir.notna() & wspd.notna()
        return wdir[mask].reset_index(drop=True), wspd[mask].reset_index(drop=True)

    except Exception as e:
        print(f"  [warn] {abbr}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Wind rose computation
# ---------------------------------------------------------------------------

def compute_windrose(wdir):
    """Returns array of shape (N_DIR_BINS,) with frequency fraction per sector."""
    counts = np.zeros(N_DIR_BINS)
    sector = 360.0 / N_DIR_BINS
    dir_indices = (((wdir.values % 360) + sector / 2) // sector % N_DIR_BINS).astype(int)
    for di in dir_indices:
        counts[di] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts

# ---------------------------------------------------------------------------
# Drawing a single wind rose into an inset axes
# ---------------------------------------------------------------------------

def draw_windrose(fig, centre_lon, centre_lat, rose_data):
    ax_main = fig.axes[0]
    bbox   = ax_main.get_position()
    x_span = LON_MAX - LON_MIN
    y_span = LAT_MAX - LAT_MIN

    cx_frac = bbox.x0 + (centre_lon - LON_MIN) / x_span * bbox.width
    cy_frac = bbox.y0 + (centre_lat - LAT_MIN) / y_span * bbox.height
    rose_w  = ROSE_SIZE / x_span * bbox.width
    rose_h  = ROSE_SIZE / y_span * bbox.height

    ax_rose = fig.add_axes(
        [cx_frac - rose_w, cy_frac - rose_h, rose_w * 2, rose_h * 2],
        projection="polar"
    )
    ax_rose.set_theta_zero_location("N")
    ax_rose.set_theta_direction(-1)

    sector_rad = 2 * np.pi / N_DIR_BINS
    angles = np.linspace(0, 2 * np.pi, N_DIR_BINS, endpoint=False)

    ax_rose.bar(
        angles, rose_data,
        width=sector_rad * 0.9,
        color="#2c7fb8",
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )

    ax_rose.set_xticklabels([])
    ax_rose.set_yticklabels([])
    ax_rose.set_yticks([])
    ax_rose.spines["polar"].set_linewidth(0.6)
#   ax_rose.spines["polar"].set_color("#555555")
    ax_rose.spines["polar"].set_visible(False)
    ax_rose.set_facecolor("none")
    ax_rose.patch.set_alpha(0.0)
#   ax_rose.grid(color="grey", linewidth=0.3, alpha=0.5)
    ax_rose.grid(False)

    return ax_rose


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def main():
    # Load coastline
    world = gpd.read_file(COASTLINE_PATH)
    malta = world.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor(SEA_COLOUR)
    malta.plot(ax=ax, color=LAND_COLOUR, edgecolor=BORDER_COLOUR, linewidth=0.8)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    ax.set_xticks(np.arange(14.1, 14.65, 0.1))
    ax.set_yticks(np.arange(35.8, 36.15, 0.1))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)",  fontsize=10)
    ax.set_title("Wind Rose by Station", fontsize=13, fontweight="bold", pad=10)

    missing_stations = []

    for stn in STATIONS:
        name, abbr, lat, lon, elev = stn
        print(f"  {abbr} ...", end=" ")

        wdir, wspd = load_wind_data(abbr)

        if wdir is None or len(wdir) < MIN_OBS:
            print(f"insufficient data ({len(wdir) if wdir is not None else 0} obs)")
            missing_stations.append(abbr)
            ax.scatter(lon, lat, c=MISSING_COLOUR, s=60,
                       edgecolors="white", linewidths=0.8, zorder=5)
        else:
            print(f"{len(wdir)} obs")
            rose_data = compute_windrose(wdir)
            draw_windrose(fig, lon, lat, rose_data)

        # Station label
        dx, dy = LABEL_OFFSETS.get(abbr, (0.008, 0.005))
        ax.text(lon + dx, lat + dy, abbr,
                fontsize=LABEL_FONT, fontweight="bold",
                color=LABEL_COLOUR, zorder=10)

    # Legend 
    legend_patches = []
    legend_patches.append(
        mpatches.Patch(facecolor="#2c7fb8", edgecolor="white",
                       linewidth=0.5, label="Wind frequency")
    )
    if missing_stations:
        legend_patches.append(
            mpatches.Patch(facecolor=MISSING_COLOUR, edgecolor="white",
                           linewidth=0.5, label="No data")
        )
    ax.legend(
        handles=legend_patches,
        title="Wind Direction",
        title_fontsize=8,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
