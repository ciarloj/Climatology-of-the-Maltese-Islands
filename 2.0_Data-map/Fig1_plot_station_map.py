"""
Station Map — Malta & Gozo
============================
Plots all meteorological stations on a map of Malta and Gozo.
Station dots are coloured by elevation with an accompanying colorbar.

Usage:
    python3 Fig1_plot_station_map.py

Requirements:
    pip install matplotlib cartopy numpy
    (cartopy pulls in shapely and handles coastlines automatically)

Output: ../../analysis/Fig1_station_map.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd

# ---------------------------------------------------------------------------
# Station data — (name, abbreviation, lat_deg, lon_deg, elevation_m)
# Coordinates converted from DMS to decimal degrees
# ---------------------------------------------------------------------------

def dms(d, m, s):
    """Convert degrees/minutes/seconds to decimal degrees."""
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

#    ("San Lawrenz", "LW",  dms(36,  3, 14),  dms(14, 12, 14), 130),
#    ("Mosta",       "MO",  dms(35, 54, 47),  dms(14, 25,  2),  90),

# ---------------------------------------------------------------------------
# Label offset adjustments (lon_offset, lat_offset) to avoid overlap
# Tune these if labels collide on your output
# ---------------------------------------------------------------------------

LABEL_OFFSETS = {
    "LQ":  ( 0.008, -0.010),
    "BN":  (-0.020,  0.006),  
    "DN":  ( 0.008,  0.005),  
    "MS":  (-0.010, -0.012),  
    "VL":  ( 0.004,  0.004), #
    "SL":  (-0.010, -0.014),
    "XR":  ( 0.008,  0.005),
    "XW":  ( 0.008, -0.002),   
    "GD":  ( 0.008,  0.000),  
}

#    "LW":  ( 0.008, -0.006),  
#    "MO":  ( 0.008,  0.005),

# ---------------------------------------------------------------------------
# Map appearance
# ---------------------------------------------------------------------------

DOT_SIZE      = 80       # scatter marker size
DOT_EDGECOLOR = "white"
DOT_LINEWIDTH = 0.8
CMAP          = "gist_earth"   # alternatives: "terrain", "YlOrBr", "copper", "gist_earth"
LAND_COLOUR   = "#e8d5b0"   # warm sand for land
SEA_COLOUR    = "#c9e8f5"   # light blue for sea
BORDER_COLOUR = "#8b7355"   # coastline colour
LABEL_FONT    = 9
LABEL_COLOUR  = "#1a1a2e"

OUTPUT_PATH   = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/analysis/spatial/Fig1_station_map.png"

# ---------------------------------------------------------------------------
# Map bounds — extend slightly beyond station extents
# ---------------------------------------------------------------------------

LON_MIN, LON_MAX = 14.10, 14.62
LAT_MIN, LAT_MAX = 35.77, 36.10

# ---------------------------------------------------------------------------
# Try to use cartopy; fall back to a plain lat/lon scatter if unavailable
# ---------------------------------------------------------------------------

def plot_with_cartopy(stations, elev_norm, cmap):
    """Uses geopandas + Natural Earth — no cartopy needed."""
    # high resolution source: https://www.soest.hawaii.edu/pwessel/gshhg/
    world = gpd.read_file("/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp")

    malta = world.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor(SEA_COLOUR)
    malta.plot(ax=ax, color=LAND_COLOUR, edgecolor=BORDER_COLOUR, linewidth=0.8)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    _draw_stations(ax, stations, elev_norm, cmap, cartopy_proj=None)
    return fig, ax

#def plot_with_cartopy(stations, elev_norm, cmap):
#    import contextily as ctx
#    fig, ax = plt.subplots(figsize=(9, 8))
#    ax.set_facecolor(SEA_COLOUR)
#    ax.set_xlim(LON_MIN, LON_MAX)
#    ax.set_ylim(LAT_MIN, LAT_MAX)
#    _draw_stations(ax, stations, elev_norm, cmap, cartopy_proj=None)
#    ctx.add_basemap(
#        ax,
#        crs="EPSG:4326",
#        source=ctx.providers.OpenStreetMap.Mapnik,
#        zoom=11,                  # increase for more detail, 11-12 is ideal for Malta
#        alpha=0.6,                # blend with your sea colour
#    )
#    return fig, ax

def _draw_stations(ax, stations, elev_norm, cmap, cartopy_proj):
    import matplotlib.pyplot as plt

    transform_kw = {}
    if cartopy_proj is not None:
        transform_kw["transform"] = cartopy_proj

    lons  = [s[3] for s in stations]
    lats  = [s[2] for s in stations]
    elevs = [s[4] for s in stations]
    abbrv = [s[1] for s in stations]
    names = [s[0] for s in stations]

    colours = [cmap(elev_norm(e)) for e in elevs]

    sc = ax.scatter(
        lons, lats,
        c=elevs,
        cmap=cmap,
        norm=elev_norm,
        s=DOT_SIZE,
        edgecolors=DOT_EDGECOLOR,
        linewidths=DOT_LINEWIDTH,
        zorder=5,
        **transform_kw,
    )

    for lon, lat, abbr, name, elev in zip(lons, lats, abbrv, names, elevs):
        dx, dy = LABEL_OFFSETS.get(abbr, (0.008, 0.005))
        ax.text(
            lon + dx, lat + dy,
            abbr,
            fontsize=LABEL_FONT,
            fontweight="bold",
            color=LABEL_COLOUR,
            zorder=6,
            **transform_kw,
        )

    return sc


# ---------------------------------------------------------------------------
# Legend box (name, abbrev, elevation)
# ---------------------------------------------------------------------------

def add_legend_box(ax, stations):
    lines = [f"{name} ({abbr}) — {elev} m"
             for _, abbr, _, _, elev in stations
             for name in [next(s[0] for s in stations if s[1] == abbr)]]
    # Simpler build:
    lines = [f"{s[1]:2s}  {s[0]:11s} ({s[4]:>3} m)" for s in stations]
    text  = "\n".join(lines)
    ax.text(
        0.77, 0.98, text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                  edgecolor="#cccccc"),
        family="monospace",
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    elevs    = [s[4] for s in STATIONS]
    elev_min = 0
    elev_max = max(elevs) + 20   # small headroom above max

   #cmap      = cm.get_cmap(CMAP)
    cmap      = matplotlib.colormaps[CMAP]
    elev_norm = mcolors.Normalize(vmin=elev_min, vmax=elev_max)

    # Try cartopy, fall back gracefully
    try:
        fig, ax = plot_with_cartopy(STATIONS, elev_norm, cmap)
    except ImportError:
        fig, ax = plot_without_cartopy(STATIONS, elev_norm, cmap)

    # Axes labels
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)",  fontsize=10)
    ax.set_title("Meteorological Stations — Malta & Gozo",
                 fontsize=13, fontweight="bold", pad=10)

    # Gridlines
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    ax.set_xticks(np.arange(14.1, 14.65, 0.1))
    ax.set_yticks(np.arange(35.8, 36.15, 0.1))
    ax.tick_params(labelsize=8)


    # Colorbar
    cbar_ax = fig.add_axes([0.235, 0.06, 0.55, 0.025])   # [left, bottom, width, height]
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=elev_norm,
                      orientation="horizontal")
    cb.set_label("Elevation (m)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Legend box
    add_legend_box(ax, STATIONS)

    plt.subplots_adjust(bottom=0.15)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
