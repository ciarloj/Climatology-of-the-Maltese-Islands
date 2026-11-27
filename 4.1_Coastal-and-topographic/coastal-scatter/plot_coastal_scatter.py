"""
Coastal Distance vs Elevation — Station Climate Scatter
=========================================================
Plots stations on a 2D space: x = distance from coastline, y = elevation.
Dot colour shows the value of a climate variable or index.
One plot per variable/statistic combination.

Usage:
    python3 plot_coastal_scatter.py

Output: ../../analysis/spatial/coastal_scatter_{stat}_{varname}.png
Input:  ../../Data/Daily_{station}.csv
        GSHHG coastline shapefile (defined in COASTLINE_PATH)

Coastal distances are computed by reprojecting station coordinates and
the coastline geometry into UTM zone 33N (EPSG:32633), then calculating
the minimum planar distance from each station point to the coast.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import FancyArrow

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

class HandlerArrow(HandlerPatch):
    """Draws a real arrow in the legend for a given compass bearing."""
    def __init__(self, bearing_deg=0, **kwargs):
        self.bearing_deg = bearing_deg
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        rad    = np.radians(self.bearing_deg)
        cx, cy = width / 2, height / 2
        length = min(width, height) * 1.8
        dx = np.sin(rad) * length / 2
        dy = np.cos(rad) * length / 2
        arrow = FancyArrow(
            cx - dx, cy - dy, 2 * dx, 2 * dy,
            width=length * 0.06,
            head_width=length * 0.40,
            head_length=length * 0.40,
            fc="#1a1a2e", ec="#1a1a2e",
            transform=trans,
            length_includes_head=True,
        )
        return [arrow]

# ---------------------------------------------------------------------------
# Station data — (name, abbreviation, lat_deg, lon_deg, elevation_m)
# ---------------------------------------------------------------------------

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
    "LQ":  ["Luqa"],
    "BN":  ["Bengħajsa", "Benghajsa"],   # accent-safe fallback
    "DN":  ["Dingli"],
    "MS":  ["Msida", "Birkirkara"],      # <-- alternative name
    "VL":  ["Valletta"],
    "SL":  ["Selmun"],
    "XR":  ["Xagħra", "Xaghra"],
    "XW":  ["Xewkija"],
    "GD":  ["Giordan"],
}

URBAN_STATIONS = {"VL", "MS", "LQ", "XR"}   # adjust as you see fit
# VL = Valletta (capital city centre)
# MS = Msida (dense suburban, harbour area)
# LQ = Luqa (airport — classically classified as urban-influenced)
# XR = Xaghra (town)
# LW = San Lawrenz (town)
# MO = Mosta (town)

SHOW_COAST_ARROWS = True   # set to False to hide arrows on plot and in legend

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COASTLINE_PATH = f"{workdir}/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
OUTPUT_DIR     = f"{workdir}/analysis/coastal-scatter"

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

MISSING_VAL = "-"
DATE_COL    = "Date"

COL_MAP = {
    "Max_Temperature":   "Max Temperature",
    "Min_Temperature":   "Min Temperature",
    "Precipitation":     "Precipitation",
    "Pressure":          "Pressure",
    "Relative_Humidity": "Relative Humidity",
    "Solar":             "Solar",
    "Temperature":       "Temperature",
    "Wind_Direction":    "Wind Direction",
    "Wind_Gust":         "Wind Gust",
    "Wind_Speed":        "Wind Speed",
}

VARLABELS = {
    "DTR":               "Diurnal Temperature Range",
    "Max_Temperature":   "Max Temperature",
    "Min_Temperature":   "Min Temperature",
    "Precipitation":     "Annual Precipitation",
    "Pressure":          "Pressure",
    "Relative_Humidity": "Relative Humidity",
    "Solar":             "Solar Radiation",
    "Temperature":       "Mean Temperature",
    "Wind_Speed":        "Wind Speed",
    "Wind_Gust":         "Wind Gust",
    "Tropical_Nights":   "Tropical Nights",
}

UNITS = {
    "DTR":               "°C",
    "Max_Temperature":   "°C",
    "Min_Temperature":   "°C",
    "Precipitation":     "mm/year",
    "Pressure":          "hPa",
    "Relative_Humidity": "%",
    "Solar":             "W/m²",
    "Temperature":       "°C",
    "Wind_Speed":        "m/s",
    "Wind_Gust":         "m/s",
    "Tropical_Nights":   "nights/year",
}

# Colourmap per variable — default is "plasma"
_wind_colours = ["#cce5ff", "#4da6ff", "#0055cc", "#001a4d", "#000000"]
WIND_CMAP = LinearSegmentedColormap.from_list("wind_intensity", _wind_colours)

VAR_CMAP = {
    "DTR":               "RdYlBu_r",
    "Relative_Humidity": "YlGnBu",
    "Precipitation":     "YlGnBu",
    "Wind_Speed":        WIND_CMAP,
    "Wind_Gust":         WIND_CMAP,
    "Temperature":       "RdYlBu_r",
    "Max_Temperature":   "RdYlBu_r",
    "Min_Temperature":   "RdYlBu_r",
    "Tropical_Nights":   "YlOrRd",
}
DEFAULT_CMAP = "plasma"

# ---------------------------------------------------------------------------
# Which variables to produce, and how to compute them
# Each entry: (varname_key, stat_label, compute_func)
# compute_func(df) -> float, where df is the full daily DataFrame for one station
# ---------------------------------------------------------------------------

TROPICAL_NIGHT_TMIN = 20.0   # °C

def _col(df, varname):
    """Return the column from df matching varname, with fallback."""
    target = COL_MAP.get(varname, varname)
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    matches = [c for c in df.columns if norm(c) == norm(target)]
    return matches[0] if matches else None


def _series(df, varname):
    col = _col(df, varname)
    if col is None:
        return None
    return pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")

# Simpler: define VARIABLES as plain tuples and handle None gracefully in main
VARIABLES = [
    ("DTR",               "mean"),
    ("Wind_Speed",        "mean"),
    ("Relative_Humidity", "mean"),
    ("Precipitation",     "annual_sum"),
    ("Min_Temperature",   "mean"),
    ("Max_Temperature",   "mean"),
]


def compute_stat(df, varname, stat):
    """Compute the statistic for one station DataFrame. Returns float or None."""
    try:
        if varname == "DTR":
            tmax = _series(df, "Max_Temperature")
            tmin = _series(df, "Min_Temperature")
            if tmax is None or tmin is None:
                return None
            dtr = (tmax - tmin).dropna()
            return float(dtr.mean()) if len(dtr) > 0 else None

        elif varname == "Tropical_Nights":
            tmin = _series(df, "Min_Temperature")
            if tmin is None:
                return None
            dates = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
            mask  = tmin.notna()
            per_year = (
                pd.DataFrame({"year": dates[mask].dt.year,
                               "tn": (tmin[mask] >= TROPICAL_NIGHT_TMIN).astype(float)})
                .groupby("year")["tn"].sum()
            )
            return float(per_year.mean()) if len(per_year) > 0 else None

        elif varname == "Precipitation" and stat in ("annual_mean", "annual_sum"):
            s = _series(df, "Precipitation")
            if s is None:
                return None
            dates = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
            clim   = s.groupby(dates.dt.month).transform("mean")
            annual = s.fillna(clim).groupby(dates.dt.year).sum()
#           annual = (
#               pd.DataFrame({"year": dates.dt.year, "val": s})
#               .groupby("year")["val"].sum()
#           )
            return float(annual.mean()) if len(annual) > 0 else None

        else:
            s = _series(df, varname)
            if s is None:
                return None
            s = s.dropna()
            if len(s) == 0:
                return None
            if stat == "mean":
                return float(s.mean())
            elif stat == "max":
                return float(s.max())
            elif stat == "min":
                return float(s.min())
            else:
                return float(s.mean())

    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Coastal distance computation
# ---------------------------------------------------------------------------

def compute_coastal_distances_m():
    """
    For each station, compute the minimum distance (in metres) to the GSHHG
    coastline by reprojecting both the station points and the coastline
    geometry into UTM zone 33N (EPSG:32633).

    Returns a dict {abbreviation: distance_m}.
    """
    print("[coast] Loading GSHHG coastline ...")
    coast_wgs = gpd.read_file(COASTLINE_PATH)

    # Restrict to Malta bounding box to speed up distance calculation
    LON_MIN, LON_MAX = 14.05, 14.65
    LAT_MIN, LAT_MAX = 35.75, 36.12
    coast_local = coast_wgs.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]

    if coast_local.empty:
        print("  [warn] No coastline found in Malta bounding box — using full dataset")
        coast_local = coast_wgs

    # Reproject to UTM 33N for metric distances
    UTM33N = "EPSG:32633"
    coast_utm = coast_local.to_crs(UTM33N)

    # Merge all coastline geometries into one for efficient distance queries
    from shapely.ops import unary_union
    coast_geom = unary_union(coast_utm.geometry).boundary

    # Build station GeoDataFrame
    stn_gdf = gpd.GeoDataFrame(
        [{"abbr": abbr, "geometry": Point(lon, lat)}
         for _, abbr, lat, lon, _ in STATIONS],
        crs="EPSG:4326"
    ).to_crs(UTM33N)

    distances  = {}
    directions = {}   # bearing in degrees from station to nearest coast point

    for _, row in stn_gdf.iterrows():
        abbr    = row["abbr"]
        pt      = row.geometry
        d       = pt.distance(coast_geom)
        nearest = coast_geom.interpolate(coast_geom.project(pt))   # closest point on coast

        # Bearing from station to nearest coast point (degrees, 0=N, 90=E)
        dx = nearest.x - pt.x
        dy = nearest.y - pt.y
        bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360

        distances[abbr]  = d
        directions[abbr] = bearing
        print(f"  {abbr:4s}  {d/1000:.2f} km   bearing {bearing:.0f}°")

    # Save station metadata CSV
 #   rows = []
 #   for name, abbr, lat, lon, elev in STATIONS:
 #       rows.append({
 #           "Station":          name,
 #           "Abbreviation":     abbr,
 #           "Latitude":         round(lat, 6),
 #           "Longitude":        round(lon, 6),
 #           "Elevation_m":      elev,
 #           "Coast_distance_km": round(distances.get(abbr, np.nan) / 1000, 3),
 #           "Coast_bearing_deg":  round(directions.get(abbr, np.nan), 1),
 #       })
 #   meta_df = pd.DataFrame(rows)
 #   meta_path = os.path.join(OUTPUT_DIR, "station_metadata.csv")
 #   os.makedirs(OUTPUT_DIR, exist_ok=True)
 #   meta_df.to_csv(meta_path, index=False)
 #   print(f"  [saved] {meta_path}")

    return distances, directions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_scatter(varname, stat, stat_values, distances_m, directions, elevations, output_dir):
    """
    stat_values  : dict {abbr: float or None}
    distances_m  : dict {abbr: float}  — distance from coast in metres
    elevations   : dict {abbr: float}  — elevation in metres
    """
    vlab = VARLABELS.get(varname, varname)
    unit = UNITS.get(varname, "")

    # Separate valid from missing
    valid   = {k: v for k, v in stat_values.items() if v is not None}
    missing = {k for k, v in stat_values.items() if v is None}

    if not valid:
        print(f"  [skip] No valid data for {stat} / {varname}")
        return

    values  = np.array([valid[k] for k in valid])
    val_min = values.min()
    val_max = values.max()
    pad     = (val_max - val_min) * 0.05 if val_max != val_min else 0.5
    norm    = mcolors.Normalize(vmin=val_min - pad, vmax=val_max + pad)

    raw_cmap = VAR_CMAP.get(varname, DEFAULT_CMAP)
    cmap_obj = raw_cmap if isinstance(raw_cmap, mcolors.Colormap) else plt.get_cmap(raw_cmap)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot missing stations in grey
    for abbr in missing:
        if abbr not in distances_m:
            continue
        marker = "s" if abbr in URBAN_STATIONS else "o"
        ax.scatter(distances_m[abbr] / 1000, elevations[abbr],
                   c="#aaaaaa", s=80, marker=marker, edgecolors="black",
                   linewidths=0.8, zorder=4)

    # Plot valid stations coloured by value
    for abbr, val in valid.items():
        if abbr not in distances_m:
            continue
        marker = "s" if abbr in URBAN_STATIONS else "o"
        ax.scatter(distances_m[abbr] / 1000, elevations[abbr],
                   c=[cmap_obj(norm(val))], s=120, marker=marker,
                   edgecolors="black", linewidths=0.8, zorder=5)

    # Station labels
    for abbr in list(valid.keys()) + list(missing):
        if abbr not in distances_m:
            continue
        ax.annotate(
            abbr,
            (distances_m[abbr] / 1000, elevations[abbr]),
            textcoords="offset points", xytext=(5, -10),
            fontsize=7.5, fontweight="bold", color="#1a1a2e",
        )

    ax.set_xlabel("Distance from coastline (km)", fontsize=11)
    ax.set_ylabel("Elevation (m)", fontsize=11)
    ax.set_title(f"{stat.replace('_', ' ').title()} {vlab}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Fixed arrow length in inches, converted to data units for visual consistency
    fig_w, fig_h   = fig.get_size_inches()
    ax_bbox        = ax.get_position()
    ax_w_in        = ax_bbox.width  * fig_w
    ax_h_in        = ax_bbox.height * fig_h
    x_range        = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range        = ax.get_ylim()[1] - ax.get_ylim()[0]
    arrow_len_in   = 0.28                              # fixed length in inches
    ARROW_LEN_X    = arrow_len_in * x_range / ax_w_in
    ARROW_LEN_Y    = arrow_len_in * y_range / ax_h_in

    if SHOW_COAST_ARROWS:
        for abbr, bearing in directions.items():
            if abbr not in distances_m:
                continue
            x0  = distances_m[abbr] / 1000
            y0  = elevations[abbr]
            rad = np.radians(bearing)
            dx  = np.sin(rad) * ARROW_LEN_X
            dy  = np.cos(rad) * ARROW_LEN_Y
            ax.annotate(
                "", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a2e",
                                lw=1.2, mutation_scale=8),
                zorder=3,
            )

    ax.grid(True, linewidth=0.4, color="grey", alpha=0.4, linestyle="--")

    # --- Legend ---
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, linestyle="none", label="Rural station"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="black", markeredgewidth=1.5,
               markersize=9, linestyle="none", label="Urban station"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#aaaaaa",
               markeredgecolor="black", markeredgewidth=1.0,
               markersize=9, linestyle="none", label="No data"),
    ]

    handler_map = {}

    if SHOW_COAST_ARROWS:
        coast_patch = FancyArrow(0, 0, 1, 0)   # dummy — handler controls appearance
        north_patch = FancyArrow(0, 0, 1, 0)   # dummy
        legend_elements.append(coast_patch)
        legend_elements.append(north_patch)
        handler_map[coast_patch] = HandlerArrow(bearing_deg=45)
        handler_map[north_patch] = HandlerArrow(bearing_deg=0)
        # Manually set labels since FancyArrow doesn't carry a label
        coast_patch.set_label("Direction to nearest coast")
        north_patch.set_label("North (0°)")

    ax.legend(handles=legend_elements, handler_map=handler_map,
              loc="upper right", fontsize=8, framealpha=0.90,
              title="Legend", title_fontsize=8)

    # Colorbar
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.025])
    cb = ColorbarBase(cbar_ax, cmap=cmap_obj, norm=norm, orientation="horizontal")
    cb.set_label(f"{vlab} ({unit})", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.subplots_adjust(bottom=0.18)
    os.makedirs(output_dir, exist_ok=True)
    fname    = f"coastal_scatter_{stat}_{varname}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Compute coastal distances once
    distances_m, directions = compute_coastal_distances_m()

    # Build elevation and name lookup
    elevations = {abbr: elev for _, abbr, _, _, elev in STATIONS}

    # Load all daily files once
    print("\n[load] Reading daily data files ...")
    daily_data = {}
#    for name, abbr, lat, lon, elev in STATIONS:
#        filename = STATION_FILENAMES.get(abbr)
#        if filename is None:
#            continue
#        filepath = f"{workdir}/Data/Daily_{filename}.csv"
#        if not os.path.exists(filepath):
#            print(f"  {abbr}  [file not found]")
#            continue
#        try:
#            df = pd.read_csv(filepath, header=0)
#            daily_data[abbr] = df
#            print(f"  {abbr:4s}  {len(df)} rows")
#        except Exception as e:
#            print(f"  {abbr}  [error: {e}]")

    for name, abbr, lat, lon, elev in STATIONS:
        candidates = STATION_FILENAMES.get(abbr, [])
        if isinstance(candidates, str):        # tolerate old single-string entries
            candidates = [candidates]
        filepath = None
        for cand in candidates:
            trial = f"../../Data/Daily_{cand}.csv"
            if os.path.exists(trial):
                filepath = trial
                break
        if filepath is None:
            print(f"  {abbr}  [file not found: tried {candidates}]")
            continue
#       try:
#           df = pd.read_csv(filepath, header=0)
#           daily_data[abbr] = df
        try:
            df = pd.read_csv(filepath, header=0)
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
            df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
            daily_data[abbr] = df
            print(f"  {abbr:4s}  {len(df)} rows  ({os.path.basename(filepath)})")
        except Exception as e:
            print(f"  {abbr}  [error: {e}]")

    # Produce one plot per variable/stat combination
    all_stat_values = {}
    for varname, stat in VARIABLES:
        print(f"\n[var] {varname} / {stat}")
        stat_values = {}

        for abbr in [s[1] for s in STATIONS]:
            if abbr not in daily_data:
                stat_values[abbr] = None
                continue
            val = compute_stat(daily_data[abbr], varname, stat)
            stat_values[abbr] = val
            tag = f"{val:.3f}" if val is not None else "[missing]"
            print(f"  {abbr:4s}  {tag}")

        all_stat_values[(varname, stat)] = stat_values
        create_scatter(varname, stat, stat_values, distances_m, directions, elevations, OUTPUT_DIR)

    # ----- Combined metadata + variable-value table -----
    name_by_abbr = {abbr: name for name, abbr, *_ in STATIONS}
    latlon_by_abbr = {abbr: (lat, lon) for _, abbr, lat, lon, _ in STATIONS}

    rows = []
    for name, abbr, lat, lon, elev in STATIONS:
        dist_m = distances_m.get(abbr)
        row = {
            "Station":           name,
            "Abbreviation":      abbr,
            "Latitude":          round(lat, 6),
            "Longitude":         round(lon, 6),
            "Elevation_m":       elev,
            "Coast_distance_km": round(dist_m / 1000, 3) if dist_m is not None else None,
        }
        # one column per plotted variable, using the same "{varname}_{stat}" label
        for (varname, stat), vals in all_stat_values.items():
            v = vals.get(abbr)
            row[f"{varname}_{stat}"] = round(v, 3) if v is not None else None
        rows.append(row)

    meta_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(OUTPUT_DIR, "station_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"  [saved] {meta_path}  ({len(meta_df.columns)} columns)")

    print(f"\n[done] All plots written to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
