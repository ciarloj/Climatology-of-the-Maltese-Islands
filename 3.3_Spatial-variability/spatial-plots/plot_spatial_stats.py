"""
Spatial Statistics Map — Malta & Gozo
=======================================
Produces one map per variable/statistic combination, colouring station dots
by the computed statistic read from Daily_{station}.csv.
Missing stations are shown as grey dots.

Usage:
    python3 plot_spatial_stats.py

Output: ../../analysis/spatial/spatial_{stat}_{varname}.png
Input:  ../../Data/Daily_{station}.csv
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
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

_wind_colours = ["#cce5ff", "#4da6ff", "#0055cc", "#001a4d", "#000000"]
WIND_CMAP = LinearSegmentedColormap.from_list("wind_intensity", _wind_colours)

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

# Map station abbreviation to filename stem used in Daily_{station}.csv
# Key = abbreviation, value = station name as used in the filename
STATION_FILENAMES = {
    "LQ":  ["Luqa"],
    "BN":  ["Bengħajsa", "Benghajsa"],
    "DN":  ["Dingli"],
    "MS":  ["Msida","Birkirkara"],
    "VL":  ["Valletta"],
    "SL":  ["Selmun"],
    "XR":  ["Xagħra","Xaghra"],
    "XW":  ["Xewkija"],
    "GD":  ["Giordan"],
}

# ---------------------------------------------------------------------------
# Label offsets
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
    "LW":  ( 0.008, -0.006),
    "MO":  ( 0.008,  0.005),
    "GD":  ( 0.008,  0.000),
}

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

MISSING_VAL = "-"
DATE_COL    = "Date"

# Column names in Daily_{station}.csv
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

# ---------------------------------------------------------------------------
# Statistic rules
# "average" : all variables EXCEPT Max_Temperature and Min_Temperature
# "max"     : all variables EXCEPT Temperature and Min_Temperature
# "min"     : all variables EXCEPT Temperature and Max_Temperature
# ---------------------------------------------------------------------------

STAT_RULES = {
    "prevailing": {
        "label":   "Prevailing",
        "exclude": {"Max_Temperature", "Min_Temperature", "Precipitation",
                    "Pressure", "Relative_Humidity", "Solar", "Temperature",
                    "Wind_Gust", "Wind_Speed"},   # only Wind_Direction runs this
        "func":    lambda s: s.mean(),   # unused for Wind_Direction, just a placeholder
    },
#   "median": {
#       "label":   "Median",
#       "exclude": {"Max_Temperature", "Min_Temperature", "Precipitation", "Pressure", 
#           "Relative_Humidity", "Solar", "Temperature", "Wind_Gust", "Wind_Speed"},
#       "func":    lambda s: s.median(),
#   }, 
    "average": {
        "label":   "Mean",
        "exclude": {"Wind_Direction"},
        "func":    lambda s: s.mean(),
    },
    "max": {
        "label":   "Maximum",
        "exclude": {"Temperature", "Min_Temperature", "Wind_Direction"},
        "func":    lambda s: s.max(),
    },
    "min": {
        "label":   "Minimum",
        "exclude": {"Temperature", "Max_Temperature", "Wind_Direction"},
        "func":    lambda s: s.min(),
    },
}

# For these variables, compute per-year aggregation first, then apply the stat
ANNUAL_SUM_VARS = {"Precipitation"}

# Stations to skip for specific variables — set to empty list to include all
SKIP_STATIONS = {
   "Pressure":     {"GD"},   # use abbreviations
}

ARROW_VARS = {"Wind_Direction"}

# ---------------------------------------------------------------------------
# Map appearance
# ---------------------------------------------------------------------------

DOT_SIZE         = 120
DOT_EDGECOLOR    = "white"
DOT_LINEWIDTH    = 0.8
MISSING_COLOUR   = "#aaaaaa"   # grey for stations without data
MISSING_SIZE     = 80
CMAP             = "plasma"  # alternatives: "viridis", "plasma", "YlOrRd", RdYlBu_r, "RdYlGn_r"
VAR_CMAP = {
    "Precipitation":     "YlGnBu",    # yellow=dry, blue=wet
    "Relative_Humidity": "YlGnBu",    # same
    "Wind_Speed":        WIND_CMAP,      # placeholder, overridden below
    "Wind_Gust":         WIND_CMAP,      # placeholder, overridden below
    "Temperature":       "RdYlBu_r",
    "Max_Temperature":   "RdYlBu_r",
    "Min_Temperature":   "RdYlBu_r",
}
LAND_COLOUR      = "#e8d5b0"
SEA_COLOUR       = "#c9e8f5"
BORDER_COLOUR    = "#8b7355"
LABEL_FONT       = 9
LABEL_COLOUR     = "#1a1a2e"

COASTLINE_PATH   = f"{workdir}/analysis/spatial/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
OUTPUT_DIR       = f"{workdir}/analysis/spatial"

LON_MIN, LON_MAX = 14.10, 14.62
LAT_MIN, LAT_MAX = 35.77, 36.10

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_station_series(abbr, varname):
    """
    Load the variable column from Daily_{station}.csv for one station.
    Returns a cleaned numeric pd.Series or None if unavailable.
    """
    candidates = STATION_FILENAMES.get(abbr, [])
    if isinstance(candidates, str):        # tolerate old single-string entries
        candidates = [candidates]
    filepath = None
    for cand in candidates:
        trial = f"{workdir}/Data/Daily_{cand}.csv"
        if os.path.exists(trial):
            filepath = trial
            break
    if filepath is None:
        return None

    try:
        df  = pd.read_csv(filepath, header=0)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
        col = COL_MAP.get(varname)
        if col is None or col not in df.columns:
            # Try normalised fallback
            norm = lambda s: s.lower().replace(" ", "_")
            match = [c for c in df.columns if norm(c) == norm(col or varname)]
            if not match:
                return None
            col = match[0]
        series = pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")
        if varname in ANNUAL_SUM_VARS:
            series_raw = pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")
            df_tmp = pd.DataFrame({"date": df[DATE_COL], "val": series_raw})
#           annual = df_tmp.groupby(df_tmp["date"].dt.year)["val"].sum()
#           return annual.dropna()
            clim   = df_tmp.groupby(df_tmp["date"].dt.month)["val"].transform("mean")
            filled = df_tmp["val"].fillna(clim)
            return filled.groupby(df_tmp["date"].dt.year).sum()
        return series.dropna()
    except Exception as e:
        print(f"  [warn] {abbr} / {varname}: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_coastline():
    world = gpd.read_file(COASTLINE_PATH)
    return world.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX]


def create_spatial_plot(varname, stat_key, stat_values, output_dir):
    """
    stat_values: dict of {abbreviation: float or None}
    """
    stat_cfg  = STAT_RULES[stat_key]
    vlab      = VARLABELS.get(varname, varname)
    unit      = UNITS.get(varname, "")
    stat_label = stat_cfg["label"]

    # Separate valid and missing stations
    valid   = {k: v for k, v in stat_values.items() if v is not None}
    missing = {k for k, v in stat_values.items() if v is None}

    if not valid:
        print(f"  [skip] No valid data for {stat_key} / {varname}")
        return

    values    = np.array(list(valid.values()))
    val_min   = values.min()
    val_max   = values.max()

    # Pad range slightly so extreme dots aren't at the very end of the colorbar
    pad = (val_max - val_min) * 0.05 if val_max != val_min else 0.5
    norm = mcolors.Normalize(vmin=val_min - pad, vmax=val_max + pad)
  # cmap_obj = cm.get_cmap(VAR_CMAP.get(varname, CMAP))
    raw_cmap = VAR_CMAP.get(varname, CMAP)
    cmap_obj = raw_cmap if isinstance(raw_cmap, mcolors.Colormap) else plt.get_cmap(raw_cmap)

    # --- Build figure ---
    malta = load_coastline()
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor(SEA_COLOUR)
    malta.plot(ax=ax, color=LAND_COLOUR, edgecolor=BORDER_COLOUR, linewidth=0.8)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)

    # --- Plot missing stations (grey) ---
    for stn in STATIONS:
        abbr = stn[1]
        if abbr not in missing:
            continue
        ax.scatter(stn[3], stn[2],
                   c=MISSING_COLOUR, s=MISSING_SIZE,
                   edgecolors="white", linewidths=0.8, zorder=5)

    # --- Plot valid stations (coloured) ---
    lons_v  = [s[3] for s in STATIONS if s[1] in valid]
    lats_v  = [s[2] for s in STATIONS if s[1] in valid]
    vals_v  = [valid[s[1]] for s in STATIONS if s[1] in valid]

    if varname in ARROW_VARS:
        # Draw arrows pointing in the wind direction (meteorological: 0°=N, 90°=E)
        arrow_len = 0.025   # in degrees — adjust to taste
        for stn in STATIONS:
            abbr = stn[1]
            if abbr not in valid:
                continue
            lat, lon = stn[2], stn[3]
            deg = valid[abbr]
            # Convert met degrees to math angle: 0°N clockwise → standard counterclockwise
            rad = np.radians(90 - deg)
            dx  = arrow_len * np.cos(rad)
            dy  = arrow_len * np.sin(rad)
            ax.annotate(
#               "", xy=(lon + dx, lat + dy), xytext=(lon, lat),
                "", xy=(lon, lat), xytext=(lon + dx, lat + dy),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a2e", lw=1.5),
                zorder=6,
            )
            # Draw a dot at origin so the station position is still visible
            ax.scatter(lon, lat, c="#1a1a2e", s=30, zorder=7)
    else:
        sc = ax.scatter(
            lons_v, lats_v,
            c=vals_v, cmap=cmap_obj, norm=norm,
            s=DOT_SIZE,
            edgecolors=DOT_EDGECOLOR, linewidths=DOT_LINEWIDTH,
            zorder=6,
        )

    # --- Labels ---
    for stn in STATIONS:
        name, abbr, lat, lon, elev = stn
        dx, dy = LABEL_OFFSETS.get(abbr, (0.008, 0.005))
        ax.text(lon + dx, lat + dy, abbr,
                fontsize=LABEL_FONT, fontweight="bold",
                color=LABEL_COLOUR, zorder=7)

    # --- Axes ---
    ax.set_xlabel("Longitude (°E)", fontsize=10)
    ax.set_ylabel("Latitude (°N)",  fontsize=10)
    ax.set_title(
        f"{stat_label} {vlab}",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.grid(True, linewidth=0.4, color="grey", alpha=0.5, linestyle="--")
    ax.set_xticks(np.arange(14.1, 14.65, 0.1))
    ax.set_yticks(np.arange(35.8, 36.15, 0.1))
    ax.tick_params(labelsize=8)

    # --- Colorbar ---
    if varname not in ARROW_VARS:
        cbar_ax = fig.add_axes([0.235, 0.06, 0.55, 0.025])
        cb = ColorbarBase(cbar_ax, cmap=cmap_obj, norm=norm, orientation="horizontal")
        cb.set_label(f"{stat_label} {vlab} ({unit})", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        plt.subplots_adjust(bottom=0.15)
#   cbar_ax = fig.add_axes([0.235, 0.06, 0.55, 0.025])
#   cb = ColorbarBase(cbar_ax, cmap=cmap_obj, norm=norm, orientation="horizontal")
#   cb.set_label(f"{stat_label} {vlab} ({unit})", fontsize=9)
#   cb.ax.tick_params(labelsize=8)

    # --- Missing marker in legend ---
    if missing:
        ax.scatter([], [], c=MISSING_COLOUR, s=MISSING_SIZE,
                   edgecolors="white", linewidths=0.8, label="No data")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    plt.subplots_adjust(bottom=0.15)
    os.makedirs(output_dir, exist_ok=True)
    fname    = f"spatial_{stat_key}_{varname}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for stat_key, stat_cfg in STAT_RULES.items():
        excluded = stat_cfg["exclude"]
        func     = stat_cfg["func"]

        for varname in COL_MAP.keys():
            if varname in excluded:
                continue

            print(f"\n[{stat_key}] {varname}")
            stat_values = {}

            for stn in STATIONS:
                abbr = stn[1]
                if abbr in SKIP_STATIONS.get(varname, set()):
                    stat_values[abbr] = None
                    print(f"  {abbr}  [skipped for {varname}]")
                    continue
                series = load_station_series(abbr, varname)
                if series is not None and len(series) > 0:
                    if varname == "Wind_Direction" or varname == "Wind Direction":
                        # Circular mean: convert to radians, average unit vectors
                        rad = np.radians(series.values)
                        if stat_key == "prevailing":
                            # Bin directions into 36 x 10° bins, find the most frequent
                            # Use circular binning so 355°-5° wraps correctly
                            bins      = np.arange(0, 370, 10)
                            counts, _ = np.histogram(series.values % 360, bins=bins)
                            peak_bin  = np.argmax(counts)
                            # Return the centre of the most frequent bin
                            stat_values[abbr] = float(peak_bin * 10 + 5)
                        else:
                            mean_deg = np.degrees(np.arctan2(
                                np.nanmean(np.sin(rad)),
                                np.nanmean(np.cos(rad))
                            )) % 360
                            stat_values[abbr] = float(mean_deg)
                    else:
                        stat_values[abbr] = float(func(series))
                    print(f"  {abbr}  {stat_values[abbr]:.3f}")
                else:
                    stat_values[abbr] = None
                    print(f"  {abbr}  [missing]")

            create_spatial_plot(varname, stat_key, stat_values, OUTPUT_DIR)

    print(f"\n[done] All plots written to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
