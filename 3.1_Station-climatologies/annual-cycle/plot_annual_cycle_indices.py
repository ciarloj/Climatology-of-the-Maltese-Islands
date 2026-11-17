"""
Climate Indices Annual Cycle Plot
===================================
Produces one plot per climate index showing the smoothed 31-day rolling
climatology for all stations as individual coloured lines, plus optional
group averages.

Usage:
    python3 plot_annual_cycle_indices.py

Output: ../../analysis/annual-cycle/annual-cycle-index_{index_name}.png
Input:  ../../Data/Daily_{station}.csv

All thresholds and window sizes are defined in the CONFIGURATION section.
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

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

ALL_STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]

GOZO_STATIONS  = ["Giordan", "Xagħra", "Xewkija"]
MALTA_STATIONS = ["Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# File / column configuration
# ---------------------------------------------------------------------------

DATE_COL    = "Date"
MISSING_VAL = "-"

# Column names in Daily_{station}.csv
COL_TMAX    = "Max Temperature"
COL_TMIN    = "Min Temperature"
COL_PRECIP  = "Precipitation"
COL_WIND    = "Wind Speed"

# ---------------------------------------------------------------------------
# CONFIGURATION — thresholds and window
# ---------------------------------------------------------------------------

ROLL_WINDOW = 31        # days — centred rolling window for all indices

# Wind
WINDY_THRESHOLD   = 10.0   # m/s  — days with wind speed >= this are "windy"

# Precipitation
WET_THRESHOLD     = 1.0    # mm   — days with precip >= this are "wet/rainy"
HEAVY_THRESHOLD   = 20.0   # mm   — days with precip >= this are "heavy rain"

# Temperature
SUMMER_DAY_TMAX   = 25.0   # °C   — Tmax >= this → summer day
TROPICAL_NIGHT_TMIN = 20.0 # °C   — Tmin >= this → tropical night

# Tx90p: 90th percentile of Tmax computed per station over the full record
# (used as the threshold for each station individually)

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------
# Each entry: {
#   "label":    human-readable name for title and y-axis,
#   "unit":     y-axis unit string,
#   "col":      which column(s) from the CSV are needed (list),
#   "compute":  function(df, window) -> pd.Series indexed by date
# }
# The compute function receives the full daily DataFrame for one station
# and returns a daily Series of the index value (before climatology averaging).

def _roll(series, window):
    """Centred rolling sum/mean with min_periods=1."""
    return series.rolling(window, center=True, min_periods=1)

def compute_windy_days(df, window):
    """Count of days >= WINDY_THRESHOLD in rolling window."""
    v = pd.to_numeric(df[COL_WIND].replace(MISSING_VAL, np.nan), errors="coerce")
    ind = (v >= WINDY_THRESHOLD).astype(float)
    ind[v.isna()] = np.nan
    return ind

#def compute_accumulated_wind(df, window):
#    """Sum of wind speed in rolling window (m/s · days)."""
#    v = pd.to_numeric(df[COL_WIND].replace(MISSING_VAL, np.nan), errors="coerce")
#    return v

def compute_rainy_days(df, window):
    """Fraction of observed days with precip >= WET_THRESHOLD."""
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    ind = (v >= WET_THRESHOLD).astype(float)
    ind[v.isna()] = np.nan          # keep gaps as gaps, not as non-events
    return ind

def compute_precip_intensity(df, window):
    """
    SDII in rolling window: total precip on wet days / number of wet days.
    NaN where there are no wet days in the window.
    """
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    wet = v.where(v >= WET_THRESHOLD)
    total  = _roll(wet.fillna(0), window).sum()
    count  = _roll((v >= WET_THRESHOLD).astype(float), window).sum()
    return total.where(count > 0) / count.where(count > 0)

def compute_heavy_rain_days(df, window):
    """Count of days with precip >= HEAVY_THRESHOLD in rolling window."""
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    ind = (v >= HEAVY_THRESHOLD).astype(float)
    ind[v.isna()] = np.nan          # keep gaps as gaps, not as non-events
    return ind

def compute_cdd(df, window):
    """
    Maximum consecutive dry days (precip < WET_THRESHOLD) within each
    rolling window of `window` days.
    """
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    dry = (v < WET_THRESHOLD).astype(float)
    dry[v.isna()] = np.nan

    half = window // 2
    result = np.full(len(dry), np.nan)
    dry_vals = dry.values

    for i in range(half, len(dry_vals) - half):
        segment = dry_vals[i - half: i + half + 1]
        if np.all(np.isnan(segment)):
            continue
        # Count longest consecutive run of 1s
        max_run = cur_run = 0
        for val in segment:
            if val == 1.0:
                cur_run += 1
                max_run = max(max_run, cur_run)
            elif val == 0.0:
                cur_run = 0
        result[i] = max_run

    return pd.Series(result, index=df.index)

def compute_summer_days(df, window):
    """Count of days with Tmax >= SUMMER_DAY_TMAX in rolling window."""
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    ind = (v >= SUMMER_DAY_TMAX).astype(float)
    ind[v.isna()] = np.nan          # keep gaps as gaps, not as non-events
    return ind

def compute_tropical_nights(df, window):
    """Count of days with Tmin >= TROPICAL_NIGHT_TMIN in rolling window."""
    v = pd.to_numeric(df[COL_TMIN].replace(MISSING_VAL, np.nan), errors="coerce")
    ind = (v >= TROPICAL_NIGHT_TMIN).astype(float)
    ind[v.isna()] = np.nan          # keep gaps as gaps, not as non-events
    return ind

def compute_tx90p(df, window):
    """
    Fraction of days in rolling window where Tmax exceeds the station's
    own 90th percentile of Tmax (computed over the full record).
    Returned as a count of days (out of window) for consistency.
    """
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    p90 = v.quantile(0.90)
    return (v > p90).astype(float)

def compute_tmax_above_p90(df, window):
    """Days where Tmax exceeds its own p90 over the full record."""
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    return (v > v.quantile(0.90)).astype(float)

def compute_tmin_below_p10(df, window):
    """Days where Tmin falls below its own p10 over the full record."""
    v = pd.to_numeric(df[COL_TMIN].replace(MISSING_VAL, np.nan), errors="coerce")
    return (v < v.quantile(0.10)).astype(float)

def compute_tmax_below_p10(df, window):
    """Days where Tmax falls below its own p10 — cold days."""
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    return (v < v.quantile(0.10)).astype(float)

def compute_tmin_above_p90(df, window):
    """Days where Tmin exceeds its own p90 — warm nights."""
    v = pd.to_numeric(df[COL_TMIN].replace(MISSING_VAL, np.nan), errors="coerce")
    return (v > v.quantile(0.90)).astype(float)

# Master index registry — edit this list to add/remove/reorder indices
INDICES = [
    # Wind
    {
        "name":    "windy_days",
        "label":   "Windy Days",
        "unit":    f"fraction of days",
        "col":     [COL_WIND],
        "compute": compute_windy_days,
    },
#    {
#        "name":    "accumulated_wind",
#        "label":   "Accumulated Wind",
#        "unit":    f"m/s · days per {ROLL_WINDOW}-day window",
#        "col":     [COL_WIND],
#        "compute": compute_accumulated_wind,
#    },
    # Precipitation
    {
        "name":    "rainy_days",
        "label":   "Rainy Days",
        "unit":    f"fraction of days",
        "col":     [COL_PRECIP],
        "compute": compute_rainy_days,
    },
    {
        "name":    "precip_intensity",
        "label":   "Precipitation Intensity (SDII)",
        "unit":    "mm/wet-day",
        "col":     [COL_PRECIP],
        "compute": compute_precip_intensity,
    },
    {
        "name":    "heavy_rain_days",
        "label":   f"Heavy Rain Days (R{int(HEAVY_THRESHOLD)}mm)",
        "unit":    f"fraction of days",
        "col":     [COL_PRECIP],
        "compute": compute_heavy_rain_days,
    },
    {
        "name":    "cdd",
        "label":   "Consecutive Dry Days (CDD)",
        "unit":    f"days",
        "col":     [COL_PRECIP],
        "compute": compute_cdd,
    },
    # Temperature
    {
        "name":    "summer_days",
        "label":   f"Summer Days (Tmax ≥ {SUMMER_DAY_TMAX}°C)",
        "unit":    f"fraction of days",
        "col":     [COL_TMAX],
        "compute": compute_summer_days,
    },
    {
        "name":    "tropical_nights",
        "label":   f"Tropical Nights (Tmin ≥ {TROPICAL_NIGHT_TMIN}°C)",
        "unit":    f"fraction of nights",
        "col":     [COL_TMIN],
        "compute": compute_tropical_nights,
    },
    {
        "name":    "tx90p",
        "label":   "Warm Days (Tx90p)",
        "unit":    f"fraction of days",
        "col":     [COL_TMAX],
        "compute": compute_tx90p,
    },
# Temperature percentile indices
    {
        "name":    "tmax_above_p90",
        "label":   "Hot Days (Tmax > p90)",
        "unit":    "fraction of days",
        "col":     [COL_TMAX],
        "compute": compute_tmax_above_p90,
    },
    {
        "name":    "tmax_below_p10",
        "label":   "Cold Days (Tmax < p10)",
        "unit":    "fraction of days",
        "col":     [COL_TMAX],
        "compute": compute_tmax_below_p10,
    },
    {
        "name":    "tmin_above_p90",
        "label":   "Warm Nights (Tmin > p90)",
        "unit":    "fraction of nights",
        "col":     [COL_TMIN],
        "compute": compute_tmin_above_p90,
    },
    {
        "name":    "tmin_below_p10",
        "label":   "Cold Nights (Tmin < p10)",
        "unit":    "fraction of nights",
        "col":     [COL_TMIN],
        "compute": compute_tmin_below_p10,
    },
]

# ---------------------------------------------------------------------------
# Which group averages to show, per index
# ---------------------------------------------------------------------------

SHOW_AVG = {
    "windy_days":        {"all": True,  "malta": True,  "gozo": True},
#    "accumulated_wind":  {"all": True,  "malta": True,  "gozo": True},
    "rainy_days":        {"all": True,  "malta": True,  "gozo": True},
    "precip_intensity":  {"all": True,  "malta": True,  "gozo": True},
    "heavy_rain_days":   {"all": True,  "malta": True,  "gozo": True},
    "cdd":               {"all": True,  "malta": True,  "gozo": True},
    "summer_days":       {"all": True,  "malta": True,  "gozo": True},
    "tropical_nights":   {"all": True,  "malta": True,  "gozo": True},
    "tmax_above_p90":    {"all": True,  "malta": True,  "gozo": True},
    "tmax_below_p10":    {"all": True,  "malta": True,  "gozo": True},
    "tmin_above_p90":    {"all": True,  "malta": True,  "gozo": True},
    "tmin_below_p10":    {"all": True,  "malta": True,  "gozo": True},
}

# ---------------------------------------------------------------------------
# Smoothing (applied on top of the 31-day rolling index)
# ---------------------------------------------------------------------------

SMOOTH_WINDOW = 31

# ---------------------------------------------------------------------------
# Mode 1: daily index → smooth climatology with 31-day rolling mean
# Mode 2: daily index → monthly aggregation, no smoothing
# ---------------------------------------------------------------------------

PLOT_MODE = 2   # change to 2 for monthly

# ---------------------------------------------------------------------------
# Colours — consistent across all plots
# ---------------------------------------------------------------------------

STATION_COLOURS = {
    "Giordan":      "#1f78b4",
    "San-Lawrenz":  "#33a02c",
    "Xagħra":       "#6a3d9a",
    "Xewkija":      "#a6cee3",
    "Selmun":       "#e31a1c",
    "Dingli":       "#ff7f00",
    "Zokrija":      "#b15928",
    "Birkirkara":   "#fb9a99",
    "Valletta":     "#fdbf6f",
    "Luqa":         "#cab2d6",
    "Bengħajsa":    "#b2df8a",
}

AVG_ALL_COLOUR   = "#000000"
AVG_MALTA_COLOUR = "#d62728"
AVG_GOZO_COLOUR  = "#1f77b4"
AVG_LINEWIDTH    = 2.5
STN_LINEWIDTH    = 1.2

BLANK_COLOUR     = "#ffffff"
BLANK_LINEWIDTH  = 0.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_daily(station):
    filepath = f"{workdir}/Data/Daily_{station}.csv"
    if not os.path.exists(filepath):
        print(f"  [warn] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath, header=0)
    if DATE_COL not in df.columns:
        print(f"  [warn] '{DATE_COL}' not found in {filepath}")
        return None
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df[df['Date'] >= pd.Timestamp("2007-01-01")]
    df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL)
    return df


def has_columns(df, cols, station):
    """Return True if all required columns are present."""
    for col in cols:
        if col not in df.columns:
            print(f"  [warn] Column '{col}' not found for {station}. Skipping index.")
            return False
    return True

def compute_climatology(index_series, mode, agg="mean"):
    if mode == 1:
        # Group by doy, average across years, then smooth
        doy  = index_series.index.dayofyear
        clim = index_series.groupby(doy).mean().reindex(range(1, 367))
        if SMOOTH_WINDOW > 1:
            vals   = clim.values
            pad    = SMOOTH_WINDOW // 2
            padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
            smoothed = pd.Series(padded).rolling(SMOOTH_WINDOW, center=True,
                                                  min_periods=1).mean().values
            clim = pd.Series(smoothed[pad:-pad], index=clim.index)
        return clim

    elif mode == 2:
        if agg == "sum":
            # Count events within each month of each year, then average those
            # monthly counts across years -> mean number of days per month
            per_month = index_series.groupby(
                [index_series.index.year, index_series.index.month]
            ).sum(min_count=1)
            return per_month.groupby(level=1).mean()
        return index_series.groupby(index_series.index.month).mean()

def group_mean(clim_dict, members):
    available = [clim_dict[s] for s in members if s in clim_dict]
    if not available:
        return None
    return pd.concat(available, axis=1).mean(axis=1)


def month_tick_positions():
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    midpoints = [16, 47, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
    return midpoints, months


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_index_plot(index_def, clim_dict, output_dir):
    """Draw the annual cycle plot for one index."""

    name  = index_def["name"]
    label = index_def["label"]
    unit  = index_def["unit"]

    if not clim_dict:
        print(f"  [skip] No data for {name}.")
        return

    show = SHOW_AVG.get(name, {"all": False, "malta": False, "gozo": False})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.margins(x=0)

    if PLOT_MODE == 1:
        x = np.arange(1, 367)
        mode_tag = "daily"
    else:
        x = np.arange(1, 13)
        mode_tag = "monthly"

    # --- Draw station lines ---
    if PLOT_MODE == 1:
        ax.set_xlim(1, 366)
        tick_pos, tick_labels = month_tick_positions()
    else:
        ax.set_xlim(0.5, 12.5)
        plt.tight_layout()
        tick_pos   = np.arange(1, 13)
        tick_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)

    # --- Draw station lines ---
    for station in ALL_STATIONS:
        if station not in clim_dict:
            continue
        ax.plot(x, clim_dict[station].reindex(x).values,
                color=STATION_COLOURS.get(station, "#888888"),
                linewidth=STN_LINEWIDTH, alpha=0.85, zorder=3)

    # --- Group averages ---
    avg_all = avg_malta = avg_gozo = None

    if show.get("all"):
        avg_all = group_mean(clim_dict, [s for s in ALL_STATIONS if s in clim_dict])
        if avg_all is not None:
            ax.plot(x, avg_all.reindex(x).values,
                    color=AVG_ALL_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    if show.get("malta"):
        avg_malta = group_mean(clim_dict, [s for s in MALTA_STATIONS if s in clim_dict])
        if avg_malta is not None:
            ax.plot(x, avg_malta.reindex(x).values,
                    color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    if show.get("gozo"):
        avg_gozo = group_mean(clim_dict, [s for s in GOZO_STATIONS if s in clim_dict])
        if avg_gozo is not None:
            ax.plot(x, avg_gozo.reindex(x).values,
                    color=AVG_GOZO_COLOUR, linewidth=AVG_LINEWIDTH,
                    linestyle="--", zorder=5)

    # --- Legend: fixed 5-col x 3-row layout ---
    NCOL = 5
    NROW = 3
    spacer = mlines.Line2D([], [], color="none", label="")

    station_handles = []
    for station in ALL_STATIONS:
        if station in clim_dict:
            station_handles.append(
                mlines.Line2D([], [], color=STATION_COLOURS.get(station, "#888888"),
                              linewidth=STN_LINEWIDTH,
                              label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " "))
                )
        else:
            station_handles.append(
                mlines.Line2D([], [], color=BLANK_COLOUR, linewidth=BLANK_LINEWIDTH,
                              label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " "))
            )

    avg_handles = []
    if avg_gozo  is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_GOZO_COLOUR,  linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (Gozo)"))
    else:                     avg_handles.append(mlines.Line2D([], [], color=BLANK_COLOUR, linewidth=BLANK_LINEWIDTH, label="Avg (Gozo)"))
    if avg_malta is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (Malta)"))
    else:                     avg_handles.append(mlines.Line2D([], [], color=BLANK_COLOUR, linewidth=BLANK_LINEWIDTH, label="Avg (Malta)"))
    if avg_all   is not None: avg_handles.append(mlines.Line2D([], [], color=AVG_ALL_COLOUR,   linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (all stations)"))
    else:                     avg_handles.append(mlines.Line2D([], [], color=BLANK_COLOUR, linewidth=BLANK_LINEWIDTH, label="Avg (all stations)"))

    # Transpose station handles into row-major display order
    stn_slots = [spacer] * (4 * NROW)
    for i, h in enumerate(station_handles):
        row = i // 4
        col = i % 4
        stn_slots[col * NROW + row] = h

    legend_handles = stn_slots + avg_handles
    title_pad = 8 + NROW * 14

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=NCOL,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
        handlelength=1.8,
    )

    # --- Axes ---
    ax.set_xlabel("months", fontsize=11)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_title(f"{label}", fontsize=13, fontweight="bold", pad=title_pad)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

#   plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"annual-cycle-index_{name}_{mode_tag}.png")
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

    for index_def in INDICES:
        name    = index_def["name"]
        req_cols = index_def["col"]
        compute  = index_def["compute"]
        print(f"\n[index] {name}")

        clim_dict = {}
        for station, df in daily_data.items():
          # print(f"  [debug] {station} columns: {list(df.columns)}")
            if not has_columns(df, req_cols, station):
                continue
            try:
                index_series = compute(df, ROLL_WINDOW)
                clim = compute_climatology(index_series, PLOT_MODE,
                                           index_def.get("agg", "mean"))
                clim_dict[station] = clim
                print(f"  {station:20s}  valid doys: {clim.notna().sum()}")
            except Exception as e:
                print(f"  [warn] {station}: {e}")

        create_index_plot(index_def, clim_dict, output_dir)

    print(f"\n[done] All plots written to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_output_dir = sys.argv[1] if len(sys.argv) > 1 else f"{workdir}/analysis/annual-cycle"
    main(run_output_dir)
