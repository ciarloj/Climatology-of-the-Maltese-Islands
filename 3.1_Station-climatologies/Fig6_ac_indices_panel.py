"""
Figure panel: climate-index annual cycles (5x2)
===============================================
Combines the per-index annual-cycle plots into one A4-portrait panel.
Reuses the computation logic from plot_annual_cycle_indices.py; only the
drawing is retargeted to a shared grid of axes.

Usage:
    python3 figure_indices_panel.py

Output: {workdir}/analysis/annual-cycle/figure_indices_panel.png
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

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

# Which indices, in panel order (left->right, top->bottom).
# Names must match the "name" field in the INDICES registry below.
PANEL_INDICES = [
#    "cdd", "precip_intensity",
#    "rainy_days", # "heavy_rain_days",
#    "tmax_above_p90", "tmax_below_p10",
#    "tmin_above_p90", "tmin_below_p10",
#    "windy_days",
    "precip_intensity", "rainy_days",
    "tmax_above_p90", "tmax_below_p10",
    "cdd", "windy_days",
]
NROWS = 3
NCOLS = 2

ALL_STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]
GOZO_STATIONS  = ["Giordan", "Xagħra", "Xewkija"]
MALTA_STATIONS = ["Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa"]
DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

DATE_COL    = "Date"
MISSING_VAL = "-"
COL_TMAX    = "Max Temperature"
COL_TMIN    = "Min Temperature"
COL_PRECIP  = "Precipitation"
COL_WIND    = "Wind Speed"

ROLL_WINDOW         = 31
WINDY_THRESHOLD     = 10.0
WET_THRESHOLD       = 1.0
HEAVY_THRESHOLD     = 20.0
SUMMER_DAY_TMAX     = 25.0
TROPICAL_NIGHT_TMIN = 20.0

PLOT_MODE     = 2     # monthly
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
AVG_LINEWIDTH = 1.5
STN_LINEWIDTH = 1.2

# ---------------------------------------------------------------------------
# Index computations — copied unchanged from plot_annual_cycle_indices.py
# (each returns a daily 0/1 or value series; NaN preserved for gaps)
# ---------------------------------------------------------------------------

def _count(series, mask):
    ind = mask.astype(float)
    ind[series.isna()] = np.nan
    return ind

def compute_windy_days(df, w):
    v = pd.to_numeric(df[COL_WIND].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v >= WINDY_THRESHOLD)

def compute_rainy_days(df, w):
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v >= WET_THRESHOLD)

def compute_heavy_rain_days(df, w):
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v >= HEAVY_THRESHOLD)

def compute_precip_intensity(df, w):
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    roll  = lambda s: s.rolling(w, center=True, min_periods=1)
    wet   = v.where(v >= WET_THRESHOLD)
    total = roll(wet.fillna(0)).sum()
    count = roll((v >= WET_THRESHOLD).astype(float)).sum()
    return total.where(count > 0) / count.where(count > 0)

def compute_cdd(df, w):
    v = pd.to_numeric(df[COL_PRECIP].replace(MISSING_VAL, np.nan), errors="coerce")
    dry = (v < WET_THRESHOLD).astype(float)
    half = w // 2
    result = np.full(len(dry), np.nan)
    dv = dry.values
    for i in range(half, len(dv) - half):
        seg = dv[i - half: i + half + 1]
        if np.all(np.isnan(seg)):
            continue
        mx = cur = 0
        for val in seg:
            if val == 1.0:
                cur += 1; mx = max(mx, cur)
            else:
                cur = 0
        result[i] = mx
    return pd.Series(result, index=df.index)

def compute_tmax_above_p90(df, w):
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v > v.quantile(0.90))

def compute_tmax_below_p10(df, w):
    v = pd.to_numeric(df[COL_TMAX].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v < v.quantile(0.10))

def compute_tmin_above_p90(df, w):
    v = pd.to_numeric(df[COL_TMIN].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v > v.quantile(0.90))

def compute_tmin_below_p10(df, w):
    v = pd.to_numeric(df[COL_TMIN].replace(MISSING_VAL, np.nan), errors="coerce")
    return _count(v, v < v.quantile(0.10))

# Registry: name -> (letter label, y-axis unit, required columns, compute fn)
INDICES = {
    "cdd":             ("Consecutive Dry Days",    "max consecutive dry days", [COL_PRECIP], compute_cdd),
    "precip_intensity":("Precipitation Intensity","mm/wet day",               [COL_PRECIP], compute_precip_intensity),
    "rainy_days":      ("Rainy Days",                    "fraction of days",         [COL_PRECIP], compute_rainy_days),
#    "heavy_rain_days": (f"Heavy Rain Days (R{int(HEAVY_THRESHOLD)}mm)", "fraction of days", [COL_PRECIP], compute_heavy_rain_days),
    "tmax_above_p90":  ("Hot Days (Tmax > p90)",         "fraction of days",         [COL_TMAX], compute_tmax_above_p90),
    "tmax_below_p10":  ("Cold Days (Tmax < p10)",        "fraction of days",         [COL_TMAX], compute_tmax_below_p10),
#    "tmin_above_p90":  ("Warm Nights (Tmin > p90)",      "fraction of nights",       [COL_TMIN], compute_tmin_above_p90),
#    "tmin_below_p10":  ("Cold Nights (Tmin < p10)",      "fraction of nights",       [COL_TMIN], compute_tmin_below_p10),
    "windy_days":      (f"Windy Days (≥{WINDY_THRESHOLD} m/s)", "fraction of days",   [COL_WIND], compute_windy_days),
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_daily(station):
    fp = f"{workdir}/Data/Daily_{station}.csv"
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, header=0)
    if DATE_COL not in df.columns:
        return None
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
    return df.dropna(subset=[DATE_COL]).set_index(DATE_COL)

def has_columns(df, cols):
    return all(c in df.columns for c in cols)

def compute_climatology(series):
    # monthly aggregation (mode 2)
    return series.groupby(series.index.month).mean()

def group_mean(clim_dict, members):
    available = [clim_dict[s] for s in members if s in clim_dict]
    if not available:
        return None
    return pd.concat(available, axis=1).mean(axis=1)

def build_clim_dict(index_name, daily_data):
    _, _, req_cols, compute = INDICES[index_name]
    clim = {}
    for station, df in daily_data.items():
        if not has_columns(df, req_cols):
            continue
        try:
            s = compute(df, ROLL_WINDOW)
            clim[station] = compute_climatology(s)
        except Exception as e:
            print(f"  [warn] {station}/{index_name}: {e}")
    return clim

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_index(ax, index_name, clim_dict, show_xlabels):
    _, unit, _, _ = INDICES[index_name]
    x = np.arange(1, 13)
    months = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    ax.set_xlim(0.5, 12.5)

    for station in ALL_STATIONS:
        if station not in clim_dict:
            continue
        ax.plot(x, clim_dict[station].reindex(x).values,
                color=STATION_COLOURS.get(station, "#888888"),
                linewidth=STN_LINEWIDTH, alpha=0.85, zorder=3)

    for members, colour in [
        ([s for s in GOZO_STATIONS  if s in clim_dict], AVG_GOZO_COLOUR),
        ([s for s in MALTA_STATIONS if s in clim_dict], AVG_MALTA_COLOUR),
        ([s for s in ALL_STATIONS   if s in clim_dict], AVG_ALL_COLOUR),
    ]:
        a = group_mean(clim_dict, members)
        if a is not None:
            ax.plot(x, a.reindex(x).values, color=colour,
                    linewidth=AVG_LINEWIDTH, linestyle="--", zorder=5)

    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(months, fontsize=9)
        ax.set_xlabel("months", fontsize=10)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(unit, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    ax.tick_params(labelsize=8)

def build_legend_handles():
    handles = []
    for station in ALL_STATIONS:
        handles.append(mlines.Line2D([], [], color=STATION_COLOURS.get(station, "#888888"),
                       linewidth=STN_LINEWIDTH,
                       label=DISPLAY_NAME_OVERRIDES.get(station, station).replace("-", " ")))
    handles.append(mlines.Line2D([], [], color=AVG_GOZO_COLOUR,  linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (Gozo)"))
    handles.append(mlines.Line2D([], [], color=AVG_MALTA_COLOUR, linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (Malta)"))
    handles.append(mlines.Line2D([], [], color=AVG_ALL_COLOUR,   linewidth=AVG_LINEWIDTH, linestyle="--", label="Avg (all stations)"))
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

    # A4 portrait ~ 8.27 x 11.69 in; leave headroom for legend + caption
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 6.0, NROWS * 3.8), squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, name in enumerate(PANEL_INDICES):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        clim = build_clim_dict(name, daily_data)
        is_bottom = idx >= len(PANEL_INDICES) - NCOLS
        draw_index(ax, name, clim, show_xlabels=is_bottom)
        label = INDICES[name][0]
        ax.set_title(f"({letters[idx]}) {label}", fontsize=10,
                     fontweight="bold", loc="left")

    # Hide unused cells; give the last real plot its x-labels since its
    # column partner below is empty
    used = len(PANEL_INDICES)
    for idx in range(used, NROWS * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)
    if used % NCOLS != 0:                       # last row is partial
        last = axes[(used - 1) // NCOLS][(used - 1) % NCOLS]
        last.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=9)
        last.set_xlabel("months", fontsize=10)

    fig.legend(handles=build_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=8,
               frameon=True, handlelength=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = f"{workdir}/analysis/annual-cycle/figure_indices_panel.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
