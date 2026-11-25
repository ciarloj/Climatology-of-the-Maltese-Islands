"""
Figure panel: UHI annual-cycle (3x1)
====================================
Combines the min-temp / mean-temp / DTR UHI annual-cycle plots into one
stacked panel with a single shared legend and a shared bottom x-axis.
Reuses the computation logic from plot_uhi_annual_cycle.py.

Usage:
    python3 Fig11_uhi_panel.py

Output: {workdir}/analysis/uhi/Fig11_uhi_panel.png
"""

import os
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

# Panel order (top -> bottom). Titles use the short label only.
# (varname, csv_col, short_title, unit)
PANEL_VARS = [
    ("Min_Temp",    "Min Temperature", "Min Temperature",  "°C"),
    ("Temperature", "Temperature",     "Mean Temperature", "°C"),
    ("DTR",         None,              "Diurnal Temp Range", "°C"),
]

ALL_STATIONS = [
    ("Giordan","GD"), ("Xagħra","XR"), ("Xewkija","XW"), ("Selmun","SL"),
    ("Dingli","DN"), ("Msida","MS"), ("Valletta","VL"), ("Luqa","LQ"),
    ("Bengħajsa","BN"),
]
STATION_FILENAMES = {
    "GD":"Giordan", "XR":"Xagħra", "XW":"Xewkija", "SL":"Selmun", "DN":"Dingli",
    "MS":"Birkirkara", "VL":"Valletta", "LQ":"Luqa", "BN":"Bengħajsa",
}
URBAN_STATIONS     = {"VL", "MS", "XR"}
RURAL_STATIONS     = {"XW", "SL", "DN"}
MONITORED_STATIONS = {"LQ", "BN", "GD"}

# Elevation adjustment (Section 4.2): Tmin and mean T adjusted to sea level
ELEVATION_ADJUST = True    # False = unadjusted version (SM)
ELEVATIONS = {"GD":167, "XR":155, "XW":104, "SL":129, "DN":250,
              "MS":66,  "VL":31,  "LQ":85,  "BN":36}   # Table 1, m a.s.l.
LAPSE_RATE = 0.0065        # °C per m (6.5 °C/km)

URBAN_COLOURS     = {"VL":"#e31a1c", "MS":"#33a02c", "XR":"#377eb8"}
MONITORED_COLOURS = {"LQ":"#6a3d9a", "BN":"#ff7f00", "GD":"#777777"}

DATE_COL      = "Date"
MISSING_VAL   = "-"
SMOOTH_WINDOW = 31
OUTPUT_DIR    = f"{workdir}/analysis/uhi"

# ---------------------------------------------------------------------------
# Computation — copied from plot_uhi_annual_cycle.py
# ---------------------------------------------------------------------------

def load_daily(abbr):
    fn = STATION_FILENAMES.get(abbr)
    if fn is None:
        return None
    fp = f"{workdir}/Data/Daily_{fn}.csv"
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, header=0)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
        return df.dropna(subset=[DATE_COL]).set_index(DATE_COL)
    except Exception:
        return None

def find_col(df, target):
    if target is None:
        return None
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    m = [c for c in df.columns if norm(c) == norm(target)]
    return m[0] if m else None

def get_daily_series(df, varname, col_name):
    if varname == "DTR":
        ct = find_col(df, "Max Temperature"); cn = find_col(df, "Min Temperature")
        if ct is None or cn is None:
            return None
        tmax = pd.to_numeric(df[ct].replace(MISSING_VAL, np.nan), errors="coerce")
        tmin = pd.to_numeric(df[cn].replace(MISSING_VAL, np.nan), errors="coerce")
        return tmax - tmin
    col = find_col(df, col_name)
    if col is None:
        return None
    return pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")

def doy_climatology(series):
    return series.groupby(series.index.dayofyear).mean().reindex(range(1, 367))

def smooth(series, window):
    if window <= 1:
        return series
    vals = series.values
    pad = window // 2
    padded = np.concatenate([vals[-pad:], vals, vals[:pad]])
    sm = pd.Series(padded).rolling(window, center=True, min_periods=1).mean().values
    return pd.Series(sm[pad:-pad], index=series.index)

def month_tick_positions():
    months = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    mids   = [16,47,75,106,136,167,197,228,258,289,319,350]
    return mids, months

def compute_clims(varname, col_name, daily_data):
    clims = {}
    for _, abbr in ALL_STATIONS:
        if abbr not in daily_data:
            continue
        s = get_daily_series(daily_data[abbr], varname, col_name)
        if s is None:
            continue
        if ELEVATION_ADJUST and varname != "DTR":
            s = s + LAPSE_RATE * ELEVATIONS[abbr]
        clims[abbr] = smooth(doy_climatology(s.dropna()), SMOOTH_WINDOW)
    return clims

# ---------------------------------------------------------------------------
# Drawing one variable onto an axis
# ---------------------------------------------------------------------------

def draw_uhi(ax, varname, col_name, ylabel, unit, daily_data, show_xlabel):
    clims = compute_clims(varname, col_name, daily_data)
    if not clims:
        return
    x = np.arange(1, 367)
    tick_pos, tick_labels = month_tick_positions()

    rural_abbrs = [a for a in RURAL_STATIONS if a in clims]
    if not rural_abbrs:
        return
    rural_mean = pd.concat([clims[a] for a in rural_abbrs], axis=1).mean(axis=1)

    urban_clims     = {a: clims[a] for a in URBAN_STATIONS if a in clims}
    monitored_clims = {a: clims[a] for a in MONITORED_STATIONS if a in clims}

    deltas = {a: c.reindex(x) - rural_mean.reindex(x) for a, c in urban_clims.items()}
    mono_deltas = {a: c.reindex(x) - rural_mean.reindex(x)
                   for a, c in monitored_clims.items()}

    delta_df = pd.DataFrame(deltas)
    mean_delta = delta_df.mean(axis=1)
    std_delta  = delta_df.std(axis=1)

    ax.fill_between(x, mean_delta.values - std_delta.values,
                    mean_delta.values + std_delta.values,
                    color="#cccccc", alpha=0.5, zorder=1)
    for a, d in deltas.items():
        ax.plot(x, d.values, color=URBAN_COLOURS.get(a, "#888888"),
                linewidth=1.0, alpha=0.8, zorder=3)
    for a, d in mono_deltas.items():
        ax.plot(x, d.values, color=MONITORED_COLOURS.get(a, "#777777"),
                linewidth=1.4, linestyle=(0, (4, 2)), alpha=0.9, zorder=4)
    ax.plot(x, mean_delta.values, color="#000000", linewidth=2.5,
            linestyle="--", zorder=5)
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle=":", zorder=2)

    ax.set_xlim(1, 366)
    ax.set_xticks(tick_pos)
    if show_xlabel:
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.set_xlabel("months", fontsize=11)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(f"Difference in {ylabel} (Δ{unit})", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

def build_legend_handles():
    handles = []
    for a in sorted(URBAN_STATIONS):
        handles.append(mlines.Line2D([], [], color=URBAN_COLOURS.get(a, "#888888"),
                       linewidth=1.2, label=a))
    for a in sorted(MONITORED_STATIONS):
        handles.append(mlines.Line2D([], [], color=MONITORED_COLOURS.get(a, "#777777"),
                       linewidth=1.4, linestyle=(0, (4, 2)), label=a))
    handles.append(mlines.Line2D([], [], color="#000000", linewidth=2.5,
                   linestyle="--", label="urban mean"))
    handles.append(mpatches.Patch(facecolor="#cccccc", edgecolor="none",
                   alpha=0.5, label="urban spread"))
    handles.append(mlines.Line2D([], [], color="#555555", linewidth=0.8,
                   linestyle=":", label="rural baseline"))
    return handles

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[load] Reading daily data files ...")
    daily_data = {}
    for _, abbr in ALL_STATIONS:
        df = load_daily(abbr)
        if df is not None:
            daily_data[abbr] = df

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 12.0), squeeze=False)
    letters = list(string.ascii_lowercase)

    for idx, (varname, col_name, title, unit) in enumerate(PANEL_VARS):
        ax = axes[idx][0]
        is_bottom = (idx == len(PANEL_VARS) - 1)
        draw_uhi(ax, varname, col_name, title, unit, daily_data, show_xlabel=is_bottom)
        ax.set_title(f"({letters[idx]}) {title}", fontsize=11,
                     fontweight="bold", loc="left")

    # Single shared legend across the top
    fig.legend(handles=build_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=9, fontsize=9, frameon=True) 

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.align_ylabels(axes[:, 0])
    fname = ("Fig11_uhi_panel.png" if ELEVATION_ADJUST
             else "figure_uhi_panel_no-elevation-adj.png")
    out_path = os.path.join(OUTPUT_DIR, fname)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
