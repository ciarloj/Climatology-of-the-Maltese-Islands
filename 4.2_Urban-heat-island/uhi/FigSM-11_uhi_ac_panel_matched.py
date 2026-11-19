"""
Figure panel: UHI annual-cycle (3x1), MATCHED-DATES version (SM)
================================================================
Identical to Fig11_uhi_ac_panel.py, except that for each variable the
climatologies of the eight MIA stations (urban, rural baseline, LQ, and BN)
are computed only from days on which ALL of them report. Giordan, which sits
outside both groups, keeps its own dates so that its lower coverage does not
shrink the overlap.

The console also reports, per variable, the number of matched days and the
largest change in each departure (matched minus all-valid-days version), so
the sensitivity can be quoted in the text.

Usage:
    python3 Fig11_uhi_ac_panel_matched.py

Output: {workdir}/analysis/uhi/figure_uhi_panel_matched.png
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
# Stations whose dates are matched: all MIA stations. GD keeps its own dates.
MATCH_STATIONS     = URBAN_STATIONS | RURAL_STATIONS | {"LQ", "BN"}
N_PERIOD_DAYS      = len(pd.date_range("2007-01-01", "2025-12-31"))

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

def compute_clims(varname, col_name, daily_data, matched=True, verbose=True):
    series = {}
    for _, abbr in ALL_STATIONS:
        if abbr not in daily_data:
            continue
        s = get_daily_series(daily_data[abbr], varname, col_name)
        if s is None:
            continue
        series[abbr] = s.dropna()

    if matched:
        # Keep only the days on which every MIA station reports this variable
        in_match = [a for a in series if a in MATCH_STATIONS]
        if in_match:
            common = pd.concat([series[a] for a in in_match], axis=1,
                               join="inner").index
            for a in in_match:
                series[a] = series[a].loc[common]
            if verbose:
                print(f"  [match] {varname:12s} {len(common)} days common to "
                      f"{len(in_match)} MIA stations "
                      f"({100 * len(common) / N_PERIOD_DAYS:.1f}% of 2007-2025)")

    return {a: smooth(doy_climatology(s), SMOOTH_WINDOW) for a, s in series.items()}

def compute_departures(clims):
    """Departures from the rural-baseline mean, for urban and monitored stations."""
    x = np.arange(1, 367)
    rural_abbrs = [a for a in RURAL_STATIONS if a in clims]
    if not rural_abbrs:
        return None
    rural_mean = pd.concat([clims[a] for a in rural_abbrs], axis=1).mean(axis=1)
    deltas = {a: clims[a].reindex(x) - rural_mean.reindex(x)
              for a in URBAN_STATIONS if a in clims}
    mono_deltas = {a: clims[a].reindex(x) - rural_mean.reindex(x)
                   for a in MONITORED_STATIONS if a in clims}
    return deltas, mono_deltas

def report_sensitivity(varname, col_name, daily_data):
    """Print the largest |matched - all-valid-days| change in each departure."""
    out_m = compute_departures(compute_clims(varname, col_name, daily_data,
                                             matched=True, verbose=False))
    out_a = compute_departures(compute_clims(varname, col_name, daily_data, matched=False))
    if out_m is None or out_a is None:
        return
    (dm, mm), (da, ma) = out_m, out_a
    diffs = {"urban mean": (pd.DataFrame(dm).mean(axis=1)
                            - pd.DataFrame(da).mean(axis=1)).abs().max()}
    for a in sorted(dm):
        diffs[a] = (dm[a] - da[a]).abs().max()
    for a in sorted(mm):
        diffs[a] = (mm[a] - ma[a]).abs().max()
    txt = ", ".join(f"{k} {v:.2f}" for k, v in diffs.items())
    print(f"  [sensitivity] {varname}: max |matched - all days| (°C): {txt}")

# ---------------------------------------------------------------------------
# Drawing one variable onto an axis
# ---------------------------------------------------------------------------

def draw_uhi(ax, varname, col_name, ylabel, unit, daily_data, show_xlabel):
    clims = compute_clims(varname, col_name, daily_data, matched=True)
    if not clims:
        return
    x = np.arange(1, 367)
    tick_pos, tick_labels = month_tick_positions()

    out = compute_departures(clims)
    if out is None:
        return
    deltas, mono_deltas = out

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
        report_sensitivity(varname, col_name, daily_data)
        ax.set_title(f"({letters[idx]}) {title}", fontsize=11,
                     fontweight="bold", loc="left")

    # Single shared legend across the top
    fig.legend(handles=build_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=9, fontsize=9, frameon=True) 

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, "figure_uhi_panel_matched.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
