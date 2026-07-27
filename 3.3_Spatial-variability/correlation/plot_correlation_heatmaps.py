"""
Station Correlation Matrix Heatmaps
====================================
For each meteorological variable, produces a heatmap of inter-station
Pearson correlations. Cells with p < 0.05 are hatched. Missing stations
(no data for the variable) are shown in grey. The diagonal is black.

Time window restricted to 2007-2025 inclusive for consistency with
Giordan station coverage.

Usage:
    python3 plot_correlation_heatmaps.py

Output: ../analysis/corr_analysis/corr_{varname}.png
Input:  ../Data/Daily_{station}.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Station configuration — same order as elsewhere in the project
# ---------------------------------------------------------------------------

STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# Variable configuration
# ---------------------------------------------------------------------------

VARNAMES = [
    "Max Temperature", "Min Temperature", "Precipitation", "Pressure",
    "Relative Humidity", "Solar", "Temperature",
    "Wind Direction", "Wind Gust", "Wind Speed",
]

DATE_COL    = "Date"
MISSING_VAL = "-"

# Time window
DATE_MIN = pd.Timestamp("2007-01-01")
DATE_MAX = pd.Timestamp("2025-12-31")

# Minimum paired observations to compute a correlation
MIN_OBS = 30

# Stations to exclude from specific variables (keys must match VARNAMES)
EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},   # pressure completeness too low for consideration
}

# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------

# Diverging colormap, red=+1, white=0, blue=-1
CMAP        = "Reds"
MISSING_CLR = "#bbbbbb"     # grey for missing data
DIAG_CLR    = "#000000"     # black diagonal
LABEL_FONT  = 9

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

workdir    = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"
DATA_DIR   = f"{workdir}/Data"
OUTPUT_DIR = f"{workdir}/analysis/corr_analysis"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_col(df, target):
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    for c in df.columns:
        if norm(c) == norm(target):
            return c
    return None


def read_station(station):
    filepath = f"{DATA_DIR}/Daily_{station}.csv"
    if not os.path.exists(filepath):
        print(f"  [warn] File not found: {filepath}")
        return None
    df = pd.read_csv(filepath, na_values=["-", " -", "- "])

    date_col = find_col(df, DATE_COL)
    if date_col is None:
        print(f"  [warn] No Date column in {filepath}")
        return None
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[(df[date_col] >= DATE_MIN) & (df[date_col] <= DATE_MAX)]
    df = df.set_index(date_col)

    # Coerce all other columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Build paired-station correlations
# ---------------------------------------------------------------------------

def build_matrices(varname, station_data):
    """
    Returns (corr_df, pval_df, has_data_dict) — all keyed on STATIONS order.
    Cells where one or both stations lack the variable are NaN; has_data_dict
    flags which stations had any data for this variable.
    """
    n = len(STATIONS)
    corr = pd.DataFrame(np.full((n, n), np.nan), index=STATIONS, columns=STATIONS)
    pval = pd.DataFrame(np.full((n, n), np.nan), index=STATIONS, columns=STATIONS)

    # Identify which stations have the variable
    series_dict = {}
    has_data    = {}
    for stn in STATIONS:
        if stn in EXCLUDE_STATIONS.get(varname, set()):
            has_data[stn] = False
            continue
        df = station_data.get(stn)
        if df is None:
            has_data[stn] = False
            continue
        col = find_col(df, varname)
        if col is None:
            has_data[stn] = False
            continue
        s = df[col].dropna()
        if len(s) < MIN_OBS:
            has_data[stn] = False
            continue
        series_dict[stn] = s
        has_data[stn]    = True

    # Compute pairwise
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if s1 not in series_dict or s2 not in series_dict:
                continue
            if i == j:
                corr.loc[s1, s2] = 1.0
                pval.loc[s1, s2] = 0.0
                continue
            # Align on common dates
            joined = pd.concat([series_dict[s1], series_dict[s2]], axis=1,
                               join="inner").dropna()
            if len(joined) < MIN_OBS:
                continue
            try:
                r, p = pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])
                corr.loc[s1, s2] = r
                pval.loc[s1, s2] = p
            except Exception:
                pass

    return corr, pval, has_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_heatmap(varname, corr, pval, has_data, output_dir, cbar_min):
    n = len(STATIONS)
    display_labels = [DISPLAY_NAME_OVERRIDES.get(s, s).replace("-", " ")
                      for s in STATIONS]

    fig, ax = plt.subplots(figsize=(8, 7))
    norm = mcolors.Normalize(vmin=cbar_min, vmax=1.0)
    cmap = plt.get_cmap(CMAP)

    # Draw cell by cell so we can mix coloured, grey, hatched, black
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            # Diagonal — black
            if i == j:
                rect = mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                          facecolor=DIAG_CLR,
                                          edgecolor="white", linewidth=0.5)
                ax.add_patch(rect)
                continue

            # Missing data
            r = corr.loc[s1, s2]
            if pd.isna(r) or not has_data.get(s1, False) or not has_data.get(s2, False):
                rect = mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                          facecolor=MISSING_CLR,
                                          edgecolor="white", linewidth=0.5)
                ax.add_patch(rect)
                continue

            # Normal coloured cell
            colour = cmap(norm(r))
            rect = mpatches.Rectangle((j, n - 1 - i), 1, 1,
                                      facecolor=colour,
                                      edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)

    # Axes setup
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(display_labels, rotation=40, ha="right",
                       fontsize=LABEL_FONT)
    # Y labels go top-to-bottom in the same station order as rows
    ax.set_yticklabels(list(reversed(display_labels)), fontsize=LABEL_FONT)

    # Remove default frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Title
    ax.set_title(f"Inter-station Correlation: {varname}\n"
                 f"({DATE_MIN.year}–{DATE_MAX.year}, daily)",
                 fontsize=12, fontweight="bold", pad=38)

    # Colorbar
    cbar_ax = fig.add_axes([0.20, 0.06, 0.6, 0.025])
    cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                      orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson correlation coefficient", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # Legend for hatch + missing + diagonal — placed top right outside cells
    legend_patches = [
        mpatches.Patch(facecolor=MISSING_CLR, edgecolor="white",
                       label="No data"),
        mpatches.Patch(facecolor=DIAG_CLR, edgecolor="white",
                       label="Self (diagonal)"),
    ]
    ax.legend(handles=legend_patches,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.01),
              ncol=2,
              fontsize=8,
              framealpha=0.95,
              frameon=True)

    plt.subplots_adjust(bottom=0.18, right=0.92)

    os.makedirs(output_dir, exist_ok=True)
    safe_var = varname.replace(" ", "_").replace("/", "-")
    out_path = os.path.join(output_dir, f"corr_{safe_var}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[load] Reading station files (2007–2025) ...")
    station_data = {}
    for stn in STATIONS:
        df = read_station(stn)
        if df is not None and not df.empty:
            station_data[stn] = df
            print(f"  {stn:15s}  {len(df)} rows")
        else:
            print(f"  {stn:15s}  [no data in window]")

    # Pass 1 — compute all matrices and find global minimum
    results = []   # list of (varname, corr, pval, has_data)
    global_min = 1.0
    for varname in VARNAMES:
        print(f"\n[var] {varname}")
        corr, pval, has_data = build_matrices(varname, station_data)
        if not any(has_data.values()):
            print(f"  [skip] No data for {varname}")
            continue
        # Find min off-diagonal correlation
        n = len(STATIONS)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = corr.iat[i, j]
                if pd.notna(r):
                    global_min = min(global_min, float(r))
        results.append((varname, corr, pval, has_data))

    # Floor at 0 (avoid blue) but if global minimum is above 0 use it for tighter range
    cbar_min = max(0.0, global_min)
    print(f"\n[colorbar] vmin = {cbar_min:.3f}, vmax = 1.0")

    # Pass 2 — draw plots with shared colour range
    for varname, corr, pval, has_data in results:
        create_heatmap(varname, corr, pval, has_data, OUTPUT_DIR, cbar_min)

    print(f"\n[done] All plots written to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
