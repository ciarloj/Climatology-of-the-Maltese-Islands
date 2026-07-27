"""
Figure panel: inter-station correlation heatmaps (3x2)
======================================================
Combines per-variable correlation heatmaps into one panel with a single
shared colourbar and a single shared legend.
Reuses the computation logic from plot_correlation_heatmaps.py.

Usage:
    python3 figure_correlation_panel.py

Output: {workdir}/analysis/corr_analysis/figure_correlation_panel.png
"""

import os
import string
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
# Configuration
# ---------------------------------------------------------------------------

STATIONS = [
    "Giordan", "Xagħra", "Xewkija",
    "Selmun", "Dingli", "Birkirkara", "Valletta", "Luqa", "Bengħajsa",
]

DISPLAY_NAME_OVERRIDES = {
    "Giordan": "GD", "Xagħra": "XR", "Xewkija": "XW", "Selmun": "SL",
    "Dingli": "DN", "Birkirkara": "MS", "Valletta": "VL", "Luqa": "LQ",
    "Bengħajsa": "BN",
}

# Panel order (left->right, top->bottom)
PANEL_VARS = [
    "Precipitation", "Temperature",
    "Relative Humidity", "Wind Speed",
]
# Display names for titles (keep short; unit-free since r is dimensionless)
TITLE_NAMES = {
    "Precipitation": "Precipitation", "Temperature": "Temperature",
    "Relative Humidity": "Relative Humidity", "Wind Direction": "Wind Direction",
    "Wind Speed": "Wind Speed", "Wind Gust": "Wind Gust",
}
NROWS = 2
NCOLS = 2

DATE_COL = "Date"
DATE_MIN = pd.Timestamp("2007-01-01")
DATE_MAX = pd.Timestamp("2025-12-31")
MIN_OBS  = 30

EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},
}

HATCH_SIGNIFICANT = False    # hatch cells with p < 0.05; set False to disable
PVAL_THRESHOLD    = 0.05

CMAP        = "Reds"
MISSING_CLR = "#bbbbbb"
DIAG_CLR    = "#000000"
LABEL_FONT  = 8

workdir  = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"
DATA_DIR = f"{workdir}/Data"
OUTPUT_DIR = f"{workdir}/analysis/corr_analysis"

# ---------------------------------------------------------------------------
# Computation — copied from plot_correlation_heatmaps.py
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
    fp = f"{DATA_DIR}/Daily_{station}.csv"
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, na_values=["-", " -", "- "])
    dc = find_col(df, DATE_COL)
    if dc is None:
        return None
    df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[dc])
    df = df[(df[dc] >= DATE_MIN) & (df[dc] <= DATE_MAX)].set_index(dc)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def build_matrices(varname, station_data):
    n = len(STATIONS)
    corr = pd.DataFrame(np.full((n, n), np.nan), index=STATIONS, columns=STATIONS)
    pval = pd.DataFrame(np.full((n, n), np.nan), index=STATIONS, columns=STATIONS)
    series_dict, has_data = {}, {}
    for stn in STATIONS:
        if stn in EXCLUDE_STATIONS.get(varname, set()):
            has_data[stn] = False; continue
        df = station_data.get(stn)
        if df is None:
            has_data[stn] = False; continue
        col = find_col(df, varname)
        if col is None:
            has_data[stn] = False; continue
        s = df[col].dropna()
        if len(s) < MIN_OBS:
            has_data[stn] = False; continue
        series_dict[stn] = s
        has_data[stn] = True

    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if s1 not in series_dict or s2 not in series_dict:
                continue
            if i == j:
                corr.loc[s1, s2] = 1.0
                pval.loc[s1, s2] = 0.0
                continue
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
# Drawing one heatmap onto an axis
# ---------------------------------------------------------------------------

def draw_heatmap(ax, corr, pval, has_data, norm, cmap, show_xlabels, show_ylabels):
    n = len(STATIONS)
    display_labels = [DISPLAY_NAME_OVERRIDES.get(s, s).replace("-", " ")
                      for s in STATIONS]

    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if i == j:
                fc = DIAG_CLR
                hatch = None
            else:
                r = corr.loc[s1, s2]
                if pd.isna(r) or not has_data.get(s1, False) or not has_data.get(s2, False):
                    fc = MISSING_CLR
                    hatch = None
                else:
                    fc = cmap(norm(r))
                    p = pval.loc[s1, s2]
                    hatch = "///" if (HATCH_SIGNIFICANT and pd.notna(p)
                                      and p < PVAL_THRESHOLD) else None
            ax.add_patch(mpatches.Rectangle((j, n - 1 - i), 1, 1,
                         facecolor=fc, edgecolor="white", linewidth=0.5, hatch=hatch))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    if show_xlabels:
        ax.set_xticklabels(display_labels, fontsize=LABEL_FONT)
    else:
        ax.set_xticklabels([])
    if show_ylabels:
        ax.set_yticklabels(list(reversed(display_labels)), fontsize=LABEL_FONT)
    else:
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[load] Reading station files (2007–2025) ...")
    station_data = {}
    for stn in STATIONS:
        df = read_station(stn)
        if df is not None and not df.empty:
            station_data[stn] = df

    # Pass 1 — compute matrices, find shared colour minimum across the panel vars
    mats = {}
    global_min = 1.0
    for varname in PANEL_VARS:
        corr, pval, has_data = build_matrices(varname, station_data)
        mats[varname] = (corr, pval, has_data)
        n = len(STATIONS)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = corr.iat[i, j]
                if pd.notna(r):
                    global_min = min(global_min, float(r))
    cbar_min = max(0.0, global_min)
    print(f"[colorbar] vmin = {cbar_min:.3f}, vmax = 1.0")

    norm = mcolors.Normalize(vmin=cbar_min, vmax=1.0)
    cmap = matplotlib.colormaps[CMAP]

    # Pass 2 — draw
    fig = plt.figure(figsize=(9.5, 10.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.06],
                          hspace=0.12, wspace=0.05, top=0.94, bottom=0.06,
                          left=0.10, right=0.97)
    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r][c] = fig.add_subplot(gs[r, c])
    cbar_ax = fig.add_subplot(gs[2, :])   # colourbar spans both columns, own row
    letters = list(string.ascii_lowercase)

    for idx, varname in enumerate(PANEL_VARS):
        row, col = idx // NCOLS, idx % NCOLS
        ax = axes[row][col]
        corr, pval, has_data = mats[varname]
        is_bottom = idx >= len(PANEL_VARS) - NCOLS
        is_left   = (col == 0)
        draw_heatmap(ax, corr, pval, has_data, norm, cmap,
                     show_xlabels=is_bottom, show_ylabels=is_left)
        ax.set_title(f"({letters[idx]}) {TITLE_NAMES.get(varname, varname)}",
                     fontsize=10, fontweight="bold", loc="left")

    for idx in range(len(PANEL_VARS), NROWS * NCOLS):
        axes[idx // NCOLS][idx % NCOLS].set_visible(False)

    # Shared legend (top) — missing + diagonal
    legend_patches = [
        mpatches.Patch(facecolor=MISSING_CLR, edgecolor="white", label="No data"),
        mpatches.Patch(facecolor=DIAG_CLR, edgecolor="white", label="Self (diagonal)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               bbox_to_anchor=(0.54, 0.99), ncol=2, fontsize=9, frameon=True)

    # Shared colourbar tied to the whole grid — constrained layout reserves its space
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Pearson correlation coefficient", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    out_path = os.path.join(OUTPUT_DIR, "figure_correlation_panel.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
