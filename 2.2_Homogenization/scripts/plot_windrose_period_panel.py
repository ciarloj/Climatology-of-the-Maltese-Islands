"""
Wind Rose Panel — by Station and Period
=======================================
Panel of wind roses: one ROW per station, three COLUMNS for the periods
2007-2012 / 2013-2015 / 2016-2025. Purpose-built to show the 2013-2015
wind-direction fault (a bias toward north) against the good years either side.
The middle (suspect) column header is drawn in red.

Usage:
    python3 plot_windrose_period_panel.py

Output: {workdir}/analysis/windrose/windrose_period_panel.png
Input:  {workdir}/Data/Daily_{station}.csv
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ---------------------------------------------------------------------------
workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

STATIONS = [
    ("Giordan",     "GD"),
    ("Xagħra",      "XR"),
    ("Xewkija",     "XW"),
    ("Selmun",      "SL"),
    ("Dingli",      "DN"),
    ("Birkirkara",  "BK"),
    ("Valletta",    "VL"),
    ("Luqa",        "LQ"),
    ("Bengħajsa",   "BN"),
]

STATION_FILENAMES = {
    "GD": "Giordan", "XR": "Xagħra", "XW": "Xewkija", "SL": "Selmun",
    "DN": "Dingli", "BK": "Birkirkara", "VL": "Valletta", "LQ": "Luqa",
    "BN": "Bengħajsa",
}

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# Columns: (label, year_start, year_end, is_suspect)
PERIODS = [
    ("2007–2012", 2007, 2012, False),
    ("2013–2015", 2013, 2015, True),
    ("2016–2025", 2016, 2025, False),
]

DATE_COL    = "Date"
MISSING_VAL = "-"
COL_WIND_DIR   = "Wind Direction"
COL_WIND_SPEED = "Wind Speed"

MIN_OBS      = 365    # min valid pairs to include a station (row)
MIN_CELL_OBS = 90     # min valid pairs to draw a period (cell)

N_DIR_BINS = 16
SPEED_BINS    = [0, 5, 10, 15, 20, np.inf]
SPEED_LABELS  = ["0–5", "5–10", "10–15", "15–20", ">20"]
SPEED_COLOURS = ["#c7e9b4", "#41ab5d", "#41b6c4", "#2c7fb8", "#253494"]

OUTPUT_PATH = f"{workdir}/analysis/windrose/windrose_period_panel.png"


# ---------------------------------------------------------------------------
def load_wind_data(abbr):
    """Return a date-indexed DataFrame with columns wdir, wspd (valid pairs)."""
    filename = STATION_FILENAMES.get(abbr)
    if filename is None:
        return None
    filepath = f"{workdir}/Data/Daily_{filename}.csv"
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, header=0)

        def find_col(target):
            if target in df.columns:
                return target
            norm = lambda s: s.lower().replace(" ", "_")
            m = [c for c in df.columns if norm(c) == norm(target)]
            return m[0] if m else None

        dir_col, spd_col, date_col = find_col(COL_WIND_DIR), find_col(COL_WIND_SPEED), find_col(DATE_COL)
        if None in (dir_col, spd_col, date_col):
            return None

        dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        wdir  = pd.to_numeric(df[dir_col].replace(MISSING_VAL, np.nan), errors="coerce")
        wspd  = pd.to_numeric(df[spd_col].replace(MISSING_VAL, np.nan), errors="coerce")
        out = pd.DataFrame({"wdir": wdir.values, "wspd": wspd.values}, index=dates)
        return out[out["wdir"].notna() & out["wspd"].notna() & out.index.notna()]
    except Exception as e:
        print(f"  [warn] {abbr}: {e}")
        return None


def compute_windrose(wdir, wspd):
    """(N_DIR_BINS, N_SPEED_BINS) array of frequency fractions."""
    n_spd  = len(SPEED_BINS) - 1
    counts = np.zeros((N_DIR_BINS, n_spd))
    sector = 360.0 / N_DIR_BINS
    di = (((wdir % 360) + sector / 2) // sector % N_DIR_BINS).astype(int)
    si = np.digitize(wspd, SPEED_BINS[1:-1])
    for d, s in zip(di, si):
        counts[d, s] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def draw_windrose(ax, rose_data, subtitle=None, r_max=None):
    sector_rad = 2 * np.pi / N_DIR_BINS
    angles     = np.linspace(0, 2 * np.pi, N_DIR_BINS, endpoint=False)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bottoms = np.zeros(N_DIR_BINS)
    for si in range(len(SPEED_BINS) - 1):
        h = rose_data[:, si]
        ax.bar(angles, h, width=sector_rad * 0.85, bottom=bottoms,
               color=SPEED_COLOURS[si], edgecolor="white", linewidth=0.4, zorder=3)
        bottoms += h

    # r_max (if given) fixes the radial scale so periods in a row are comparable
    max_r = r_max if (r_max is not None and r_max > 0) else bottoms.max()
    if max_r > 0:
        nice_max = np.ceil(max_r / 0.05) * 0.05
        rticks   = np.linspace(0, nice_max, 4)[1:]
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{int(r*100)}%" for r in rticks], fontsize=5, color="#444444")
        ax.set_ylim(0, nice_max * 1.1)

    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=6)
    ax.set_facecolor("#f8f8f8")
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("#888888")
    ax.grid(color="grey", linewidth=0.3, alpha=0.4)
    if subtitle:
        ax.set_title(subtitle, fontsize=6.5, color="#555555", pad=2)


def empty_cell(ax, n):
    ax.text(0.5, 0.5, f"n={n}\n(insufficient)", transform=ax.transAxes,
            ha="center", va="center", fontsize=6, color="#aaaaaa")
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#fcfcfc")


# ---------------------------------------------------------------------------
def main():
    stations_data = []
    for name, abbr in STATIONS:
        print(f"  {abbr} ...", end=" ")
        df = load_wind_data(abbr)
        if df is None or len(df) < MIN_OBS:
            print(f"skipped ({0 if df is None else len(df)} obs)")
            continue
        print(f"{len(df)} obs")
        stations_data.append((name, abbr, df))

    if not stations_data:
        print("[error] No stations with sufficient data.")
        return

    n_rows, n_cols = len(stations_data), len(PERIODS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.9, n_rows * 2.7),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (name, abbr, df) in enumerate(stations_data):
        disp = DISPLAY_NAME_OVERRIDES.get(name, name)
        # Build all three period roses first so the radial scale can be shared
        cells = []
        for (plabel, y0, y1, suspect) in PERIODS:
            sub = df[(df.index.year >= y0) & (df.index.year <= y1)]
            n = len(sub)
            if n < MIN_CELL_OBS:
                cells.append((None, n))
            else:
                cells.append((compute_windrose(sub["wdir"].values, sub["wspd"].values), n))
        row_max = max((rose.sum(axis=1).max() for rose, n in cells if rose is not None),
                      default=0)
        for j, (rose, n) in enumerate(cells):
            ax = axes[i, j]
            if rose is None:
                empty_cell(ax, n)
            else:
                draw_windrose(ax, rose, subtitle=f"n={n:,}", r_max=row_max)
        # row label (station)
        axes[i, 0].annotate(disp, xy=(-0.40, 0.5), xycoords="axes fraction",
                            ha="center", va="center", rotation=90,
                            fontsize=11, fontweight="bold")

    # column headers (periods); suspect one in red
    for j, (plabel, y0, y1, suspect) in enumerate(PERIODS):
        axes[0, j].annotate(plabel, xy=(0.5, 1.30), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=13, fontweight="bold",
                            color=("#c1121f" if suspect else "#222222"))

    legend_patches = [
        mpatches.Patch(facecolor=SPEED_COLOURS[i], edgecolor="white", linewidth=0.5,
                       label=f"{SPEED_LABELS[i]} m/s")
        for i in range(len(SPEED_LABELS))
    ]
    fig.suptitle("Wind Rose by Station and Period", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.955, bottom=0.05, left=0.06, hspace=0.45, wspace=0.30)
    fig.legend(handles=legend_patches, title="Wind Speed (m/s)", title_fontsize=9,
               loc="lower center", ncol=len(SPEED_LABELS), fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
