"""
Wind Rose Panel Plot
=====================
Produces a panel of wind roses, one per station, showing wind speed
and direction distributions. Stations without sufficient data are omitted.

Usage:
    python3 plot_windrose_panel.py

Output: ../../analysis/windrose/windrose_panel.png
Input:  ../../Data/Daily_{station}.csv
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ---------------------------------------------------------------------------
# Station order — stations without data will be skipped automatically
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
    "GD":  "Giordan",
    "XR":  "Xagħra",
    "XW":  "Xewkija",
    "SL":  "Selmun",
    "DN":  "Dingli",
    "BK":  "Birkirkara",
    "VL":  "Valletta",
    "LQ":  "Luqa",
    "BN":  "Bengħajsa",
}

DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATE_COL    = "Date"
MISSING_VAL = "-"

COL_WIND_DIR   = "Wind Direction"
COL_WIND_SPEED = "Wind Speed"

# Minimum number of valid paired observations to include a station
MIN_OBS = 365

# Directional bins
N_DIR_BINS = 16   # 22.5° per sector

# Speed bins (m/s) and appearance — match reference image style
SPEED_BINS    = [0, 5, 10, 15, 20, np.inf]
SPEED_LABELS  = ["0–5", "5–10", "10–15", "15–20", ">20"]
SPEED_COLOURS = ["#c7e9b4", "#41ab5d", "#41b6c4", "#2c7fb8", "#253494"]

# Panel layout — number of columns; rows calculated automatically
N_COLS = 3

# Output
OUTPUT_PATH = f"{workdir}/analysis/windrose/windrose_panel.png"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wind_data(abbr):
    filename = STATION_FILENAMES.get(abbr)
    if filename is None:
        return None, None
    filepath = f"{workdir}/Data/Daily_{filename}.csv"
    if not os.path.exists(filepath):
        return None, None
    try:
        df = pd.read_csv(filepath, header=0)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]

        def find_col(target):
            if target in df.columns:
                return target
            norm = lambda s: s.lower().replace(" ", "_")
            matches = [c for c in df.columns if norm(c) == norm(target)]
            return matches[0] if matches else None

        dir_col = find_col(COL_WIND_DIR)
        spd_col = find_col(COL_WIND_SPEED)
        if dir_col is None or spd_col is None:
            return None, None

        wdir = pd.to_numeric(df[dir_col].replace(MISSING_VAL, np.nan), errors="coerce")
        wspd = pd.to_numeric(df[spd_col].replace(MISSING_VAL, np.nan), errors="coerce")

        mask = wdir.notna() & wspd.notna()
        return wdir[mask].reset_index(drop=True), wspd[mask].reset_index(drop=True)

    except Exception as e:
        print(f"  [warn] {abbr}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Wind rose computation
# ---------------------------------------------------------------------------

def compute_windrose(wdir, wspd):
    """
    Returns array of shape (N_DIR_BINS, N_SPEED_BINS) with frequency
    fractions — each cell is the fraction of total observations.
    """
    n_spd  = len(SPEED_BINS) - 1
    counts = np.zeros((N_DIR_BINS, n_spd))

    sector      = 360.0 / N_DIR_BINS
    dir_indices = (((wdir.values % 360) + sector / 2) // sector % N_DIR_BINS).astype(int)
    spd_indices = np.digitize(wspd.values, SPEED_BINS[1:-1])

    for di, si in zip(dir_indices, spd_indices):
        counts[di, si] += 1

    total = counts.sum()
    return counts / total if total > 0 else counts


# ---------------------------------------------------------------------------
# Draw one wind rose into a given polar axes
# ---------------------------------------------------------------------------

def draw_windrose(ax, rose_data, title, n_obs):
    """
    rose_data : (N_DIR_BINS, N_SPEED_BINS) frequency array
    """
    sector_rad = 2 * np.pi / N_DIR_BINS
    angles     = np.linspace(0, 2 * np.pi, N_DIR_BINS, endpoint=False)
    n_spd      = len(SPEED_BINS) - 1

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    bottoms = np.zeros(N_DIR_BINS)
    for si in range(n_spd):
        heights = rose_data[:, si]
        ax.bar(
            angles, heights,
            width=sector_rad * 0.85,
            bottom=bottoms,
            color=SPEED_COLOURS[si],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        bottoms += heights

    # Radial axis: format as percentage
    max_r = bottoms.max()
    if max_r > 0:
        nice_max = np.ceil(max_r / 0.05) * 0.05   # round up to nearest 5%
        rticks   = np.linspace(0, nice_max, 4)[1:]  # 3 rings, skip 0
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{int(r*100)}%" for r in rticks],
                           fontsize=5.5, color="#444444")
        ax.set_ylim(0, nice_max * 1.1)

    # Compass labels — N E S W only to keep it uncluttered
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=7)

    ax.set_facecolor("#f8f8f8")
    ax.spines["polar"].set_linewidth(0.6)
    ax.spines["polar"].set_color("#888888")
    ax.grid(color="grey", linewidth=0.3, alpha=0.4)

    ax.set_title(f"{title}", fontsize=8, fontweight="bold", pad=6)
#   ax.text(0.5, 1.16, f"(n={n_obs:,})",
#           transform=ax.transAxes,
#           ha="center", va="bottom",
#           fontsize=7, fontstyle="italic", fontweight="normal")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data for all stations, skip those without sufficient obs
    valid_stations = []
    for name, abbr in STATIONS:
        print(f"  {abbr} ...", end=" ")
        wdir, wspd = load_wind_data(abbr)
        if wdir is None or len(wdir) < MIN_OBS:
            n = len(wdir) if wdir is not None else 0
            print(f"skipped ({n} obs)")
            continue
        print(f"{len(wdir)} obs")
        rose_data = compute_windrose(wdir, wspd)
        valid_stations.append((name, abbr, rose_data, len(wdir)))

    if not valid_stations:
        print("[error] No stations with sufficient data.")
        return

    n_valid = len(valid_stations)
    n_cols  = min(N_COLS, n_valid)
    n_rows  = -(-n_valid // n_cols)   # ceiling division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.2),
        subplot_kw={"projection": "polar"},
    )

    # Flatten axes array for easy indexing
    axes_flat = np.array(axes).flatten() if n_valid > 1 else [axes]

    for idx, (name, abbr, rose_data, n_obs) in enumerate(valid_stations):
        display_name = DISPLAY_NAME_OVERRIDES.get(name, name)
        draw_windrose(axes_flat[idx], rose_data, display_name, n_obs)

    # Hide any unused axes
    for idx in range(n_valid, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legend — placed below the panel
    legend_patches = [
        mpatches.Patch(facecolor=SPEED_COLOURS[i], edgecolor="white",
                       linewidth=0.5, label=f"{SPEED_LABELS[i]} m/s")
        for i in range(len(SPEED_LABELS))
    ]

    fig.suptitle("Wind Rose by Station", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.55)
    fig.legend(
        handles=legend_patches,
        title="Wind Speed (m/s)",
        title_fontsize=9,
        loc="upper center",
        ncol=len(SPEED_LABELS),
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
