"""
Seasonal Wind Rose Panel Plot
==============================
Produces a panel of wind roses: rows = stations, columns = seasons (DJF, MAM, JJA, SON).
Stations without sufficient data are omitted entirely.

Usage:
    python3 plot_windrose_seasonal.py

Output: ../../analysis/windrose/windrose_seasonal.png
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
# Seasons — month lists and display labels
# ---------------------------------------------------------------------------

SEASONS = [
    ("DJF", [12, 1, 2]),
    ("MAM", [3, 4, 5]),
    ("JJA", [6, 7, 8]),
    ("SON", [9, 10, 11]),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATE_COL    = "Date"
MISSING_VAL = "-"

COL_WIND_DIR   = "Wind Direction"
COL_WIND_SPEED = "Wind Speed"

# Minimum valid observations per season to draw a rose (otherwise greyed out)
MIN_OBS_SEASON = 90   # roughly one season's worth

N_DIR_BINS = 16

SPEED_BINS    = [0, 5, 10, 15, 20, np.inf]
SPEED_LABELS  = ["0–5", "5–10", "10–15", "15–20", ">20"]
SPEED_COLOURS = ["#c7e9b4", "#41ab5d", "#41b6c4", "#2c7fb8", "#253494"]

# Minimum total obs across all seasons for a station to be included at all
MIN_OBS_TOTAL = 365

OUTPUT_PATH = f"{workdir}/analysis/windrose/windrose_seasonal.png"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wind_data(abbr):
    """Load wind dir and speed, return as a DataFrame with a parsed date index."""
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
            matches = [c for c in df.columns if norm(c) == norm(target)]
            return matches[0] if matches else None

        date_col = find_col(DATE_COL)
        dir_col  = find_col(COL_WIND_DIR)
        spd_col  = find_col(COL_WIND_SPEED)
        if None in (date_col, dir_col, spd_col):
            return None

        df["_date"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        wdir = pd.to_numeric(df[dir_col].replace(MISSING_VAL, np.nan), errors="coerce")
        wspd = pd.to_numeric(df[spd_col].replace(MISSING_VAL, np.nan), errors="coerce")

        mask = wdir.notna() & wspd.notna() & df["_date"].notna()
        out  = pd.DataFrame({
            "date": df["_date"][mask].values,
            "wdir": wdir[mask].values,
            "wspd": wspd[mask].values,
        })
        out["month"] = pd.to_datetime(out["date"]).dt.month
        return out

    except Exception as e:
        print(f"  [warn] {abbr}: {e}")
        return None


# ---------------------------------------------------------------------------
# Wind rose computation
# ---------------------------------------------------------------------------

def compute_windrose(wdir, wspd):
    n_spd  = len(SPEED_BINS) - 1
    counts = np.zeros((N_DIR_BINS, n_spd))
    sector = 360.0 / N_DIR_BINS

    dir_indices = (((wdir % 360) + sector / 2) // sector % N_DIR_BINS).astype(int)
    spd_indices = np.digitize(wspd, SPEED_BINS[1:-1])

    for di, si in zip(dir_indices, spd_indices):
        counts[di, si] += 1

    total = counts.sum()
    return counts / total if total > 0 else counts


# ---------------------------------------------------------------------------
# Draw one wind rose
# ---------------------------------------------------------------------------

def draw_windrose(ax, rose_data, n_obs):
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

    max_r = bottoms.max()
    if max_r > 0:
        nice_max = np.ceil(max_r / 0.05) * 0.05
        rticks   = np.linspace(0, nice_max, 4)[1:]
        ax.set_yticks(rticks)
        ax.set_yticklabels([f"{int(r*100)}%" for r in rticks],
                           fontsize=4.5, color="#444444")
        ax.set_ylim(0, nice_max * 1.1)

    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=6)
    ax.set_facecolor("#f8f8f8")
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("#888888")
    ax.grid(color="grey", linewidth=0.3, alpha=0.4)


def draw_missing(ax):
    """Draw a greyed-out placeholder for a season with insufficient data."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("#e0e0e0")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_linewidth(0.5)
    ax.spines["polar"].set_color("#aaaaaa")
    ax.grid(False)
    ax.text(0, 0, "n/a", ha="center", va="center",
            fontsize=7, color="#888888", transform=ax.transData)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load and filter stations
    valid_stations = []
    for name, abbr in STATIONS:
        print(f"  {abbr} ...", end=" ")
        df = load_wind_data(abbr)
        if df is None or len(df) < MIN_OBS_TOTAL:
            n = len(df) if df is not None else 0
            print(f"skipped ({n} total obs)")
            continue
        print(f"{len(df)} obs")
        valid_stations.append((name, abbr, df))

    if not valid_stations:
        print("[error] No stations with sufficient data.")
        return

    n_rows = len(valid_stations)
    n_cols = len(SEASONS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.2),
        subplot_kw={"projection": "polar"},
    )

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Column headers (seasons) — drawn as text above the top row
    season_names = [s[0] for s in SEASONS]
    for col, season_name in enumerate(season_names):
        axes[0, col].set_title(season_name, fontsize=10, fontweight="bold", pad=28)

    # Row labels (station names) — drawn to the left of each row
    for row, (name, abbr, df) in enumerate(valid_stations):
        display_name = DISPLAY_NAME_OVERRIDES.get(name, name)
        axes[row, 0].set_ylabel(display_name, fontsize=8, fontweight="bold",
                                labelpad=12, rotation=90, va="center")


    # Draw roses
    for row, (name, abbr, df) in enumerate(valid_stations):
        for col, (season_name, months) in enumerate(SEASONS):
            ax      = axes[row, col]
            mask    = df["month"].isin(months)
            subset  = df[mask]
            n_obs   = len(subset)

            if n_obs < MIN_OBS_SEASON:
                draw_missing(ax)
            else:
                rose_data = compute_windrose(subset["wdir"].values,
                                             subset["wspd"].values)
                draw_windrose(ax, rose_data, n_obs)

            # Only show compass labels on edges to reduce clutter
            if row < n_rows - 1:
                ax.set_xticklabels([])   # hide S label except last row
            if col > 0:
                ax.set_yticklabels([])   # hide % labels except first col

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=SPEED_COLOURS[i], edgecolor="white",
                       linewidth=0.5, label=f"{SPEED_LABELS[i]}")
        for i in range(len(SPEED_LABELS))
    ]

    fig.suptitle("Seasonal Wind Rose by Station", fontsize=13,
                 fontweight="bold", y=1.01)
#   plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.02, hspace=0.15, wspace=0.25)
    fig.legend(
        handles=legend_patches,
        title="Wind Speed (m/s)",
        title_fontsize=9,
        loc="upper center",
        ncol=len(SPEED_LABELS),
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.00),
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
