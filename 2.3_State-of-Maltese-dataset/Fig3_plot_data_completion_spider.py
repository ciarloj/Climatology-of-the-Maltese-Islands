"""
Data Completion Spider Plot
=============================
Produces a single spider/radar plot summarising data completion across
all stations and variables. Each spoke is a station, each line is a variable.

Usage:
    python3 plot_data_completion_spider.py [mode]

Modes (same as plot_data_completion.py):
    as-is     - Plot %N directly from stats files
    base1997  - Normalise all stations to 1997-2025 baseline
    base2007  - Normalise Giordan station to 2007-2025 baseline (DEFAULT)

Input files:
    ../../analysis/basic-stats/stats_{varname}.csv  (one per variable)
    ../../Data/Daily_Giordan.csv (required for base2007 mode)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from datetime import date

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"

VARNAMES = [
    "Precipitation", "Pressure", "Relative_Humidity",
    "Solar", "Temperature", "Wind_Direction", "Wind_Gust", "Wind_Speed",
]

# Spider spokes — comment out stations to exclude them
_STATION_SEQ = [
    "Giordan",
    "Xagħra",
    "Xewkija",
    "Selmun",
    "Dingli",
    "Birkirkara",
    "Valletta",
    "Luqa",
    "Bengħajsa",
]

# Default mode
DEFAULT_MODE = "base2007"

# Variable colours
VAR_COLOURS = {
    "Max_Temperature":   "#e31a1c",
    "Min_Temperature":   "#fb9a99",
    "Precipitation":     "#1f78b4",
    "Pressure":          "#b15928",
    "Relative_Humidity": "#a6cee3",
    "Solar":             "#ff7f00",
    "Temperature":       "#6a3d9a",
    "Wind_Direction":    "#33a02c",
    "Wind_Gust":         "#cab2d6",
    "Wind_Speed":        "#b2df8a",
}

# Human-readable variable labels
VARLABELS = {
    "Max_Temperature":   "Max Temperature",
    "Min_Temperature":   "Min Temperature",
    "Relative_Humidity": "Relative Humidity",
    "Solar":             "Solar Radiation",
    "Wind_Direction":    "Wind Direction",
    "Wind_Gust":         "Wind Gust",
    "Wind_Speed":        "Wind Speed",
}

GIORDAN_STATION = "Giordan"

# ---------------------------------------------------------------------------
# Pre-computed day counts
# ---------------------------------------------------------------------------

def _days_inclusive(start: date, end: date) -> int:
    return (end - start).days + 1

N0          = _days_inclusive(date(1997, 1, 1),  date(2006, 12, 31))
N_2007_2025 = _days_inclusive(date(2007, 1, 1),  date(2025, 12, 31))
N_2023_2025 = _days_inclusive(date(2023, 1, 1),  date(2025, 12, 31))

# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------

def load_stats(varname: str) -> pd.DataFrame:
    filepath = f"{workdir}/analysis/basic-stats/stats_{varname}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Expected file not found: {filepath}")
    return pd.read_csv(filepath, header=0, index_col=0)


def get_stat_row(df, row_label):
    if row_label not in df.index:
        raise KeyError(f"Row label '{row_label}' not found.")
    return df.loc[row_label]


def pct_n_as_is(df, station):
    val = get_stat_row(df, "%N")[station]
    if pd.isna(val) or str(val).strip() == "":
        return 0.0
    return float(val)


def pct_n_base1997(df, station):
    if station == GIORDAN_STATION:
        return pct_n_as_is(df, station)
    pct_n_val = get_stat_row(df, "%N")[station]
    n_val     = get_stat_row(df, "n")[station]
    if pd.isna(pct_n_val) or pd.isna(n_val):
        return 0.0
    pct_n  = float(pct_n_val)
    n      = float(n_val)
    n_full = n * pct_n / 100.0
    n1     = N0 + n
    return n_full * 100.0 / n1


def pct_n_base2007(df, station, varname):
    if station != GIORDAN_STATION:
        return pct_n_as_is(df, station)

    giordan_path = os.path.join(workdir, "Data", "Daily_Giordan.csv")
    if not os.path.exists(giordan_path):
        raise FileNotFoundError(f"Giordan file not found at: {giordan_path}")

    raw = pd.read_csv(giordan_path, header=0)
    date_col = raw.columns[7] if raw.columns[7] == "Date" else "Date"
    if date_col not in raw.columns:
        raise ValueError("Date column not found in Giordan file.")

    var_cols = list(raw.columns[:7])
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    start      = pd.Timestamp("2007-01-01")
    end_window = pd.Timestamp("2022-12-31")
    window = raw[(raw[date_col] >= start) & (raw[date_col] <= end_window)]
    n      = len(window)

    if varname not in var_cols:
        match = [c for c in var_cols
                 if c.lower().replace(" ", "_") == varname.lower().replace(" ", "_")]
        if not match:
            raise KeyError(f"Variable '{varname}' not in {var_cols}.")
        var_col = match[0]
    else:
        var_col = varname

    valid_mask = (window[var_col] != "-") & (window[var_col].notna())
    n_full     = int(valid_mask.sum())
    n_total    = n + N_2023_2025
    return (n_full * 100.0 / n_total) if n_total > 0 else 0.0


def compute_pct(df, station, varname, mode):
    if mode == "as-is":
        return pct_n_as_is(df, station)
    elif mode == "base1997":
        return pct_n_base1997(df, station)
    elif mode == "base2007":
        return pct_n_base2007(df, station, varname)
    return 0.0


# ---------------------------------------------------------------------------
# Spider plotting
# ---------------------------------------------------------------------------

def create_spider_plot(data, stations, mode, output_dir):
    n_spokes = len(stations)
    if n_spokes < 3:
        print("[error] Need at least 3 stations for a spider plot.")
        return

    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    for varname in VARNAMES:
        if varname not in data:
            continue
        values = data[varname]
        values_closed = values + [values[0]]
        colour = VAR_COLOURS.get(varname, "#888888")

        thick_vars = {"Precipitation", "Pressure"}
        lw = 1.6 if varname in thick_vars else 1.6
        ax.plot(angles_closed, values_closed,
                color=colour, linewidth=lw, alpha=0.85,
                marker="o", markersize=6, zorder=3)
       # ax.fill(angles_closed, values_closed,
       #         color=colour, alpha=0.06, zorder=2)

    # Spokes — station names
    ax.set_xticks(angles)
    #DISPLAY_NAME_OVERRIDES = {"Birkirkara": "Msida"}
    DISPLAY_NAME_OVERRIDES = {
        "Giordan": "GD", "Xagħra": "XR", "Xewkija": "XW", "Selmun": "SL",
        "Dingli": "DN", "Birkirkara": "MS", "Valletta": "VL", "Luqa": "LQ",
        "Bengħajsa": "BN",
    }
    display_labels = [DISPLAY_NAME_OVERRIDES.get(s, s).replace("-", " ") for s in stations]
    ax.set_xticklabels(display_labels, fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=8)

    # Radial range and ticks
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"],
                       fontsize=7, color="#666666")
    ax.set_rlabel_position(180 / n_spokes)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.grid(True, linewidth=0.5, color="grey", alpha=0.5, linestyle="--")
    ax.spines["polar"].set_color("#999999")
    ax.spines["polar"].set_linewidth(0.8)

    # Title
    if mode == "as-is":
        modelab = "all data"
    elif mode == "base1997":
        modelab = "1997-2025 baseline"
    elif mode == "base2007":
        modelab = "2007-2025 baseline"
    else:
        modelab = mode

#    ax.set_title(f"Data Completion Summary — {modelab}",
#                 fontsize=13, fontweight="bold", pad=24)

    # Legend
    handles = []
    for varname in VARNAMES:
        if varname not in data:
            continue
        colour = VAR_COLOURS.get(varname, "#888888")
        label  = VARLABELS.get(varname, varname)
        handles.append(
            mlines.Line2D([], [], color=colour, linewidth=1.6,
                          marker="o", markersize=6, label=label)
        )

    leg = fig.legend(handles=handles,
               loc="center left",
               bbox_to_anchor=(0.85, 0.85),
               ncol=1,
               fontsize=9,
               frameon=True,
               framealpha=0.9,
               title_fontsize=10)

    os.makedirs(output_dir, exist_ok=True)
    fname    = f"Fig3_spider_data-completion_{mode}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(mode, output_dir=f"{workdir}/analysis/basic-stats"):
    valid_modes = ("as-is", "base2007", "base1997")
    if mode not in valid_modes:
        sys.exit(f"Invalid mode '{mode}'. Choose from: {valid_modes}")

    print(f"[run] Mode = {mode}")
    print(f"[stations] {_STATION_SEQ}")

    data = {}

    for varname in VARNAMES:
        print(f"\n[var] {varname}")
        try:
            df = load_stats(varname)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        pcts = []
        for station in _STATION_SEQ:
            if station not in df.columns:
                print(f"  [warn] {station} not in stats file — using 0")
                pcts.append(0.0)
                continue
            try:
                pct = compute_pct(df, station, varname, mode)
                pct = max(0.0, min(100.0, pct))
                print(f"  {station:15s}  %N = {pct:.2f}%")
                pcts.append(pct)
            except (KeyError, ValueError, FileNotFoundError) as e:
                print(f"  [warn] {station}: {e} – using 0")
                pcts.append(0.0)

        data[varname] = pcts

    create_spider_plot(data, _STATION_SEQ, mode, output_dir)
    print(f"\n[done]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE
    main(run_mode)
