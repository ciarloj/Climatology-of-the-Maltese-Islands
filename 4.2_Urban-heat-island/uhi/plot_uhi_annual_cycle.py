"""
Urban Heat Island — Annual Cycle of Intensity 
=====================================================================
Plots the daily UHI intensity as an annual cycle:
  - Thin coloured lines: each urban station minus the rural mean baseline
  - Thick dashed black line: mean of all urban stations minus rural mean baseline
  - Grey shading: ±1 standard deviation across urban stations
  - Horizontal reference line at zero

UHI intensity is defined here as the Canopy Layer Urban Heat Island (CUHI),
computed as the air temperature difference between urban and rural stations.
Method follows:
  Stewart, I.D. (2011). A systematic review and scientific critique of
  methodology in modern urban heat island literature.
  International Journal of Climatology, 31(2), 200-217.
  DOI: 10.1002/joc.2141

The rural mean is treated as the common baseline across all urban comparisons.
This approach is discussed in:
  Lowry, W.P. (1977). Empirical estimation of urban effects on climate:
  A problem analysis. Journal of Applied Meteorology, 16(2), 129-135.
  DOI: 10.1175/1520-0450(1977)016<0129:EEOUEO>2.0.CO;2

Caution: on a small island like Malta, urban stations tend to also be closer
to the coast, which can suppress Tmin independently of urbanisation. The UHI
signal here should be interpreted as an upper bound — the regression residual
approach (plot_uhi_regression.py) partially controls for this confounding.
See:
  Pérez-Zanón, N. et al. (2015). On the definition of urban heat island
  intensity: the "rural" reference.
  Frontiers in Earth Science, 3, 24.
  DOI: 10.3389/feart.2015.00024

Baseline membership: two stations are deliberately excluded from BOTH the
urban and rural baselines because their land use does not cleanly fit either
class — Luqa (international airport, LCZ 8, open low-rise) and Bengħajsa
(LCZ 9/10, adjacent to the Marsaxlokk Freeport). They are still plotted, as
dashed "monitored" lines, so their behaviour can be compared, but they do not
enter the urban ensemble mean or the rural reference.

Variable roles: not every variable isolates the UHI equally. Each is tagged
with a role (see VARIABLES):
  primary   - minimum temperature: canonical nocturnal UHI signal.
  secondary - mean temperature: contains the UHI but blended with the daytime
              signal; supporting evidence only.
  coastal   - maximum temperature, DTR and relative humidity: dominated by
              coastal proximity / sea breeze rather than urban fabric on a
              small island. Plotted for completeness but flagged on the figure
              and NOT to be explored as UHI in the paper.

Usage:
    python3 plot_uhi_annual_cycle.py

Output: ../../analysis/uhi/uhi_annual_cycle_{varname}.png
Input:  ../../Data/Daily_{station}.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Station configuration
# ---------------------------------------------------------------------------

workdir = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate/"

ALL_STATIONS = [
    ("Giordan",     "GD"),
    ("Xagħra",      "XR"),
    ("Xewkija",     "XW"),
    ("Selmun",      "SL"),
    ("Dingli",      "DN"),
    ("Msida",       "MS"),
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
    "MS":  "Birkirkara",
    "VL":  "Valletta",
    "LQ":  "Luqa",
    "BN":  "Bengħajsa",
}

# --- Baseline membership (land-use based; edit these sets to reconfigure) ---
# URBAN and RURAL define the two baselines from which UHI intensity is built.
# MONITORED stations are plotted for reference but excluded from BOTH baselines
# and from the urban ensemble statistics (see docstring).
URBAN_STATIONS    = {"VL", "MS", "XR"}          # compact urban core (LCZ 2)
RURAL_STATIONS    = {"XW", "SL", "DN"}          # sheltered rural / sparsely built (LCZ 9)
MONITORED_STATIONS = {"LQ", "BN", "GD"}         # plotted but excluded from both baselines

# NOTE on Giordan (GD): it is an exposed, coastal headland site whose minima are
# governed by sea proximity rather than the sheltered inland-rural conditions the
# baseline is meant to represent, and it is the noisiest/most anomalous station in
# the network (see Section 3.4). It is therefore kept OUT of the rural baseline and
# plotted only as a monitored line. To restore it to the baseline (the mean UHI is
# largely insensitive to the choice), move "GD" back into RURAL_STATIONS.

# Colours for individual urban baseline station lines
URBAN_COLOURS = {
    "VL": "#e31a1c",   # red
    "MS": "#33a02c",   # green
    "XR": "#377eb8",   # blue
}

# Colours for monitored (non-baseline) station lines — drawn dashed
MONITORED_COLOURS = {
    "LQ": "#6a3d9a",   # violet
    "BN": "#ff7f00",   # orange
    "GD": "#777777",   # grey
}

# ---------------------------------------------------------------------------
# Variables to process
# Each entry: (varname, column_name_in_csv, y_axis_label, unit)
# ---------------------------------------------------------------------------

# Each entry: (varname, column_name_in_csv, y_axis_label, unit, uhi_role, note)
# uhi_role: "primary" | "secondary" | "coastal"  (see docstring)
VARIABLES = [
    ("Wind_Speed",    "Wind Speed",    "Wind Speed",    "m/s", "coastal", "Not urban-driven — near-null here is the result."),
    ("Precipitation", "Precipitation", "Precipitation", "mm",  "coastal", "Not urban-driven — shown for the matched comparison."),
    ("Min_Temp",          "Min Temperature",   "Min Temperature",    "°C", "primary",
     "Canonical nocturnal UHI signal — primary indicator."),
    ("Temperature",       "Temperature",       "Mean Temperature",   "°C", "secondary",
     "Contains the UHI but blends day and night — supporting only."),
    ("Max_Temp",          "Max Temperature",   "Max Temperature",    "°C", "coastal",
     "Daytime sea-breeze / exposure dominates, not urban fabric."),
    ("DTR",               None,                "Diurnal Temp Range", "°C", "coastal",
     "Coastal / continentality dominates on Malta — not a clean UHI metric."),
    ("Relative_Humidity", "Relative Humidity", "Relative Humidity",  "%",  "coastal",
     "Coastal moisture and T-dependence dominate — weak UHI signal."),
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATE_COL    = "Date"
MISSING_VAL = "-"
SMOOTH_WINDOW = 31   # days — centred rolling mean (matches other scripts)
OUTPUT_DIR  = f"{workdir}/analysis/uhi"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_daily(abbr):
    filename = STATION_FILENAMES.get(abbr)
    if filename is None:
        return None
    filepath = f"{workdir}/Data/Daily_{filename}.csv"
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, header=0)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
        df = df[df[DATE_COL] >= pd.Timestamp("2007-01-01")]
        df = df.dropna(subset=[DATE_COL]).set_index(DATE_COL)
        return df
    except Exception as e:
        print(f"  [warn] {abbr}: {e}")
        return None


def find_col(df, target):
    """Find column by name with normalised fallback."""
    if target is None:
        return None
    if target in df.columns:
        return target
    norm = lambda s: s.lower().replace(" ", "_")
    matches = [c for c in df.columns if norm(c) == norm(target)]
    return matches[0] if matches else None


def get_daily_series(df, varname, col_name):
    """Return a daily numeric Series for the given variable."""
    if varname == "DTR":
        col_tmax = find_col(df, "Max Temperature")
        col_tmin = find_col(df, "Min Temperature")
        if col_tmax is None or col_tmin is None:
            return None
        tmax = pd.to_numeric(df[col_tmax].replace(MISSING_VAL, np.nan), errors="coerce")
        tmin = pd.to_numeric(df[col_tmin].replace(MISSING_VAL, np.nan), errors="coerce")
        return (tmax - tmin)
    else:
        col = find_col(df, col_name)
        if col is None:
            return None
        return pd.to_numeric(df[col].replace(MISSING_VAL, np.nan), errors="coerce")


def doy_climatology(series):
    """
    Group by day-of-year, compute mean across all years.
    Returns Series indexed 1-366.
    """
    doy  = series.index.dayofyear
    clim = series.groupby(doy).mean()
    return clim.reindex(range(1, 367))


def smooth(series, window):
    """Centred rolling mean with wrap-around padding."""
    if window <= 1:
        return series
    vals = series.values
    pad  = window // 2
    padded   = np.concatenate([vals[-pad:], vals, vals[:pad]])
    smoothed = pd.Series(padded).rolling(window, center=True,
                                          min_periods=1).mean().values
    return pd.Series(smoothed[pad:-pad], index=series.index)


def month_tick_positions():
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    mids   = [16, 47, 75, 106, 136, 167, 197, 228, 258, 289, 319, 350]
    return mids, months


# ---------------------------------------------------------------------------
# Main plotting
# ---------------------------------------------------------------------------

def create_uhi_plot(varname, col_name, ylabel, unit, uhi_role, note,
                    urban_clims, rural_mean_clim, monitored_clims, output_dir):
    """
    urban_clims     : dict {abbr: smoothed climatology (doy 1-366)} — urban baseline
    rural_mean_clim : smoothed climatology of the rural mean (doy 1-366)
    monitored_clims : dict {abbr: smoothed climatology} — plotted but NOT baselined
    uhi_role, note  : classification of the variable for UHI interpretation
    """
    if not urban_clims or rural_mean_clim is None:
        print(f"  [skip] Insufficient data for {varname}")
        return

    x = np.arange(1, 367)
    tick_pos, tick_labels = month_tick_positions()

    # Compute delta = each urban station minus rural mean (urban ensemble only)
    deltas = {}
    for abbr, clim in urban_clims.items():
        delta = clim.reindex(x) - rural_mean_clim.reindex(x)
        deltas[abbr] = delta

    # Monitored (non-baseline) deltas — same reference, but kept separate
    monitored_deltas = {}
    for abbr, clim in (monitored_clims or {}).items():
        monitored_deltas[abbr] = clim.reindex(x) - rural_mean_clim.reindex(x)

    # Urban ensemble mean and std — computed from the urban baseline ONLY
    delta_df  = pd.DataFrame(deltas)
    mean_delta = delta_df.mean(axis=1)
    std_delta  = delta_df.std(axis=1)

    fig, ax = plt.subplots(figsize=(11, 6))

    # ±1 SD shading
    ax.fill_between(x,
                    mean_delta.values - std_delta.values,
                    mean_delta.values + std_delta.values,
                    color="#cccccc", alpha=0.5, zorder=1,
                    label="Urban spread (±1 SD)")

    # Individual urban baseline station lines (solid)
    for abbr, delta in deltas.items():
        colour = URBAN_COLOURS.get(abbr, "#888888")
        ax.plot(x, delta.values, color=colour, linewidth=1.0,
                alpha=0.8, zorder=3)

    # Monitored station lines (dashed) — excluded from baseline/ensemble
    for abbr, delta in monitored_deltas.items():
        colour = MONITORED_COLOURS.get(abbr, "#777777")
        ax.plot(x, delta.values, color=colour, linewidth=1.4,
                linestyle=(0, (4, 2)), alpha=0.9, zorder=4)

    # Urban ensemble mean
    ax.plot(x, mean_delta.values,
            color="#000000", linewidth=2.5, linestyle="--",
            zorder=5, label="Urban mean")

    # Zero reference line
    ax.axhline(0, color="#555555", linewidth=0.8, linestyle=":",
               zorder=2, label="Rural baseline")

    # Axes formatting
    ax.set_xlim(1, 366)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("months", fontsize=11)
    ax.set_ylabel(f"Difference in {ylabel} (Δ{unit})", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    # Legend: urban baseline stations + monitored stations + ensemble mean + shading
    handles = []
    for abbr in sorted(deltas.keys()):
        colour = URBAN_COLOURS.get(abbr, "#888888")
        handles.append(
            mlines.Line2D([], [], color=colour, linewidth=1.2,
                          label=abbr)
        )
    for abbr in sorted(monitored_deltas.keys()):
        colour = MONITORED_COLOURS.get(abbr, "#777777")
        handles.append(
            mlines.Line2D([], [], color=colour, linewidth=1.4,
                          linestyle=(0, (4, 2)),
                          label=abbr)
        )
    handles.append(
        mlines.Line2D([], [], color="#000000", linewidth=2.5,
                      linestyle="--", label="Urban mean UHII")
    )
    handles.append(
        mpatches.Patch(facecolor="#cccccc", edgecolor="none",
                       alpha=0.5, label="±1 SD (urban spread)")
    )
    handles.append(
        mlines.Line2D([], [], color="#555555", linewidth=0.8,
                      linestyle=":", label="Rural baseline")
    )

    n_handles = len(handles)
    ncols     = min(n_handles, 6)
    nrows     = -(-n_handles // ncols)
    title_pad = 8 + nrows * 14

    ax.set_title(f"Urban vs Rural annual cycle: {ylabel}",
                 fontsize=13, fontweight="bold", pad=title_pad)

    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncols,
        fontsize=8,
        borderaxespad=0,
        frameon=True,
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    # Flag coastal (non-UHI) variables in the filename so they are easy to spot
    flag = "_FLAG-coastal" if uhi_role == "coastal" else ""
    out_path = os.path.join(output_dir, f"uhi_annual_cycle_{varname}{flag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}   [{uhi_role}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load all daily data
    print("[load] Reading daily data files ...")
    daily_data = {}
    for name, abbr in ALL_STATIONS:
        df = load_daily(abbr)
        if df is not None:
            daily_data[abbr] = df
            print(f"  {abbr:4s}  {len(df)} rows")

    # Report the station roles and variable roles once, up front
    print("\n[config] Urban baseline   :", sorted(URBAN_STATIONS))
    print("[config] Rural baseline   :", sorted(RURAL_STATIONS))
    print("[config] Monitored (excl.):", sorted(MONITORED_STATIONS))
    print("[config] Variable roles:")
    for v in VARIABLES:
        print(f"           {v[0]:20s} -> {v[4]}")

    for varname, col_name, ylabel, unit, uhi_role, note in VARIABLES:
        print(f"\n[var] {varname}  ({uhi_role})")

        # Compute climatologies for all stations
        clims = {}
        for name, abbr in ALL_STATIONS:
            if abbr not in daily_data:
                continue
            series = get_daily_series(daily_data[abbr], varname, col_name)
            if series is None:
                continue
            clim = doy_climatology(series.dropna())
            clim = smooth(clim, SMOOTH_WINDOW)
            clims[abbr] = clim
            print(f"  {abbr:4s}  valid doys: {clim.notna().sum()}")

        if not clims:
            print(f"  [skip] No data for {varname}")
            continue

        # Rural mean baseline (rural stations only; monitored excluded)
        rural_abbrs = [a for a in RURAL_STATIONS if a in clims]
        if not rural_abbrs:
            print(f"  [skip] No rural stations with data for {varname}")
            continue
        rural_mean_clim = pd.concat(
            [clims[a] for a in rural_abbrs], axis=1
        ).mean(axis=1)
        print(f"  Rural baseline from : {rural_abbrs}")

        # Urban baseline climatologies (urban stations only; monitored excluded)
        urban_clims = {a: clims[a] for a in URBAN_STATIONS if a in clims}
        if not urban_clims:
            print(f"  [skip] No urban stations with data for {varname}")
            continue

        # Monitored (non-baseline) climatologies — plotted for reference
        monitored_clims = {a: clims[a] for a in MONITORED_STATIONS if a in clims}
        if monitored_clims:
            print(f"  Monitored (plotted) : {list(monitored_clims)}")

        create_uhi_plot(varname, col_name, ylabel, unit, uhi_role, note,
                        urban_clims, rural_mean_clim, monitored_clims, OUTPUT_DIR)

    print(f"\n[done] All plots written to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
