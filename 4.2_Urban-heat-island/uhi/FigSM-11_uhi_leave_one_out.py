"""
Supplementary check for Figure 11: station sensitivity and uncertainty of the
urban-rural difference (Reviewer 1, major comment 3)
================================================================================
Two tests on the urban-mean departure from the rural baseline (the black dashed
line in Figure 11), for minimum temperature, mean temperature, and DTR:

1. Leave-one-out: each of the six group stations (urban VL, MS, XR; rural XW,
   SL, DN) is removed in turn and the urban-rural difference is recomputed.
   That gives the full set plus six variants.

2. Uncertainty: a bootstrap over years. Whole years are resampled with
   replacement (N_BOOT times) and the smoothed annual cycle is recomputed each
   time; the 2.5th and 97.5th percentiles give a 95% confidence interval.
   Resampling whole years keeps the day-to-day autocorrelation inside each
   block, so it does not inflate the confidence.

3. Elevation adjustment: both tests are run twice, without and with each
   station's temperature adjusted to sea level using a constant lapse rate
   (LAPSE_RATE, standard atmosphere 6.5 °C/km) and the Table 1 elevations.
   The adjustment is a constant per station, so it shifts each curve without
   changing its shape. It is applied to minimum and mean temperature only:
   it cancels in DTR (Tmax and Tmin shift equally). Night-time lapse rates are
   often weaker than the standard value, so for Tmin the adjusted result is
   best read as a lower bound.

All calculations use concurrent observations: only the days on which all eight
MIA stations report the variable (the same dates as the matched-dates SM figure).
The dates are NOT re-matched when a station is dropped, so the variants differ
only in which stations are in the groups, not in which days are sampled.
Every variant uses the same bootstrap resamples (the RNG is re-seeded each
time), so the variants and the two adjustment modes are directly comparable.
With matched dates, the urban-mean departure equals the smoothed day-of-year
climatology of the daily (urban mean minus rural mean) series, as in Figure 11.

Usage:
    python3 FigSM-11_uhi_leave_one_out.py

Output ({workdir}/analysis/uhi/):
    uhi_leave_one_out.csv                 one row per adjustment, variable, and
                                          variant: annual mean, seasonal minimum
                                          and maximum of the smoothed cycle, each
                                          with its 95% bootstrap CI
    figure_uhi_leave_one_out.png          unadjusted: Tmin, mean T, and DTR
    figure_uhi_leave_one_out_elevadj.png  elevation-adjusted: Tmin and mean T
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

# (varname, csv_col, short_title, unit) - same as Figure 11
PANEL_VARS = [
    ("Min_Temp",    "Min Temperature", "Min Temperature",  "°C"),
    ("Temperature", "Temperature",     "Mean Temperature", "°C"),
    ("DTR",         None,              "Diurnal Temp Range", "°C"),
]

STATION_FILENAMES = {
    "GD":"Giordan", "XR":"Xagħra", "XW":"Xewkija", "SL":"Selmun", "DN":"Dingli",
    "MS":"Birkirkara", "VL":"Valletta", "LQ":"Luqa", "BN":"Bengħajsa",
}
URBAN_STATIONS = ["VL", "MS", "XR"]
RURAL_STATIONS = ["XW", "SL", "DN"]
# Dates are matched across all MIA stations, as in the matched-dates SM figure
MATCH_STATIONS = URBAN_STATIONS + RURAL_STATIONS + ["LQ", "BN"]
N_PERIOD_DAYS  = len(pd.date_range("2007-01-01", "2025-12-31"))

N_BOOT = 1000          # bootstrap replicates
SEED   = 42            # fixed seed, so reruns give the same CI

# Elevation adjustment (Table 1 elevations, m above sea level)
ELEVATIONS = {"GD":167, "XR":155, "XW":104, "SL":129, "DN":250,
              "MS":66,  "VL":31,  "LQ":85,  "BN":36}
LAPSE_RATE = 0.0065    # °C per m (standard atmosphere, 6.5 °C/km)
ADJUSTABLE = {"Min_Temp", "Temperature"}   # cancels in DTR
# (mode, figure file name): unadjusted first, then elevation-adjusted
MODES = [("none",  "figure_uhi_leave_one_out.png"),
         ("lapse", "figure_uhi_leave_one_out_elevadj.png")]

# Leave-one-out curve styles: dropped urban stations solid, rural dashed
DROP_COLOURS = {"VL":"#e31a1c", "MS":"#33a02c", "XR":"#377eb8",
                "XW":"#a65628", "SL":"#e7298a", "DN":"#e6ab02"}

DATE_COL      = "Date"
MISSING_VAL   = "-"
SMOOTH_WINDOW = 31
OUTPUT_DIR    = f"{workdir}/analysis/uhi"

# ---------------------------------------------------------------------------
# Loading and climatology helpers - same as Figure 11
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

# ---------------------------------------------------------------------------
# Urban-rural difference on matched dates
# ---------------------------------------------------------------------------

def matched_table(varname, col_name, daily_data):
    """Day x station table of the variable, on days all MIA stations report."""
    cols = {}
    for a in MATCH_STATIONS:
        if a in daily_data:
            s = get_daily_series(daily_data[a], varname, col_name)
            if s is not None:
                cols[a] = s.dropna()
    table = pd.concat(cols, axis=1, join="inner")
    print(f"  [match] {varname:12s} {len(table)} days common to {table.shape[1]} "
          f"MIA stations ({100 * len(table) / N_PERIOD_DAYS:.1f}% of 2007-2025)")
    return table

def smoothed_cycle(values, doy):
    """Smoothed day-of-year climatology (days 1-366) of a daily series."""
    sums   = np.bincount(doy, weights=values, minlength=367)[1:]
    counts = np.bincount(doy, minlength=367)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(counts > 0, sums / counts, np.nan)
    return smooth(pd.Series(clim, index=range(1, 367)), SMOOTH_WINDOW).values

def cycle_stats(cycle):
    return np.nanmean(cycle), np.nanmin(cycle), np.nanmax(cycle)

def elevation_adjust(table):
    """Adjust each station's temperature to sea level with a constant lapse rate."""
    offsets = pd.Series({a: LAPSE_RATE * ELEVATIONS[a] for a in table.columns})
    return table + offsets

def analyse_variant(table, urban, rural):
    """Smoothed cycle, its stats, and bootstrap CIs for one urban/rural grouping."""
    rng  = np.random.default_rng(SEED)   # same resamples for every variant
    d    = (table[urban].mean(axis=1) - table[rural].mean(axis=1)).values
    doy  = table.index.dayofyear.values
    year = table.index.year.values

    cycle = smoothed_cycle(d, doy)
    est   = cycle_stats(cycle)

    years    = np.unique(year)
    year_idx = {y: np.where(year == y)[0] for y in years}
    boot_cycles, boot_stats = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(years, size=len(years), replace=True)
        idx  = np.concatenate([year_idx[y] for y in pick])
        c    = smoothed_cycle(d[idx], doy[idx])
        boot_cycles.append(c)
        boot_stats.append(cycle_stats(c))
    boot_cycles = np.array(boot_cycles)
    boot_stats  = np.array(boot_stats)
    band = (np.nanpercentile(boot_cycles, 2.5, axis=0),
            np.nanpercentile(boot_cycles, 97.5, axis=0))
    ci = [(np.nanpercentile(boot_stats[:, k], 2.5),
           np.nanpercentile(boot_stats[:, k], 97.5)) for k in range(3)]
    return cycle, est, ci, band, len(years)

def variants():
    """Full set first, then each group station dropped in turn."""
    out = [("all stations", None, URBAN_STATIONS, RURAL_STATIONS)]
    for a in URBAN_STATIONS:
        out.append((f"without {a}", a, [u for u in URBAN_STATIONS if u != a], RURAL_STATIONS))
    for a in RURAL_STATIONS:
        out.append((f"without {a}", a, URBAN_STATIONS, [r for r in RURAL_STATIONS if r != a]))
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[load] Reading daily data files ...")
    daily_data = {}
    for a in MATCH_STATIONS:
        df = load_daily(a)
        if df is not None:
            daily_data[a] = df

    rows = []
    tables = {v[0]: matched_table(v[0], v[1], daily_data) for v in PANEL_VARS}
    for mode, fig_name in MODES:
        print(f"\n[mode] elevation adjustment: {mode}")
        panel_vars = [v for v in PANEL_VARS if mode == "none" or v[0] in ADJUSTABLE]
        rows += draw_mode(mode, fig_name, panel_vars, tables)

    csv_path = os.path.join(OUTPUT_DIR, "uhi_leave_one_out.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.3f")
    print(f"[saved] {csv_path}")

def draw_mode(mode, fig_name, panel_vars, tables):
    rows = []
    n = len(panel_vars)
    fig, axes = plt.subplots(n, 1, figsize=(11.0, 4.0 * n), squeeze=False)
    letters = list(string.ascii_lowercase)
    x = np.arange(1, 367)
    tick_pos, tick_labels = month_tick_positions()

    for idx, (varname, col_name, title, unit) in enumerate(panel_vars):
        table = tables[varname]
        if mode == "lapse":
            table = elevation_adjust(table)
        ax = axes[idx][0]

        for label, dropped, urban, rural in variants():
            cycle, est, ci, band, n_years = analyse_variant(table, urban, rural)
            rows.append({
                "elevation_adjustment": mode,
                "variable": varname, "variant": label,
                "urban": " ".join(urban), "rural": " ".join(rural),
                "matched_days": len(table), "years": n_years,
                "annual_mean": est[0], "annual_ci_low": ci[0][0], "annual_ci_high": ci[0][1],
                "seasonal_min": est[1], "min_ci_low": ci[1][0], "min_ci_high": ci[1][1],
                "seasonal_max": est[2], "max_ci_low": ci[2][0], "max_ci_high": ci[2][1],
            })
            print(f"  [{varname}] {label:13s} annual {est[0]:+.2f} "
                  f"({ci[0][0]:+.2f} to {ci[0][1]:+.2f}) | "
                  f"seasonal {est[1]:+.2f} to {est[2]:+.2f} °C")

            if dropped is None:
                ax.fill_between(x, band[0], band[1], color="#cccccc", alpha=0.6, zorder=1)
                ax.plot(x, cycle, color="#000000", linewidth=2.5, linestyle="--", zorder=5)
            else:
                style = "-" if dropped in URBAN_STATIONS else (0, (4, 2))
                ax.plot(x, cycle, color=DROP_COLOURS[dropped], linewidth=1.2,
                        linestyle=style, alpha=0.9, zorder=3)

        ax.axhline(0, color="#555555", linewidth=0.8, linestyle=":", zorder=2)
        ax.set_xlim(1, 366)
        ax.set_xticks(tick_pos)
        if idx == n - 1:
            ax.set_xticklabels(tick_labels, fontsize=10)
            ax.set_xlabel("months", fontsize=11)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(f"Urban−rural difference in {title} (Δ{unit})", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.grid(axis="x", linestyle=":", alpha=0.25)
        suffix = (f", adjusted to sea level ({LAPSE_RATE * 1000:.1f} °C/km)"
                  if mode == "lapse" else "")
        ax.set_title(f"({letters[idx]}) {title}{suffix}", fontsize=11,
                     fontweight="bold", loc="left")

    handles = [mlines.Line2D([], [], color="#000000", linewidth=2.5, linestyle="--",
                             label="all stations"),
               mpatches.Patch(facecolor="#cccccc", edgecolor="none", alpha=0.6,
                              label="95% CI (bootstrap by year)")]
    for a in URBAN_STATIONS + RURAL_STATIONS:
        style = "-" if a in URBAN_STATIONS else (0, (4, 2))
        handles.append(mlines.Line2D([], [], color=DROP_COLOURS[a], linewidth=1.2,
                                     linestyle=style, label=f"without {a}"))
    handles.append(mlines.Line2D([], [], color="#555555", linewidth=0.8,
                                 linestyle=":", label="rural baseline"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=9, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 1 - 0.15 / n])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUTPUT_DIR, fig_name)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {fig_path}")
    return rows

if __name__ == "__main__":
    main()
