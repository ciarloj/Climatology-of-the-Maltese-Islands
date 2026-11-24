"""
Standard Normal Homogeneity Test (SNHT) - Relative variant, MERRA-2 reference
=============================================================================
Independent check of the homogenized station series (Reviewer 1, major
comment 1; Reviewer 2, point 9). Identical to snht_TS_ERA5.py except that:

  - The reference is MERRA-2 (NASA GMAO), an independent reanalysis, at the
    grid point nearest to the ERA5 extraction point (36.0 N, 14.375 E).
    MERRA-2 variables are converted exactly as the ERA5 ones:
        Temperature        <- T2M (monthly mean, K -> deg C)
        Max / Min Temp     <- daily T2MMAX / T2MMIN, averaged to monthly
        Pressure           <- SLP (Pa -> hPa)
        Wind Speed         <- sqrt(U10M^2 + V10M^2) from monthly means
        Relative Humidity  <- Magnus formula on monthly T2M and T2MDEW
        Solar              <- SWGDN (W m-2)
  - Critical values are the 5 % values of Khaliq and Ouarda (2007), Table 2.
    (The table in snht_TS_ERA5.py is close to the 1 % level.)
  - CV is reported to 2 decimals, since some T0 values sit close to it.
  - A comparison table joins these results with the ERA5 results, with the
    ERA5 significance re-scored against the same 5 % critical values.

Outputs go to analysis/SNHT/figures_merra2/.
"""
# %%
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import netCDF4 as nc4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "../..", "Data")
MERRA2_DIR = os.path.join(os.path.dirname(__file__), "../../../../..", "Data", "MERRA2")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "../../analysis/SNHT", "figures_merra2")
# ERA5 results to compare against (output of snht_TS_ERA5.py)
ERA5_RESULTS = os.path.join(os.path.dirname(__file__), "../../analysis/SNHT",
                            "figures_era5", "snht_era5_results.csv")

# Merged MERRA-2 files (from merge_merra2.py), matched by pattern
MERRA2_FILES = {
    "daily_stats": "M2SDNXSLV*.nc",   # daily T2MMAX, T2MMIN
    "monthly_slv": "M2TMNXSLV*.nc",   # monthly T2M, T2MDEW, SLP, U10M, V10M
    "monthly_rad": "M2TMNXRAD*.nc",   # monthly SWGDN
}
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Extraction point: same target as ERA5; nearest MERRA-2 point is 36.0 N, 14.375 E
# ---------------------------------------------------------------------------
MALTA_LAT = 35.9
MALTA_LON = 14.5

# ---------------------------------------------------------------------------
# Common analysis period (applied to both station and MERRA-2 data)
# ---------------------------------------------------------------------------
PERIOD_START = pd.Period("2007-01", freq="M")
PERIOD_END   = pd.Period("2025-12", freq="M")

# ---------------------------------------------------------------------------
# Station filtering and renaming
# ---------------------------------------------------------------------------
# Stations removed from analysis (short records / data-quality concerns)
EXCLUDE_STATIONS = {"San-Lawrenz", "Zokrija"}

# Rename applied after loading (old name -> display name)
STATION_RENAME = {"Birkirkara": "Msida"}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
# Variables to test and how to aggregate daily -> monthly
#   "sum"  for precipitation totals
#   "mean" for everything else
VARIABLES = {
    "Temperature":       "mean",
    "Max Temperature":   "mean",
    "Min Temperature":   "mean",
    "Relative Humidity": "mean",
    "Solar":             "mean",
    "Pressure":          "mean",
    "Wind Speed":        "mean",
}

# Columns to exclude entirely (circular, derived, or unsuitable for SNHT)
SKIP_COLS = {"Wind Direction", "Wind Gust", "Date"}

# Minimum number of monthly anomaly values required to run the test
MIN_OBS = 36   # 3 years

# ---------------------------------------------------------------------------
# Critical values at 5 % significance (Khaliq and Ouarda, 2007, Table 2)
# ---------------------------------------------------------------------------
CV_TABLE = {
    10:  5.637,
    20:  7.089,
    30:  7.747,
    40:  8.151,
    50:  8.432,
    60:  8.647,
    70:  8.814,
    80:  8.951,
    90:  9.067,
    100: 9.167,
    125: 9.365,
    150: 9.519,
    175: 9.641,
    200: 9.741,
    250: 9.898,
    300: 10.020,
}
CV_NS   = np.array(sorted(CV_TABLE.keys()), dtype=float)
CV_VALS = np.array([CV_TABLE[n] for n in CV_NS.astype(int)], dtype=float)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def critical_value(n: int) -> float:
    """Linearly interpolate the 5 % critical value for sample size n."""
    if n <= CV_NS[0]:
        return CV_VALS[0]
    if n >= CV_NS[-1]:
        return CV_VALS[-1]
    return float(np.interp(n, CV_NS, CV_VALS))


def snht(series: np.ndarray):
    """
    Compute the SNHT statistic T(k) for a 1-D difference/anomaly series.

    Parameters
    ----------
    series : 1-D array of floats (monthly difference values)

    Returns
    -------
    T  : array of T(k), length n-1
    k0 : 1-indexed position of the detected break (argmax of T)
    T0 : maximum test statistic
    """
    n   = len(series)
    sig = series.std(ddof=1)
    if sig == 0:
        return np.zeros(n - 1), 1, 0.0

    z = (series - series.mean()) / sig
    T = np.array([
        k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2
        for k in range(1, n)
    ])
    k0 = int(np.argmax(T)) + 1   # 1-indexed
    return T, k0, float(T.max())


def load_all_stations() -> dict:
    """
    Read all Daily_*.csv files, apply exclusions and renames, and return
    monthly anomaly series.

    Daily values are aggregated to monthly means/sums, then the long-term
    monthly climatology (mean per calendar month) is subtracted to produce
    anomalies suitable for SNHT.

    Returns
    -------
    dict: {station_name: DataFrame(index=PeriodIndex[M], columns=variables)}
    """
    pattern = os.path.join(DATA_DIR, "Daily_*.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    stations = {}
    for fpath in files:
        raw_name = os.path.basename(fpath).replace("Daily_", "").replace(".csv", "")

        if raw_name in EXCLUDE_STATIONS:
            print(f"  Skipping excluded station: {raw_name}")
            continue

        name = STATION_RENAME.get(raw_name, raw_name)

        df = pd.read_csv(fpath, na_values=["-", "N/A", "n/a", "NA", ""])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Month"] = df["Date"].dt.to_period("M")

        # Select only variables we test (skip unsuitable columns)
        keep = [c for c in df.columns if c in VARIABLES and c not in SKIP_COLS]
        if not keep:
            continue
        df_sub = df[keep + ["Month"]]

        # Aggregate to monthly
        agg_funcs = {col: VARIABLES[col] for col in keep}
        monthly   = df_sub.groupby("Month").agg(agg_funcs)

        # Mask months with < 20 daily observations as NaN
        counts = df_sub.groupby("Month")[keep].count()
        for col in keep:
            monthly.loc[counts[col] < 20, col] = np.nan

        # Restrict to common analysis period
        monthly = monthly.loc[PERIOD_START:PERIOD_END]

        # Subtract long-term monthly climatology (mean per calendar month)
        monthly["_cal_month"] = monthly.index.month
        for col in keep:
            clim = monthly.groupby("_cal_month")[col].mean()
            monthly[col] = monthly[col] - monthly["_cal_month"].map(clim)
        monthly = monthly.drop(columns="_cal_month")

        stations[name] = monthly

    return stations


def _open_merra2(key):
    """Open one merged MERRA-2 file at the grid point nearest MALTA_LAT/LON."""
    import xarray as xr
    files = sorted(glob.glob(os.path.join(MERRA2_DIR, MERRA2_FILES[key])))
    if not files:
        raise FileNotFoundError(
            f"No MERRA-2 file matching {MERRA2_FILES[key]} in {MERRA2_DIR}")
    ds = xr.open_dataset(files[0]).sel(lat=MALTA_LAT, lon=MALTA_LON, method="nearest")
    return ds


def _monthly(da) -> pd.Series:
    """1-D xarray time series -> pandas Series of monthly means (PeriodIndex)."""
    s = da.to_series()
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    return s.groupby(level=0).mean()


def load_merra2_reference(raw: bool = False) -> pd.DataFrame:
    """
    Load MERRA-2 at the Malta grid point and return monthly anomaly series for
    each variable in VARIABLES, built exactly as the ERA5 reference.
    """
    slv = _open_merra2("monthly_slv")
    rad = _open_merra2("monthly_rad")
    day = _open_merra2("daily_stats")
    print(f"  MERRA-2 grid point used: {float(slv.lat):.3f} N, {float(slv.lon):.3f} E")

    t_c = _monthly(slv["T2M"]) - 273.15
    d_c = _monthly(slv["T2MDEW"]) - 273.15
    u   = _monthly(slv["U10M"])
    v   = _monthly(slv["V10M"])

    df = pd.DataFrame({
        "Temperature":       t_c,
        "Max Temperature":   _monthly(day["T2MMAX"]) - 273.15,
        "Min Temperature":   _monthly(day["T2MMIN"]) - 273.15,
        "Pressure":          _monthly(slv["SLP"]) / 100.0,
        "Wind Speed":        np.sqrt(u**2 + v**2),
        "Relative Humidity": 100.0 * (np.exp(17.625 * d_c / (243.04 + d_c)) /
                                      np.exp(17.625 * t_c / (243.04 + t_c))),
        "Solar":             _monthly(rad["SWGDN"]),
    })
    df.index = pd.PeriodIndex(df.index, freq="M")
    df = df.sort_index().loc[PERIOD_START:PERIOD_END]

    if not raw:
        # Subtract long-term monthly climatology to produce anomalies
        df["_cal_month"] = df.index.month
        for col in [c for c in VARIABLES if c in df.columns]:
            clim    = df.groupby("_cal_month")[col].mean()
            df[col] = df[col] - df["_cal_month"].map(clim)
        df = df.drop(columns="_cal_month")

    return df


def compare_with_era5(results: list) -> None:
    """
    Join MERRA-2 and ERA5 results per station and variable. ERA5 significance
    is re-scored against the same 5 % critical values used here.
    """
    if not os.path.exists(ERA5_RESULTS):
        print(f"ERA5 results not found ({ERA5_RESULTS}); comparison skipped.")
        return
    era = pd.read_csv(ERA5_RESULTS)
    era["CV (5%)"]     = era["n"].apply(lambda n: round(critical_value(int(n)), 2))
    era["Significant"] = np.where(era["T0"] > era["CV (5%)"], "YES", "no")
    mer = pd.DataFrame(results)
    cols = ["Station", "Variable", "n", "T0", "CV (5%)", "Break year", "Shift", "Significant"]
    m = era[[c for c in cols if c in era.columns]].merge(mer[cols], on=["Station", "Variable"], how="outer",
                        suffixes=(" ERA5", " MERRA-2"))

    def gap(a, b):
        try:
            pa, pb = pd.Period(a, freq="M"), pd.Period(b, freq="M")
            return abs((pa.year - pb.year) * 12 + (pa.month - pb.month))
        except Exception:
            return np.nan
    m["Break gap (months)"] = [gap(a, b) for a, b in
                               zip(m["Break year ERA5"], m["Break year MERRA-2"])]
    m["Both significant"] = np.where((m["Significant ERA5"] == "YES") &
                                     (m["Significant MERRA-2"] == "YES"), "YES", "no")
    m = m.sort_values(["Station", "Variable"])
    out = os.path.join(FIG_DIR, "snht_reference_comparison.csv")
    m.to_csv(out, index=False)

    flagged = m[(m["Significant ERA5"] == "YES") | (m["Significant MERRA-2"] == "YES")]
    print(f"\n{'=' * 72}\nSERIES SIGNIFICANT AGAINST EITHER REFERENCE (5 %)\n{'=' * 72}")
    print(flagged.to_string(index=False) if len(flagged) else "None.")
    print(f"\nComparison saved to: {out}")


# ---------------------------------------------------------------------------
# SNHT pass
# ---------------------------------------------------------------------------

def run_snht_pass(stations: dict, ref: pd.DataFrame,
                  save_figures: bool = False) -> list:
    """
    Run a single SNHT pass over all stations and variables using MERRA-2 as the
    background reference series.

    Parameters
    ----------
    stations     : candidate stations
    ref     : MERRA-2 anomaly DataFrame (PeriodIndex[M], one col per variable)
    save_figures : if True, save per-station SNHT figures for significant breaks

    Returns
    -------
    list of result dicts
    """
    results = []
    for station in sorted(stations.keys()):
        df_cand = stations[station]

        for variable in VARIABLES:
            if variable not in df_cand.columns:
                continue
            if variable not in ref.columns:
                continue

            cand_s = df_cand[variable].dropna()
            ref_s  = ref[variable].dropna()

            # Align on common months, drop any NaN
            common = cand_s.index.intersection(ref_s.index)
            if len(common) == 0:
                continue

            cand_a = cand_s.loc[common]
            ref_a  = ref_s.loc[common]
            valid  = cand_a.notna() & ref_a.notna()
            cand_c = cand_a[valid]
            ref_c  = ref_a[valid]

            if len(cand_c) < MIN_OBS:
                continue

            diff        = (cand_c - ref_c).values
            periods     = cand_c.index
            T, k0, T0   = snht(diff)
            cv          = critical_value(len(diff))
            significant = T0 > cv
            bp_idx      = k0 - 1 if k0 <= len(periods) else len(periods) - 1
            break_year  = str(periods[bp_idx])
            shift       = diff[k0:].mean() - diff[:k0].mean()   # post- minus pre-break

            results.append({
                "Station":     station,
                "Variable":    variable,
                "n":           len(diff),
                "T0":          round(T0, 2),
                "CV (5%)":     round(cv, 2),
                "Break year":  break_year,
                "Shift":       round(shift, 2),
                "Significant": "YES" if significant else "no",
            })

            if significant and save_figures:
                plot_snht(station, variable, cand_c, ref_c,
                          T, k0, T0, cv, periods, break_year)

    return results


# ---------------------------------------------------------------------------
# Per-station figure
# ---------------------------------------------------------------------------

def plot_snht(candidate, variable, cand_s, ref_s, T, k0, T0, cv, periods, break_year):
    """Save a 2-panel figure: monthly anomaly time series + T(k) statistic."""
    fig = plt.figure(figsize=(11, 7))
    gs  = gridspec.GridSpec(2, 1, hspace=0.42)

    ts_idx   = periods.to_timestamp()
    break_ts = pd.Period(break_year, freq="M").to_timestamp()

    # Top: monthly anomaly time series
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cand_s.index.to_timestamp(), cand_s.values,
             color="steelblue", lw=1.2, label=candidate)
    ax1.plot(ref_s.index.to_timestamp(), ref_s.values,
             color="darkorange", lw=1.2, ls="--", label="MERRA-2 reference")
    ax1.axvline(break_ts, color="crimson", lw=1.5, ls=":", label=f"Break {break_year}")
    ax1.set_title(f"{candidate}  \u2014  {variable}  (monthly anomaly)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"\u0394 {variable}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Bottom: T(k) statistic
    ax2 = fig.add_subplot(gs[1])
    t_ts = ts_idx[:-1]
    ax2.plot(t_ts, T, color="black", lw=1.5, label="T(k)")
    ax2.axhline(cv, color="crimson", lw=1.5, ls="--",
                label=f"Critical value (5 %) = {cv:.2f}")
    if k0 - 1 < len(t_ts):
        ax2.scatter([t_ts[k0 - 1]], [T0], color="crimson", zorder=5,
                    label=f"T\u2080 = {T0:.2f} at {break_year}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("T(k)")
    ax2.set_title("SNHT statistic", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    safe = lambda s: s.replace(" ", "_")
    out  = os.path.join(FIG_DIR, f"snht_merra2_{safe(candidate)}_{safe(variable)}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figure saved: {out}")


# ---------------------------------------------------------------------------
# All-stations overview figure
# ---------------------------------------------------------------------------

MAX_COLS = 3   # maximum number of variable columns per row group
PLOT_VARS = ["Temperature", "Relative Humidity", "Wind Speed"]


def plot_all_timeseries(stations: dict, results: list,
                        ref: pd.DataFrame) -> None:
    """
    One figure, grid of subplots.  Variables are wrapped into groups of
    MAX_COLS columns; each group stacks n_stations rows.  Layout:

        | var1 | var2 | var3 |
        | stn1 | stn1 | stn1 |
        | stn2 | stn2 | stn2 |
        ...
        | var4 | var5 | var6 |
        | stn1 | stn1 | stn1 |
        ...

    Each panel shows:
      - Candidate station monthly anomalies (blue, solid) — full record
      - MERRA-2 reference anomalies (orange, dashed)
      - Vertical red line  where T0 > CV  (significant break)
      - Vertical grey line where T0 <= CV (non-significant peak)
    """
    station_list = sorted(stations.keys())
    var_list     = [v for v in PLOT_VARS if
                    any(v in df.columns for df in stations.values())]
    n_stations   = len(station_list)

    # Split variables into groups of MAX_COLS
    var_groups   = [var_list[i:i + MAX_COLS] for i in range(0, len(var_list), MAX_COLS)]
    n_groups     = len(var_groups)
    total_rows   = n_stations * n_groups

    result_lookup = {(r["Station"], r["Variable"]): r for r in results}

    fig, axes = plt.subplots(
        total_rows, MAX_COLS,
        figsize=(5 * MAX_COLS, 2.6 * total_rows),
        sharex=False, sharey=False
    )
    # Guarantee 2-D array
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    for g, var_group in enumerate(var_groups):
        for i, stn in enumerate(station_list):
            row = g * n_stations + i

            for j in range(MAX_COLS):
                ax = axes[row, j]

                # Blank unused columns in last group
                if j >= len(var_group):
                    ax.axis("off")
                    continue

                var = var_group[j]

                if var not in stations[stn].columns or var not in ref.columns:
                    ax.axis("off")
                    continue

                cand    = stations[stn][var].dropna()
                ref_m2 = ref[var].dropna()

                if len(cand) < 2:
                    ax.axis("off")
                    continue

                # Full station record
                ax.plot(cand.index.to_timestamp(), cand.values,
                        color="steelblue", lw=1.0, label=stn)

                # MERRA-2 overlay for overlapping period
                common = cand.index.intersection(ref_m2.index)
                valid  = cand.loc[common].notna() & ref_m2.loc[common].notna()
                ref_c  = ref_m2.loc[common][valid]
                if not ref_c.empty:
                    ax.plot(ref_c.index.to_timestamp(), ref_c.values,
                            color="darkorange", lw=1.2, ls="--", label="MERRA-2")

                res = result_lookup.get((stn, var))
                if res:
                    by  = pd.Period(res["Break year"], freq="M").to_timestamp()
                    sig = res["Significant"] == "YES"
                    ax.axvline(by,
                               color="crimson" if sig else "grey",
                               lw=1.5 if sig else 1.0,
                               ls="-" if sig else "--",
                               alpha=0.85,
                               label=f"Break {res['Break year']}" + (" *" if sig else ""))

                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.25)

                # Column title: first station of each group
                if i == 0:
                    ax.set_title(var, fontsize=8, fontweight="bold", pad=3)
                # Row label: first column only
                if j == 0:
                    ax.set_ylabel(stn, fontsize=8)

    handles = [
        plt.Line2D([0], [0], color="steelblue",  lw=1.5, label="Station"),
        plt.Line2D([0], [0], color="darkorange", lw=1.2, ls="--", label="MERRA-2"),
        plt.Line2D([0], [0], color="crimson", lw=1.5, label="Significant break"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Monthly anomaly time series — candidate vs. MERRA-2 background\n"
                 "(vertical lines = SNHT break months; red = significant at 5 %)",
                 fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "snht_merra2_all_timeseries.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"All-timeseries figure saved: {out}")


# ---------------------------------------------------------------------------
# Summary heatmap figure
# ---------------------------------------------------------------------------

def plot_summary(results: list) -> None:
    """
    Heatmap of T0 values across stations (rows) and variables (columns).
    Cell text = T0 (top) and break year (bottom).
    Significant cells (T0 > CV) are outlined with a bold red border.
    """
    if not results:
        return

    df        = pd.DataFrame(results)
    stations  = sorted(df["Station"].unique())
    variables = [v for v in VARIABLES if v in df["Variable"].unique()]

    t0_mat   = pd.DataFrame(np.nan,  index=stations, columns=variables)
    year_mat = pd.DataFrame("",      index=stations, columns=variables)
    sig_mat  = pd.DataFrame(False,   index=stations, columns=variables)

    for _, row in df.iterrows():
        s, v = row["Station"], row["Variable"]
        if v not in variables:
            continue
        t0_mat.loc[s, v]   = row["T0"]
        year_mat.loc[s, v] = str(row["Break year"])
        sig_mat.loc[s, v]  = row["Significant"] == "YES"

    n_rows, n_cols = len(stations), len(variables)
    fig, ax = plt.subplots(figsize=(4 + n_cols * 1.6, 1.5 + n_rows * 0.7))

    t0_vals = t0_mat.values.astype(float)
    vmax    = np.nanmax(t0_vals)

    im = ax.imshow(t0_vals, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    for i, stn in enumerate(stations):
        for j, var in enumerate(variables):
            t0   = t0_mat.loc[stn, var]
            year = year_mat.loc[stn, var]
            sig  = sig_mat.loc[stn, var]

            if np.isnan(t0):
                ax.text(j, i, "\u2014", ha="center", va="center",
                        fontsize=9, color="grey")
            else:
                color = "white" if t0 > vmax * 0.6 else "black"
                ax.text(j, i - 0.12, f"{t0:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
                ax.text(j, i + 0.22, year, ha="center", va="center",
                        fontsize=7.5, color=color)

            if sig:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="crimson", lw=2.5
                ))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(variables, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(stations, fontsize=10)
    ax.set_title("SNHT Summary — T\u2080 statistic (break year)\n"
                 "Red outline = significant at 5 %  |  MERRA-2 background reference",
                 fontsize=12, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("T\u2080", fontsize=10)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "snht_merra2_summary.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary figure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading station data ...")
    stations = load_all_stations()
    print(f"Stations loaded: {', '.join(sorted(stations.keys()))}\n")

    print("Loading MERRA-2 reference ...")
    ref = load_merra2_reference()
    print(f"MERRA-2 period: {ref.index[0]} – {ref.index[-1]}\n")

    # ------------------------------------------------------------------
    # Single SNHT pass: candidate station vs. MERRA-2 background
    # ------------------------------------------------------------------
    print("=" * 72)
    print("SNHT  —  reference: MERRA-2 background")
    print("=" * 72)
    results = run_snht_pass(stations, ref=ref, save_figures=True)

    # Summary table
    print(f"\n{'=' * 72}")
    print("SNHT RESULTS SUMMARY  (MERRA-2 background reference)")
    print("=" * 72)
    if not results:
        print("No results produced (check data availability).")
    else:
        df_res = pd.DataFrame(results)
        df_res["_sig"] = df_res["Significant"] == "YES"
        df_res = df_res.sort_values(["_sig", "T0"], ascending=[False, False])
        df_res = df_res.drop(columns="_sig")
        print(df_res.to_string(index=False))

        csv_out = os.path.join(FIG_DIR, "snht_merra2_results.csv")
        df_res.to_csv(csv_out, index=False)
        print(f"\nResults saved to: {csv_out}")

    plot_all_timeseries(stations, results, ref=ref)
    plot_summary(results)
    compare_with_era5(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
