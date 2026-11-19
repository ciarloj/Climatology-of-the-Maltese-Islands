"""
Supplementary check for Figure 9: temporal autocorrelation and effective sample size
====================================================================================
Consecutive daily values are not independent, so the number of paired days (n)
overstates the information behind each inter-station correlation. For each
station and variable this script computes the lag-1 autocorrelation (r1) of the
daily series, and for each station pair the effective sample size

    n_eff = n * (1 - r1_a * r1_b) / (1 + r1_a * r1_b)      (Bretherton et al., 1999)

capped at n. The significance of each Pearson r is then re-tested with a
t-test on n_eff - 2 degrees of freedom.

Station list, variables, date window, missing-value handling, and pairwise
overlap rules are the same as in Fig9_correlation_heatmap_panel.py, so n and r
match the published figure.

Usage:
    python3 Fig9_autocorrelation_neff.py

Output ({workdir}/analysis/corr_analysis/):
    figure_correlation_r1.csv              r1 per station (rows) and variable (columns)
    figure_correlation_neff_{var}.csv      n_eff per station pair, same layout as Figure 9
                                           (blank = no data; diagonal left blank)
    figure_correlation_neff_summary.csv    per variable: ranges of r1, r, n, n_eff, the
                                           largest p-value, and pairs with p >= 0.05
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as t_dist

# ---------------------------------------------------------------------------
# Configuration (kept in sync with Fig9_correlation_heatmap_panel.py)
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

PANEL_VARS = [
    "Precipitation", "Temperature",
    "Relative Humidity", "Wind Speed",
]

DATE_COL = "Date"
DATE_MIN = pd.Timestamp("2007-01-01")
DATE_MAX = pd.Timestamp("2025-12-31")
MIN_OBS  = 30

EXCLUDE_STATIONS = {
    "Pressure": {"Giordan"},
}

PVAL_THRESHOLD = 0.05

workdir    = "/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/WP2/climate assessment/Maltese Islands Climate"
DATA_DIR   = f"{workdir}/Data"
OUTPUT_DIR = f"{workdir}/analysis/corr_analysis"

# ---------------------------------------------------------------------------
# Loading (same as Fig9_correlation_heatmap_panel.py)
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

# ---------------------------------------------------------------------------
# Autocorrelation and effective sample size
# ---------------------------------------------------------------------------

def lag1_autocorr(s):
    """Lag-1 autocorrelation over consecutive calendar days (gaps stay missing)."""
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
    return s.autocorr(lag=1)

def effective_n(n, r1a, r1b):
    prod = r1a * r1b
    return min(float(n), n * (1 - prod) / (1 + prod))

def pvalue_neff(r, n_eff):
    """Two-sided p-value for Pearson r tested on n_eff - 2 degrees of freedom."""
    df = n_eff - 2
    if df <= 0:
        return np.nan
    if abs(r) >= 1:
        return 0.0
    tstat = r * np.sqrt(df / (1 - r ** 2))
    return 2 * t_dist.sf(abs(tstat), df)

def analyse_variable(varname, station_data):
    series = {}
    for stn in STATIONS:
        if stn in EXCLUDE_STATIONS.get(varname, set()):
            continue
        df = station_data.get(stn)
        if df is None:
            continue
        col = find_col(df, varname)
        if col is None:
            continue
        s = df[col].dropna()
        if len(s) >= MIN_OBS:
            series[stn] = s

    r1 = {stn: lag1_autocorr(s) for stn, s in series.items()}

    n_st = len(STATIONS)
    neff = pd.DataFrame(np.full((n_st, n_st), np.nan), index=STATIONS, columns=STATIONS)
    pairs = []
    for i, s1 in enumerate(STATIONS):
        for j, s2 in enumerate(STATIONS):
            if j <= i or s1 not in series or s2 not in series:
                continue
            joined = pd.concat([series[s1], series[s2]], axis=1, join="inner").dropna()
            n = len(joined)
            if n < MIN_OBS:
                continue
            r, _ = pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])
            ne = effective_n(n, r1[s1], r1[s2])
            p = pvalue_neff(r, ne)
            neff.loc[s1, s2] = neff.loc[s2, s1] = ne
            pairs.append({"pair": f"{DISPLAY_NAME_OVERRIDES[s1]}-{DISPLAY_NAME_OVERRIDES[s2]}",
                          "r": r, "n": n, "n_eff": ne, "p_eff": p})
    return r1, neff, pd.DataFrame(pairs)

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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = [DISPLAY_NAME_OVERRIDES[s] for s in STATIONS]
    r1_table = pd.DataFrame(index=labels, columns=PANEL_VARS, dtype=float)
    summary = []

    for varname in PANEL_VARS:
        r1, neff, pairs = analyse_variable(varname, station_data)
        for stn, v in r1.items():
            r1_table.loc[DISPLAY_NAME_OVERRIDES[stn], varname] = v

        out = neff.copy()
        out.index, out.columns = labels, labels
        fp = os.path.join(OUTPUT_DIR, f"figure_correlation_neff_{varname.replace(' ', '_')}.csv")
        out.to_csv(fp, float_format="%.0f")

        if pairs.empty:
            print(f"  [skip] {varname}: no valid station pairs")
            continue
        failing = pairs[pairs["p_eff"] >= PVAL_THRESHOLD]
        r1_vals = pd.Series(r1).dropna()
        summary.append({
            "variable": varname,
            "r1_min": r1_vals.min(), "r1_max": r1_vals.max(),
            "r_min": pairs["r"].min(), "r_max": pairs["r"].max(),
            "n_min": pairs["n"].min(), "n_max": pairs["n"].max(),
            "neff_min": pairs["n_eff"].min(), "neff_max": pairs["n_eff"].max(),
            "p_eff_max": pairs["p_eff"].max(),
            "pairs_total": len(pairs),
            "pairs_p_ge_0.05": len(failing),
            "failing_pairs": "; ".join(failing["pair"]) if len(failing) else "",
        })
        print(f"  [{varname}] r1 {r1_vals.min():.2f}-{r1_vals.max():.2f} | "
              f"n {pairs['n'].min()}-{pairs['n'].max()} | "
              f"n_eff {pairs['n_eff'].min():.0f}-{pairs['n_eff'].max():.0f} | "
              f"max p_eff {pairs['p_eff'].max():.2g} | "
              f"pairs with p >= {PVAL_THRESHOLD}: {len(failing)}/{len(pairs)}"
              + (f" ({'; '.join(failing['pair'])})" if len(failing) else ""))

    r1_table.to_csv(os.path.join(OUTPUT_DIR, "figure_correlation_r1.csv"), float_format="%.3f")
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "figure_correlation_neff_summary.csv"),
                                 index=False, float_format="%.4g")
    print(f"[saved] CSVs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
