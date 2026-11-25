"""
compare_raw_final.py
--------------------
Effect of the homogenization on the spatial patterns (Reviewer 1, major comment 1).

For each variable, compares the 2007-2025 station means of the original (raw)
data and the final homogenized data:
  - mean per station (all valid days in each version) and the difference
  - station rank in each version, and the Spearman rank correlation across stations
  - inter-station range (highest minus lowest station mean)
DTR (Max minus Min Temperature) is included because Fig. 11 uses it.

USAGE (from the folder holding snht_TS_ERA5.py)
    python compare_raw_final.py <raw_folder> [<final_folder>]
<final_folder> defaults to the Data folder used by the SNHT scripts.

OUTPUT
    compare_raw_final.csv in the current folder (one row per station and variable),
    plus a one-line summary per variable on screen.
"""
import os
import sys
import numpy as np
import pandas as pd

FINAL_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")
STATIONS = {"Giordan": "GD", "Xagħra": "XR", "Xewkija": "XW", "Selmun": "SL", "Dingli": "DN",
            "Birkirkara": "MS", "Valletta": "VL", "Luqa": "LQ", "Bengħajsa": "BN"}
VARIABLES = ["Temperature", "Max Temperature", "Min Temperature", "DTR",
             "Relative Humidity", "Solar", "Pressure", "Wind Speed", "Wind Gust"]
START, END, MISSING = "2007-01-01", "2025-12-31", "-"


def station_means(folder):
    out = {}
    for fname, abbr in STATIONS.items():
        p = os.path.join(folder, f"Daily_{fname}.csv")
        if not os.path.exists(p):
            print(f"  [skip] {p} not found")
            continue
        df = pd.read_csv(p, dtype=str)
        d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df[(d >= START) & (d <= END)]
        num = {c: pd.to_numeric(df[c].replace(MISSING, np.nan), errors="coerce")
               for c in df.columns if c != "Date"}
        if "Max Temperature" in num and "Min Temperature" in num:
            num["DTR"] = num["Max Temperature"] - num["Min Temperature"]
        out[abbr] = {v: (num[v].mean(), int(num[v].notna().sum())) for v in VARIABLES if v in num}
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python compare_raw_final.py <raw_folder> [<final_folder>]")
    raw_dir = sys.argv[1]
    fin_dir = sys.argv[2] if len(sys.argv) > 2 else FINAL_DEFAULT
    raw, fin = station_means(raw_dir), station_means(fin_dir)

    rows = []
    for v in VARIABLES:
        for s in STATIONS.values():
            if v in raw.get(s, {}) and v in fin.get(s, {}):
                (mr, nr), (mf, nf) = raw[s][v], fin[s][v]
                if nr and nf:
                    rows.append({"variable": v, "station": s, "raw_mean": mr, "final_mean": mf,
                                 "difference": mf - mr, "raw_days": nr, "final_days": nf})
    df = pd.DataFrame(rows)
    df["raw_rank"] = df.groupby("variable")["raw_mean"].rank(ascending=False).astype(int)
    df["final_rank"] = df.groupby("variable")["final_mean"].rank(ascending=False).astype(int)

    print(f"{'variable':18s} {'n':>2s} {'rho':>5s} {'range raw':>10s} {'range final':>12s}   largest change")
    for v, g in df.groupby("variable", sort=False):
        rho = g["raw_mean"].corr(g["final_mean"], method="spearman")
        big = g.loc[g["difference"].abs().idxmax()]
        moved = (g["raw_rank"] != g["final_rank"]).sum()
        print(f"{v:18s} {len(g):2d} {rho:5.2f} {g['raw_mean'].max() - g['raw_mean'].min():10.2f} "
              f"{g['final_mean'].max() - g['final_mean'].min():12.2f}   "
              f"{big['station']} {big['difference']:+.2f}; {moved} station(s) change rank")

    out = "compare_raw_final.csv"
    df.round(3).to_csv(out, index=False, encoding="utf-8")
    print(f"\n[saved] {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
