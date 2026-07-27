"""
rh_saturation_scan.py
---------------------
Find stretches where the RH sensor is pinned at/near 100% (saturated/clipped).
Those sectors cannot be rescued by any shift -- the upper tail is gone -- so they
must be set to missing. The mildly wet-biased sectors (low saturation) are left
for the additive homogeniser.

For each station it reports, per year, the fraction of days with RH >= THRESH and
flags years above FLAG_FRAC; for flagged years it lists the offending months so
you can copy date ranges straight into apply_set_missing.py.

Run:  python rh_saturation_scan.py
"""
import os, glob
import numpy as np
import pandas as pd
from snht_TS_ERA5 import DATA_DIR          # the same Data folder the other scripts use

RH_COL    = "Relative Humidity"
THRESH    = 99.5         # a daily value >= this is treated as "saturated"
FLAG_FRAC = 0.15         # year/month flagged if this fraction of days are saturated
MISSING   = "-"


def scan(path):
    name = os.path.basename(path).replace("Daily_", "").replace(".csv", "")
    df = pd.read_csv(path, na_values=[MISSING, "N/A", "n/a", "NA", ""])
    if RH_COL not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    rh = pd.to_numeric(df[RH_COL], errors="coerce")
    sat = (rh >= THRESH)
    yr = df["Date"].dt.year
    by_year = pd.DataFrame({"n": rh.notna().groupby(yr).sum(),
                            "sat": sat.groupby(yr).sum()})
    by_year["frac"] = by_year["sat"] / by_year["n"].replace(0, np.nan)

    print(f"\n{name}")
    flagged_years = []
    for y, row in by_year.iterrows():
        mark = "  <== FLAG" if row["frac"] >= FLAG_FRAC else ""
        if row["frac"] >= FLAG_FRAC or row["sat"] > 0:
            print(f"   {int(y)}: {row['frac']*100:5.1f}% saturated "
                  f"({int(row['sat'])}/{int(row['n'])} days){mark}")
        if row["frac"] >= FLAG_FRAC:
            flagged_years.append(int(y))

    # month detail for flagged years -> ready-to-use ranges
    if flagged_years:
        mo = df["Date"].dt.to_period("M")
        fm = (sat.groupby(mo).sum() / rh.notna().groupby(mo).sum().replace(0, np.nan))
        hot = fm[(fm >= FLAG_FRAC) & (fm.index.year.isin(flagged_years))]
        if len(hot):
            lo, hi = hot.index.min(), hot.index.max()
            print(f"   -> saturated months span {lo} .. {hi}  "
                  f"(set-missing candidate range {lo.start_time.date()} .. {hi.end_time.date()})")
    return by_year


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Daily_*.csv")))
    files = [f for f in files if "_corrected" not in f] or files
    print(f"RH saturation scan (THRESH={THRESH}%, flag at {FLAG_FRAC*100:.0f}% of days)")
    print(f"Looking in: {DATA_DIR}")
    if not files:
        print("  -> No Daily_*.csv found here. Run it beside snht_TS_ERA5.py "
              "(or set DATA_DIR).")
        return
    found_rh = 0
    for f in files:
        if scan(f) is not None:
            found_rh += 1
    if found_rh == 0:
        print(f"\n  -> Found {len(files)} files but none had a '{RH_COL}' column "
              "-- check the column name.")


if __name__ == "__main__":
    main()
