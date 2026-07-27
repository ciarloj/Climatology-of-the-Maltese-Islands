"""
apply_step_fix.py
-----------------
Estimate and remove a single ADDITIVE step (offset) inhomogeneity from one
station/variable, reusing the SNHT + ERA5 machinery already defined in
snht_TS_ERA5.py.

It does four things and prints a report:
  1. estimates the offset from the candidate-minus-ERA5 monthly anomaly difference
  2. re-runs SNHT on the de-stepped difference to VERIFY the break is gone
  3. applies the same constant offset to the RAW daily series (post-break only)
  4. writes a non-destructive corrected file:  Daily_<station>_corrected.csv
     (the original file is never touched)

Currently configured for the Xewkija mean-sea-level-pressure step detected at
the 2015/2016 boundary. To reuse it for another *additive-step* case, change
the four CONFIG values below. It is NOT appropriate for multiplicative fixes
(e.g. solar) or set-to-missing fixes (e.g. dead sensors).

HOW TO RUN
  1. Save this file in the SAME folder as snht_TS_ERA5.py.
  2. From that folder, run:   python apply_step_fix.py
"""
import os
import pandas as pd

# Reuse the loaders / test / data path from the detection script.
# Importing does NOT run its main(): that is guarded by `if __name__ == "__main__"`.
from snht_TS_ERA5 import (
    load_all_stations,
    load_era5_reference,
    snht,
    critical_value,
    DATA_DIR,
)

# ---------------------------------------------------------------------------
# CONFIG  -- the only things you change per case
# ---------------------------------------------------------------------------
STATION  = "Xewkija"                        # station display name (as used by the loaders)
VARIABLE = "Pressure"                       # column name (as used by the loaders)
# BREAK = last UN-shifted month; the offset is applied from the FOLLOWING month onward.
# SNHT nominally flagged 2015-12, but inspection of the monthly means shows Dec-2015 is
# already fully shifted (Oct 1003 -> Nov 1008 transitional -> Dec 1017), so we set the
# last clean month to Nov-2015 and correct from 2015-12-01.
BREAK    = pd.Period("2015-11", freq="M")
RAW_FILE = f"Daily_{STATION}.csv"           # raw daily file inside DATA_DIR (confirm exact name)
# ---------------------------------------------------------------------------


def main():
    # --- 1. estimate the offset on the candidate - ERA5 anomaly difference --
    stations = load_all_stations()
    era5     = load_era5_reference()

    if STATION not in stations:
        raise SystemExit(f"Station '{STATION}' not found. Available: {sorted(stations)}")
    if VARIABLE not in stations[STATION].columns:
        raise SystemExit(f"'{VARIABLE}' not available for {STATION}. "
                         f"Has: {list(stations[STATION].columns)}")

    cand   = stations[STATION][VARIABLE].dropna()
    ref    = era5[VARIABLE].dropna()
    common = cand.index.intersection(ref.index)
    diff   = (cand.loc[common] - ref.loc[common]).dropna()

    seg1 = diff[diff.index <= BREAK]    # up to and including the break month
    seg2 = diff[diff.index >  BREAK]    # the shifted segment
    if len(seg1) == 0 or len(seg2) == 0:
        raise SystemExit("One side of the break is empty -- check the BREAK month.")

    offset = float(seg2.mean() - seg1.mean())   # +ve => post-break sits too high

    # --- 2. verify: remove the step from the difference and re-run SNHT ------
    diff_corr  = diff.copy()
    mask_after = diff_corr.index > BREAK
    diff_corr.loc[mask_after] = diff_corr.loc[mask_after] - offset

    cv        = critical_value(len(diff))
    t0_before = snht(diff.values)[2]
    t0_after  = snht(diff_corr.values)[2]
    passed    = t0_after < cv

    print("=" * 64)
    print(f"STEP FIX  --  {STATION} / {VARIABLE}")
    print("=" * 64)
    print(f"Break month            : {BREAK}  (shift starts {BREAK + 1})")
    print(f"Segment sizes          : n1={len(seg1)}  n2={len(seg2)} months")
    print(f"Estimated offset       : {offset:+.2f}   (post-break minus pre-break)")
    print(f"SNHT T0  before        : {t0_before:8.1f}")
    print(f"SNHT T0  after         : {t0_after:8.1f}")
    print(f"Critical value (5%)    : {cv:8.1f}")
    print("-" * 64)
    if passed:
        print("VERIFICATION PASSED -- step removed; the corrected file is valid.")
    else:
        print("WARNING -- T0 still above the critical value after one offset.")
        print("           A constant step did not fully homogenise the series")
        print("           (possible second break or a drift). Inspect before use.")
    print("=" * 64)

    # --- 3 & 4. apply to RAW daily data and write a corrected copy ----------
    raw_path = os.path.join(DATA_DIR, RAW_FILE)
    if not os.path.exists(raw_path):
        raise SystemExit(f"Raw file not found: {raw_path}\n"
                         f"Check RAW_FILE -- files in {DATA_DIR} use the 'Daily_<name>.csv' pattern.")

    raw = pd.read_csv(raw_path, na_values=["-", "N/A", "n/a", "NA", ""])
    if VARIABLE not in raw.columns:
        raise SystemExit(f"Column '{VARIABLE}' not in {RAW_FILE}. Columns: {list(raw.columns)}")

    # Parse dates ONLY for masking; leave the Date column on disk byte-for-byte unchanged.
    dates      = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    apply_mask = dates > BREAK.to_timestamp(how="end")   # strictly after the break month
    raw.loc[apply_mask, VARIABLE] = raw.loc[apply_mask, VARIABLE] - offset

    out_path = os.path.join(DATA_DIR, f"Daily_{STATION}_corrected.csv")
    raw.to_csv(out_path, index=False, na_rep="-")
    print(f"Adjusted {int(apply_mask.sum())} daily '{VARIABLE}' records (dates after {BREAK}).")
    print(f"Corrected file written : {out_path}")
    print("Original file left unchanged.")


if __name__ == "__main__":
    main()
