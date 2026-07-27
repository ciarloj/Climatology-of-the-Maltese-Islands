#!/usr/bin/env python3
"""Flag physically-impossible daily solar values and set them missing.

Daily-mean GHI cannot exceed the clear-sky ceiling for that day of year. We
build that ceiling empirically: a high per-day-of-year quantile pooled across
all stations and years, smoothed seasonally, then flag anything above it by a
margin. A LOW quantile (0.99, not 0.999) is used on purpose so a station's own
spikes don't inflate the envelope that is meant to catch them.

Validated on this network: Q=0.99, MARGIN=1.10 catches the Luqa spring-2019
spikes (418-584 W/m2, vs an April ceiling ~350) while leaving the legitimate
~405 summer peak untouched.

Runs as a DRY RUN by default (prints what it would remove). Set APPLY=True to
write the missing flags into the *_corrected.csv files (other columns kept).
"""
import os
import glob
import numpy as np
import pandas as pd

from snht_TS_ERA5 import DATA_DIR

VAR      = "Solar"
QUANTILE = 0.99     # envelope percentile (robust to the outliers themselves)
MARGIN   = 1.10     # flag values above envelope * MARGIN
SMOOTH   = 10       # +/- days seasonal smoothing of the envelope
MISSING  = "-"
APPLY    = False    # False = report only; True = set flagged values missing


def read(fname, prefer_corrected=True):
    cor = os.path.join(DATA_DIR, fname.replace(".csv", "_corrected.csv"))
    raw = os.path.join(DATA_DIR, fname)
    p = cor if (prefer_corrected and os.path.exists(cor)) else raw
    return p, pd.read_csv(p, dtype=str)


files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(DATA_DIR, "Daily_*.csv")))
         if "_corrected" not in f]

# --- pooled seasonal envelope ---
pool = []
for fname in files:
    _, df = read(fname)
    if VAR not in df.columns:
        continue
    d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    v = pd.to_numeric(df[VAR].replace(MISSING, np.nan), errors="coerce")
    pool.append(pd.DataFrame({"doy": d.dt.dayofyear.values, VAR: v.values}))
pool = pd.concat(pool).dropna()

env = pool.groupby("doy")[VAR].quantile(QUANTILE).reindex(range(1, 367)).interpolate()
ext = pd.concat([env, env, env]).reset_index(drop=True)              # wrap for circular smoothing
sm  = ext.rolling(2 * SMOOTH + 1, center=True, min_periods=1).mean()
envelope = pd.Series(sm.iloc[366:732].values, index=range(1, 367))

# --- flag per station ---
print(f"{'DRY RUN - ' if not APPLY else ''}solar outliers (Q={QUANTILE}, margin={MARGIN}):\n")
total = 0
for fname in files:
    path, df = read(fname)
    if VAR not in df.columns:
        continue
    d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    v = pd.to_numeric(df[VAR].replace(MISSING, np.nan), errors="coerce")
    thr  = d.dt.dayofyear.map(envelope) * MARGIN
    flag = (v > thr).fillna(False)
    if flag.sum():
        for dt, val, t in zip(d[flag], v[flag], thr[flag]):
            print(f"  {fname:24s} {dt.date()}  {val:6.0f}  (ceiling {t:.0f})")
        total += int(flag.sum())
        if APPLY:
            df.loc[flag.values, VAR] = MISSING
            df.to_csv(path, index=False)
            print(f"    -> wrote {os.path.basename(path)}")

print(f"\n{'set missing' if APPLY else 'would flag'}: {total} value(s)")
if not APPLY:
    print("Review the list, then set APPLY=True to write the missing flags.")
