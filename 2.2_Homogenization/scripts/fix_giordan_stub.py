#!/usr/bin/env python3
"""Correct Giordan's cold-biased early Temperature stub (2007-01 .. 2008-11).

Why a dedicated script: the stub sits immediately before a multi-year data gap.
homogenize.py's binary segmentation cannot estimate an offset across it - the
all-missing gap segment yields a NaN adjustment - so the stub is left ~2 C too
cold (T0 ~ 47).  The stub's internal variability tracks ERA5 fine; it is a clean
level offset, so the right fix is a single additive shift, NOT masking 5 years.

Method: on the deseasonalised (Giordan - ERA5) difference, measure the level in
the stub window and in a clean post-gap reference window, and add the difference
to the stub's daily values only.  Everything else is preserved byte-for-byte.

Run AFTER any homogenize.py round (reads Daily_Giordan_corrected.csv if present).
"""
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import netCDF4 as nc4

from snht_TS_ERA5 import DATA_DIR, ERA5_DIR, MALTA_LAT, MALTA_LON

STATION = "Daily_Giordan"
VAR     = "Temperature"
MISSING = "-"

STUB = ("2007-01-01", "2008-11-30")     # cold-biased early real data (before the gap)
REF  = ("2012-01-01", "2014-12-31")     # clean post-gap window - verify it tracks ERA5

# --- ERA5 monthly mean 2 m temperature (raw, deg C), read straight from the zips ---
def era5_monthly_t2m():
    out = {}
    for fpath in sorted(glob.glob(os.path.join(ERA5_DIR, "era5_monthly_*.nc"))):
        if any(t in os.path.basename(fpath) for t in ("tmax", "tmin")):
            continue                                    # skip the new min/max files
        with zipfile.ZipFile(fpath) as z:
            ua = next(n for n in z.namelist() if "avgua" in n)
            ds = nc4.Dataset(ua, memory=z.read(ua))
            la = int(np.argmin(np.abs(ds["latitude"][:]  - MALTA_LAT)))
            lo = int(np.argmin(np.abs(ds["longitude"][:] - MALTA_LON)))
            tt = nc4.num2date(ds["valid_time"][:], ds["valid_time"].units)
            for k, t in enumerate(tt):
                out[pd.Period(f"{t.year}-{t.month:02d}", freq="M")] = \
                    float(ds["t2m"][k, la, lo]) - 273.15
            ds.close()
    return pd.Series(out).sort_index()


# --- station daily -> monthly mean (>= 20 valid days, matching the pipeline) ---
corrected = os.path.join(DATA_DIR, f"{STATION}_corrected.csv")
src = corrected if os.path.exists(corrected) else os.path.join(DATA_DIR, f"{STATION}.csv")
print("reading", os.path.basename(src))

df    = pd.read_csv(src, dtype=str)
dates = pd.to_datetime(df["Date"], format="%d/%m/%Y")
vals  = pd.to_numeric(df[VAR].replace(MISSING, np.nan), errors="coerce")

g   = pd.Series(vals.values, index=dates)
cnt = g.resample("MS").count()
mon = g.resample("MS").mean().where(cnt >= 20)
mon.index = mon.index.to_period("M")

# --- deseasonalised difference, level in each window, offset ---
diff = (mon - era5_monthly_t2m()).dropna()
clim = diff.groupby(diff.index.month).transform("mean")
anom = diff - clim

def win(a, b):
    return (anom.index >= pd.Period(a[:7], "M")) & (anom.index <= pd.Period(b[:7], "M"))

m_stub, m_ref = win(*STUB), win(*REF)
if m_stub.sum() == 0 or m_ref.sum() == 0:
    raise SystemExit(f"empty window (stub n={m_stub.sum()}, ref n={m_ref.sum()}) "
                     f"- check STUB/REF dates against the data")

offset = anom[m_ref].mean() - anom[m_stub].mean()
print(f"stub level  = {anom[m_stub].mean():+.3f} C  (n={m_stub.sum()})")
print(f"ref  level  = {anom[m_ref].mean():+.3f} C  (n={m_ref.sum()})")
print(f"--> applying offset = {offset:+.3f} C to {STUB[0]} .. {STUB[1]}")

# --- apply to the stub's daily values only; preserve everything else ---
in_stub = (dates >= pd.Timestamp(STUB[0])) & (dates <= pd.Timestamp(STUB[1]))
out = df.copy()
new = pd.to_numeric(df.loc[in_stub, VAR].replace(MISSING, np.nan), errors="coerce") + offset
out.loc[in_stub, VAR] = new.map(lambda x: MISSING if pd.isna(x) else f"{x:.1f}")

dst = os.path.join(DATA_DIR, f"{STATION}_corrected.csv")
out.to_csv(dst, index=False)
print("wrote", dst, f"({int(in_stub.sum())} daily rows adjusted)")
