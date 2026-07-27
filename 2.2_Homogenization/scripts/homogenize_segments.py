#!/usr/bin/env python3
"""Piecewise homogenisation for gappy / bounded-segment records.

Use when multi-year gaps or a bounded depression defeat homogenize.py's
automatic segmentation. You supply the homogeneous segments by hand; each is
aligned onto a chosen reference segment on the deseasonalised station-minus-ERA5
difference. Other columns, gaps, and the column's numeric precision are preserved.

  MODE = "additive"        offset errors (temperature, pressure), and the safer
                           default for STEP breaks against the noisy ERA5 wind
                           reference - it aligns the arithmetic mean directly.
  MODE = "multiplicative"  proportional / gain faults where the variance scales
                           with the mean (e.g. the Xagħra/Selmun wind depression,
                           confirmed: mean x0.54, SD x0.53). One factor per segment.

STACKING: reads <station>_corrected.csv if present so prior corrections to OTHER
variables survive. READ_RAW resets ONLY the target column to its raw values
(use it for a first/!redo correction of this variable); set False to stack.

Re-run SNHT afterwards. If a segment is internally inhomogeneous, split it.

  --- Giordan (additive, temperature) config, for reference: ---
  STATION="Daily_Giordan"; VAR="Temperature"; MODE="additive"; READ_RAW=True
  SEGMENTS=[("2007-01-01","2008-12-31"),("2011-01-01","2014-07-31"),
            ("2014-08-01","2022-12-31")]; REF_INDEX=2
"""
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import netCDF4 as nc4

from snht_TS_ERA5 import DATA_DIR, ERA5_DIR, MALTA_LAT, MALTA_LON

# ----------------------------- configure here ------------------------------
STATION  = "Daily_Xagħra"
VAR      = "Wind Speed"           # "Temperature" | "Wind Speed"
MODE     = "multiplicative"       # "additive" | "multiplicative"
MISSING  = "-"
READ_RAW = True                   # reset ONLY this column from raw, keep the rest

# Xagħra depression -> post-2016 baseline. Recovery is sharp 2015-12->2016-01,
# so S2 must STOP at the last low month and S3 START at the first recovered one;
# extending S2 past the recovery is what spikes the boundary. Refine off daily.
SEGMENTS = [
    ("2008-01-01", "2011-12-31"),   # S1 pre        (good)
    ("2012-07-01", "2015-12-31"),   # S2 depression (~x0.54)
    ("2016-01-01", "2025-12-31"),   # S3 post       (good)  <- reference
]
REF_INDEX = 2
# ---------------------------------------------------------------------------


def era5_monthly(var):
    out = {}
    for fpath in sorted(glob.glob(os.path.join(ERA5_DIR, "era5_monthly_*.nc"))):
        if any(t in os.path.basename(fpath) for t in ("tmax", "tmin")):
            continue
        with zipfile.ZipFile(fpath) as z:
            ua = next(n for n in z.namelist() if "avgua" in n)
            ds = nc4.Dataset(ua, memory=z.read(ua))
            la = int(np.argmin(np.abs(ds["latitude"][:]  - MALTA_LAT)))
            lo = int(np.argmin(np.abs(ds["longitude"][:] - MALTA_LON)))
            tt = nc4.num2date(ds["valid_time"][:], ds["valid_time"].units)
            for k, t in enumerate(tt):
                p = pd.Period(f"{t.year}-{t.month:02d}", freq="M")
                if var == "Temperature":
                    out[p] = float(ds["t2m"][k, la, lo]) - 273.15
                elif var == "Wind Speed":
                    u = float(ds["u10"][k, la, lo]); v = float(ds["v10"][k, la, lo])
                    out[p] = float(np.sqrt(u * u + v * v))
                else:
                    raise ValueError(f"unsupported VAR {var!r}")
            ds.close()
    return pd.Series(out).sort_index()


def column_decimals(series_str):
    decs = [len(v.split(".")[1]) for v in series_str.dropna()
            if isinstance(v, str) and v not in ("-", "") and "." in v]
    return max(set(decs), key=decs.count) if decs else 2


# --- read corrected-if-exists (keep other cols); reset ONLY VAR from raw ---
raw_path = os.path.join(DATA_DIR, f"{STATION}.csv")
cor_path = os.path.join(DATA_DIR, f"{STATION}_corrected.csv")
df = pd.read_csv(cor_path if os.path.exists(cor_path) else raw_path, dtype=str)
if READ_RAW:
    df[VAR] = pd.read_csv(raw_path, dtype=str)[VAR].values
print(f"reading {os.path.basename(cor_path if os.path.exists(cor_path) else raw_path)}"
      f" | VAR={VAR} | MODE={MODE} | reset-from-raw={READ_RAW}")

dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
vals  = pd.to_numeric(df[VAR].replace(MISSING, np.nan), errors="coerce")
ndec  = column_decimals(df[VAR])

g   = pd.Series(vals.values, index=dates)
cnt = g.resample("MS").count()
mon = g.resample("MS").mean().where(cnt >= 20)
mon.index = mon.index.to_period("M")

era5 = era5_monthly(VAR)
if MODE == "additive":
    diff = (mon - era5).dropna()
elif MODE == "multiplicative":
    pair = pd.concat([mon.rename("s"), era5.rename("e")], axis=1).dropna()
    pair = pair[(pair["s"] > 0) & (pair["e"] > 0)]
    diff = np.log(pair["s"]) - np.log(pair["e"])
else:
    raise ValueError(MODE)
anom = diff - diff.groupby(diff.index.month).transform("mean")


def level(a, b):
    m = (anom.index >= pd.Period(a[:7], "M")) & (anom.index <= pd.Period(b[:7], "M"))
    return anom[m].mean(), int(m.sum())


ref_level, ref_n = level(*SEGMENTS[REF_INDEX])
print(f"\nreference = S{REF_INDEX+1} "
      f"{SEGMENTS[REF_INDEX][0][:7]}..{SEGMENTS[REF_INDEX][1][:7]}  n={ref_n}  "
      f"({ndec} dp)\n")

for i, (a, b) in enumerate(SEGMENTS):
    lv, n = level(a, b)
    if n == 0:
        print(f"  S{i+1} {a[:7]}..{b[:7]}: NO valid months - skipped"); continue
    d = ref_level - lv
    if MODE == "additive":
        adj, desc = d, f"offset {d:+.3f}"
    else:
        adj, desc = float(np.exp(d)), f"factor x{np.exp(d):.3f}"
    tag = "  (reference)" if i == REF_INDEX else f"  -> {desc}"
    print(f"  S{i+1} {a[:7]}..{b[:7]}: n={n}{tag}")
    if i == REF_INDEX or abs(d) < 1e-9:
        continue
    inseg = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    cur   = pd.to_numeric(df.loc[inseg, VAR].replace(MISSING, np.nan), errors="coerce")
    new   = cur + adj if MODE == "additive" else cur * adj
    df.loc[inseg, VAR] = new.map(lambda x: MISSING if pd.isna(x) else f"{x:.{ndec}f}")

df.to_csv(cor_path, index=False)
print(f"\nwrote {cor_path}")
