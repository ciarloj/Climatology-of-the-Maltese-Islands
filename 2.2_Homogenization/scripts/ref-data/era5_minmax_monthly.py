#!/usr/bin/env python3
"""Download ERA5 daily-maximum / daily-minimum 2 m temperature and aggregate to
monthly means, for use as the SNHT reference for station Max / Min Temperature.

Run once.  Two stages:
  retrieve()   - daily statistics from the CDS derived dataset, one file per year
  aggregate()  - monthly mean of those daily values

Output (written next to your other era5_monthly_*.nc files, in ERA5_DIR):
  era5_monthly_tmax.nc   variable t2m_max   (Kelvin, valid_time/latitude/longitude)
  era5_monthly_tmin.nc   variable t2m_min

IMPORTANT - the CDS is post-migration and the derived-daily dataset has had beta
quirks.  Open the dataset's *Download* tab, tick "2m temperature" + "Daily
maximum" + "1-hourly", and use **Show API request** to confirm the exact keys /
values in retrieve() below (chiefly `daily_statistic`, `frequency`, `time_zone`).
The variable and time-coordinate names in the returned files are auto-detected in
aggregate(), so those do not need editing.

If retrieve() raises NotImplementedError ("beta version ... not implemented"),
use the hourly fallback at the bottom of this file instead.
"""
import os
import glob
import numpy as np
import xarray as xr

# keep paths identical to the SNHT script
from snht_TS_ERA5 import ERA5_DIR, MALTA_LAT, MALTA_LON  # noqa: E402

# small box around Malta (N, W, S, E) - must contain MALTA_LAT/MALTA_LON
AREA   = [MALTA_LAT + 0.4, MALTA_LON - 0.4, MALTA_LAT - 0.4, MALTA_LON + 0.4]
YEARS  = [str(y) for y in range(2007, 2025)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS   = [f"{d:02d}" for d in range(1, 32)]

RAW_DIR = os.path.join(ERA5_DIR, "era5_daily_minmax")          # raw daily downloads
os.makedirs(RAW_DIR, exist_ok=True)

# (CDS daily_statistic value, short tag, output variable name)
STATS = [("daily_maximum", "tmax", "t2m_max"),
         ("daily_minimum", "tmin", "t2m_min")]


def retrieve():
    """Download one daily-statistics file per (statistic, year). Resumable."""
    import cdsapi
    c = cdsapi.Client()
    for stat, tag, _ in STATS:
        for yr in YEARS:
            out = os.path.join(RAW_DIR, f"era5_daily_{tag}_{yr}.nc")
            if os.path.exists(out):
                print("skip (exists):", os.path.basename(out))
                continue
            print("downloading", stat, yr)
            c.retrieve(
                "derived-era5-single-levels-daily-statistics",
                {
                    "product_type":    "reanalysis",
                    "variable":        ["2m_temperature"],
                    "year":            yr,
                    "month":           MONTHS,
                    "day":             DAYS,
                    "daily_statistic": stat,          # <-- verify on Download tab
                    "time_zone":       "utc+00:00",
                    "frequency":       "1_hourly",
                    "area":            AREA,
                },
                out,
            )


def aggregate():
    """Monthly mean of the daily values -> era5_monthly_t{max,min}.nc."""
    for stat, tag, var in STATS:
        files = sorted(glob.glob(os.path.join(RAW_DIR, f"era5_daily_{tag}_*.nc")))
        if not files:
            raise FileNotFoundError(f"no daily files for {tag} in {RAW_DIR}")
        ds = xr.open_mfdataset(files, combine="by_coords")

        tname = "valid_time" if "valid_time" in ds.coords else "time"
        vname = list(ds.data_vars)[0]            # the single 2 m-temperature field
        print(f"{tag}: variable={vname!r}  time={tname!r}  ({len(files)} files)")

        monthly = ds[vname].resample(**{tname: "1MS"}).mean().rename(var)
        if tname != "valid_time":
            monthly = monthly.rename({tname: "valid_time"})

        dst = os.path.join(ERA5_DIR, f"era5_monthly_{tag}.nc")
        monthly.to_dataset(name=var).to_netcdf(dst)
        print("wrote", dst)


if __name__ == "__main__":
    retrieve()
    aggregate()


# ---------------------------------------------------------------------------
# HOURLY FALLBACK  (use only if the derived-daily dataset errors out)
# Download hourly 2 m temperature, then compute daily max/min locally:
#
#   import cdsapi, xarray as xr, glob, os
#   c = cdsapi.Client()
#   for yr in YEARS:
#       c.retrieve("reanalysis-era5-single-levels", {
#           "product_type": "reanalysis", "variable": ["2m_temperature"],
#           "year": yr, "month": MONTHS, "day": DAYS,
#           "time": [f"{h:02d}:00" for h in range(24)],
#           "area": AREA, "data_format": "netcdf",
#       }, os.path.join(RAW_DIR, f"era5_hourly_t2m_{yr}.nc"))
#
#   ds = xr.open_mfdataset(sorted(glob.glob(os.path.join(RAW_DIR,"era5_hourly_t2m_*.nc"))))
#   t  = "valid_time" if "valid_time" in ds.coords else "time"
#   v  = list(ds.data_vars)[0]
#   for var, op in [("t2m_max","max"), ("t2m_min","min")]:
#       daily   = getattr(ds[v].resample(**{t:"1D"}), op)()
#       monthly = daily.resample(**{t:"1MS"}).mean().rename(var)
#       if t != "valid_time": monthly = monthly.rename({t:"valid_time"})
#       monthly.to_dataset(name=var).to_netcdf(
#           os.path.join(ERA5_DIR, f"era5_monthly_{'tmax' if var=='t2m_max' else 'tmin'}.nc"))
# ---------------------------------------------------------------------------
