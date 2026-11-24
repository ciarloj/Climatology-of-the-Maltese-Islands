#!/usr/bin/env python3
"""
Download ERA5 monthly-mean surface data for Malta.

Variables downloaded (all from reanalysis-era5-single-levels-monthly-means):
  - 2m_temperature                      -> Temperature
  - 2m_dewpoint_temperature             -> Relative Humidity (computed offline)
  - total_precipitation                 -> Precipitation
  - mean_sea_level_pressure             -> Pressure
  - 10m_u_component_of_wind             -> Wind Speed (u)
  - 10m_v_component_of_wind             -> Wind Speed (v)
  - surface_solar_radiation_downwards   -> Solar

These map directly to the variables in the Malta station CSVs used by snht_TS.py.
ERA5 is used as the large-scale background state / reference series.

Output
------
  data/ERA5/era5_monthly_{YEAR}.nc   — one NetCDF per year (12 months x area)

Usage
-----
  python download_era5.py             # full period 1979–2024
  python download_era5.py 2010 2020   # subset

Requirements
------------
  pip install cdsapi
  ~/.cdsapirc must contain valid CDS credentials (new API, cds.climate.copernicus.eu)
  cdsapi >= 0.7.0  (new .retrieve(...).download() syntax)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
YEAR_START = 2007
YEAR_END   = 2025

# Bounding box — same as radar-era5-conv-anal.py: N W S E
AREA = [37, 13, 34, 16]

OUT_DIR = Path("/mnt/d/Storage/James/Dropbox/PALEOSIM/Research/Data/ERA5")

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "total_precipitation",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]

# ---------------------------------------------------------------------------

def out_path(year: int) -> Path:
    return OUT_DIR / f"era5_monthly_{year}.nc"


def download_year(year: int, client) -> str:
    """Download one year of monthly means. Returns 'ok', 'cached', or 'fail'."""
    dst = out_path(year)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"  {year}  cached ({dst.stat().st_size / 1e3:.0f} kB)")
        return "cached"

    print(f"  {year}  requesting ...", end="", flush=True)
    try:
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable":     VARIABLES,
                "year":         [str(year)],
                "month":        MONTHS,
                "time":         ["00:00"],
                "area":         AREA,
                "data_format":  "netcdf",
                "download_format": "unarchived",
            },
        ).download(str(dst))
        size_kb = dst.stat().st_size / 1e3
        print(f"  ok  ({size_kb:.0f} kB)")
        return "ok"
    except Exception as exc:
        print(f"  FAILED: {exc}")
        if dst.exists():
            dst.unlink()          # remove partial file
        return "fail"


def main() -> None:
    args = sys.argv[1:]
    y0 = int(args[0]) if len(args) > 0 else YEAR_START
    y1 = int(args[1]) if len(args) > 1 else YEAR_END

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ERA5 monthly means — Malta background state")
    print(f"Period    : {y0}–{y1}  ({y1 - y0 + 1} years)")
    print(f"Variables : {len(VARIABLES)}")
    print(f"Area      : N{AREA[0]} W{AREA[1]} S{AREA[2]} E{AREA[3]}")
    print(f"Output    : {OUT_DIR}\n")

    import cdsapi
    client = cdsapi.Client()

    counts = {"ok": 0, "cached": 0, "fail": 0}
    for year in range(y0, y1 + 1):
        status = download_year(year, client)
        counts[status] += 1

    print(f"\n{'='*50}")
    print(f"  Downloaded : {counts['ok']}")
    print(f"  Cached     : {counts['cached']}")
    print(f"  Failed     : {counts['fail']}")
    print(f"{'='*50}")

    if counts["fail"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
