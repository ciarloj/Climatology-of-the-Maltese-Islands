"""
Concatenate a folder of downloaded MERRA-2 files into a single NetCDF file.
==========================================================================
Usage:
    python3 merge_merra2.py <input_folder> <output_file.nc>
Examples:
    python3 merge_merra2.py merra2/slv          merra2/M2SDNXSLV_2007-2025.nc
    python3 merge_merra2.py merra2/slv_monthly  merra2/M2TMNXSLV_2007-2025.nc
    python3 merge_merra2.py merra2/rad_monthly  merra2/M2TMNXRAD_2007-2025.nc

- Reads every .nc4 file in the folder (not subfolders, so failed/ is ignored),
  joins them along time in date order, and writes one file.
- Nothing is deleted or changed in the input folder.
- Prints the time range and flags any missing or duplicated time steps.
Needs: pip install xarray netCDF4
"""

import sys
import glob
import os
import pandas as pd
import xarray as xr

def main(folder, out_file):
    files = sorted(glob.glob(os.path.join(folder, "*.nc4")))
    if not files:
        sys.exit(f"No .nc4 files found in {folder}")
    print(f"[read] {len(files)} files in {folder}")

    parts = []
    for i, f in enumerate(files, 1):
        with xr.open_dataset(f) as ds:
            parts.append(ds.load())
        if i % 500 == 0 or i == len(files):
            print(f"\r  {i}/{len(files)}", end="", flush=True)
    print()

    merged = xr.concat(parts, dim="time").sortby("time")

    # Checks: duplicated or missing time steps
    t = pd.DatetimeIndex(merged["time"].values)
    dupes = t[t.duplicated()]
    if len(dupes):
        print(f"[warn] {len(dupes)} duplicated time steps, e.g. {dupes[0]}")
    step = pd.infer_freq(t.normalize()[:10]) if len(t) >= 10 else None
    if step in ("D", "MS"):
        expected = pd.date_range(t.min().normalize(), t.max(), freq=step)
        missing = expected.difference(t.normalize())
        print(f"[check] {'daily' if step == 'D' else 'monthly'} series: "
              f"{len(t)} of {len(expected)} time steps present"
              + (f"; missing e.g. {missing[0].date()}" if len(missing) else ""))

    print(f"[info] {t.min().date()} to {t.max().date()} | "
          f"variables: {', '.join(merged.data_vars)} | "
          f"lat {merged['lat'].values.round(2).tolist()} | "
          f"lon {merged['lon'].values.round(3).tolist()}")

    merged.to_netcdf(out_file)
    print(f"[saved] {out_file} ({os.path.getsize(out_file) / 1e6:.1f} MB)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 merge_merra2.py <input_folder> <output_file.nc>")
    main(sys.argv[1], sys.argv[2])
