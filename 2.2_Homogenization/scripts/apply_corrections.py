#!/usr/bin/env python3
"""Apply a log of manual corrections: set-missing, additive offset, or
multiplicative factor, per (station, variable, date range). Applied in order.

Built for inheriting corrections onto same-sensor DERIVED variables:
  Max / Min Temperature  <- the average-Temperature log (same thermometer)
  Wind Gust              <- the Wind-Speed log (same anemometer)
Copy the dates / offsets / factors straight out of your homogenize.py and
homogenize_segments.py printouts and tabulate them below.

Reads <station>_corrected.csv if present (other columns + their precision are
preserved); each target column is reset from raw first so re-runs don't stack.

SCIENCE / caveats:
  - Break DATES and set-missing intervals transfer EXACTLY (shared instrument).
  - A multiplicative (gain) factor transfers exactly to gust.
  - An additive offset transfers exactly only for a uniform sensor bias. For
    max/min a radiation-shield / diurnal-dependent bias can differ between
    max, min and mean, so treat inherited offsets as a first pass and
    re-estimate the magnitudes against ERA5 daily max/min when it lands -
    the break DATES stay put, only the size of each step is refit.
  - Gust has no clean ERA5 reference; inherit from wind speed and document it.
"""
import os
from collections import defaultdict
import numpy as np
import pandas as pd

from snht_TS_ERA5 import DATA_DIR

MISSING = "-"
RESET_FROM_RAW = True   # reset each target column from raw before applying

# (station_file, variable, start, end, type, value)
#   type "missing"  -> value ignored
#   type "add"      -> value = additive offset (same units as the column)
#   type "multiply" -> value = factor
CORRECTIONS = [
    # ---- Max inheriting the average-Temperature log ----
      ( "Daily_Bengħajsa.csv",  "Max Temperature", "2007-01-01", "2015-12-31", "add", 0.44),
      ( "Daily_Bengħajsa.csv",  "Max Temperature", "2016-01-01", "2023-10-31", "add", 0.24),
      ( "Daily_Dingli.csv",     "Max Temperature", "2007-01-01", "2014-08-31", "add", 0.31),	
      ( "Daily_Dingli.csv",     "Max Temperature", "2007-01-01", "2010-03-31", "add", -0.29),	
      ( "Daily_Giordan.csv",    "Max Temperature", "2007-01-01", "2014-07-31", "add", -1.7),	
      ( "Daily_Giordan.csv",    "Max Temperature", "2007-01-01", "2008-11-30", "add", 2.309),	
      ( "Daily_Giordan.csv",    "Max Temperature", "2007-01-01", "2008-12-31", "add", -0.406),	
      ( "Daily_Giordan.csv",    "Max Temperature", "2011-01-01", "2014-07-31", "add", -0.924),	
      ( "Daily_Luqa.csv",       "Max Temperature", "2007-01-01", "2015-12-31", "add", 0.31),	
      ( "Daily_Luqa.csv",       "Max Temperature", "2007-01-01", "2015-02-28", "add", -0.26),	
      ( "Daily_Luqa.csv",       "Max Temperature", "2015-12-01", "2016-01-31", "missing", None),	
      ( "Daily_Luqa.csv",       "Max Temperature", "2015-09-25", "2015-11-30", "missing", None),
      ( "Daily_Birkirkara.csv", "Max Temperature", "2007-01-01", "2012-05-31", "add", 0.22),
      ( "Daily_Birkirkara.csv", "Max Temperature", "2012-06-01", "2014-03-31", "add", 1.36),
      ( "Daily_Birkirkara.csv", "Max Temperature", "2014-04-01", "2015-12-31", "add", 0.6),
      ( "Daily_Selmun.csv",     "Max Temperature", "2007-01-01", "2012-06-30", "add", 0.53),
      ( "Daily_Selmun.csv",     "Max Temperature", "2007-01-01", "2008-04-30", "add", -0.31),
      ( "Daily_Valletta.csv",   "Max Temperature", "2007-01-01", "2011-05-31", "add", -0.2),
      ( "Daily_Valletta.csv",   "Max Temperature", "2011-06-01", "2014-09-30", "add", 0.24),
      ( "Daily_Xagħra.csv",     "Max Temperature", "2007-01-01", "2021-04-30", "add", 0.23),
      ( "Daily_Xewkija.csv",    "Max Temperature", "2007-01-01", "2010-02-28", "add", 0.17),
      ( "Daily_Xewkija.csv",    "Max Temperature", "2010-03-01", "2012-05-31", "add", 0.39),
      ( "Daily_Xewkija.csv",    "Max Temperature", "2012-06-01", "2019-05-31", "add", 0.04),
      ( "Daily_Xewkija.csv",    "Max Temperature", "2019-06-01", "2023-10-31", "add", -0.36),
    # ---- Min inheriting the average-Temperature log ----
      ( "Daily_Bengħajsa.csv",  "Min Temperature", "2007-01-01", "2015-12-31", "add", 0.44),
      ( "Daily_Bengħajsa.csv",  "Min Temperature", "2016-01-01", "2023-10-31", "add", 0.24),
      ( "Daily_Dingli.csv",     "Min Temperature", "2007-01-01", "2014-08-31", "add", 0.31),
      ( "Daily_Dingli.csv",     "Min Temperature", "2007-01-01", "2010-03-31", "add", -0.29),
      ( "Daily_Giordan.csv",    "Min Temperature", "2007-01-01", "2014-07-31", "add", -1.7),
      ( "Daily_Giordan.csv",    "Min Temperature", "2007-01-01", "2008-11-30", "add", 2.309),
      ( "Daily_Giordan.csv",    "Min Temperature", "2007-01-01", "2008-12-31", "add", -0.406),
      ( "Daily_Giordan.csv",    "Min Temperature", "2011-01-01", "2014-07-31", "add", -0.924),
      ( "Daily_Luqa.csv",       "Min Temperature", "2007-01-01", "2015-12-31", "add", 0.31),
      ( "Daily_Luqa.csv",       "Min Temperature", "2007-01-01", "2015-02-28", "add", -0.26),
      ( "Daily_Luqa.csv",       "Min Temperature", "2015-12-01", "2016-01-31", "missing", None),
      ( "Daily_Luqa.csv",       "Min Temperature", "2015-09-25", "2015-11-30", "missing", None),
      ( "Daily_Birkirkara.csv", "Min Temperature", "2007-01-01", "2012-05-31", "add", 0.22),
      ( "Daily_Birkirkara.csv", "Min Temperature", "2012-06-01", "2014-03-31", "add", 1.36),
      ( "Daily_Birkirkara.csv", "Min Temperature", "2014-04-01", "2015-12-31", "add", 0.6),
      ( "Daily_Selmun.csv",     "Min Temperature", "2007-01-01", "2012-06-30", "add", 0.53),
      ( "Daily_Selmun.csv",     "Min Temperature", "2007-01-01", "2008-04-30", "add", -0.31),
      ( "Daily_Valletta.csv",   "Min Temperature", "2007-01-01", "2011-05-31", "add", -0.2),
      ( "Daily_Valletta.csv",   "Min Temperature", "2011-06-01", "2014-09-30", "add", 0.24),
      ( "Daily_Xagħra.csv",     "Min Temperature", "2007-01-01", "2021-04-30", "add", 0.23),
      ( "Daily_Xewkija.csv",    "Min Temperature", "2007-01-01", "2010-02-28", "add", 0.17),
      ( "Daily_Xewkija.csv",    "Min Temperature", "2010-03-01", "2012-05-31", "add", 0.39),
      ( "Daily_Xewkija.csv",    "Min Temperature", "2012-06-01", "2019-05-31", "add", 0.04),
      ( "Daily_Xewkija.csv",    "Min Temperature", "2019-06-01", "2023-10-31", "add", -0.36),
    # ---- Wind Gust inheriting the Wind-Speed log ----
      ( "Daily_Valletta.csv",   "Wind Gust",       "2007-01-01", "2013-11-30", "add", 2.30),
      ( "Daily_Birkirkara.csv", "Wind Gust",       "2012-12-01", "2025-12-31", "add", 0.94),
      ( "Daily_Xagħra.csv",     "Wind Gust",       "2008-01-01", "2011-12-31", "multiply", 1.039),
      ( "Daily_Xagħra.csv",     "Wind Gust",       "2012-07-01", "2015-12-31", "multiply", 1.783),
      ( "Daily_Selmun.csv",     "Wind Gust",       "2008-01-01", "2011-12-31", "multiply", 1.151),
      ( "Daily_Selmun.csv",     "Wind Gust",       "2012-06-01", "2015-12-31", "multiply", 1.961),
]


def col_decimals(series_str):
    decs = [len(v.split(".")[1]) for v in series_str.dropna()
            if isinstance(v, str) and v not in ("-", "") and "." in v]
    return max(set(decs), key=decs.count) if decs else 2


by_file = defaultdict(list)
for row in CORRECTIONS:
    by_file[row[0]].append(row)

if not CORRECTIONS:
    print("CORRECTIONS list is empty - fill it from your logs.")

for fname, rows in by_file.items():
    raw = os.path.join(DATA_DIR, fname)
    cor = os.path.join(DATA_DIR, fname.replace(".csv", "_corrected.csv"))
    df    = pd.read_csv(cor if os.path.exists(cor) else raw, dtype=str)
    rawdf = pd.read_csv(raw, dtype=str)
    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    if RESET_FROM_RAW:
        for var in {r[1] for r in rows}:
            df[var] = rawdf[var].values

    print(f"\n{fname}:")
    for _, var, a, b, typ, val in rows:
        nd = col_decimals(rawdf[var])
        m  = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
        if typ == "missing":
            df.loc[m, var] = MISSING
            print(f"  {var:16s} {a}..{b}  set missing  ({int(m.sum())} days)")
        elif typ in ("add", "multiply"):
            cur = pd.to_numeric(df.loc[m, var].replace(MISSING, np.nan), errors="coerce")
            new = cur + val if typ == "add" else cur * val
            df.loc[m, var] = new.map(lambda x: MISSING if pd.isna(x) else f"{x:.{nd}f}")
            print(f"  {var:16s} {a}..{b}  {typ} {val}  ({int(cur.notna().sum())} values)")
        else:
            raise ValueError(f"unknown type {typ!r}")

    df.to_csv(cor, index=False)
    print(f"  wrote {os.path.basename(cor)}")
