"""
apply_residual_fixes.py
-----------------------
Homogenization round 3 (v3). Removes the residual breaks found when the
homogenized (v2) data was re-tested with SNHT at the corrected 5 % critical
values (Khaliq and Ouarda, 2007) against ERA5 and MERRA-2.

Selection rule: a break is corrected when it is significant against at least
one reference AND either (a) significant against the other too, or (b) the
other reference puts its largest T(k) within 2 months of the same month.
Break months and offsets come from the ERA5 test (the reference used for all
earlier corrections); MERRA-2 gives the same months and offsets within
0.02 C for TMX/TMN.
Segments at the record ends shorter than 24 months are set to missing rather
than adjusted (the same rule as the Msida 2007 solar spike in round 2).
Wind speed and gust use a gain factor (multiplicative; WMO, 2020).
Not corrected (one reference only, different month):
    Luqa Pressure (MERRA-2 only, 2021-06) and Valletta RH (MERRA-2 only, 2016-06).

TMX and TMN are now corrected separately: the TAV offsets they inherited
removed the mean shift at the 2015/16 changeover but not the extremes
(TMX shifted by about +0.9 C and TMN by about -0.3 C at most stations).

SAFETY
- Never writes to DATA_DIR. Copies the whole Data folder to Data_v3 and
  applies the fixes there.
- Only the cells covered by FIXES change; values are handled as text, so every
  other cell is written back exactly as read. Missing cells stay missing.
- Re-running reproduces the same output (it always starts from v2). It stops
  if Data_v3 already exists, or if Data is already the v3 data.

DOCUMENTATION (written next to the existing change logs)
    ../change_logs/23_residual_fixes.log       full report
    ../change_logs/residual_fixes_summary.csv  rows in Changes_summary.csv format

USAGE (from the folder holding snht_TS_ERA5.py)
    python apply_residual_fixes.py
Then rename  Data -> Data_v2  and  Data_v3 -> Data,  and rerun both SNHT scripts.
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd

from snht_TS_ERA5 import DATA_DIR

if len(sys.argv) > 1:                 # optional: python apply_residual_fixes.py <data_dir>
    DATA_DIR = sys.argv[1]

DATA_DIR = os.path.normpath(DATA_DIR)
OUT_DIR  = DATA_DIR + "_v3"
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "change_logs")
MARKER   = "_HOMOGENIZATION_v3.txt"
MISSING  = "-"
START    = "2007-01-01"
CLIP     = {"Relative Humidity": (0, 100), "Wind Speed": (0, None),
            "Wind Gust": (0, None), "Solar": (0, None)}

# (file, column, start, end, type, value, comment)
#   "add"     value added to every valid cell in [start, end]
#   "gain"    value = SNHT shift (post minus pre); cells multiplied by
#             1 + value / mean of the segment (multiplicative, as for all solar fixes)
#   "inherit" value = another column in the same file; cells multiplied by the
#             factor already computed for that column (gust <- wind speed)
#   "missing" cells set to "-"
# Pre-break segments are adjusted onto the post-break level (anchor = last
# segment, as in rounds 1-2), except where noted.
FIXES = [
    # ---- TMX: 2015/16 changeover (2021 at Giordan) ----
    ("Daily_Bengħajsa.csv",  "Max Temperature", START, "2015-12-31", "add",  0.95, "TMX break 2015-12"),
    ("Daily_Xagħra.csv",     "Max Temperature", START, "2015-12-31", "add",  1.09, "TMX break 2015-12"),
    ("Daily_Luqa.csv",       "Max Temperature", START, "2015-08-31", "add",  1.01, "TMX break 2015-08"),
    ("Daily_Xewkija.csv",    "Max Temperature", START, "2015-12-31", "add",  0.92, "TMX break 2015-12"),
    ("Daily_Dingli.csv",     "Max Temperature", START, "2015-12-31", "add",  0.96, "TMX break 2015-12"),
    ("Daily_Selmun.csv",     "Max Temperature", START, "2015-12-31", "add",  0.91, "TMX break 2015-12"),
    ("Daily_Birkirkara.csv", "Max Temperature", START, "2015-08-31", "add",  0.52, "TMX break 2015-08"),
    ("Daily_Valletta.csv",   "Max Temperature", START, "2015-12-31", "add",  0.38, "TMX break 2015-12"),
    ("Daily_Giordan.csv",    "Max Temperature", START, "2021-11-30", "add",  1.69, "TMX break 2021-11"),
    # ---- TMN: 2015/16 changeover ----
    ("Daily_Xewkija.csv",    "Min Temperature", START, "2016-10-31", "add", -0.40, "TMN break 2016-10"),
    ("Daily_Xagħra.csv",     "Min Temperature", START, "2016-02-29", "add", -0.36, "TMN break 2016-02"),
    ("Daily_Bengħajsa.csv",  "Min Temperature", START, "2015-03-31", "add", -0.46, "TMN break 2015-03"),
    ("Daily_Selmun.csv",     "Min Temperature", START, "2016-02-29", "add", -0.25, "TMN break 2016-02"),
    ("Daily_Dingli.csv",     "Min Temperature", START, "2016-05-31", "add", -0.25, "TMN break 2016-05"),
    # ---- TAV ----
    ("Daily_Luqa.csv",       "Temperature",     START, "2015-09-30", "add",  0.16, "TAV break 2015-09"),
    ("Daily_Giordan.csv",    "Temperature",     START, "2021-11-30", "add",  0.92, "TAV break 2021-11"),
    ("Daily_Xagħra.csv",     "Temperature",     "2024-09-01", "2025-12-31", "missing", None,
     "TAV break 2024-08; post-break segment 16 months (< 24-month minimum), too short to adjust"),
    # ---- RH ----
    ("Daily_Giordan.csv",    "Relative Humidity", START, "2015-01-31", "add", 3.22, "RH break 2015-01"),
    ("Daily_Xewkija.csv",    "Relative Humidity", START, "2018-04-30", "add", 1.67, "RH break 2018-04 (MERRA-2: 2016-11)"),
    ("Daily_Bengħajsa.csv",  "Relative Humidity", START, "2018-04-30", "add", 1.65, "RH break 2018-04"),
    ("Daily_Dingli.csv",     "Relative Humidity", START, "2016-07-31", "add", 1.41, "RH break 2016-07"),
    # ---- Solar ----
    ("Daily_Luqa.csv",       "Solar", START, "2007-07-31", "missing", None,
     "SR break 2007-07 at record start; 7 months too short for a gain factor"),
    ("Daily_Birkirkara.csv", "Solar", START, "2012-12-31", "gain", 5.42, "SR break 2012-12 (before dead-sensor gap)"),
    # ---- Wind (gust inherits the wind-speed fix, as in rounds 1-2) ----
    ("Daily_Selmun.csv",     "Wind Speed", START, "2017-02-28", "gain", -0.59, "WS break 2017-02 (gain factor)"),
    ("Daily_Selmun.csv",     "Wind Gust",  START, "2017-02-28", "inherit", "Wind Speed", "inherits the WS factor (same anemometer)"),
]


def col_decimals(col):
    decs = [len(v.split(".")[1]) for v in col if v not in (MISSING, "") and "." in v]
    return max(set(decs), key=decs.count) if decs else 1


def main():
    log_lines, summary, factors = [], [], {}
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    # ---- checks before anything is written ----
    if os.path.exists(os.path.join(DATA_DIR, MARKER)):
        sys.exit(f"{DATA_DIR} already holds the v3 data ({MARKER}); nothing done.")
    if os.path.exists(OUT_DIR):
        sys.exit(f"{OUT_DIR} already exists. Delete or rename it, then rerun.")
    files = {}
    for f, col, *_ in FIXES:
        if f not in files:
            p = os.path.join(DATA_DIR, f)
            if not os.path.exists(p):
                sys.exit(f"Missing file: {p}")
            files[f] = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
        if col not in files[f].columns:
            sys.exit(f"Column '{col}' not in {f}. Columns: {list(files[f].columns)}")

    # ---- copy the whole folder, then overwrite the fixed files ----
    shutil.copytree(DATA_DIR, OUT_DIR)
    log(f"Copied {DATA_DIR}\n    -> {OUT_DIR}\n")

    for fname, df in files.items():
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        log(f"{fname}")
        for f, col, a, b, typ, val, comment in FIXES:
            if f != fname:
                continue
            nd   = col_decimals(df[col])
            rng  = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
            num  = pd.to_numeric(df.loc[rng, col].replace(MISSING, np.nan), errors="coerce")
            ok   = num.notna()
            idx  = num[ok].index
            if not len(idx):
                log(f"  WARNING {col}: no valid values in {a}..{b}; nothing changed")
                continue

            if typ == "missing":
                df.loc[idx, col] = MISSING
                sval, stype = np.nan, "missing"
                log(f"  {col:17s} {a}..{b}  set missing  ({len(idx)} values)")
            else:
                if typ == "add":
                    new, sval, stype = num[ok] + val, val, "additive"
                elif typ == "gain":
                    factor = 1 + val / num[ok].mean()
                    factors[(fname, col)] = factor
                    new, sval, stype = num[ok] * factor, round(factor, 4), "multiplicative"
                elif typ == "inherit":
                    factor = factors[(fname, val)]
                    new, sval, stype = num[ok] * factor, round(factor, 4), "multiplicative"
                else:
                    raise ValueError(f"unknown type {typ!r}")
                lo, hi = CLIP.get(col, (None, None))
                clipped = (int((new < lo).sum()) if lo is not None else 0) + \
                          (int((new > hi).sum()) if hi is not None else 0)
                new = new.clip(lower=lo, upper=hi)
                df.loc[idx, col] = new.map(lambda x: f"{x:.{nd}f}")
                what = (f"add {val:+.2f}" if typ == "add" else
                        f"x {sval} (shift {val:+.2f})" if typ == "gain" else
                        f"x {sval} (from {val})")
                log(f"  {col:17s} {a}..{b}  {what:22s} {len(idx):5d} values  "
                    f"mean {num[ok].mean():.2f} -> {new.mean():.2f}"
                    + (f"  ({clipped} clipped to {lo}..{hi})" if clipped else ""))

            summary.append({
                "station": fname.replace("Daily_", "").replace(".csv", ""),
                "variable": col,
                "start": pd.Timestamp(a).strftime("%d/%m/%Y"),
                "end": pd.Timestamp(b).strftime("%d/%m/%Y"),
                "type": stype, "value": sval, "comment": f"v3: {comment}",
            })

        df.to_csv(os.path.join(OUT_DIR, fname), index=False)
        log(f"  wrote {os.path.join(OUT_DIR, fname)}\n")

    with open(os.path.join(OUT_DIR, MARKER), "w") as fh:
        fh.write("v3 data: v2 plus apply_residual_fixes.py\n")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "23_residual_fixes.log")
    sum_path = os.path.join(LOG_DIR, "residual_fixes_summary.csv")
    pd.DataFrame(summary).to_csv(sum_path, index=False, encoding="utf-8")
    log(f"{len(summary)} fixes applied. Report: {log_path}\nSummary rows: {sum_path}")
    log("Next: rename Data -> Data_v2 and Data_v3 -> Data, then rerun both SNHT scripts.")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
