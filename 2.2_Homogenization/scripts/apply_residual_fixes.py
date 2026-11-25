"""
apply_residual_fixes.py
-----------------------
Homogenization rounds 3-10. Removes the residual breaks found when the
homogenized data was re-tested with SNHT at the corrected 5 % critical values
(Khaliq and Ouarda, 2007) against ERA5 and MERRA-2. Each round uses the
re-test after the previous one; SNHT reports only the largest break per
series, so removing it reveals the next one.

Selection rule: every break significant against ERA5 (the reference used in
rounds 1-2 and throughout the paper) is corrected, as is a break significant
against MERRA-2 where ERA5 puts its largest T(k) within 2 months (the same
break). Break months and offsets come from ERA5; where both references are
significant at different months, the ERA5 break is corrected.
Other MERRA-2-only breaks are corrected with MERRA-2's month and offset unless
the correction produces a significant break against ERA5, i.e. the references
contradict each other. A trial (first version of round 7) showed this for
Bengħajsa, Dingli, and Valletta RH, which are reported, not corrected. Luqa
pressure is treated the same way: MERRA-2 shows +0.21 hPa from 2021-06, the
first of its reprocessed MERRA2_401 months, while ERA5, which tracks Luqa
pressure closely, shows no step.
Segments shorter than 24 months are set to missing: at the record ends, and
between two breaks of opposite sign found in successive rounds.
Pre-break segments are moved onto the post-break level (anchor = last).
Wind speed and gust use a gain factor (multiplicative; WMO, 2020).
Stopping rule: stop when no break meets the selection rule.

TMX and TMN are corrected separately from round 3: the TAV offsets they
inherited removed the mean shift at the 2015/16 changeover but not the
extremes (TMX shifted by about +0.9 C and TMN by about -0.3 C).

DATA
- Applies the fixes in place to the Daily_*.csv files in DATA_DIR. Keep your
  own copy of the previous version: running the same round twice applies it twice.
- Only the cells covered by FIXES change; values are handled as text, so every
  other cell is written back exactly as read. Missing cells stay missing.
  Adjusted cells keep the column's decimals (one decimal for whole-number columns).

DOCUMENTATION (written next to the existing change logs)
    round 3: ../change_logs/23_residual_fixes.log, residual_fixes_summary.csv
    round 4: ../change_logs/24_residual_fixes_r4.log, residual_fixes_summary_r4.csv
    round 5: ../change_logs/25_residual_fixes_r5.log, residual_fixes_summary_r5.csv
    round 6: ../change_logs/26_residual_fixes_r6.log, residual_fixes_summary_r6.csv
    round 7: ../change_logs/27_residual_fixes_r7.log, residual_fixes_summary_r7.csv
    round 8: ../change_logs/28_residual_fixes_r8.log, residual_fixes_summary_r8.csv
    round 9: ../change_logs/29_residual_fixes_r9.log, residual_fixes_summary_r9.csv
    round 10: ../change_logs/30_residual_fixes_r10.log, residual_fixes_summary_r10.csv

USAGE (from the folder holding snht_TS_ERA5.py)
    python apply_residual_fixes.py 10       one round
    python apply_residual_fixes.py 3-10     rounds 3 to 10 in order, starting from the v2 data
Then rerun both SNHT scripts.
"""
import os
import sys
import numpy as np
import pandas as pd

# Same Data folder as snht_TS_ERA5.py (not imported, so no figure folders are created)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Data")

ARG = sys.argv[1] if len(sys.argv) > 1 else "10"       # "<round>" or "<first>-<last>"
if len(sys.argv) > 2:                                   # optional data folder (testing)
    DATA_DIR = sys.argv[2]

DATA_DIR = os.path.normpath(DATA_DIR)
LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "change_logs")
LOG_NAME = {3: "23_residual_fixes.log", 4: "24_residual_fixes_r4.log",
            5: "25_residual_fixes_r5.log", 6: "26_residual_fixes_r6.log",
            7: "27_residual_fixes_r7.log", 8: "28_residual_fixes_r8.log",
            9: "29_residual_fixes_r9.log", 10: "30_residual_fixes_r10.log"}
SUM_NAME = {3: "residual_fixes_summary.csv", 4: "residual_fixes_summary_r4.csv",
            5: "residual_fixes_summary_r5.csv", 6: "residual_fixes_summary_r6.csv",
            7: "residual_fixes_summary_r7.csv", 8: "residual_fixes_summary_r8.csv",
            9: "residual_fixes_summary_r9.csv", 10: "residual_fixes_summary_r10.csv"}
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
FIXES = {3: [
    # ---- TMX: 2015/16 changeover (2021 at Giordan) ----
    ("Daily_Bengħajsa.csv",  "Max Temperature", START, "2015-12-31", "add",  0.95, "TMX break 2015-12"),
    ("Daily_Xagħra.csv",     "Max Temperature", START, "2015-12-31", "add",  1.09, "TMX break 2015-12"),
    ("Daily_Luqa.csv",       "Max Temperature", START, "2015-08-31", "add",  1.01, "TMX break 2015-08"),
    ("Daily_Xewkija.csv",    "Max Temperature", START, "2015-12-31", "add",  0.92, "TMX break 2015-12"),
    ("Daily_Dingli.csv",     "Max Temperature", START, "2015-12-31", "add",  0.96, "TMX break 2015-12"),
    ("Daily_Selmun.csv",     "Max Temperature", START, "2015-12-31", "add",  0.91, "TMX break 2015-12"),
    ("Daily_Birkirkara.csv", "Max Temperature", START, "2015-08-31", "add",  0.52, "TMX break 2015-08"),
    ("Daily_Valletta.csv",   "Max Temperature", START, "2015-12-31", "add",  0.38, "TMX break 2015-12"),
    ("Daily_Giordan.csv",    "Max Temperature", "2021-12-01", "2022-12-31", "missing", None,
     "TMX break 2021-11; 13 months before the record ends (2022-12), too short to adjust"),
    # ---- TMN: 2015/16 changeover ----
    ("Daily_Xewkija.csv",    "Min Temperature", START, "2016-10-31", "add", -0.40, "TMN break 2016-10"),
    ("Daily_Xagħra.csv",     "Min Temperature", START, "2016-02-29", "add", -0.36, "TMN break 2016-02"),
    ("Daily_Bengħajsa.csv",  "Min Temperature", START, "2015-03-31", "add", -0.46, "TMN break 2015-03"),
    ("Daily_Selmun.csv",     "Min Temperature", START, "2016-02-29", "add", -0.25, "TMN break 2016-02"),
    ("Daily_Dingli.csv",     "Min Temperature", START, "2016-05-31", "add", -0.25, "TMN break 2016-05"),
    # ---- TAV ----
    ("Daily_Luqa.csv",       "Temperature",     START, "2015-09-30", "add",  0.16, "TAV break 2015-09"),
    ("Daily_Giordan.csv",    "Temperature",     "2021-12-01", "2022-12-31", "missing", None,
     "TAV break 2021-11; 13 months before the record ends (2022-12), too short to adjust"),
    ("Daily_Xagħra.csv",     "Temperature",     "2024-09-01", "2025-12-31", "missing", None,
     "TAV break 2024-08; post-break segment 16 months (< 24-month minimum), too short to adjust"),
    # ---- RH ----
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
], 4: [
    # ---- record-start segments shorter than 24 months: set missing ----
    ("Daily_Bengħajsa.csv",  "Max Temperature", START, "2008-01-31", "missing", None,
     "TMX break 2008-01 (MERRA-2 2007-12); 13-month start segment"),
    ("Daily_Bengħajsa.csv",  "Min Temperature", START, "2007-08-31", "missing", None,
     "TMN break 2007-08 (MERRA-2 2007-07); 8-month start segment"),
    ("Daily_Dingli.csv",     "Max Temperature", START, "2008-04-30", "missing", None,
     "TMX break 2008-04 (MERRA-2 significant at 2007-11); 16-month start segment"),
    # ---- TMX ----
    ("Daily_Birkirkara.csv", "Max Temperature", START, "2023-06-30", "add",  0.42, "TMX break 2023-06 (both references)"),
    ("Daily_Valletta.csv",   "Max Temperature", START, "2023-06-30", "add",  0.29, "TMX break 2023-06 (MERRA-2 significant at 2007-09)"),
    # ---- TMN ----
    ("Daily_Xagħra.csv",     "Min Temperature", START, "2023-10-31", "add", -0.29, "TMN break 2023-10 (MERRA-2 significant at 2024-07)"),
    # ---- RH ----
    ("Daily_Dingli.csv",     "Relative Humidity", START, "2022-03-31", "add", -2.52, "RH break 2022-03 (MERRA-2 2022-01)"),
    ("Daily_Bengħajsa.csv",  "Relative Humidity", START, "2015-09-30", "add", -1.68, "RH break 2015-09 (MERRA-2 2015-08)"),
    # ---- Solar ----
    ("Daily_Birkirkara.csv", "Solar", START, "2019-05-31", "gain", -5.72, "SR break 2019-05 (both references)"),
    # Not corrected (one reference only): Luqa Solar (ERA5 2013-10; MERRA-2 2014-03, not
    # significant), Giordan RH (ERA5 2014-01; MERRA-2 2020-09), Xewkija TMN (MERRA-2 2024-07;
    # ERA5 2025-03), Luqa Pressure and Valletta RH (MERRA-2 only, as in round 3).
], 5: [
    # ---- record-end segments shorter than 24 months: set missing ----
    ("Daily_Birkirkara.csv", "Max Temperature", START, "2007-11-30", "missing", None,
     "TMX break 2007-11 (MERRA-2 2007-09); 11-month start segment"),
    ("Daily_Xewkija.csv",    "Max Temperature", START, "2008-01-31", "missing", None,
     "TMX break 2008-01 (MERRA-2 significant at 2021-08); 13-month start segment"),
    ("Daily_Birkirkara.csv", "Solar", "2025-04-01", "2025-12-31", "missing", None,
     "SR break 2025-03 (MERRA-2 2025-01); 9-month end segment"),
    # ---- TMX ----
    ("Daily_Dingli.csv",     "Max Temperature", START, "2023-06-30", "add",  0.37, "TMX break 2023-06 (both references)"),
    ("Daily_Valletta.csv",   "Max Temperature", START, "2013-05-31", "add", -0.22, "TMX break 2013-05 (both references)"),
    # ---- TMN ----
    ("Daily_Xagħra.csv",     "Min Temperature", START, "2013-09-30", "add",  0.13, "TMN break 2013-09 (both references)"),
    # Not corrected (one reference only, or the other more than 2 months away):
    # Bengħajsa RH (MERRA-2 2017-01; ERA5 2017-05), Luqa Solar (ERA5 2013-10; MERRA-2 2014-03),
    # Giordan RH (ERA5 only), Bengħajsa TMN, Dingli RH, Xewkija TMN, Luqa Pressure, and
    # Valletta RH (MERRA-2 only).
], 6: [
    ("Daily_Luqa.csv",       "Solar", START, "2013-10-31", "gain", -3.92, "SR break 2013-10 (MERRA-2 2014-03)"),
    # Not corrected (MERRA-2 only; ERA5 shows no break at that date): Luqa Pressure 2021-06
    # (+0.21 hPa), Bengħajsa RH 2017-01 (+2.1 %), Dingli RH 2018-04 (+1.9 %), Valletta RH
    # 2016-06 (+1.7 %), Bengħajsa TMN 2022-09 (-0.23 C), Xewkija TMN 2024-07 (-0.23 C).
], 7: [
    # ---- short anomalous segments between breaks of opposite sign: set missing ----
    ("Daily_Giordan.csv",    "Relative Humidity", "2014-02-01", "2015-01-31", "missing", None,
     "12 months between RH breaks 2014-01 (-), 2014-09 (+), and 2015-01 (+)"),
    ("Daily_Xewkija.csv",    "Max Temperature", "2021-02-01", "2021-08-31", "missing", None,
     "7 months between TMX breaks 2021-01 (+) and 2021-08 (-)"),
    # These two were first corrected as steps (Giordan RH in rounds 3 and 6; Xewkija TMX in
    # rounds 4 and 6, reversed in round 5); the step corrections were dropped when the re-tests
    # showed short anomalous segments instead.
    # Not corrected (MERRA-2 only). A first version of this round also corrected them
    # (Bengħajsa RH box, Xewkija TMN end, Luqa SR 2021-04, Dingli RH 2018-04, Valletta RH
    # 2016-06, Bengħajsa TMN 2022-09); the ERA5 re-test then showed new significant breaks
    # for the three RH series (Valletta T0 26.2, Dingli 18.1, Bengħajsa 15.2), which are
    # therefore not corrected. The other three did not disturb ERA5 and are applied in round 9.
], 8: [
    # From the ERA5 re-test after round 7
    ("Daily_Giordan.csv",    "Relative Humidity", START, "2020-11-30", "add", -3.58,
     "RH break 2020-11 (MERRA-2 2020-09); 25 months to the record end (2022-12)"),
    ("Daily_Xewkija.csv",    "Max Temperature", START, "2023-06-30", "add",  0.27, "TMX break 2023-06"),
], 9: [
    # ---- Giordan: 2011-12 breaks (both references), found again once the 2021-12 to
    #      2022-12 months were set missing in round 3 ----
    ("Daily_Giordan.csv",    "Max Temperature", START, "2011-12-31", "add", -0.87, "TMX break 2011-12 (MERRA-2 2011-11)"),
    ("Daily_Giordan.csv",    "Temperature",     START, "2011-12-31", "add", -0.58, "TAV break 2011-12 (MERRA-2 2011-11)"),
    ("Daily_Giordan.csv",    "Relative Humidity", START, "2008-11-30", "missing", None,
     "RH break 2008-11 (both references); 23-month start segment before the 2009-2011 gap"),
    # ---- Xewkija TMX: MERRA-2 significant, ERA5 maximum at the same month ----
    ("Daily_Xewkija.csv",    "Max Temperature", START, "2017-08-31", "add", -0.17, "TMX break 2017-08 (MERRA-2 significant)"),
    # ---- MERRA-2-only breaks whose trial correction did not disturb ERA5 ----
    ("Daily_Luqa.csv",       "Solar", START, "2021-04-30", "gain", 5.29, "SR break 2021-04 (MERRA-2; ERA5 2020-03)"),
    ("Daily_Bengħajsa.csv",  "Min Temperature", START, "2022-09-30", "add", -0.23, "TMN break 2022-09 (MERRA-2; ERA5 2025-03)"),
    ("Daily_Xewkija.csv",    "Min Temperature", "2024-08-01", "2025-12-31", "missing", None,
     "TMN break 2024-07 (MERRA-2; ERA5 2025-03); 17-month end segment"),
], 10: [
    ("Daily_Giordan.csv",    "Temperature",     START, "2014-01-31", "add",  0.45, "TAV break 2014-01 (MERRA-2 2014-01)"),
    # Remaining MERRA-2 flags, not corrected (references contradict): Bengħajsa, Dingli, and
    # Valletta RH, and Luqa Pressure.
]}


def col_decimals(col):
    decs = [len(v.split(".")[1]) for v in col if v not in (MISSING, "") and "." in v]
    # Whole-number columns get one decimal in adjusted cells, so offsets such as
    # -3.58 are not rounded to whole units (which would bias every value by up to 0.5)
    return max(set(decs), key=decs.count) if decs else 1


def run_round(ROUND):
    log_lines, summary, factors = [], [], {}
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    # ---- checks before anything is written ----
    if ROUND not in FIXES:
        sys.exit(f"Unknown round {ROUND}; available: {sorted(FIXES)}")
    fixes = FIXES[ROUND]
    files = {}
    for f, col, *_ in fixes:
        if f not in files:
            p = os.path.join(DATA_DIR, f)
            if not os.path.exists(p):
                sys.exit(f"Missing file: {p}")
            files[f] = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
        if col not in files[f].columns:
            sys.exit(f"Column '{col}' not in {f}. Columns: {list(files[f].columns)}")

    log(f"Round {ROUND}: fixing files in {DATA_DIR}\n")

    for fname, df in files.items():
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        log(f"{fname}")
        for f, col, a, b, typ, val, comment in fixes:
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
                "type": stype, "value": sval, "comment": f"v{ROUND}: {comment}",
            })

        df.to_csv(os.path.join(DATA_DIR, fname), index=False)
        log(f"  wrote {fname}\n")

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_NAME[ROUND])
    sum_path = os.path.join(LOG_DIR, SUM_NAME[ROUND])
    pd.DataFrame(summary).to_csv(sum_path, index=False, encoding="utf-8")
    log(f"{len(summary)} fixes applied. Report: {log_path}\nSummary rows: {sum_path}")
    log("Next: rerun both SNHT scripts.")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n")


def main():
    a, _, b = ARG.partition("-")
    rounds = range(int(a), int(b or a) + 1)
    for r in rounds:
        if r not in FIXES:
            sys.exit(f"Unknown round {r}; available: {sorted(FIXES)}")
    for r in rounds:
        run_round(r)
        print()


if __name__ == "__main__":
    main()
