"""
apply_set_missing.py
--------------------
Option-1 (set-to-missing) treatment: blank out explicit date ranges of a column
for flagged dead-sensor / unusable periods, writing the file's missing marker "-".

Why explicit date ranges (not a threshold): every edit is a documented
[start, end] per station/column, so the change is fully traceable for the
paper's data-availability record, and it can never clip a genuine low day
elsewhere in the series.

Fidelity: every cell is read as a literal string, so ONLY the flagged cells
change -- all other columns and values are written back exactly as they were
(no NaN->blank, no whole-number->"X.0" drift).

Accumulation: if a <station>_corrected.csv already exists it is used as the
input, so multiple fixes to the same station layer cleanly. Re-running is
idempotent.

Place this file in the same folder as snht_TS_ERA5.py, then run:
    python apply_set_missing.py
"""
import os
import pandas as pd
from snht_TS_ERA5 import DATA_DIR

MISSING = "-"

# (raw_file, column, start, end, reason)   -- start/end inclusive
INTERVALS = [
    ("Daily_Luqa.csv", "Temperature", "2015-09-25", "2015-11-30", "cold drift into Dec-2015 sensor failure"),
]


def corrected_name(raw_file):
    base = raw_file[:-4] if raw_file.lower().endswith(".csv") else raw_file
    return base + "_corrected.csv"


def first_real(values):
    """first value that is not blank or the missing marker."""
    for v in values:
        if v not in ("", MISSING):
            return v
    return "n/a"


def main():
    # group edits by file so all edits for one file write a single corrected output
    by_file = {}
    for raw, col, a, b, why in INTERVALS:
        by_file.setdefault(raw, []).append((col, a, b, why))

    for raw_file, edits in by_file.items():
        out_file = corrected_name(raw_file)
        in_path = os.path.join(DATA_DIR, out_file)          # prefer an existing corrected file
        if not os.path.exists(in_path):
            in_path = os.path.join(DATA_DIR, raw_file)       # else start from the raw file
        if not os.path.exists(in_path):
            print(f"SKIP {raw_file}: not found in {DATA_DIR}")
            continue

        df = pd.read_csv(in_path, dtype=str, keep_default_na=False)   # literal strings, no reformatting
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        print(f"\n{raw_file}  (input: {os.path.basename(in_path)})")

        for col, a, b, why in edits:
            if col not in df.columns:
                print(f"  ! '{col}' not in this file -- skipped")
                continue
            in_range = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
            already  = int(df.loc[in_range, col].isin(["", MISSING]).sum())
            before   = first_real(df.loc[dates < pd.Timestamp(a), col].iloc[::-1])
            after    = first_real(df.loc[dates > pd.Timestamp(b), col])
            df.loc[in_range, col] = MISSING
            covered = int(in_range.sum())
            print(f"  {col}: {a} -> {b} | {covered} days covered "
                  f"({covered - already} newly set, {already} already missing) | "
                  f"neighbours: {before} ... {after} | {why}")

        out_path = os.path.join(DATA_DIR, out_file)
        df.to_csv(out_path, index=False)
        print(f"  written: {out_path}")


if __name__ == "__main__":
    main()
